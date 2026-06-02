"""
Grouped Query Attention (GQA) with RoPE and KV cache.

Two attention classes are defined here:
  - LlamaAttention: standard GQA, no Q/K norm.
  - Qwen3Attention: GQA with per-head Q/K RMSNorm applied before RoPE.

Llama-3.2-1B uses Grouped Query Attention (GQA) where n_heads=32 query heads
share n_kv_heads=8 key/value heads in groups of n_groups=4.

GQA reduces KV cache memory by n_groups times: instead of storing 32 K and V heads
per token per layer, only 8 are stored. At attention time, each KV head is
repeated n_groups times to match the Q head count before the matmul.

LlamaAttention computation (A == D for Llama):
  1. Q, K, V projections:  (B, S, D) → (B, S, H, Dh), (B, S, Hkv, Dh), (B, S, Hkv, Dh)
  2. Apply RoPE to Q and K (offset by position_offset)
  3. Write K, V to the KV cache at position_offset; read back the full valid slice
  4. Transpose Q, K_full, V_full for matmul: (B, H/Hkv, S/T, Dh)
  5. Repeat K and V heads: (B, Hkv, T, Dh) → (B, H, T, Dh)  [mx.repeat, axis=1]
  6. Attention scores: Q @ K.T / sqrt(Dh) → (B, H, S, T)
  7. Apply causal mask: tokens at position i can only attend to positions <= i
  8. Softmax: (B, H, S, T)
  9. Weighted sum: attn_weights @ V → (B, H, S, Dh)
  10. Merge heads: (B, S, A)
  11. Output projection: (B, S, D)

Qwen3Attention adds Q/K RMSNorm between steps 1 and 2 (after reshape, before RoPE).
For Qwen3-0.6B, A = H * Dh = 2048 != D = 1024.

Causal mask:
  During prefill (S > 1): a lower-triangular mask prevents attending to future
  tokens. Position i attends to positions 0..i only.
  During decode (S = 1): the single new token attends to all T cached positions;
  no masking needed (all positions are in the past).
"""

from __future__ import annotations

import math

import mlx.core as mx

from tiny_duo_infer.cache import KVCache
from tiny_duo_infer.config import ModelConfig
from tiny_duo_infer.layers.normalization import RMSNorm
from tiny_duo_infer.layers.rope import apply_rope
from tiny_duo_infer.models.base import Linear, Module


class LlamaAttention(Module):
    """
    Grouped Query Attention with RoPE and KV cache.

    Implements the full attention mechanism for one transformer layer.
    Uses pre-computed RoPE tables (passed in from LlamaModel) and writes
    to the shared KVCache via the update() protocol.

    Attributes:
        q_proj:  Linear (D → H * Dh)
        k_proj:  Linear (D → Hkv * Dh)
        v_proj:  Linear (D → Hkv * Dh)
        o_proj:  Linear (H * Dh → D)
        cos_sin: (cos_table, sin_table) from precompute_freqs — shared ref.
    """

    def __init__(self, config: ModelConfig, cos_sin: tuple[mx.array, mx.array]) -> None:
        """
        Args:
            config:  model config (n_heads, n_kv_heads, d_model, head_dim).
            cos_sin: (cos_table, sin_table) precomputed by LlamaModel at init.
        """
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.cos_sin = cos_sin

        self.q_proj = Linear(config.d_model, config.n_heads * config.head_dim)
        self.k_proj = Linear(config.d_model, config.n_kv_heads * config.head_dim)
        self.v_proj = Linear(config.d_model, config.n_kv_heads * config.head_dim)
        self.o_proj = Linear(config.n_heads * config.head_dim, config.d_model)

    def forward(
        self,
        x: mx.array,
        cache: KVCache,
        layer_idx: int,
        position_offset: int,
    ) -> mx.array:
        """
        Args:
            x:               (B, S, D) input hidden states.
            cache:           KVCache for this request.
            layer_idx:       index of this layer, passed to cache.update().
            position_offset: absolute position of x[:, 0, :].
                             0 during prefill; cache.current_len during decode.
        Returns:
            (B, S, D) attention output.
        """
        B, S, _D = x.shape
        H, Hkv, Dh = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).reshape(B, S, H, Dh)  # (B, S, H, Dh)
        k = self.k_proj(x).reshape(B, S, Hkv, Dh)  # (B, S, Hkv, Dh)
        v = self.v_proj(x).reshape(B, S, Hkv, Dh)  # (B, S, Hkv, Dh)

        cos, sin = self.cos_sin
        q = apply_rope(q, cos, sin, offset=position_offset)  # (B, S, H, Dh)
        k = apply_rope(k, cos, sin, offset=position_offset)  # (B, S, Hkv, Dh)

        # KV cache stores transposed K/V as (B, Hkv, T, Dh). update() writes
        # this layer's new positions and returns the full valid prefix.
        new_k = mx.transpose(k, (0, 2, 1, 3))  # (B, Hkv, S, Dh)
        new_v = mx.transpose(v, (0, 2, 1, 3))  # (B, Hkv, S, Dh)
        k_full, v_full = cache.update(
            layer_idx, new_k, new_v, position_offset
        )  # (B, Hkv, T, Dh)
        T = k_full.shape[2]

        q_t = mx.transpose(q, (0, 2, 1, 3))  # (B, H, S, Dh)

        # GQA expands each KV head along the head axis so every group of query
        # heads attends to the KV head it shares in the original Llama layout.
        k_expanded = mx.repeat(k_full, repeats=self.n_groups, axis=1)  # (B, H, T, Dh)

        scores = (q_t @ mx.transpose(k_expanded, (0, 1, 3, 2))) / math.sqrt(Dh)
        scores = _apply_causal_mask(scores, position_offset)
        weights = mx.softmax(scores, axis=-1)  # (B, H, S, T)

        v_expanded = mx.repeat(v_full, repeats=self.n_groups, axis=1)  # (B, H, T, Dh)

        attended = weights @ v_expanded  # (B, H, S, Dh)
        merged = mx.transpose(attended, (0, 2, 1, 3)).reshape(B, S, H * Dh)
        return self.o_proj(merged)


class Qwen3Attention(Module):
    """
    Qwen3 Grouped Query Attention with Q/K RMSNorm before RoPE.

    Identical to LlamaAttention except for per-head Q/K normalization.
    The norm is applied after projection and head reshape, before RoPE rotation.
    Applying Q/K norm after RoPE changes the attention scores and is incorrect
    for Qwen3.

    Qwen3-0.6B has H=16, Dh=128, so the attention projection width
    A = H * Dh = 2048 != D = 1024. q_proj and o_proj use A, not D.

    Attributes:
        q_proj:  Linear (D → A)
        k_proj:  Linear (D → Hkv * Dh)
        v_proj:  Linear (D → Hkv * Dh)
        o_proj:  Linear (A → D)
        q_norm:  RMSNorm(Dh) — one shared weight applied to each query head
        k_norm:  RMSNorm(Dh) — one shared weight applied to each key head
        cos_sin: (cos_table, sin_table) from precompute_freqs — shared ref.
    """

    def __init__(self, config: ModelConfig, cos_sin: tuple[mx.array, mx.array]) -> None:
        """
        Args:
            config:  model config (n_heads, n_kv_heads, d_model, head_dim).
            cos_sin: (cos_table, sin_table) precomputed by Qwen3Model at init.
        """
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.cos_sin = cos_sin

        self.q_proj = Linear(config.d_model, config.n_heads * config.head_dim)
        self.k_proj = Linear(config.d_model, config.n_kv_heads * config.head_dim)
        self.v_proj = Linear(config.d_model, config.n_kv_heads * config.head_dim)
        self.o_proj = Linear(config.n_heads * config.head_dim, config.d_model)

        # Per-head Q/K normalization. Weight shape is (Dh,) — shared across all
        # heads. Applied independently to each head's Dh vector before RoPE.
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: mx.array,
        cache: KVCache,
        layer_idx: int,
        position_offset: int,
    ) -> mx.array:
        """
        Args:
            x:               (B, S, D) input hidden states.
            cache:           KVCache for this request.
            layer_idx:       index of this layer, passed to cache.update().
            position_offset: absolute position of x[:, 0, :].
                             0 during prefill; cache.current_len during decode.
        Returns:
            (B, S, D) attention output.
        """
        B, S, _D = x.shape
        H, Hkv, Dh = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).reshape(B, S, H, Dh)  # (B, S, H, Dh)
        k = self.k_proj(x).reshape(B, S, Hkv, Dh)  # (B, S, Hkv, Dh)
        v = self.v_proj(x).reshape(B, S, Hkv, Dh)  # (B, S, Hkv, Dh)

        # Q/K norm must come before RoPE. Applying it after RoPE changes the
        # attention scores and is incorrect for Qwen3.
        q = self.q_norm(q)  # (B, S, H, Dh)
        k = self.k_norm(k)  # (B, S, Hkv, Dh)

        cos, sin = self.cos_sin
        q = apply_rope(q, cos, sin, offset=position_offset)
        k = apply_rope(k, cos, sin, offset=position_offset)

        new_k = mx.transpose(k, (0, 2, 1, 3))  # (B, Hkv, S, Dh)
        new_v = mx.transpose(v, (0, 2, 1, 3))  # (B, Hkv, S, Dh)
        k_full, v_full = cache.update(layer_idx, new_k, new_v, position_offset)
        T = k_full.shape[2]

        q_t = mx.transpose(q, (0, 2, 1, 3))  # (B, H, S, Dh)

        k_expanded = mx.repeat(k_full, repeats=self.n_groups, axis=1)  # (B, H, T, Dh)
        v_expanded = mx.repeat(v_full, repeats=self.n_groups, axis=1)  # (B, H, T, Dh)

        scores = (q_t @ mx.transpose(k_expanded, (0, 1, 3, 2))) / math.sqrt(Dh)
        scores = _apply_causal_mask(scores, position_offset)
        weights = mx.softmax(scores, axis=-1)  # (B, H, S, T)

        attended = weights @ v_expanded  # (B, H, S, Dh)
        merged = mx.transpose(attended, (0, 2, 1, 3)).reshape(B, S, H * Dh)  # (B, S, A)
        return self.o_proj(merged)  # (B, S, D)


def _apply_causal_mask(scores: mx.array, position_offset: int) -> mx.array:
    """
    Mask attention scores so each query position sees only past/current keys.

    Args:
        scores:          (B, H, S, T) raw attention scores.
        position_offset: absolute position of the first query token.

    Returns:
        (B, H, S, T) scores with future-key positions set to a large negative
        value before softmax.
    """
    _B, _H, S, T = scores.shape
    query_positions = position_offset + mx.arange(S)  # (S,)
    key_positions = mx.arange(T)  # (T,)
    future_mask = key_positions[None, :] > query_positions[:, None]  # (S, T)
    future_mask = future_mask.reshape(1, 1, S, T)
    return mx.where(future_mask, mx.array(-1e9, dtype=scores.dtype), scores)
