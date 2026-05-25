"""
Grouped Query Attention (GQA) with RoPE and KV cache.

Llama-3.2-1B uses Grouped Query Attention (GQA) where n_heads=32 query heads
share n_kv_heads=8 key/value heads in groups of n_groups=4.

GQA reduces KV cache memory by n_groups×: instead of storing 32 K and V heads
per token per layer, only 8 are stored. At attention time, each KV head is
repeated n_groups times to match the Q head count before the matmul.

Attention computation:
  1. Q, K, V projections:  (B, S, D) → (B, S, H, Dh), (B, S, Hkv, Dh), (B, S, Hkv, Dh)
  2. Apply RoPE to Q and K (offset by position_offset)
  3. Write K, V to the KV cache at position_offset; read back the full valid slice
  4. Transpose Q, K_full, V_full for matmul: (B, H/Hkv, S/T, Dh)
  5. Repeat K and V heads: (B, Hkv, T, Dh) → (B, H, T, Dh)  [mx.repeat, axis=1]
  6. Attention scores: Q @ K.T / sqrt(Dh) → (B, H, S, T)
  7. Apply causal mask: tokens at position i can only attend to positions <= i
  8. Softmax: (B, H, S, T)
  9. Weighted sum: attn_weights @ V → (B, H, S, Dh)
  10. Merge heads: (B, S, D)
  11. Output projection: (B, S, D)

Causal mask:
  During prefill (S > 1): a lower-triangular mask prevents attending to future
  tokens. Position i attends to positions 0..i only.
  During decode (S = 1): the single new token attends to all T cached positions;
  no masking needed (all positions are in the past).
"""

from __future__ import annotations

from tiny_duo_infer.config import ModelConfig
from tiny_duo_infer.models.base import Module


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

    def __init__(self, config: ModelConfig, cos_sin: tuple[any, any]) -> None:
        """
        Args:
            config:  model config (n_heads, n_kv_heads, d_model, head_dim).
            cos_sin: (cos_table, sin_table) precomputed by LlamaModel at init.
        """
        raise NotImplementedError

    def forward(
        self,
        x: any,
        cache: any,
        layer_idx: int,
        position_offset: int,
    ) -> any:
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
        raise NotImplementedError
