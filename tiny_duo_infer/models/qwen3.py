"""
Qwen3 model assembly: Qwen3Block and Qwen3Model.

Qwen3 follows the same high-level decoder-only transformer structure as Llama:
embedding -> transformer blocks -> final RMSNorm -> lm_head. The key
architecture difference is attention: Qwen3 uses per-head Q/K RMSNorm
before RoPE and can have attention width A = H * Dh that differs from
hidden size D.

Architecture overview:
  Qwen3Model:
    embed_tokens: Embedding
    layers: list of Qwen3Block
    final_norm: RMSNorm
    lm_head: Linear

  Qwen3Block:
    input_norm:     RMSNorm   — pre-norm before attention
    attn:           Qwen3Attention (GQA, Q/K norm, RoPE, KV cache)
    post_attn_norm: RMSNorm   — pre-norm before FFN
    ffn:            SwiGLUFFN

Attribute names match qwen3_converter.py project-key fragments exactly:
  layers.{i}.input_norm.weight, layers.{i}.attn.q_norm.weight, etc.

Forward pass signature:
  Qwen3Model.forward(input_ids, cache, position_offset) -> logits
  Qwen3Block.forward(x, cache, layer_idx, position_offset) -> x
"""

from __future__ import annotations

import mlx.core as mx

from tiny_duo_infer.cache import KVCache
from tiny_duo_infer.config import ModelConfig
from tiny_duo_infer.layers.attention import Qwen3Attention
from tiny_duo_infer.layers.feedforward import SwiGLUFFN
from tiny_duo_infer.layers.normalization import RMSNorm
from tiny_duo_infer.layers.rope import precompute_freqs
from tiny_duo_infer.models.base import Embedding, Linear, Module


class Qwen3Block(Module):
    """
    One Qwen3 transformer block with Qwen3Attention.

    Residual structure is the same pre-norm pattern used by Llama:
        x = x + attn(input_norm(x))
        x = x + ffn(post_attn_norm(x))

    The attention sub-layer is explicit Qwen3Attention so the Q/K norm step is
    visible and easy to compare with LlamaBlock.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        cos_sin: tuple[mx.array, mx.array],
    ) -> None:
        """
        Args:
            config:    model configuration (n_heads, d_model, head_dim, etc.).
            layer_idx: position of this block in the transformer stack.
            cos_sin:   shared RoPE tables precomputed by Qwen3Model.
        """
        self.layer_idx = layer_idx
        self.input_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = Qwen3Attention(config, cos_sin)
        self.post_attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)

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
            cache:           KVCache instance for this request.
            layer_idx:       this block's layer index, passed to cache.update().
            position_offset: absolute position of the first token in this step.
        Returns:
            (B, S, D) output hidden states.
        """
        residual = x
        x = self.attn(self.input_norm(x), cache, layer_idx, position_offset)
        x = residual + x

        residual = x
        x = self.ffn(self.post_attn_norm(x))
        x = residual + x

        return x


class Qwen3Model(Module):
    """
    Full Qwen3 model: embedding -> N Qwen3Blocks -> final norm -> lm_head.

    Forward returns logits (B, S, V) and has the same signature as LlamaModel so
    Engine can dispatch by model class without changing prefill/decode logic.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Construct the Qwen3 model from config. Weights are populated later.

        RoPE tables are precomputed once and shared across Qwen3Blocks.
        """
        cos_sin = precompute_freqs(
            config.head_dim, config.max_seq_len, config.rope_theta
        )

        self.embed_tokens = Embedding(config.vocab_size, config.d_model)
        self.layers = [Qwen3Block(config, i, cos_sin) for i in range(config.n_layers)]
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        input_ids: mx.array,
        cache: KVCache,
        position_offset: int,
    ) -> mx.array:
        """
        Run the full Qwen3 forward pass.

        Args:
            input_ids:       (B, S) integer token IDs.
            cache:           KVCache instance for this request.
            position_offset: absolute position of the first input token.
        Returns:
            (B, S, V) logits for each input position.
        """
        x = self.embed_tokens(input_ids)  # (B, S, D)

        for i, block in enumerate(self.layers):
            x = block(x, cache, layer_idx=i, position_offset=position_offset)

        x = self.final_norm(x)
        return self.lm_head(x)

    def load_weights(self, weights: dict[str, mx.array]) -> None:
        """
        Load flat dot-path weights into Qwen3 sub-modules.

        Mirrors LlamaModel.load_weights because both models store transformer
        blocks in a Python list rather than as named Module attributes.
        """
        layer_weights: dict[int, dict[str, mx.array]] = {}
        other_weights: dict[str, mx.array] = {}

        for key, value in weights.items():
            if key.startswith("layers."):
                _, idx_str, rest = key.split(".", 2)
                idx = int(idx_str)
                layer_weights.setdefault(idx, {})[rest] = value
            else:
                other_weights[key] = value

        if other_weights:
            super().load_weights(other_weights)

        for idx, layer_dict in layer_weights.items():
            self.layers[idx].load_weights(layer_dict)
