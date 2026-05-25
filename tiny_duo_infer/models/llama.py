"""
Llama model assembly: LlamaBlock and LlamaModel.

Assembles the individual layer implementations (RMSNorm, LlamaAttention,
SwiGLUFFN) into the full Llama-3.2-1B transformer architecture.

Architecture overview:
  LlamaModel:
    embed_tokens: Embedding
    layers: list of LlamaBlock
    final_norm: RMSNorm  (final layer norm)
    lm_head: Linear  (tied to embed_tokens.weight in Llama-3.2-1B)

  LlamaBlock (one transformer layer):
    input_norm:     RMSNorm   — pre-norm before attention
    attn:           LlamaAttention (GQA, RoPE, KV cache)
    post_attn_norm: RMSNorm   — pre-norm before FFN
    ffn:            SwiGLUFFN

  Attribute names match llama_converter.py project-key fragments exactly:
    layers.{i}.input_norm.weight, layers.{i}.attn.q_proj.weight, etc.

  Residual connections: Llama uses pre-norm, so the residual bypasses the norm.
      x = x + attn(norm1(x))
      x = x + ffn(norm2(x))

Forward pass signature:
  LlamaModel.forward(input_ids, cache, position_offset) -> logits
  LlamaBlock.forward(x, cache, layer_idx, position_offset) -> x
"""

from __future__ import annotations

import mlx.core as mx

from tiny_duo_infer.cache import KVCache
from tiny_duo_infer.config import ModelConfig
from tiny_duo_infer.layers.attention import LlamaAttention
from tiny_duo_infer.layers.feedforward import SwiGLUFFN
from tiny_duo_infer.layers.normalization import RMSNorm
from tiny_duo_infer.layers.rope import precompute_freqs
from tiny_duo_infer.models.base import Embedding, Linear, Module


class LlamaBlock(Module):
    """
    One transformer block: pre-norm → attention → residual → pre-norm → FFN → residual.

    Llama uses pre-normalization: the residual bypasses the norm entirely.
        x = x + attn(input_norm(x))        — attention sub-layer
        x = x + ffn(post_attn_norm(x))     — FFN sub-layer

    Attributes:
        input_norm:     RMSNorm applied before attention.
        attn:           LlamaAttention (GQA, RoPE, KV cache).
        post_attn_norm: RMSNorm applied before the FFN.
        ffn:            SwiGLUFFN.
        layer_idx:      this block's position in the stack (0-indexed).
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        cos_sin: tuple[mx.array, mx.array],
    ) -> None:
        """
        Args:
            config:    model configuration (n_heads, d_model, etc.).
            layer_idx: position of this block in the transformer stack (0-indexed).
            cos_sin:   (cos_table, sin_table) precomputed RoPE tables from LlamaModel.
                       All blocks share the same tables — passed by reference.
        """
        self.layer_idx = layer_idx
        self.input_norm     = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn           = LlamaAttention(config, cos_sin)
        self.post_attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn            = SwiGLUFFN(config)

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
                             0 during prefill; current_len during decode.
        Returns:
            (B, S, D) output hidden states.
        """
        # Attention sub-layer: pre-norm, then residual connection
        residual = x
        x = self.attn(self.input_norm(x), cache, layer_idx, position_offset)
        x = residual + x  # (B, S, D)

        # FFN sub-layer: pre-norm, then residual connection
        residual = x
        x = self.ffn(self.post_attn_norm(x))
        x = residual + x  # (B, S, D)

        return x


class LlamaModel(Module):
    """
    Full Llama model: embedding → N transformer blocks → final norm → lm_head.

    Attributes:
        embed_tokens: Embedding (vocab_size, d_model).
        layers:       list of LlamaBlock, one per transformer layer.
        final_norm:   RMSNorm applied before lm_head (project key: final_norm.weight).
        lm_head:      Linear (d_model → vocab_size) — weight tied to embed_tokens in
                      Llama-3.2-1B (handled by llama_converter, not here).

    Forward returns logits (B, S, V) for all positions in the sequence.
    During decode, S=1 so logits are (B, 1, V); only logits[0, 0, :] is used.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Construct the model from config. Weights are uninitialised until
        load_weights() is called.

        RoPE tables are precomputed here once and shared across all LlamaBlocks.
        This avoids recomputing the same sin/cos arrays n_layers times.

        Args:
            config: model configuration (n_layers, d_model, vocab_size, etc.).
        """
        # Precompute RoPE tables once; all blocks share this reference
        cos_sin = precompute_freqs(config.head_dim, config.max_seq_len, config.rope_theta)

        self.embed_tokens = Embedding(config.vocab_size, config.d_model)
        self.layers       = [LlamaBlock(config, i, cos_sin) for i in range(config.n_layers)]
        self.final_norm   = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head      = Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        input_ids: mx.array,
        cache: KVCache,
        position_offset: int,
    ) -> mx.array:
        """
        Run the full model forward pass.

        Args:
            input_ids:       (B, S) integer token IDs.
            cache:           KVCache instance for this request.
            position_offset: absolute position of the first input token.
                             0 during prefill; cache.current_len during decode.
        Returns:
            (B, S, V) logits for each input position.
        """
        # Token embedding: integer IDs → dense vectors
        x = self.embed_tokens(input_ids)  # (B, S, D)

        # Transformer blocks — each block reads and writes the KV cache
        for i, block in enumerate(self.layers):
            x = block(x, cache, layer_idx=i, position_offset=position_offset)
            # (B, S, D) preserved through every block

        # Final pre-lm-head normalisation
        x = self.final_norm(x)  # (B, S, D)

        # Project to vocabulary logits
        logits = self.lm_head(x)  # (B, S, V)

        return logits

    def load_weights(self, weights: dict[str, mx.array]) -> None:
        """
        Load weights by routing flat dot-path keys to sub-modules.

        Overrides Module.load_weights to handle `self.layers` being a list
        rather than a named Module attribute. All other keys (embed_tokens.*,
        final_norm.*, lm_head.*) are routed by the standard Module logic.

        Key format for layer weights: 'layers.{i}.{rest}'
            e.g. 'layers.0.attn.q_proj.weight'
        """
        layer_weights: dict[int, dict[str, mx.array]] = {}
        other_weights: dict[str, mx.array] = {}

        for key, value in weights.items():
            if key.startswith("layers."):
                # Split 'layers.{i}.{rest}' → index i, remainder rest
                _, idx_str, rest = key.split(".", 2)
                idx = int(idx_str)
                layer_weights.setdefault(idx, {})[rest] = value
            else:
                other_weights[key] = value

        # Route embed_tokens.*, final_norm.*, lm_head.* through standard Module routing
        if other_weights:
            super().load_weights(other_weights)

        # Route each layer's weights to the corresponding LlamaBlock
        for idx, layer_dict in layer_weights.items():
            self.layers[idx].load_weights(layer_dict)
