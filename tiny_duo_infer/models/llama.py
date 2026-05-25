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

from tiny_duo_infer.config import ModelConfig
from tiny_duo_infer.models.base import Module


class LlamaBlock(Module):
    """
    One transformer block: pre-norm → attention → residual → pre-norm → FFN → residual.

    Attributes:
        input_norm:     RMSNorm applied before attention.
        attn:           LlamaAttention (GQA, RoPE, KV cache).
        post_attn_norm: RMSNorm applied before the FFN.
        ffn:            SwiGLUFFN.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        """
        Args:
            config:    model configuration (n_heads, d_model, etc.).
            layer_idx: which layer index this block occupies (0-indexed).
                       Passed through to LlamaAttention for KV cache writes.
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
            cache:           KVCache instance for this request.
            layer_idx:       this block's layer index, passed to cache.update().
            position_offset: absolute position of the first token in this step.
                             0 during prefill; current_len during decode.
        Returns:
            (B, S, D) output hidden states.
        """
        raise NotImplementedError


class LlamaModel(Module):
    """
    Full Llama model: embedding → N transformer blocks → final norm → lm_head.

    Attributes:
        embed_tokens: Embedding (vocab_size, d_model).
        layers:       list of LlamaBlock, one per transformer layer.
        final_norm:   RMSNorm applied before lm_head (project key: final_norm.weight).
        lm_head:      Linear (vocab_size, d_model) — weight tied to embed_tokens.

    Forward returns logits (B, S, V) for all positions in the sequence.
    During decode, S=1 so logits are (B, 1, V); only logits[0, 0, :] is used.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Construct the model from config. Weights are all uninitialised until
        load_weights() is called.
        """
        raise NotImplementedError

    def forward(
        self,
        input_ids: any,
        cache: any,
        position_offset: int,
    ) -> any:
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
        raise NotImplementedError
