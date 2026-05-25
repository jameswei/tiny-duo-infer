"""
HuggingFace → project weight key mapping and shape validation.

Translates the HuggingFace Llama checkpoint key names to the flat dot-path
keys used by the project module tree. Also validates tensor shapes against
the model config and handles tied embeddings.

HF key mapping for Llama-3.2-1B (12 patterns):

    HF key                                          Project key
    ─────────────────────────────────────────────   ─────────────────────────────────────
    model.embed_tokens.weight                       embed_tokens.weight
    model.layers.{i}.input_layernorm.weight         layers.{i}.input_layernorm.weight
    model.layers.{i}.self_attn.q_proj.weight        layers.{i}.self_attn.q_proj.weight
    model.layers.{i}.self_attn.k_proj.weight        layers.{i}.self_attn.k_proj.weight
    model.layers.{i}.self_attn.v_proj.weight        layers.{i}.self_attn.v_proj.weight
    model.layers.{i}.self_attn.o_proj.weight        layers.{i}.self_attn.o_proj.weight
    model.layers.{i}.post_attention_layernorm.weight layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.mlp.gate_proj.weight           layers.{i}.mlp.gate_proj.weight
    model.layers.{i}.mlp.up_proj.weight             layers.{i}.mlp.up_proj.weight
    model.layers.{i}.mlp.down_proj.weight           layers.{i}.mlp.down_proj.weight
    model.norm.weight                               norm.weight
    lm_head.weight                                  lm_head.weight  (tied = embed_tokens.weight)

Tied embeddings:
    Llama-3.2-1B ties lm_head.weight to embed_tokens.weight. The HF checkpoint
    may omit lm_head.weight entirely or include it as a duplicate. The converter
    ensures lm_head.weight is present in the output dict by copying embed_tokens.weight
    if needed (no extra memory: same underlying array).
"""

from __future__ import annotations

from tiny_duo_infer.config import ModelConfig


def convert(
    hf_weights: dict[str, any],
    config: ModelConfig,
) -> dict[str, any]:
    """
    Translate HF key names to project key names and validate shapes.

    Args:
        hf_weights: flat dict of HF key → mx.array from loader.load_weights().
        config:     model config for shape assertions.

    Returns:
        flat dict of project key → mx.array, ready for LlamaModel.load_weights().

    Raises:
        ValueError: if a required key is missing or a shape does not match config.
    """
    raise NotImplementedError
