"""
HuggingFace → project weight key mapping and shape validation.

Translates the HuggingFace Llama checkpoint key names to the flat dot-path
keys used by the project module tree. Also validates tensor shapes against
the model config and handles tied embeddings.

HF key mapping for Llama-3.2-1B (12 patterns):

    HF key                                          Project key
    ------------------------------------------------ -------------------------------------
    model.embed_tokens.weight                       embed_tokens.weight
    model.layers.{i}.input_layernorm.weight         layers.{i}.input_norm.weight
    model.layers.{i}.self_attn.q_proj.weight        layers.{i}.attn.q_proj.weight
    model.layers.{i}.self_attn.k_proj.weight        layers.{i}.attn.k_proj.weight
    model.layers.{i}.self_attn.v_proj.weight        layers.{i}.attn.v_proj.weight
    model.layers.{i}.self_attn.o_proj.weight        layers.{i}.attn.o_proj.weight
    model.layers.{i}.post_attention_layernorm.weight layers.{i}.post_attn_norm.weight
    model.layers.{i}.mlp.gate_proj.weight           layers.{i}.ffn.gate_proj.weight
    model.layers.{i}.mlp.up_proj.weight             layers.{i}.ffn.up_proj.weight
    model.layers.{i}.mlp.down_proj.weight           layers.{i}.ffn.down_proj.weight
    model.norm.weight                               final_norm.weight
    lm_head.weight                                  lm_head.weight  (tied = embed_tokens.weight)

Tied embeddings:
    Llama-3.2-1B ties lm_head.weight to embed_tokens.weight. The HF checkpoint
    may omit lm_head.weight entirely or include it as a duplicate. The converter
    ensures lm_head.weight is present in the output dict by copying embed_tokens.weight
    if needed (no extra memory: same underlying array).

dot-path keys will be loaded into the module tree, like this:
    LlamaModel
    ├── embed_tokens              → Embedding.weight          (V, D)
    ├── layers[0..15]
    │   ├── input_norm            → RMSNorm.weight            (D,)
    │   ├── attn                  → LlamaAttention
    │   │   ├── q_proj            → Linear.weight             (D, D)
    │   │   ├── k_proj            → Linear.weight             (Hkv*Dh, D)
    │   │   ├── v_proj            → Linear.weight             (Hkv*Dh, D)
    │   │   └── o_proj            → Linear.weight             (D, D)
    │   ├── post_attn_norm        → RMSNorm.weight            (D,)
    │   └── ffn                   → SwiGLUFFN
    │       ├── gate_proj         → Linear.weight             (I, D)
    │       ├── up_proj           → Linear.weight             (I, D)
    │       └── down_proj         → Linear.weight             (D, I)
    ├── final_norm                → RMSNorm.weight            (D,)
    └── lm_head                   → Linear.weight             (V, D)

"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import mlx.core as mx

from tiny_duo_infer.config import ModelConfig


def convert(
    hf_weights: Mapping[str, mx.array],
    config: ModelConfig,
) -> dict[str, mx.array]:
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
    expected = _expected_weight_specs(config)
    converted: dict[str, mx.array] = {}

    for hf_key, tensor in hf_weights.items():
        if hf_key == "lm_head.weight":
            _validate_shape(hf_key, tensor, (config.vocab_size, config.d_model))
            converted["lm_head.weight"] = tensor
            continue

        spec = expected.get(hf_key)
        if spec is None:
            warnings.warn(
                f"unexpected HF weight key ignored: {hf_key}",
                UserWarning,
                stacklevel=2,
            )
            continue

        project_key, expected_shape = spec
        _validate_shape(hf_key, tensor, expected_shape)
        converted[project_key] = tensor

    _fill_tied_lm_head(converted)
    _validate_required_project_keys(converted, expected)
    return converted


def _expected_weight_specs(
    config: ModelConfig,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """
    Build the HF-key to project-key table for one concrete model config.

    The mapping is generated from `n_layers` so missing or extra layer indexes
    are caught deterministically. Shape tuples use HF's stored layout:
    `(out_dim, in_dim)` for matrix weights and `(dim,)` for norm weights.
    """
    specs: dict[str, tuple[str, tuple[int, ...]]] = {
        "model.embed_tokens.weight": (
            "embed_tokens.weight",
            (config.vocab_size, config.d_model),
        ),
        "model.norm.weight": ("final_norm.weight", (config.d_model,)),
    }

    q_shape = (config.d_model, config.d_model)
    kv_shape = (config.n_kv_heads * config.head_dim, config.d_model)
    o_shape = (config.d_model, config.d_model)
    gate_up_shape = (config.intermediate_size, config.d_model)
    down_shape = (config.d_model, config.intermediate_size)

    for layer_idx in range(config.n_layers):
        hf_prefix = f"model.layers.{layer_idx}"
        project_prefix = f"layers.{layer_idx}"
        specs.update(
            {
                f"{hf_prefix}.input_layernorm.weight": (
                    f"{project_prefix}.input_norm.weight",
                    (config.d_model,),
                ),
                f"{hf_prefix}.self_attn.q_proj.weight": (
                    f"{project_prefix}.attn.q_proj.weight",
                    q_shape,
                ),
                f"{hf_prefix}.self_attn.k_proj.weight": (
                    f"{project_prefix}.attn.k_proj.weight",
                    kv_shape,
                ),
                f"{hf_prefix}.self_attn.v_proj.weight": (
                    f"{project_prefix}.attn.v_proj.weight",
                    kv_shape,
                ),
                f"{hf_prefix}.self_attn.o_proj.weight": (
                    f"{project_prefix}.attn.o_proj.weight",
                    o_shape,
                ),
                f"{hf_prefix}.post_attention_layernorm.weight": (
                    f"{project_prefix}.post_attn_norm.weight",
                    (config.d_model,),
                ),
                f"{hf_prefix}.mlp.gate_proj.weight": (
                    f"{project_prefix}.ffn.gate_proj.weight",
                    gate_up_shape,
                ),
                f"{hf_prefix}.mlp.up_proj.weight": (
                    f"{project_prefix}.ffn.up_proj.weight",
                    gate_up_shape,
                ),
                f"{hf_prefix}.mlp.down_proj.weight": (
                    f"{project_prefix}.ffn.down_proj.weight",
                    down_shape,
                ),
            }
        )

    return specs


def _fill_tied_lm_head(converted: dict[str, mx.array]) -> None:
    """
    Reuse token embeddings as lm_head.weight when HF omits a separate head.

    Llama-3.2-1B ties these weights. Assigning the same array object is
    intentional: it records the tied relationship without copying memory.
    """
    if "lm_head.weight" not in converted and "embed_tokens.weight" in converted:
        converted["lm_head.weight"] = converted["embed_tokens.weight"]


def _validate_required_project_keys(
    converted: dict[str, mx.array],
    expected: Mapping[str, tuple[str, tuple[int, ...]]],
) -> None:
    """
    Ensure every required project key exists after mapping and tied-head fill.

    `lm_head.weight` is included here even though it may be synthesized from
    embeddings rather than read directly from the checkpoint.
    """
    required_project_keys = {project_key for project_key, _shape in expected.values()}
    required_project_keys.add("lm_head.weight")

    missing_keys = sorted(required_project_keys - converted.keys())
    if missing_keys:
        preview = ", ".join(missing_keys[:8])
        suffix = "" if len(missing_keys) <= 8 else f", ... ({len(missing_keys)} total)"
        raise ValueError(f"missing required project weight key(s): {preview}{suffix}")


def _validate_shape(
    hf_key: str,
    tensor: Any,
    expected_shape: tuple[int, ...],
) -> None:
    """Raise a clear error if a tensor's stored shape does not match config."""
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"shape mismatch for {hf_key!r}: expected {expected_shape}, got {actual_shape}"
        )
