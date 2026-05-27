"""
HuggingFace → project weight key mapping for Qwen3 checkpoints.

Qwen3 uses the same broad decoder-only transformer layout as Llama, with two
important converter-level differences:

1. Attention projection width is `A = n_heads * head_dim`, which may differ
   from hidden size `D`.
2. Each attention layer has per-head Q/K RMSNorm weights:
   `self_attn.q_norm.weight` and `self_attn.k_norm.weight`, each shaped `(Dh,)`.

Unlike the Llama converter, Qwen3-0.6B support treats `lm_head.weight` as a
required checkpoint tensor. Do not synthesize it from `embed_tokens.weight`;
doing so would hide incomplete Qwen3 model artifacts.
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
    Translate Qwen3 HuggingFace key names to project key names and validate shapes.

    Args:
        hf_weights: flat dict of HF key → mx.array from loader.load_weights().
        config:     Qwen3 model config for shape assertions.

    Returns:
        flat dict of project key → mx.array, ready for Qwen3Model.load_weights().

    Raises:
        ValueError: if config is not Qwen3, a required key is missing, or a
            tensor shape does not match the config.
    """
    if config.model_type != "qwen3":
        raise ValueError(
            f"qwen3_converter requires model_type 'qwen3', got {config.model_type!r}"
        )

    expected = _expected_weight_specs(config)
    converted: dict[str, mx.array] = {}

    for hf_key, tensor in hf_weights.items():
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

    _validate_required_project_keys(converted, expected)
    return converted


def _expected_weight_specs(
    config: ModelConfig,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """
    Build the Qwen3 HF-key to project-key table for one concrete config.

    Matrix weights use HuggingFace's stored layout `(out_dim, in_dim)`.
    Qwen3 Q/K norm weights use one `(head_dim,)` vector shared across heads.
    """
    attention_width = config.n_heads * config.head_dim
    kv_width = config.n_kv_heads * config.head_dim

    specs: dict[str, tuple[str, tuple[int, ...]]] = {
        "model.embed_tokens.weight": (
            "embed_tokens.weight",
            (config.vocab_size, config.d_model),
        ),
        "model.norm.weight": ("final_norm.weight", (config.d_model,)),
        "lm_head.weight": ("lm_head.weight", (config.vocab_size, config.d_model)),
    }

    q_shape = (attention_width, config.d_model)
    kv_shape = (kv_width, config.d_model)
    o_shape = (config.d_model, attention_width)
    gate_up_shape = (config.intermediate_size, config.d_model)
    down_shape = (config.d_model, config.intermediate_size)
    qk_norm_shape = (config.head_dim,)

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
                f"{hf_prefix}.self_attn.q_norm.weight": (
                    f"{project_prefix}.attn.q_norm.weight",
                    qk_norm_shape,
                ),
                f"{hf_prefix}.self_attn.k_norm.weight": (
                    f"{project_prefix}.attn.k_norm.weight",
                    qk_norm_shape,
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


def _validate_required_project_keys(
    converted: dict[str, mx.array],
    expected: Mapping[str, tuple[str, tuple[int, ...]]],
) -> None:
    """Ensure every required Qwen3 project key exists after mapping."""
    required_project_keys = {project_key for project_key, _shape in expected.values()}
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
    """Raise a clear error if a Qwen3 tensor's stored shape does not match config."""
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"shape mismatch for {hf_key!r}: expected {expected_shape}, got {actual_shape}"
        )
