"""
Weight-only quantization conversion step for the model loading pipeline.

Called after the HF-key converter (llama_converter or qwen3_converter) has
produced the project-key dict, and before the model tree is populated via
load_weights().  This is step 3 of the loading pipeline:

  1. load safetensors
  2. convert HF keys/shapes for Llama or Qwen3
  3. quantize eligible project weights  ← this module
  4. load values into the model tree via model.load_weights()

Only 2-D matrix weights used by Linear projections are eligible.
Embeddings, RMSNorm weights, Qwen3 Q/K norm weights, and any 1-D tensor
stay full precision.
"""

from __future__ import annotations

import mlx.core as mx

from tiny_duo_infer.quantization import QuantizationConfig, QuantizedWeight


_ELIGIBLE_SUFFIXES: tuple[str, ...] = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)

_ELIGIBLE_EXACT: frozenset[str] = frozenset({"lm_head.weight"})


def _is_eligible(key: str, tensor: mx.array) -> bool:
    """Return True if this project-key tensor should be quantized."""
    if tensor.ndim != 2:
        return False
    if key in _ELIGIBLE_EXACT:
        return True
    return any(key.endswith(suffix) for suffix in _ELIGIBLE_SUFFIXES)


def quantize_weights(
    project_weights: dict[str, mx.array],
    config: QuantizationConfig,
) -> dict[str, mx.array | QuantizedWeight]:
    """
    Convert eligible Linear projection weights to QuantizedWeight objects.

    Iterates over the project weight dict produced by a model-family converter.
    Eligible 2-D matrix weights (attention projections, FFN projections,
    lm_head) are replaced with QuantizedWeight objects via mx.quantize().
    All other weights (embeddings, RMSNorm, Qwen3 Q/K norm, 1-D tensors)
    pass through unchanged.

    When Llama's lm_head.weight is tied to embed_tokens.weight (same array
    object), quantizing lm_head.weight replaces only the lm_head entry in
    the output dict.  embed_tokens.weight remains a full-precision mx.array.

    Args:
        project_weights: flat dict of project key → mx.array from a converter.
        config:          quantization config (bits, group_size, mode).

    Returns:
        New dict with the same keys; eligible weights replaced by QuantizedWeight.

    Raises:
        ValueError: if any eligible weight has in_features not divisible by
                    config.group_size.  The message names the key, in_features,
                    and group_size.
    """
    result: dict[str, mx.array | QuantizedWeight] = {}
    for key, tensor in project_weights.items():
        if not _is_eligible(key, tensor):
            result[key] = tensor
            continue

        out_features = int(tensor.shape[0])
        in_features = int(tensor.shape[1])
        if in_features % config.group_size != 0:
            raise ValueError(
                f"quantize_weights: in_features={in_features} for weight {key!r} "
                f"is not divisible by group_size={config.group_size}."
            )

        qweight, scales, biases = mx.quantize(
            tensor, group_size=config.group_size, bits=config.bits
        )
        result[key] = QuantizedWeight(
            qweight=qweight,
            scales=scales,
            biases=biases,
            bits=config.bits,
            group_size=config.group_size,
            mode=config.mode,
            out_features=out_features,
            in_features=in_features,
        )

    return result
