"""
Weight-only quantization config and quantized weight representation.

Phase 1.8 adds MLX-native INT4 and INT8 weight-only quantization for Linear
layers.  Only matrix weights (attention projections, FFN projections, lm_head)
are quantized; activations, embeddings, norms, and KV-cache buffers remain
full precision.

Two public types are defined here:

  QuantizationConfig — user-facing config that selects bits, group size, and
                        mode; validated at construction; passed to
                        Engine.from_model_path().

  QuantizedWeight    — internal representation of a single quantized Linear
                        weight; stores the packed weight data returned by
                        mx.quantize() together with shape metadata so every
                        call site is self-describing.  Consumed by
                        Linear.forward() via mx.quantized_matmul().

The normal runtime path for a quantized projection is:

    y = mx.quantized_matmul(
        x,
        qw.qweight, qw.scales, qw.biases,
        transpose=True,
        group_size=qw.group_size,
        bits=qw.bits,
        mode=qw.mode,
    )

mx.dequantize() is intentionally NOT used in the normal path — it would
convert weights back to full precision, eliminating the memory benefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx


_VALID_BITS: frozenset[int] = frozenset({4, 8})
_VALID_MODES: frozenset[str] = frozenset({"affine"})


@dataclass
class QuantizationConfig:
    """User-facing quantization configuration for weight-only INT4/INT8 quantization.

    Passed to Engine.from_model_path() to enable in-memory quantization of
    eligible Linear weights at load time.  Full-precision loading remains the
    default when no config is provided (quantization=None).

    Affine quantization maps each group of in_features values to an integer
    range using per-group scale and bias:
        q = round((x - bias) / scale)
    The fused mx.quantized_matmul() reconstructs the full-precision product
    without materialising full-precision weights.

    Attributes:
        bits:       Bit width.  Must be 4 or 8.
        group_size: Elements per quantization group along the input dimension.
                    Each group gets its own scale and bias.  Must be positive
                    and must evenly divide in_features for every eligible matrix.
                    Default 64 matches the MLX convention and works for all
                    real Llama-3.2-1B and Qwen3-0.6B linear dimensions.
                    Tiny test fixtures with in_features=32 must use group_size=32
                    (or another divisor of 32).
        mode:       Quantization scheme.  Phase 1.8 supports "affine" only.
    """

    bits: Literal[4, 8]
    group_size: int = 64
    mode: Literal["affine"] = "affine"

    def __post_init__(self) -> None:
        if self.bits not in _VALID_BITS:
            raise ValueError(
                f"QuantizationConfig bits must be one of {sorted(_VALID_BITS)}, "
                f"got {self.bits}."
            )
        if self.group_size <= 0:
            raise ValueError(
                f"QuantizationConfig group_size must be positive, got {self.group_size}."
            )
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"QuantizationConfig mode must be one of {sorted(_VALID_MODES)!r}, "
                f"got {self.mode!r}."
            )


@dataclass
class QuantizedWeight:
    """Quantized representation of a single Linear weight matrix.

    Produced by calling mx.quantize() on a full-precision (out, in) weight
    during model loading, and consumed by Linear.forward() via
    mx.quantized_matmul().

    Storing a QuantizedWeight on Linear.weight instead of a plain mx.array
    is the signal that the quantized forward path should be used — Linear
    inspects the type of self.weight to choose the path.

    MLX affine quantization packs multiple values per 32-bit word:
      - qweight shape: (out_features, in_features * bits // 32)
      - scales shape:  (out_features, in_features // group_size)
      - biases shape:  (out_features, in_features // group_size)

    All three arrays are returned directly by mx.quantize() and stored here
    without modification.

    Attributes:
        qweight:      Packed quantized weight array from mx.quantize().
        scales:       Per-group scale factors; shape (out_features, in_features // group_size).
        biases:       Per-group zero-point biases; same shape as scales.
        bits:         Bit width (4 or 8).
        group_size:   Input-dimension elements per quantization group.
        mode:         Quantization mode ("affine").
        out_features: Original weight shape[0] — output dimension of the projection.
        in_features:  Original weight shape[1] — input dimension of the projection.
    """

    qweight: mx.array
    scales: mx.array
    biases: mx.array
    bits: int
    group_size: int
    mode: str
    out_features: int
    in_features: int
