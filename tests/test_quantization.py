"""
Tests for tiny_duo_infer.quantization.

Covers QuantizationConfig validation, QuantizedWeight construction and metadata,
and the Phase 1.8 quantization fields added to GenerationStats.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from tiny_duo_infer.quantization import QuantizationConfig, QuantizedWeight
from tiny_duo_infer.generation import GenerationStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stats(**overrides) -> GenerationStats:
    """Minimal valid GenerationStats for testing quantization fields."""
    defaults: dict = dict(
        context_policy="allow_context_stop",
        original_prompt_tokens=5,
        accepted_prompt_tokens=5,
        truncated_prompt_tokens=0,
        rejected_prompt_tokens=0,
        prompt_tokens=5,
        generated_tokens=3,
        stop_reason="eos",
        prompt_prepare_ms=1.0,
        prefill_ms=10.0,
        time_to_first_token_ms=12.0,
        decode_ms=15.0,
        total_ms=27.0,
        decode_tokens_per_sec=200.0,
        kv_cache_allocated_bytes=4096,
        kv_cache_active_bytes=256,
        max_seq_len=64,
        active_seq_len=8,
    )
    defaults.update(overrides)
    return GenerationStats(**defaults)


def _make_quantized_weight(
    out_features: int = 8,
    in_features: int = 64,
    group_size: int = 32,
    bits: int = 4,
) -> QuantizedWeight:
    """Construct a QuantizedWeight from a random matrix via mx.quantize()."""
    w = mx.random.normal(shape=(out_features, in_features))
    qweight, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    return QuantizedWeight(
        qweight=qweight,
        scales=scales,
        biases=biases,
        bits=bits,
        group_size=group_size,
        mode="affine",
        out_features=out_features,
        in_features=in_features,
    )


# ---------------------------------------------------------------------------
# QuantizationConfig — valid construction
# ---------------------------------------------------------------------------


def test_quant_config_bits4_default_group_size():
    cfg = QuantizationConfig(bits=4)
    assert cfg.bits == 4
    assert cfg.group_size == 64
    assert cfg.mode == "affine"


def test_quant_config_bits8_default_group_size():
    cfg = QuantizationConfig(bits=8)
    assert cfg.bits == 8
    assert cfg.group_size == 64
    assert cfg.mode == "affine"


def test_quant_config_custom_group_size():
    cfg = QuantizationConfig(bits=4, group_size=32)
    assert cfg.group_size == 32


def test_quant_config_group_size_1():
    # group_size=1 is the degenerate case (one element per group); valid per spec
    cfg = QuantizationConfig(bits=8, group_size=1)
    assert cfg.group_size == 1


# ---------------------------------------------------------------------------
# QuantizationConfig — validation errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_bits", [2, 3, 6, 16, 32])
def test_quant_config_rejects_invalid_bits(bad_bits):
    with pytest.raises(ValueError, match="bits must be one of"):
        QuantizationConfig(bits=bad_bits)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_group_size", [0, -1, -64])
def test_quant_config_rejects_non_positive_group_size(bad_group_size):
    with pytest.raises(ValueError, match="group_size must be positive"):
        QuantizationConfig(bits=4, group_size=bad_group_size)


@pytest.mark.parametrize("bad_mode", ["symmetric", "fp8", ""])
def test_quant_config_rejects_invalid_mode(bad_mode):
    with pytest.raises(ValueError, match="mode must be one of"):
        QuantizationConfig(bits=4, mode=bad_mode)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# QuantizedWeight — construction and metadata
# ---------------------------------------------------------------------------


def test_quantized_weight_stores_shape_metadata():
    qw = _make_quantized_weight(out_features=8, in_features=64, group_size=32, bits=4)
    assert qw.out_features == 8
    assert qw.in_features == 64


def test_quantized_weight_stores_config_metadata():
    qw = _make_quantized_weight(bits=4, group_size=32)
    assert qw.bits == 4
    assert qw.group_size == 32
    assert qw.mode == "affine"


def test_quantized_weight_stores_int8():
    qw = _make_quantized_weight(out_features=4, in_features=32, group_size=32, bits=8)
    assert qw.bits == 8
    assert qw.out_features == 4
    assert qw.in_features == 32


def test_quantized_weight_arrays_are_mx_arrays():
    qw = _make_quantized_weight()
    assert isinstance(qw.qweight, mx.array)
    assert isinstance(qw.scales, mx.array)
    assert isinstance(qw.biases, mx.array)


def test_quantized_weight_scales_biases_shape():
    # scales and biases shape: (out_features, in_features // group_size)
    out_f, in_f, gs = 8, 64, 32
    qw = _make_quantized_weight(out_features=out_f, in_features=in_f, group_size=gs)
    expected_groups = in_f // gs
    assert qw.scales.shape == (out_f, expected_groups)
    assert qw.biases.shape == (out_f, expected_groups)


# ---------------------------------------------------------------------------
# GenerationStats — new quantization fields (Phase 1.8)
# ---------------------------------------------------------------------------


def test_generation_stats_quantization_fields_have_no_quantization_defaults():
    stats = _make_stats()
    assert stats.quantization_mode == "none"
    assert stats.quantization_bits is None
    assert stats.quantization_group_size is None
    assert stats.quantized_linear_count == 0
    assert stats.full_precision_linear_count == 0
    assert stats.linear_weight_full_precision_bytes == 0
    assert stats.linear_weight_runtime_bytes == 0


def test_generation_stats_runtime_bytes_equals_full_precision_bytes_when_no_quant():
    # For the no-quantization case the two memory fields should be equal.
    stats = _make_stats(
        linear_weight_full_precision_bytes=1024,
        linear_weight_runtime_bytes=1024,
    )
    assert stats.linear_weight_runtime_bytes == stats.linear_weight_full_precision_bytes


def test_generation_stats_accepts_int4_quantization_mode():
    stats = _make_stats(
        quantization_mode="int4",
        quantization_bits=4,
        quantization_group_size=64,
        quantized_linear_count=10,
        full_precision_linear_count=2,
        linear_weight_full_precision_bytes=2048,
        linear_weight_runtime_bytes=512,
    )
    assert stats.quantization_mode == "int4"
    assert stats.quantization_bits == 4
    assert stats.quantization_group_size == 64
    assert stats.quantized_linear_count == 10
    assert stats.full_precision_linear_count == 2
    assert stats.linear_weight_runtime_bytes < stats.linear_weight_full_precision_bytes


def test_generation_stats_accepts_int8_quantization_mode():
    stats = _make_stats(
        quantization_mode="int8",
        quantization_bits=8,
        quantization_group_size=32,
    )
    assert stats.quantization_mode == "int8"
    assert stats.quantization_bits == 8


def test_generation_stats_accepts_none_quantization_mode_explicitly():
    stats = _make_stats(quantization_mode="none")
    assert stats.quantization_mode == "none"


def test_generation_stats_rejects_invalid_quantization_mode():
    with pytest.raises(ValueError, match="quantization_mode must be one of"):
        _make_stats(quantization_mode="fp8")


def test_generation_stats_rejects_empty_quantization_mode():
    with pytest.raises(ValueError, match="quantization_mode must be one of"):
        _make_stats(quantization_mode="")


# ---------------------------------------------------------------------------
# GenerationStats — quantization coherence invariants
# ---------------------------------------------------------------------------


def test_generation_stats_rejects_int4_mode_with_bits8():
    with pytest.raises(ValueError, match="quantization_bits must be 4"):
        _make_stats(
            quantization_mode="int4",
            quantization_bits=8,
            quantization_group_size=64,
        )


def test_generation_stats_rejects_int8_mode_with_bits4():
    with pytest.raises(ValueError, match="quantization_bits must be 8"):
        _make_stats(
            quantization_mode="int8",
            quantization_bits=4,
            quantization_group_size=64,
        )


def test_generation_stats_rejects_int4_mode_with_bits_none():
    with pytest.raises(ValueError, match="quantization_bits must be 4"):
        _make_stats(
            quantization_mode="int4",
            quantization_bits=None,
            quantization_group_size=64,
        )


def test_generation_stats_rejects_int4_mode_with_group_size_none():
    with pytest.raises(ValueError, match="quantization_group_size must be a positive integer"):
        _make_stats(
            quantization_mode="int4",
            quantization_bits=4,
            quantization_group_size=None,
        )


def test_generation_stats_rejects_int8_mode_with_group_size_zero():
    with pytest.raises(ValueError, match="quantization_group_size must be a positive integer"):
        _make_stats(
            quantization_mode="int8",
            quantization_bits=8,
            quantization_group_size=0,
        )


def test_generation_stats_rejects_none_mode_with_bits_set():
    with pytest.raises(ValueError, match="quantization_bits must be None"):
        _make_stats(quantization_mode="none", quantization_bits=4)


def test_generation_stats_rejects_none_mode_with_group_size_set():
    with pytest.raises(ValueError, match="quantization_group_size must be None"):
        _make_stats(quantization_mode="none", quantization_group_size=64)


def test_generation_stats_rejects_none_mode_with_nonzero_quantized_count():
    with pytest.raises(ValueError, match="quantized_linear_count must be 0"):
        _make_stats(quantization_mode="none", quantized_linear_count=5)


@pytest.mark.parametrize("field,value", [
    ("quantized_linear_count", -1),
    ("full_precision_linear_count", -1),
    ("linear_weight_full_precision_bytes", -1),
    ("linear_weight_runtime_bytes", -1),
])
def test_generation_stats_rejects_negative_counts_and_bytes(field, value):
    with pytest.raises(ValueError, match=f"{field} must be >= 0"):
        _make_stats(**{field: value})
