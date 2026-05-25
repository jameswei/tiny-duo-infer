"""
Tests for individual Llama layer implementations.

All tests use TINY_CONFIG (from conftest.py) with randomly initialised weights.
No real model artifacts are required.

Test categories:
  - RMSNorm: output shape, normalisation formula, weight scaling
  - RoPE: precompute_freqs output shape, apply_rope output shape, offset handling
  - LlamaAttention: output shape, KV cache writes, causal mask (prefill), no mask (decode)
  - SwiGLUFFN: output shape, gate × up element-wise, down projection
"""

import math

import mlx.core as mx
import pytest

from tiny_duo_infer.layers.normalization import RMSNorm
from tiny_duo_infer.layers.rope import apply_rope, precompute_freqs


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

def test_rmsnorm_init_records_shape_and_eps(tiny_config):
    """RMSNorm records D and eps; checkpoint loading fills weight later."""
    layer = RMSNorm(
        d_model=tiny_config["d_model"],
        eps=tiny_config["rms_norm_eps"],
    )
    assert layer.d_model == tiny_config["d_model"]
    assert layer.eps == tiny_config["rms_norm_eps"]
    assert layer.weight is None


def test_rmsnorm_forward_shape(tiny_config):
    """RMSNorm preserves the common transformer shape: (B, S, D) → (B, S, D)."""
    d_model = tiny_config["d_model"]
    layer = RMSNorm(d_model=d_model, eps=tiny_config["rms_norm_eps"])
    layer.weight = mx.ones((d_model,))
    x = mx.random.normal((1, 5, d_model))

    out = layer(x)

    assert out.shape == (1, 5, d_model)


def test_rmsnorm_forward_matches_manual_formula():
    """Verify y = x / sqrt(mean(x^2) + eps) * weight by hand."""
    layer = RMSNorm(d_model=2, eps=0.0)
    layer.weight = mx.array([1.0, 1.0])
    x = mx.array([[[3.0, 4.0]]])  # mean square = (9 + 16) / 2 = 12.5
    scale = 1.0 / (12.5 ** 0.5)
    expected = mx.array([[[3.0 * scale, 4.0 * scale]]])

    out = layer(x)
    mx.eval(out, expected)

    assert mx.allclose(out, expected, atol=1e-6).item()


def test_rmsnorm_applies_per_channel_weight():
    """The learned weight scales each hidden channel after RMS normalization."""
    layer = RMSNorm(d_model=2, eps=0.0)
    layer.weight = mx.array([2.0, 0.5])
    x = mx.array([[[3.0, 4.0]]])
    scale = 1.0 / (12.5 ** 0.5)
    expected = mx.array([[[3.0 * scale * 2.0, 4.0 * scale * 0.5]]])

    out = layer(x)
    mx.eval(out, expected)

    assert mx.allclose(out, expected, atol=1e-6).item()


def test_rmsnorm_weight_set_via_load_weights(tiny_config):
    """load_weights() assigns the scale parameter used by forward()."""
    layer = RMSNorm(
        d_model=tiny_config["d_model"],
        eps=tiny_config["rms_norm_eps"],
    )
    weight = mx.random.normal((tiny_config["d_model"],))

    layer.load_weights({"weight": weight})

    assert layer.weight is weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def test_precompute_freqs_output_shapes(tiny_config):
    """Tables are (max_seq_len, head_dim // 2) — one row per position, one column per pair."""
    cos, sin = precompute_freqs(
        head_dim=tiny_config["head_dim"],
        max_seq_len=tiny_config["max_seq_len"],
        theta=tiny_config["rope_theta"],
    )
    expected = (tiny_config["max_seq_len"], tiny_config["head_dim"] // 2)
    assert cos.shape == expected
    assert sin.shape == expected


def test_precompute_freqs_position_zero_is_identity(tiny_config):
    """At position 0 every angle is 0, so cos[0]=1 and sin[0]=0 — no rotation."""
    cos, sin = precompute_freqs(
        head_dim=tiny_config["head_dim"],
        max_seq_len=tiny_config["max_seq_len"],
        theta=tiny_config["rope_theta"],
    )
    mx.eval(cos, sin)
    assert mx.allclose(cos[0], mx.ones_like(cos[0])).item()
    assert mx.allclose(sin[0], mx.zeros_like(sin[0])).item()


def test_precompute_freqs_frequency_formula(tiny_config):
    """Verify freq_i = 1 / (theta ^ (2i / head_dim)) via cos values at position 1."""
    head_dim = tiny_config["head_dim"]
    theta = tiny_config["rope_theta"]
    cos, _ = precompute_freqs(head_dim, tiny_config["max_seq_len"], theta)
    mx.eval(cos)
    # At position 1, angle_i = 1 * freq_i, so cos[1, i] = cos(freq_i)
    for i in range(head_dim // 2):
        freq_i = 1.0 / (theta ** (2 * i / head_dim))
        assert abs(float(cos[1, i]) - math.cos(freq_i)) < 1e-5, f"frequency mismatch at pair {i}"


def test_apply_rope_output_shape_unchanged(tiny_config):
    """apply_rope returns the same shape as its input: (B, S, H, Dh)."""
    B, S = 1, 5
    H, Dh = tiny_config["n_heads"], tiny_config["head_dim"]
    x = mx.random.normal((B, S, H, Dh))
    cos, sin = precompute_freqs(Dh, tiny_config["max_seq_len"], tiny_config["rope_theta"])

    out = apply_rope(x, cos, sin, offset=0)

    assert out.shape == x.shape


def test_apply_rope_preserves_vector_norm(tiny_config):
    """RoPE is a rotation (unitary transform), so the L2 norm of each head vector is preserved."""
    B, S = 1, 5
    H, Dh = tiny_config["n_heads"], tiny_config["head_dim"]
    x = mx.random.normal((B, S, H, Dh))
    cos, sin = precompute_freqs(Dh, tiny_config["max_seq_len"], tiny_config["rope_theta"])

    out = apply_rope(x, cos, sin, offset=0)
    mx.eval(x, out)

    orig_norms = mx.sqrt((x * x).sum(axis=-1))   # (B, S, H)
    out_norms = mx.sqrt((out * out).sum(axis=-1)) # (B, S, H)
    assert mx.allclose(orig_norms, out_norms, atol=1e-5).item()


def test_apply_rope_offset_changes_embedding(tiny_config):
    """The same input at different sequence positions must produce different rotated outputs."""
    B, S = 1, 1
    H, Dh = tiny_config["n_heads"], tiny_config["head_dim"]
    x = mx.random.normal((B, S, H, Dh))
    cos, sin = precompute_freqs(Dh, tiny_config["max_seq_len"], tiny_config["rope_theta"])

    out0 = apply_rope(x, cos, sin, offset=0)
    out5 = apply_rope(x, cos, sin, offset=5)
    mx.eval(out0, out5)

    assert not mx.allclose(out0, out5).item()


def test_apply_rope_manual_rotation_formula():
    """Verify the rotation formula x0'=x0*cos-x1*sin, x1'=x0*sin+x1*cos at a known position."""
    # Minimal config: head_dim=2 gives one pair (Dh//2=1), theta=10000
    # freq_0 = 1 / (10000 ^ (0/2)) = 1.0
    # At offset=2: angle = 2 * 1.0 = 2.0
    x = mx.array([[[[3.0, 4.0]]]])   # (B=1, S=1, H=1, Dh=2)
    cos, sin = precompute_freqs(head_dim=2, max_seq_len=8, theta=10000.0)

    out = apply_rope(x, cos, sin, offset=2)
    mx.eval(out)

    c = math.cos(2.0)
    s = math.sin(2.0)
    expected_x0 = 3.0 * c - 4.0 * s
    expected_x1 = 3.0 * s + 4.0 * c
    expected = mx.array([[[[expected_x0, expected_x1]]]])
    mx.eval(expected)

    assert mx.allclose(out, expected, atol=1e-5).item()


def test_apply_rope_decode_uses_correct_position(tiny_config):
    """During decode, offset=current_len ensures the new token gets the right absolute position.

    A single decode token with offset=T must produce the same rotation as a prefill
    token at sequence position T (i.e., apply_rope on a sequence of length T+1 at position T).
    """
    H, Dh = tiny_config["n_heads"], tiny_config["head_dim"]
    T = 3  # pretend 3 tokens have been generated so far

    x_single = mx.random.normal((1, 1, H, Dh))   # decode: S=1 at position T
    cos, sin = precompute_freqs(Dh, tiny_config["max_seq_len"], tiny_config["rope_theta"])

    # Decode step: single token with offset = T
    out_decode = apply_rope(x_single, cos, sin, offset=T)

    # Prefill equivalent: build a sequence of T+1 tokens where the last matches x_single
    x_seq = mx.concatenate([mx.random.normal((1, T, H, Dh)), x_single], axis=1)  # (1, T+1, H, Dh)
    out_prefill = apply_rope(x_seq, cos, sin, offset=0)

    mx.eval(out_decode, out_prefill)

    # The last token of the prefill sequence should match the decode output
    assert mx.allclose(out_decode[0, 0], out_prefill[0, T], atol=1e-5).item()


# ---------------------------------------------------------------------------
# LlamaAttention
# ---------------------------------------------------------------------------

# TODO M1.3: implement attention tests once attention.py is implemented.


# ---------------------------------------------------------------------------
# SwiGLUFFN
# ---------------------------------------------------------------------------

# TODO M1.3: implement FFN tests once feedforward.py is implemented.
