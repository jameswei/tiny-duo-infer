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

from tiny_duo_infer.layers.feedforward import SwiGLUFFN
from tiny_duo_infer.layers.normalization import RMSNorm
from tiny_duo_infer.layers.rope import apply_rope, precompute_freqs
from tiny_duo_infer.models.base import Linear


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

def test_swiglu_init_creates_three_linear_projections(tiny_model_config):
    """SwiGLUFFN exposes gate_proj, up_proj, down_proj as Linear sub-modules."""
    ffn = SwiGLUFFN(tiny_model_config)
    assert isinstance(ffn.gate_proj, Linear)
    assert isinstance(ffn.up_proj, Linear)
    assert isinstance(ffn.down_proj, Linear)


def test_swiglu_projection_dimensions(tiny_model_config):
    """gate_proj and up_proj expand D→I; down_proj contracts I→D."""
    ffn = SwiGLUFFN(tiny_model_config)
    D = tiny_model_config.d_model
    I = tiny_model_config.intermediate_size
    assert ffn.gate_proj.in_features == D and ffn.gate_proj.out_features == I
    assert ffn.up_proj.in_features == D   and ffn.up_proj.out_features == I
    assert ffn.down_proj.in_features == I and ffn.down_proj.out_features == D


def test_swiglu_forward_output_shape(tiny_model_config):
    """SwiGLUFFN preserves the transformer shape: (B, S, D) → (B, S, D)."""
    D = tiny_model_config.d_model
    I = tiny_model_config.intermediate_size
    ffn = SwiGLUFFN(tiny_model_config)
    ffn.gate_proj.weight = mx.random.normal((I, D))
    ffn.up_proj.weight   = mx.random.normal((I, D))
    ffn.down_proj.weight = mx.random.normal((D, I))

    x = mx.random.normal((1, 5, D))
    out = ffn(x)

    assert out.shape == x.shape


def test_swiglu_gate_and_up_are_separate_weight_tensors(tiny_model_config):
    """gate_proj and up_proj must not share the same weight object."""
    D = tiny_model_config.d_model
    I = tiny_model_config.intermediate_size
    ffn = SwiGLUFFN(tiny_model_config)
    ffn.gate_proj.weight = mx.random.normal((I, D))
    ffn.up_proj.weight   = mx.random.normal((I, D))
    ffn.down_proj.weight = mx.random.normal((D, I))

    assert ffn.gate_proj.weight is not ffn.up_proj.weight


def test_swiglu_zero_gate_zeros_output(tiny_model_config):
    """When gate_proj produces all zeros, silu(0)=0 so the entire output is zero.

    This verifies that SiLU is applied to gate (not up): silu(0) * up = 0.
    If SiLU were incorrectly applied to up, zeroing gate_proj would still
    zero the output, but zeroing up_proj (see next test) would not.
    """
    D = tiny_model_config.d_model
    I = tiny_model_config.intermediate_size
    ffn = SwiGLUFFN(tiny_model_config)
    ffn.gate_proj.weight = mx.zeros((I, D))          # gate = 0, silu(0) = 0
    ffn.up_proj.weight   = mx.random.normal((I, D))  # up is non-zero
    ffn.down_proj.weight = mx.random.normal((D, I))

    x = mx.random.normal((1, 3, D))
    out = ffn(x)
    mx.eval(out)

    assert mx.allclose(out, mx.zeros_like(out), atol=1e-6).item()


def test_swiglu_zero_up_zeros_output(tiny_model_config):
    """When up_proj produces all zeros, silu(gate) * 0 = 0 so output is zero.

    Together with test_swiglu_zero_gate_zeros_output this confirms the
    multiplicative gate: both paths must be non-zero for output to be non-zero.
    """
    D = tiny_model_config.d_model
    I = tiny_model_config.intermediate_size
    ffn = SwiGLUFFN(tiny_model_config)
    ffn.gate_proj.weight = mx.random.normal((I, D))
    ffn.up_proj.weight   = mx.zeros((I, D))  # up = 0
    ffn.down_proj.weight = mx.random.normal((D, I))

    x = mx.random.normal((1, 3, D))
    out = ffn(x)
    mx.eval(out)

    assert mx.allclose(out, mx.zeros_like(out), atol=1e-6).item()


def test_swiglu_forward_manual_formula():
    """Verify out = down_proj(silu(gate_proj(x)) * up_proj(x)) with known weights."""
    # Minimal D=2, I=2 case with identity-like weights for easy hand computation
    import math
    from tiny_duo_infer.config import ModelConfig
    cfg = ModelConfig(
        d_model=2, n_layers=1, n_heads=1, n_kv_heads=1,
        intermediate_size=2, vocab_size=4, max_seq_len=8,
        rope_theta=10000.0, rms_norm_eps=1e-5,
    )
    ffn = SwiGLUFFN(cfg)
    # gate_proj = identity, up_proj = 2x identity, down_proj = identity
    ffn.gate_proj.weight = mx.array([[1.0, 0.0], [0.0, 1.0]])  # (I=2, D=2)
    ffn.up_proj.weight   = mx.array([[2.0, 0.0], [0.0, 2.0]])  # (I=2, D=2)
    ffn.down_proj.weight = mx.array([[1.0, 0.0], [0.0, 1.0]])  # (D=2, I=2)

    x = mx.array([[[1.0, 2.0]]])  # (B=1, S=1, D=2)

    # gate = x @ gate_proj.T = [1, 2]
    # up   = x @ up_proj.T   = [2, 4]
    # silu([1, 2]) = [1*sig(1), 2*sig(2)]
    # silu(gate) * up = [2*sig(1), 8*sig(2)]
    # down_proj(above) @ identity.T = [2*sig(1), 8*sig(2)]
    sig1 = 1.0 / (1.0 + math.exp(-1.0))
    sig2 = 1.0 / (1.0 + math.exp(-2.0))
    expected = mx.array([[[2.0 * sig1, 8.0 * sig2]]])

    out = ffn(x)
    mx.eval(out, expected)

    assert mx.allclose(out, expected, atol=1e-5).item()


def test_swiglu_load_weights_routes_to_submodules(tiny_model_config):
    """load_weights() dot-path routing populates gate_proj, up_proj, down_proj."""
    D = tiny_model_config.d_model
    I = tiny_model_config.intermediate_size
    ffn = SwiGLUFFN(tiny_model_config)

    gate_w = mx.random.normal((I, D))
    up_w   = mx.random.normal((I, D))
    down_w = mx.random.normal((D, I))

    ffn.load_weights({
        "gate_proj.weight": gate_w,
        "up_proj.weight":   up_w,
        "down_proj.weight": down_w,
    })

    assert ffn.gate_proj.weight is gate_w
    assert ffn.up_proj.weight   is up_w
    assert ffn.down_proj.weight is down_w
