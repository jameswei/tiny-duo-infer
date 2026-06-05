"""
Phase 1.8 integration tests: quantized generation on tiny synthetic models.

These tests assemble a complete tiny Llama or Qwen3 Engine with random weights,
apply weight-only quantization through quantize_weights(), and run a short
generation to verify:

  - Generation completes without crashing (no-quant, INT8, INT4).
  - Stop semantics (max_new_tokens, stop strings) are respected.
  - Full-precision greedy output is deterministic across two runs.
  - GenerationStats quantization fields are correctly populated.
  - Qwen3 quantization uses group_size=32 (d_model=32 fixture constraint).

No real model artifacts are required.  Slow real-model smoke tests are at the
bottom and are skipped unless --run-slow is passed.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

from tiny_duo_infer.config import ModelConfig
from tiny_duo_infer.engine import Engine
from tiny_duo_infer.generation import GenerationRequest
from tiny_duo_infer.models.llama import LlamaModel
from tiny_duo_infer.models.qwen3 import Qwen3Model
from tiny_duo_infer.quantization import QuantizationConfig
from tiny_duo_infer.weights.quantizer import compute_linear_weight_stats, quantize_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer that encodes to a fixed sequence and decodes to text."""

    eos_token_id: int = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(str(t) for t in token_ids)


def _llama_weights(config: ModelConfig) -> dict[str, mx.array]:
    """Random Llama project weights matching config dimensions."""
    D, V, I = config.d_model, config.vocab_size, config.intermediate_size
    H, Hkv, Dh = config.n_heads, config.n_kv_heads, config.head_dim
    weights: dict[str, mx.array] = {
        "embed_tokens.weight": mx.random.normal((V, D)),
        "final_norm.weight": mx.random.normal((D,)),
        "lm_head.weight": mx.random.normal((V, D)),
    }
    for i in range(config.n_layers):
        weights.update({
            f"layers.{i}.input_norm.weight": mx.random.normal((D,)),
            f"layers.{i}.attn.q_proj.weight": mx.random.normal((H * Dh, D)),
            f"layers.{i}.attn.k_proj.weight": mx.random.normal((Hkv * Dh, D)),
            f"layers.{i}.attn.v_proj.weight": mx.random.normal((Hkv * Dh, D)),
            f"layers.{i}.attn.o_proj.weight": mx.random.normal((D, H * Dh)),
            f"layers.{i}.post_attn_norm.weight": mx.random.normal((D,)),
            f"layers.{i}.ffn.gate_proj.weight": mx.random.normal((I, D)),
            f"layers.{i}.ffn.up_proj.weight": mx.random.normal((I, D)),
            f"layers.{i}.ffn.down_proj.weight": mx.random.normal((D, I)),
        })
    return weights


def _qwen3_weights(config: ModelConfig) -> dict[str, mx.array]:
    """Random Qwen3 project weights matching config dimensions."""
    D, V, I = config.d_model, config.vocab_size, config.intermediate_size
    H, Hkv, Dh = config.n_heads, config.n_kv_heads, config.head_dim
    A = H * Dh
    weights: dict[str, mx.array] = {
        "embed_tokens.weight": mx.random.normal((V, D)),
        "final_norm.weight": mx.random.normal((D,)),
        "lm_head.weight": mx.random.normal((V, D)),
    }
    for i in range(config.n_layers):
        weights.update({
            f"layers.{i}.input_norm.weight": mx.random.normal((D,)),
            f"layers.{i}.attn.q_proj.weight": mx.random.normal((A, D)),
            f"layers.{i}.attn.k_proj.weight": mx.random.normal((Hkv * Dh, D)),
            f"layers.{i}.attn.v_proj.weight": mx.random.normal((Hkv * Dh, D)),
            f"layers.{i}.attn.o_proj.weight": mx.random.normal((D, A)),
            f"layers.{i}.attn.q_norm.weight": mx.random.normal((Dh,)),
            f"layers.{i}.attn.k_norm.weight": mx.random.normal((Dh,)),
            f"layers.{i}.post_attn_norm.weight": mx.random.normal((D,)),
            f"layers.{i}.ffn.gate_proj.weight": mx.random.normal((I, D)),
            f"layers.{i}.ffn.up_proj.weight": mx.random.normal((I, D)),
            f"layers.{i}.ffn.down_proj.weight": mx.random.normal((D, I)),
        })
    return weights


def _make_llama_engine(
    config: ModelConfig,
    quantization: QuantizationConfig | None = None,
) -> Engine:
    """Build a tiny Llama Engine with synthetic weights and optional quantization."""
    weights = _llama_weights(config)
    if quantization is not None:
        weights = quantize_weights(weights, quantization)
    ws = compute_linear_weight_stats(weights)
    model = LlamaModel(config)
    model.load_weights(weights)
    return Engine(
        model=model,
        tokenizer=_FakeTokenizer(),
        config=config,
        max_seq_len=config.max_seq_len,
        quantization=quantization,
        linear_weight_stats=ws,
    )


def _make_qwen3_engine(
    config: ModelConfig,
    quantization: QuantizationConfig | None = None,
) -> Engine:
    """Build a tiny Qwen3 Engine with synthetic weights and optional quantization."""
    weights = _qwen3_weights(config)
    if quantization is not None:
        weights = quantize_weights(weights, quantization)
    ws = compute_linear_weight_stats(weights)
    model = Qwen3Model(config)
    model.load_weights(weights)
    return Engine(
        model=model,
        tokenizer=_FakeTokenizer(),
        config=config,
        max_seq_len=config.max_seq_len,
        quantization=quantization,
        linear_weight_stats=ws,
    )


# ---------------------------------------------------------------------------
# Llama — generation correctness (no-quant, INT8, INT4)
# ---------------------------------------------------------------------------


def test_llama_fp_generation_completes(tiny_model_config):
    """Tiny Llama full-precision generation runs without crashing."""
    engine = _make_llama_engine(tiny_model_config)
    req = GenerationRequest(prompt="hello", max_new_tokens=3, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stop_reason in ("eos", "max_new_tokens")
    assert resp.generated_tokens >= 0


def test_llama_int8_generation_completes(tiny_model_config):
    """Tiny Llama INT8 quantized generation runs without crashing."""
    config = QuantizationConfig(bits=8, group_size=64)
    engine = _make_llama_engine(tiny_model_config, quantization=config)
    req = GenerationRequest(prompt="hello", max_new_tokens=3, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stop_reason in ("eos", "max_new_tokens")
    assert resp.generated_tokens >= 0


def test_llama_int4_generation_completes(tiny_model_config):
    """Tiny Llama INT4 quantized generation runs without crashing."""
    config = QuantizationConfig(bits=4, group_size=64)
    engine = _make_llama_engine(tiny_model_config, quantization=config)
    req = GenerationRequest(prompt="hello", max_new_tokens=3, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stop_reason in ("eos", "max_new_tokens")
    assert resp.generated_tokens >= 0


def test_llama_fp_greedy_output_deterministic(tiny_model_config):
    """Full-precision greedy generation is deterministic across two runs."""
    mx.random.seed(42)
    engine = _make_llama_engine(tiny_model_config)
    req = GenerationRequest(prompt="hello", max_new_tokens=4, temperature=0.0)
    resp1 = engine.generate_request(req)
    resp2 = engine.generate_request(req)
    assert resp1.text == resp2.text


def test_llama_quantized_respects_max_new_tokens(tiny_model_config):
    """INT4 quantized generation does not exceed max_new_tokens."""
    config = QuantizationConfig(bits=4, group_size=64)
    engine = _make_llama_engine(tiny_model_config, quantization=config)
    req = GenerationRequest(prompt="hello", max_new_tokens=2, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.generated_tokens <= 2


def test_llama_quantized_respects_stop_string(tiny_model_config):
    """INT8 quantized generation forces stop_reason='stop_string' and trims the marker.

    The quantized engine is built normally, so every `Linear.forward()` in the
    real `LlamaModel` runs through `mx.quantized_matmul` during prefill. To
    make the next sampled token deterministic without bypassing the
    quantization path, this test wraps the real model: each forward call
    invokes the genuine quantized model first and forces its result with
    `mx.eval(real_logits)` (so quantized matmul actually executes — MLX is
    lazy), then returns synthetic logits whose argmax is token 5. The fake
    tokenizer decodes [5] -> "5", so `stop=["5"]` triggers on the first
    sampled token. The test asserts both that `stop_reason == "stop_string"`
    and that the returned text excludes the stop marker, while
    GenerationStats still report INT8 with at least one quantized linear.
    """
    config = QuantizationConfig(bits=8, group_size=64)
    engine = _make_llama_engine(tiny_model_config, quantization=config)

    V = tiny_model_config.vocab_size
    row = [100.0 if i == 5 else 0.0 for i in range(V)]  # argmax -> 5

    class _ForceArgmaxOverQuantizedModel:
        """Run the real quantized model, then override only the returned logits.

        The wrapper preserves the spec-required path: every quantized
        `Linear.forward()` (q/k/v/o_proj, gate/up/down_proj, lm_head) executes
        via `mx.quantized_matmul` because we call the inner model and then
        evaluate its output to defeat MLX laziness. Only the final logits
        tensor is replaced with a deterministic distribution so the test does
        not depend on the random INT8 numerics.
        """

        def __init__(self, inner) -> None:
            self._inner = inner

        def __call__(self, input_ids, cache, position_offset):
            real_logits = self._inner(input_ids, cache, position_offset)
            mx.eval(real_logits)
            B = int(input_ids.shape[0])
            S = int(input_ids.shape[1])
            return mx.array([[row] * S] * B)

    engine.model = _ForceArgmaxOverQuantizedModel(engine.model)

    req = GenerationRequest(prompt="hi", max_new_tokens=5, temperature=0.0, stop=["5"])
    resp = engine.generate_request(req)

    assert resp.stop_reason == "stop_string", (
        f"expected stop_reason='stop_string', got {resp.stop_reason!r}"
    )
    assert "5" not in resp.text
    assert resp.stats is not None
    assert resp.stats.quantization_mode == "int8"
    assert resp.stats.quantized_linear_count > 0


# ---------------------------------------------------------------------------
# Qwen3 — generation correctness with group_size=32 (d_model=32 requirement)
# ---------------------------------------------------------------------------


def test_qwen3_fp_generation_completes(tiny_qwen3_model_config):
    """Tiny Qwen3 full-precision generation runs without crashing."""
    engine = _make_qwen3_engine(tiny_qwen3_model_config)
    req = GenerationRequest(prompt="hello", max_new_tokens=3, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stop_reason in ("eos", "max_new_tokens")


def test_qwen3_int8_generation_completes(tiny_qwen3_model_config):
    """Tiny Qwen3 INT8 generation uses group_size=32 (d_model=32 constraint)."""
    config = QuantizationConfig(bits=8, group_size=32)
    engine = _make_qwen3_engine(tiny_qwen3_model_config, quantization=config)
    req = GenerationRequest(prompt="hello", max_new_tokens=3, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stop_reason in ("eos", "max_new_tokens")


def test_qwen3_int4_generation_completes(tiny_qwen3_model_config):
    """Tiny Qwen3 INT4 generation uses group_size=32 (d_model=32 constraint)."""
    config = QuantizationConfig(bits=4, group_size=32)
    engine = _make_qwen3_engine(tiny_qwen3_model_config, quantization=config)
    req = GenerationRequest(prompt="hello", max_new_tokens=3, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stop_reason in ("eos", "max_new_tokens")


def test_qwen3_int4_default_group_size_64_rejected(tiny_qwen3_model_config):
    """group_size=64 must be rejected for Qwen3 tiny (d_model=32 — 32 % 64 != 0)."""
    config = QuantizationConfig(bits=4, group_size=64)
    with pytest.raises(ValueError, match="not divisible by group_size"):
        _make_qwen3_engine(tiny_qwen3_model_config, quantization=config)


# ---------------------------------------------------------------------------
# GenerationStats quantization fields from integration path
# ---------------------------------------------------------------------------


def test_llama_fp_stats_quantization_mode_none(tiny_model_config):
    """Full-precision Llama stats have quantization_mode=none and zero counts."""
    engine = _make_llama_engine(tiny_model_config)
    req = GenerationRequest(prompt="hello", max_new_tokens=1, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s is not None
    assert s.quantization_mode == "none"
    assert s.quantization_bits is None
    assert s.quantization_group_size is None
    assert s.quantized_linear_count == 0
    assert s.full_precision_linear_count > 0
    assert s.linear_weight_full_precision_bytes > 0
    assert s.linear_weight_full_precision_bytes == s.linear_weight_runtime_bytes


def test_llama_int4_stats_quantization_fields(tiny_model_config):
    """INT4 Llama stats report correct mode, bits, group_size, and non-zero counts."""
    config = QuantizationConfig(bits=4, group_size=64)
    engine = _make_llama_engine(tiny_model_config, quantization=config)
    req = GenerationRequest(prompt="hello", max_new_tokens=1, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.quantization_mode == "int4"
    assert s.quantization_bits == 4
    assert s.quantization_group_size == 64
    assert s.quantized_linear_count > 0
    assert s.full_precision_linear_count == 0
    assert s.linear_weight_full_precision_bytes > 0
    assert s.linear_weight_runtime_bytes > 0


def test_qwen3_int8_stats_group_size_32(tiny_qwen3_model_config):
    """Qwen3 INT8 stats report group_size=32 (the required tiny fixture value)."""
    config = QuantizationConfig(bits=8, group_size=32)
    engine = _make_qwen3_engine(tiny_qwen3_model_config, quantization=config)
    req = GenerationRequest(prompt="hello", max_new_tokens=1, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.quantization_mode == "int8"
    assert s.quantization_bits == 8
    assert s.quantization_group_size == 32
    assert s.quantized_linear_count > 0


def test_llama_int4_memory_reduced_vs_fp(tiny_model_config):
    """INT4 quantized weights use fewer runtime bytes than full-precision."""
    config = QuantizationConfig(bits=4, group_size=64)
    engine = _make_llama_engine(tiny_model_config, quantization=config)
    req = GenerationRequest(prompt="hello", max_new_tokens=1, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.linear_weight_runtime_bytes < s.linear_weight_full_precision_bytes


# ---------------------------------------------------------------------------
# Slow real-model smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_llama_int8_smoke():
    """Real Llama-3.2-1B: INT8 quantized generation produces at least one token."""
    model_path = Path(os.environ.get("LLAMA_MODEL_PATH", "./models/llama-3.2-1b"))
    quant = QuantizationConfig(bits=8, group_size=64)
    engine = Engine.from_model_path(model_path, max_seq_len=128, quantization=quant)
    req = GenerationRequest(prompt="The capital of France is", max_new_tokens=3, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert resp.generated_tokens >= 1
    assert s is not None
    assert s.quantization_mode == "int8"
    assert s.quantization_bits == 8
    assert s.quantized_linear_count > 0
    assert s.linear_weight_runtime_bytes < s.linear_weight_full_precision_bytes


@pytest.mark.slow
def test_qwen3_int8_smoke():
    """Real Qwen3-0.6B: INT8 quantized generation produces at least one token."""
    model_path = Path(os.environ.get("QWEN_MODEL_PATH", "./models/qwen3-0.6b"))
    quant = QuantizationConfig(bits=8, group_size=64)
    engine = Engine.from_model_path(model_path, max_seq_len=128, quantization=quant)
    req = GenerationRequest(prompt="Hello", max_new_tokens=3, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert resp.generated_tokens >= 1
    assert s is not None
    assert s.quantization_mode == "int8"
    assert s.quantized_linear_count > 0
    assert s.linear_weight_runtime_bytes < s.linear_weight_full_precision_bytes
