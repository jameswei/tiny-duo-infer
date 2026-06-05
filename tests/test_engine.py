"""
Tests for tiny_duo_infer.engine.Engine.

Unit tests use TINY_CONFIG with randomly initialised weights (no real artifacts).
Slow smoke tests require local Llama-3.2-1B artifacts (--run-slow).

Test categories (unit):
  - Engine.from_model_path: loads config, tokenizer stub, model stub
  - generate(): yields tokens up to max_new_tokens
  - generate(): stops at EOS token
  - generate(): greedy output is deterministic across two calls
  - generate(): yielded text fragments decode to non-empty strings

Test categories (slow smoke):
  - Load real artifacts and generate without crashing
  - max_new_tokens is respected
  - EOS handling works when EOS appears in output
  - Greedy generation is deterministic
  - Generated token IDs decode to non-empty text
"""

import pytest
import mlx.core as mx

import tiny_duo_infer.engine as engine_module
from tiny_duo_infer.context_policy import ContextBudgetError
from tiny_duo_infer.engine import Engine
from tiny_duo_infer.generation import ChatMessage, GenerationRequest
from tiny_duo_infer.quantization import QuantizationConfig


# ---------------------------------------------------------------------------
# Unit tests (no model artifacts required)
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Tokenizer test double that records encode options."""

    # Chosen to be different from the greedy argmax of _RecordingModel's decode
    # logits (which is always vocab_size - 1 = 255). Tests that rely on EOS
    # stopping must use _SequenceModel instead of _RecordingModel.
    eos_token_id: int = 0
    bos_token_id: int = 1

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        return list(self.token_ids)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(str(t) for t in token_ids)


class _RecordingModel:
    """
    Minimal model double for Engine prefill tests.

    It writes deterministic K/V tensors through KVCache.update(), matching the
    real LlamaModel contract, and returns logits whose values identify their
    sequence position.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.calls: list[tuple[mx.array, int]] = []

    def __call__(self, input_ids, cache, position_offset: int):
        self.calls.append((input_ids, position_offset))
        B, S = input_ids.shape
        Hkv, Dh = self.config.n_kv_heads, self.config.head_dim

        for layer_idx in range(self.config.n_layers):
            fill_value = float(layer_idx + 1)
            new_k = mx.ones((B, Hkv, S, Dh)) * fill_value
            new_v = mx.ones((B, Hkv, S, Dh)) * (fill_value + 10.0)
            cache.update(layer_idx, new_k, new_v, position=position_offset)

        vocab_positions = mx.arange(self.config.vocab_size, dtype=mx.float32)
        seq_positions = mx.arange(S, dtype=mx.float32).reshape(1, S, 1)
        return seq_positions + vocab_positions.reshape(1, 1, self.config.vocab_size)


class _RecordingCache:
    """KVCache wrapper that records whether Engine asks it to materialise buffers."""

    def __init__(self, inner_cache) -> None:
        self.inner_cache = inner_cache
        self.eval_calls = 0

    def update(self, *args, **kwargs):
        return self.inner_cache.update(*args, **kwargs)

    def advance(self, *args, **kwargs):
        return self.inner_cache.advance(*args, **kwargs)

    def eval(self) -> None:
        self.eval_calls += 1
        self.inner_cache.eval()

    @property
    def current_len(self) -> int:
        return self.inner_cache.current_len

    @property
    def _keys(self):
        return self.inner_cache._keys

    @property
    def _values(self):
        return self.inner_cache._values


def _make_engine(tiny_model_config, max_seq_len: int | None = None) -> Engine:
    """Create an Engine with fake model/tokenizer components."""
    return Engine(
        model=_RecordingModel(tiny_model_config),
        tokenizer=_FakeTokenizer([7, 8, 9]),
        config=tiny_model_config,
        max_seq_len=max_seq_len or tiny_model_config.max_seq_len,
    )


def _fake_model_class(family: str, records: list[tuple[str, object]]) -> type:
    """Build a loadable model test double for Engine.from_model_path() dispatch."""

    class FakeModel:
        def __init__(self, config) -> None:
            self.config = config
            self.loaded_weights = None
            records.append((f"{family}.init", config))

        def load_weights(self, weights) -> None:
            self.loaded_weights = weights
            records.append((f"{family}.load_weights", weights))

    return FakeModel


def test_from_model_path_dispatches_llama_model_and_converter(monkeypatch, tiny_model_config):
    """Llama configs use LlamaModel plus the Llama weight converter."""
    records: list[tuple[str, object]] = []
    tokenizer = _FakeTokenizer([1, 2])
    hf_weights = {"raw": mx.array([1.0])}
    converted_weights = {"converted": mx.array([2.0])}

    def convert_llama(weights, config):
        records.append(("llama.convert", (weights, config)))
        return converted_weights

    monkeypatch.setattr(
        engine_module,
        "load_config",
        lambda model_dir: tiny_model_config,
    )
    monkeypatch.setattr(
        engine_module.Tokenizer,
        "from_pretrained",
        staticmethod(lambda model_dir: tokenizer),
    )
    monkeypatch.setattr(engine_module, "load_weights", lambda model_dir: hf_weights)
    monkeypatch.setattr(engine_module, "LlamaModel", _fake_model_class("llama", records))
    monkeypatch.setattr(engine_module, "Qwen3Model", _fake_model_class("qwen3", records))
    monkeypatch.setattr(engine_module, "convert_llama", convert_llama)

    engine = Engine.from_model_path("/fake/llama", max_seq_len=11)

    assert engine.tokenizer is tokenizer
    assert engine.config.model_type == "llama"
    assert engine.config.max_seq_len == 11
    assert records[0][0] == "llama.convert"
    assert records[0][1] == (hf_weights, engine.config)
    assert records[1] == ("llama.init", engine.config)
    assert records[2] == ("llama.load_weights", converted_weights)


def test_from_model_path_dispatches_qwen3_model_and_converter(
    monkeypatch,
    tiny_qwen3_model_config,
):
    """Qwen3 configs use Qwen3Model plus the Qwen3 weight converter."""
    records: list[tuple[str, object]] = []
    tokenizer = _FakeTokenizer([1, 2])
    hf_weights = {"raw": mx.array([1.0])}
    converted_weights = {"converted": mx.array([3.0])}

    def convert_qwen3(weights, config):
        records.append(("qwen3.convert", (weights, config)))
        return converted_weights

    monkeypatch.setattr(
        engine_module,
        "load_config",
        lambda model_dir: tiny_qwen3_model_config,
    )
    monkeypatch.setattr(
        engine_module.Tokenizer,
        "from_pretrained",
        staticmethod(lambda model_dir: tokenizer),
    )
    monkeypatch.setattr(engine_module, "load_weights", lambda model_dir: hf_weights)
    monkeypatch.setattr(engine_module, "LlamaModel", _fake_model_class("llama", records))
    monkeypatch.setattr(engine_module, "Qwen3Model", _fake_model_class("qwen3", records))
    monkeypatch.setattr(engine_module, "convert_qwen3", convert_qwen3)

    engine = Engine.from_model_path("/fake/qwen3", max_seq_len=13)

    assert engine.tokenizer is tokenizer
    assert engine.config.model_type == "qwen3"
    assert engine.config.max_seq_len == 13
    assert records[0][0] == "qwen3.convert"
    assert records[0][1] == (hf_weights, engine.config)
    assert records[1] == ("qwen3.init", engine.config)
    assert records[2] == ("qwen3.load_weights", converted_weights)


def test_from_model_path_rejects_max_seq_len_above_config(monkeypatch, tiny_model_config):
    """Dispatch is not attempted if requested max_seq_len exceeds model context."""
    records: list[tuple[str, object]] = []

    monkeypatch.setattr(
        engine_module,
        "load_config",
        lambda model_dir: tiny_model_config,
    )
    monkeypatch.setattr(
        engine_module,
        "load_weights",
        lambda model_dir: records.append(("load", model_dir)),
    )

    with pytest.raises(ValueError, match="exceeds model context length"):
        Engine.from_model_path("/fake/llama", max_seq_len=tiny_model_config.max_seq_len + 1)

    assert records == []


def test_from_model_path_quantization_none_skips_quantize_weights(monkeypatch, tiny_model_config):
    """quantization=None leaves converted weights unchanged; quantize_weights not called."""
    records: list[tuple[str, object]] = []
    tokenizer = _FakeTokenizer([1, 2])
    hf_weights = {"raw": mx.array([1.0])}
    converted_weights = {"converted": mx.array([2.0])}

    def convert_llama(weights, config):
        records.append(("llama.convert", (weights, config)))
        return converted_weights

    quantize_calls: list = []

    def fake_quantize_weights(project_weights, config):
        quantize_calls.append((project_weights, config))
        return project_weights

    monkeypatch.setattr(engine_module, "load_config", lambda _: tiny_model_config)
    monkeypatch.setattr(engine_module.Tokenizer, "from_pretrained", staticmethod(lambda _: tokenizer))
    monkeypatch.setattr(engine_module, "load_weights", lambda _: hf_weights)
    monkeypatch.setattr(engine_module, "LlamaModel", _fake_model_class("llama", records))
    monkeypatch.setattr(engine_module, "convert_llama", convert_llama)
    monkeypatch.setattr(engine_module, "quantize_weights", fake_quantize_weights)

    Engine.from_model_path("/fake/llama", max_seq_len=tiny_model_config.max_seq_len, quantization=None)

    assert quantize_calls == [], "quantize_weights must not be called when quantization=None"
    assert records[2] == ("llama.load_weights", converted_weights)


def test_from_model_path_quantization_config_calls_quantize_and_loads_result(
    monkeypatch, tiny_model_config
):
    """quantization=QuantizationConfig calls quantize_weights after conversion and loads result."""
    records: list[tuple[str, object]] = []
    tokenizer = _FakeTokenizer([1, 2])
    hf_weights = {"raw": mx.array([1.0])}
    converted_weights = {"converted": mx.array([2.0])}
    quantized_weights = {"converted_q": mx.array([3.0])}  # distinct sentinel

    def convert_llama(weights, config):
        records.append(("llama.convert", (weights, config)))
        return converted_weights

    quant_config = QuantizationConfig(bits=4, group_size=64)
    quantize_calls: list = []

    def fake_quantize_weights(project_weights, config):
        quantize_calls.append((project_weights, config))
        return quantized_weights

    monkeypatch.setattr(engine_module, "load_config", lambda _: tiny_model_config)
    monkeypatch.setattr(engine_module.Tokenizer, "from_pretrained", staticmethod(lambda _: tokenizer))
    monkeypatch.setattr(engine_module, "load_weights", lambda _: hf_weights)
    monkeypatch.setattr(engine_module, "LlamaModel", _fake_model_class("llama", records))
    monkeypatch.setattr(engine_module, "convert_llama", convert_llama)
    monkeypatch.setattr(engine_module, "quantize_weights", fake_quantize_weights)

    Engine.from_model_path("/fake/llama", max_seq_len=tiny_model_config.max_seq_len, quantization=quant_config)

    assert len(quantize_calls) == 1
    assert quantize_calls[0] == (converted_weights, quant_config)
    # model.load_weights must receive the quantized dict, not the converter output
    assert records[2] == ("llama.load_weights", quantized_weights)


def test_from_model_path_quantize_weights_error_propagates_before_model_construction(
    monkeypatch, tiny_model_config
):
    """ValueError from quantize_weights propagates; model is never constructed."""
    records: list[tuple[str, object]] = []
    tokenizer = _FakeTokenizer([1, 2])
    hf_weights = {"raw": mx.array([1.0])}
    converted_weights = {"converted": mx.array([2.0])}

    def convert_llama(weights, config):
        return converted_weights

    def fake_quantize_weights(project_weights, config):
        raise ValueError("in_features=48 not divisible by group_size=64")

    monkeypatch.setattr(engine_module, "load_config", lambda _: tiny_model_config)
    monkeypatch.setattr(engine_module.Tokenizer, "from_pretrained", staticmethod(lambda _: tokenizer))
    monkeypatch.setattr(engine_module, "load_weights", lambda _: hf_weights)
    monkeypatch.setattr(engine_module, "LlamaModel", _fake_model_class("llama", records))
    monkeypatch.setattr(engine_module, "convert_llama", convert_llama)
    monkeypatch.setattr(engine_module, "quantize_weights", fake_quantize_weights)

    with pytest.raises(ValueError, match="not divisible"):
        Engine.from_model_path(
            "/fake/llama",
            max_seq_len=tiny_model_config.max_seq_len,
            quantization=QuantizationConfig(bits=4, group_size=64),
        )

    assert records == [], "model must not be constructed when quantize_weights raises"


def test_prefill_token_ids_returns_final_position_logits(tiny_model_config):
    """Prefill returns only logits from the final prompt position: (V,)."""
    engine = _make_engine(tiny_model_config)
    logits = engine.prefill_token_ids([3, 4, 5])
    mx.eval(logits)

    assert logits.shape == (tiny_model_config.vocab_size,)
    assert logits[0].item() == 2.0
    assert logits[17].item() == 19.0


def test_prefill_token_ids_fills_cache_and_advances_once(tiny_model_config):
    """Engine commits cache.current_len after all layers write prompt K/V."""
    engine = _make_engine(tiny_model_config)
    prompt_len = 4

    engine.prefill_token_ids([1, 2, 3, 4])

    assert engine.cache is not None
    assert engine.cache.current_len == prompt_len
    for layer_idx in range(tiny_model_config.n_layers):
        mx.eval(engine.cache._keys[layer_idx], engine.cache._values[layer_idx])
        valid_keys = engine.cache._keys[layer_idx][:, :, :prompt_len, :]
        valid_values = engine.cache._values[layer_idx][:, :, :prompt_len, :]
        assert mx.allclose(
            valid_keys,
            mx.ones_like(valid_keys) * (layer_idx + 1),
        ).item()
        assert mx.allclose(
            valid_values,
            mx.ones_like(valid_values) * (layer_idx + 11),
        ).item()


def test_prefill_token_ids_materialises_cache_before_return(tiny_model_config):
    """Prefill evaluates cache buffers so decode can read committed K/V state."""
    engine = _make_engine(tiny_model_config)
    real_new_cache = engine._new_cache

    def new_recording_cache():
        return _RecordingCache(real_new_cache())

    engine._new_cache = new_recording_cache

    engine.prefill_token_ids([1, 2, 3])

    assert engine.cache.eval_calls == 1


def test_prefill_token_ids_calls_model_at_position_zero(tiny_model_config):
    """Prefill always writes the prompt starting at absolute cache position 0."""
    engine = _make_engine(tiny_model_config)
    token_ids = [10, 11, 12]

    engine.prefill_token_ids(token_ids)

    model = engine.model
    assert len(model.calls) == 1
    input_ids, position_offset = model.calls[0]
    assert position_offset == 0
    assert input_ids.shape == (1, len(token_ids))
    assert input_ids.tolist() == [token_ids]


def test_prefill_uses_tokenizer_with_special_tokens(tiny_model_config):
    """Text prefill delegates to tokenizer.encode(add_special_tokens=True)."""
    engine = _make_engine(tiny_model_config)
    tokenizer = engine.tokenizer

    engine.prefill("hello")

    assert tokenizer.encode_calls == [("hello", True)]


def test_prefill_rejects_empty_prompt_tokens(tiny_model_config):
    """An empty prompt cannot provide final-position logits."""
    engine = _make_engine(tiny_model_config)

    with pytest.raises(ValueError, match="at least one token"):
        engine.prefill_token_ids([])


def test_prefill_rejects_prompt_longer_than_cache(tiny_model_config):
    """Prompt tokens must fit in the request's fixed-size KV cache."""
    engine = _make_engine(tiny_model_config, max_seq_len=2)

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        engine.prefill_token_ids([1, 2, 3])


class _SequenceModel:
    """
    Model double that steers greedy decoding through a fixed token sequence.

    On each __call__, returns (B, S, V) logits where the single high value is
    at the index given by token_sequence[call_idx]. Greedy then always picks
    that token.

    The sequence wraps via modulo so tests can use a short list even if the
    engine makes more calls than expected.
    """

    def __init__(self, config, token_sequence: list[int]) -> None:
        self.config = config
        self.token_sequence = token_sequence
        self._call_idx = 0
        self.calls: list[tuple[mx.array, int]] = []

    def __call__(self, input_ids: mx.array, cache, position_offset: int) -> mx.array:
        self.calls.append((input_ids, position_offset))
        B, S = input_ids.shape
        Hkv, Dh = self.config.n_kv_heads, self.config.head_dim

        for layer_idx in range(self.config.n_layers):
            new_k = mx.zeros((B, Hkv, S, Dh))
            new_v = mx.zeros((B, Hkv, S, Dh))
            cache.update(layer_idx, new_k, new_v, position=position_offset)

        token = self.token_sequence[self._call_idx % len(self.token_sequence)]
        self._call_idx += 1

        # One-hot-like logits: index `token` = 1.0, all others = 0.0.
        # mx.argmax will reliably return `token` regardless of ties at 0.
        indicator = (mx.arange(self.config.vocab_size) == token).astype(mx.float32)
        return mx.broadcast_to(
            indicator.reshape(1, 1, self.config.vocab_size), (B, S, self.config.vocab_size)
        )


# ---------------------------------------------------------------------------
# Decode loop tests (unit)
# ---------------------------------------------------------------------------

def test_generate_yields_up_to_max_new_tokens(tiny_model_config):
    """generate() yields exactly max_new_tokens fragments when EOS is not seen."""
    # _RecordingModel decode logits: arange(V) → argmax = V-1 ≠ eos_token_id=0
    engine = _make_engine(tiny_model_config)
    tokens = list(engine.generate("hello", max_new_tokens=5, temperature=0.0))
    assert len(tokens) == 5


def test_generate_stops_before_yielding_eos(tiny_model_config):
    """generate() does not yield the EOS token when EOS is the first decode output."""
    eos = _FakeTokenizer.eos_token_id  # 0
    engine = _make_engine(tiny_model_config)
    # prefill call (call 0): sequence[0]=10 → first generated token = 10
    # decode call (call 1): sequence[1]=eos → EOS → stop without yielding
    engine.model = _SequenceModel(tiny_model_config, [10, eos])
    tokens = list(engine.generate("hello", max_new_tokens=10, temperature=0.0))
    assert len(tokens) == 1
    assert tokens[0] == "10"  # _FakeTokenizer.decode([10]) = "10"


def test_generate_stops_after_several_tokens_then_eos(tiny_model_config):
    """generate() stops at EOS mid-stream and does not yield beyond it."""
    eos = _FakeTokenizer.eos_token_id  # 0
    engine = _make_engine(tiny_model_config)
    # call 0 (prefill): 10 → first token
    # call 1 (decode 1): 20 → second token
    # call 2 (decode 2): eos → stop
    engine.model = _SequenceModel(tiny_model_config, [10, 20, eos])
    tokens = list(engine.generate("hello", max_new_tokens=10, temperature=0.0))
    assert len(tokens) == 2
    assert tokens == ["10", "20"]


def test_generate_decode_increments_cache_each_step(tiny_model_config):
    """cache.current_len grows by 1 for every decode step, checkable mid-stream."""
    engine = _make_engine(tiny_model_config)
    prompt_len = 3  # _FakeTokenizer always encodes to [7, 8, 9]
    gen = engine.generate("hello", max_new_tokens=4, temperature=0.0)

    # First next(): prefill runs + first yield. advance(1) happens after the
    # yield, so current_len is still prompt_len at this suspension point.
    next(gen)
    assert engine.cache.current_len == prompt_len

    # Subsequent next() calls: resume → decode step → advance(1) → yield.
    next(gen)
    assert engine.cache.current_len == prompt_len + 1

    next(gen)
    assert engine.cache.current_len == prompt_len + 2


def test_generate_materialises_cache_after_prefill_and_decode(tiny_model_config):
    """Generation evaluates cache buffers after prefill and each decode forward."""
    engine = _make_engine(tiny_model_config)
    real_new_cache = engine._new_cache

    def new_recording_cache():
        return _RecordingCache(real_new_cache())

    engine._new_cache = new_recording_cache

    list(engine.generate("hello", max_new_tokens=4, temperature=0.0))

    # max_new_tokens=4 yields four tokens. The loop runs three decode forwards:
    # after the first, second, and third yielded tokens. The fourth token is the
    # final output, so no unused decode forward is run after it.
    assert engine.cache.eval_calls == 4  # 1 prefill + 3 decode cache evals


def test_generate_decode_uses_cache_len_as_position_offset(tiny_model_config):
    """Each decode step passes cache.current_len at call time as position_offset."""
    engine = _make_engine(tiny_model_config)
    prompt_len = 3  # _FakeTokenizer returns [7, 8, 9]
    list(engine.generate("hello", max_new_tokens=3, temperature=0.0))

    model = engine.model
    # calls[0] = prefill at position_offset=0
    # calls[1], [2], [3] = decode steps at prompt_len + 0, +1, +2
    assert model.calls[0][1] == 0
    for step, call in enumerate(model.calls[1:]):
        _, pos = call
        assert pos == prompt_len + step


def test_generate_decode_input_is_single_token(tiny_model_config):
    """Each decode step feeds a (1, 1) input_ids tensor to the model."""
    engine = _make_engine(tiny_model_config)
    list(engine.generate("hello", max_new_tokens=3, temperature=0.0))

    for call in engine.model.calls[1:]:  # skip prefill
        input_ids, _ = call
        assert input_ids.shape == (1, 1)


def test_generate_greedy_is_deterministic(tiny_model_config):
    """Two greedy generate() calls with the same prompt produce identical output."""
    engine1 = _make_engine(tiny_model_config)
    engine2 = _make_engine(tiny_model_config)
    out1 = list(engine1.generate("hello", max_new_tokens=4, temperature=0.0))
    out2 = list(engine2.generate("hello", max_new_tokens=4, temperature=0.0))
    assert out1 == out2


# ---------------------------------------------------------------------------
# generate_request() tests (unit)
# ---------------------------------------------------------------------------


class _UniformModel:
    """
    Model double that returns all-zeros logits — uniform distribution after softmax.

    Used for seeded sampling tests where non-trivial randomness is needed:
    any token in the vocabulary is equally likely, so the chosen token depends
    entirely on the PRNG state (seed).
    """

    def __init__(self, config) -> None:
        self.config = config

    def __call__(self, input_ids: mx.array, cache, position_offset: int) -> mx.array:
        B, S = input_ids.shape
        Hkv, Dh = self.config.n_kv_heads, self.config.head_dim
        for layer_idx in range(self.config.n_layers):
            new_k = mx.zeros((B, Hkv, S, Dh))
            new_v = mx.zeros((B, Hkv, S, Dh))
            cache.update(layer_idx, new_k, new_v, position=position_offset)
        return mx.zeros((B, S, self.config.vocab_size))


def _make_sequence_engine(tiny_model_config, token_sequence, max_seq_len=None):
    """Engine with _SequenceModel for controlled generate_request() tests."""
    return Engine(
        model=_SequenceModel(tiny_model_config, token_sequence),
        tokenizer=_FakeTokenizer([7, 8, 9]),
        config=tiny_model_config,
        max_seq_len=max_seq_len or tiny_model_config.max_seq_len,
    )


def test_generate_request_stops_on_eos(tiny_model_config):
    """generate_request() stops at EOS and returns stop_reason='eos'."""
    eos = _FakeTokenizer.eos_token_id  # 0
    # call 0 (prefill): steer first generated token to 10
    # call 1 (decode):  steer next token to eos → stop
    engine = _make_sequence_engine(tiny_model_config, [10, eos])
    req = GenerationRequest(prompt="hello", max_new_tokens=10, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stop_reason == "eos"
    assert resp.text == "10"
    assert resp.generated_tokens == 1


def test_generate_request_stops_on_max_new_tokens(tiny_model_config):
    """generate_request() stops after max_new_tokens with correct stop_reason."""
    engine = _make_sequence_engine(tiny_model_config, [10, 20, 30, 40, 50])
    req = GenerationRequest(prompt="hello", max_new_tokens=3, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stop_reason == "max_new_tokens"
    assert resp.generated_tokens == 3
    assert resp.text == "102030"


def test_generate_request_stops_on_stop_string(tiny_model_config):
    """generate_request() stops when a stop string appears in the decoded output."""
    # decode([99]) = "99"; stop string "99" appears in accumulated text "1099"
    engine = _make_sequence_engine(tiny_model_config, [10, 99, 30])
    req = GenerationRequest(prompt="hello", max_new_tokens=10, temperature=0.0, stop=["99"])
    resp = engine.generate_request(req)
    assert resp.stop_reason == "stop_string"
    assert "99" not in resp.text
    assert resp.text == "10"


def test_generate_request_stop_string_excluded_from_text(tiny_model_config):
    """Text returned by generate_request() does not include the matched stop marker."""
    engine = _make_sequence_engine(tiny_model_config, [10, 20, 99, 30])
    req = GenerationRequest(prompt="hello", max_new_tokens=10, temperature=0.0, stop=["99"])
    resp = engine.generate_request(req)
    assert resp.text == "1020"
    assert "99" not in resp.text


def test_generate_request_stops_on_context_length(tiny_model_config):
    """generate_request() stops with 'context_length' when the KV cache is full."""
    # max_seq_len=4, prompt=[7,8,9] (len=3): one decode slot at position 3.
    # max_new_tokens=4 (<=max_seq_len so context policy admits the request).
    # Step 0: token 10 at position 3 is valid → yielded, cache advances to 4.
    # Step 1: context_length check fires BEFORE decoding token 20 (position 4
    #   is out of bounds) → stop with generated_tokens=1.
    engine = _make_sequence_engine(tiny_model_config, [10, 20, 30], max_seq_len=4)
    req = GenerationRequest(prompt="hello", max_new_tokens=4, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stop_reason == "context_length"
    assert resp.generated_tokens == 1
    assert resp.text == "10"


def test_generate_request_records_prompt_token_count(tiny_model_config):
    """prompt_tokens equals the length of the encoded prompt."""
    engine = _make_sequence_engine(tiny_model_config, [10])
    # _FakeTokenizer always encodes to [7, 8, 9] → 3 tokens
    req = GenerationRequest(prompt="hello", max_new_tokens=1, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.prompt_tokens == 3


def test_generate_request_records_generated_token_count(tiny_model_config):
    """generated_tokens counts tokens produced by the decode loop."""
    eos = _FakeTokenizer.eos_token_id
    engine = _make_sequence_engine(tiny_model_config, [10, 20, eos])
    req = GenerationRequest(prompt="hello", max_new_tokens=10, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.generated_tokens == 2
    assert resp.stop_reason == "eos"


def test_generate_request_eos_priority_over_stop_string(tiny_model_config):
    """EOS stops generation before the stop-string check runs for that token."""
    eos = _FakeTokenizer.eos_token_id  # 0; decode([0]) = "0"
    # Step 0: token=5, text="5", stop="0" not in "5" → continue
    # Step 1: token=eos=0 → EOS check fires before decode([0])="0" is compared
    engine = _make_sequence_engine(tiny_model_config, [5, eos])
    req = GenerationRequest(prompt="hi", max_new_tokens=10, temperature=0.0, stop=["0"])
    resp = engine.generate_request(req)
    assert resp.stop_reason == "eos"
    assert resp.text == "5"


def test_generate_request_max_new_tokens_zero(tiny_model_config):
    """max_new_tokens=0 returns empty text immediately with stop_reason='max_new_tokens'."""
    engine = _make_sequence_engine(tiny_model_config, [10])
    req = GenerationRequest(prompt="hello", max_new_tokens=0, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.text == ""
    assert resp.generated_tokens == 0
    assert resp.stop_reason == "max_new_tokens"


def test_generate_request_rejects_chat_mode_for_llama(tiny_model_config):
    """generate_request() with chat=True raises ValueError for Llama (base model, no template)."""
    engine = _make_sequence_engine(tiny_model_config, [10])
    msgs = [ChatMessage(role="user", content="hello")]
    req = GenerationRequest(messages=msgs, chat=True, max_new_tokens=5, temperature=0.0)
    with pytest.raises(ValueError, match="[Cc]hat"):
        engine.generate_request(req)


def test_generate_request_chat_encodes_chatml_prompt(tiny_qwen3_model_config):
    """generate_request() with chat=True passes the ChatML-formatted string to the tokenizer."""
    tokenizer = _FakeTokenizer([7, 8, 9])
    engine = Engine(
        model=_SequenceModel(tiny_qwen3_model_config, [10]),
        tokenizer=tokenizer,
        config=tiny_qwen3_model_config,
        max_seq_len=tiny_qwen3_model_config.max_seq_len,
    )
    msgs = [ChatMessage(role="user", content="Hello")]
    req = GenerationRequest(messages=msgs, chat=True, max_new_tokens=1, temperature=0.0)
    engine.generate_request(req)
    expected = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    assert tokenizer.encode_calls[0][0] == expected


# ---------------------------------------------------------------------------
# Seeded sampling tests (unit)
# ---------------------------------------------------------------------------


def _make_uniform_engine(tiny_model_config):
    """Engine with _UniformModel for seeded sampling tests."""
    return Engine(
        model=_UniformModel(tiny_model_config),
        tokenizer=_FakeTokenizer([7, 8, 9]),
        config=tiny_model_config,
        max_seq_len=tiny_model_config.max_seq_len,
    )


def test_generate_request_seed_makes_sampling_deterministic(tiny_model_config):
    """Same seed produces identical token sequences across two generate_request() calls."""
    engine1 = _make_uniform_engine(tiny_model_config)
    engine2 = _make_uniform_engine(tiny_model_config)
    req = GenerationRequest(prompt="hello", max_new_tokens=5, temperature=1.0, seed=42)
    resp1 = engine1.generate_request(req)
    resp2 = engine2.generate_request(req)
    assert resp1.text == resp2.text
    assert resp1.generated_tokens == resp2.generated_tokens


def test_generate_request_different_seeds_produce_different_output(tiny_model_config):
    """Different seeds produce different token sequences on uniform logits."""
    engine0 = _make_uniform_engine(tiny_model_config)
    engine1 = _make_uniform_engine(tiny_model_config)
    req0 = GenerationRequest(prompt="hello", max_new_tokens=5, temperature=1.0, seed=0)
    req1 = GenerationRequest(prompt="hello", max_new_tokens=5, temperature=1.0, seed=1)
    resp0 = engine0.generate_request(req0)
    resp1 = engine1.generate_request(req1)
    # With uniform logits over 256 tokens, the probability that 5 independent
    # draws are identical for two different seeds is (1/256)^5 ≈ 10^-12.
    assert resp0.text != resp1.text


def test_generate_request_seed_none_uses_current_prng_state(tiny_model_config):
    """seed=None does not crash and leaves PRNG state managed by the caller."""
    engine = _make_uniform_engine(tiny_model_config)
    req = GenerationRequest(prompt="hello", max_new_tokens=3, temperature=1.0, seed=None)
    resp = engine.generate_request(req)
    assert resp.generated_tokens == 3


def test_generate_request_seed_no_effect_on_greedy(tiny_model_config):
    """Greedy decoding (temperature=0.0) is deterministic regardless of seed."""
    engine0 = _make_sequence_engine(tiny_model_config, [10, 20, 30])
    engine1 = _make_sequence_engine(tiny_model_config, [10, 20, 30])
    req0 = GenerationRequest(prompt="hello", max_new_tokens=3, temperature=0.0, seed=0)
    req1 = GenerationRequest(prompt="hello", max_new_tokens=3, temperature=0.0, seed=99)
    resp0 = engine0.generate_request(req0)
    resp1 = engine1.generate_request(req1)
    assert resp0.text == resp1.text


# ---------------------------------------------------------------------------
# generate_request() stats (unit)
# ---------------------------------------------------------------------------


def test_generate_request_stats_populated(tiny_model_config):
    """generate_request() populates stats on the response."""
    eos = _FakeTokenizer.eos_token_id
    engine = _make_sequence_engine(tiny_model_config, [10, eos])
    req = GenerationRequest(prompt="hello", max_new_tokens=10, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stats is not None


def test_generate_request_stats_eos(tiny_model_config):
    """Stats for eos stop: stop_reason matches, tokens consistent."""
    eos = _FakeTokenizer.eos_token_id
    engine = _make_sequence_engine(tiny_model_config, [10, eos])
    req = GenerationRequest(prompt="hello", max_new_tokens=10, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.stop_reason == "eos" == resp.stop_reason
    assert s.generated_tokens == resp.generated_tokens == 1
    assert s.prompt_tokens == resp.prompt_tokens == s.accepted_prompt_tokens


def test_generate_request_stats_max_new_tokens(tiny_model_config):
    """Stats for max_new_tokens stop: generated_tokens equals max_new_tokens."""
    engine = _make_sequence_engine(tiny_model_config, [10, 20, 30])
    req = GenerationRequest(prompt="hello", max_new_tokens=3, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.stop_reason == "max_new_tokens" == resp.stop_reason
    assert s.generated_tokens == 3 == resp.generated_tokens


def test_generate_request_stats_stop_string(tiny_model_config):
    """Stats for stop_string stop: stop_reason and invariants hold."""
    engine = _make_sequence_engine(tiny_model_config, [10, 99, 30])
    req = GenerationRequest(prompt="hello", max_new_tokens=10, temperature=0.0, stop=["99"])
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.stop_reason == "stop_string" == resp.stop_reason
    assert s.prompt_tokens == s.accepted_prompt_tokens


def test_generate_request_stats_context_length(tiny_model_config):
    """Stats for context_length stop: stop_reason and invariants hold."""
    engine = _make_sequence_engine(tiny_model_config, [10, 20, 30], max_seq_len=4)
    req = GenerationRequest(prompt="hello", max_new_tokens=4, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.stop_reason == "context_length" == resp.stop_reason
    assert s.generated_tokens == 1 == resp.generated_tokens


def test_generate_request_stats_prompt_tokens_equals_accepted(tiny_model_config):
    """prompt_tokens == accepted_prompt_tokens (spec invariant)."""
    engine = _make_sequence_engine(tiny_model_config, [10])
    req = GenerationRequest(prompt="hello", max_new_tokens=1, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    # _FakeTokenizer encodes to [7,8,9] → 3 tokens, no truncation
    assert s.prompt_tokens == 3
    assert s.accepted_prompt_tokens == 3
    assert s.original_prompt_tokens == 3
    assert s.truncated_prompt_tokens == 0


def test_generate_request_stats_active_seq_len(tiny_model_config):
    """active_seq_len == accepted_prompt_tokens + generated_tokens."""
    engine = _make_sequence_engine(tiny_model_config, [10, 20])
    req = GenerationRequest(prompt="hello", max_new_tokens=2, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.active_seq_len == s.accepted_prompt_tokens + s.generated_tokens


def test_generate_request_stats_kv_bytes(tiny_model_config):
    """kv_cache_allocated_bytes uses max_seq_len; kv_cache_active_bytes uses active_seq_len."""
    engine = _make_sequence_engine(tiny_model_config, [10, 20, 30])
    req = GenerationRequest(prompt="hello", max_new_tokens=2, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    # allocated uses full max_seq_len, active uses a smaller active_seq_len
    assert s.kv_cache_allocated_bytes > s.kv_cache_active_bytes
    # ratio must equal max_seq_len / active_seq_len
    assert s.kv_cache_allocated_bytes / s.kv_cache_active_bytes == (
        s.max_seq_len / s.active_seq_len
    )


def test_generate_request_stats_zero_tokens_zero_throughput(tiny_model_config):
    """max_new_tokens=0 produces coherent zero throughput."""
    engine = _make_sequence_engine(tiny_model_config, [10])
    req = GenerationRequest(prompt="hello", max_new_tokens=0, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.generated_tokens == 0
    assert s.decode_tokens_per_sec == 0.0
    assert s.decode_ms == 0.0


def test_generate_request_stats_timing_non_negative(tiny_model_config):
    """All timing fields are non-negative."""
    engine = _make_sequence_engine(tiny_model_config, [10, 20])
    req = GenerationRequest(prompt="hello", max_new_tokens=2, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.prompt_prepare_ms >= 0.0
    assert s.prefill_ms >= 0.0
    assert s.time_to_first_token_ms >= 0.0
    assert s.decode_ms >= 0.0
    assert s.total_ms >= 0.0
    assert s.decode_tokens_per_sec >= 0.0


def test_generate_request_stats_model_type(tiny_model_config):
    """stats.model_type matches config.model_type."""
    engine = _make_sequence_engine(tiny_model_config, [10])
    req = GenerationRequest(prompt="hello", max_new_tokens=1, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stats.model_type == tiny_model_config.model_type


def test_generate_request_stats_context_policy_default(tiny_model_config):
    """Default context_policy is allow_context_stop."""
    engine = _make_sequence_engine(tiny_model_config, [10])
    req = GenerationRequest(prompt="hello", max_new_tokens=1, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stats.context_policy == "allow_context_stop"


def test_generate_request_stats_truncate_left(tiny_model_config):
    """truncate_left policy reduces accepted_prompt_tokens."""
    # _FakeTokenizer returns [7,8,9] (3 tokens). max_seq_len=4, max_new_tokens=2
    # budget = 4-2 = 2 → accepted = [8,9] (last 2 tokens), truncated = 1
    engine = _make_sequence_engine(tiny_model_config, [10], max_seq_len=4)
    req = GenerationRequest(
        prompt="hello", max_new_tokens=2, temperature=0.0,
        context_policy="truncate_left",
    )
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.context_policy == "truncate_left"
    assert s.original_prompt_tokens == 3
    assert s.accepted_prompt_tokens == 2
    assert s.truncated_prompt_tokens == 1
    assert s.prompt_tokens == s.accepted_prompt_tokens


def test_generate_request_stats_reject_raises(tiny_model_config):
    """reject policy raises ContextBudgetError when prompt+generation exceeds max_seq_len."""
    # _FakeTokenizer: 3 prompt tokens, max_new_tokens=62, max_seq_len=64 → 3+62=65 > 64
    engine = _make_sequence_engine(tiny_model_config, [10], max_seq_len=64)
    req = GenerationRequest(
        prompt="hello", max_new_tokens=62, temperature=0.0,
        context_policy="reject",
    )
    with pytest.raises(ContextBudgetError):
        engine.generate_request(req)


def test_generate_stream_final_response_has_stats(tiny_model_config):
    """generate_stream() final GenerationResponse includes populated stats."""
    from tiny_duo_infer.generation import GenerationResponse as GR
    eos = _FakeTokenizer.eos_token_id
    engine = _make_sequence_engine(tiny_model_config, [10, eos])
    req = GenerationRequest(prompt="hello", max_new_tokens=10, temperature=0.0)
    final = None
    for item in engine.generate_stream(req):
        if isinstance(item, GR):
            final = item
    assert final is not None
    assert final.stats is not None
    assert final.stats.stop_reason == "eos"


# ---------------------------------------------------------------------------
# GenerationStats quantization fields — T06
# ---------------------------------------------------------------------------


def test_generate_request_stats_quantization_mode_none_by_default(tiny_model_config):
    """Engine constructed without quantization reports mode=none, bits=None."""
    engine = _make_engine(tiny_model_config)
    req = GenerationRequest(prompt="hello", max_new_tokens=1, temperature=0.0)
    resp = engine.generate_request(req)
    assert resp.stats is not None
    assert resp.stats.quantization_mode == "none"
    assert resp.stats.quantization_bits is None
    assert resp.stats.quantization_group_size is None


def test_generate_request_stats_quantization_counts_zero_by_default(tiny_model_config):
    """Engine with no quantization has zero for all weight count/byte fields."""
    engine = _make_engine(tiny_model_config)
    req = GenerationRequest(prompt="hello", max_new_tokens=1, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.quantized_linear_count == 0
    assert s.full_precision_linear_count == 0
    assert s.linear_weight_full_precision_bytes == 0
    assert s.linear_weight_runtime_bytes == 0


def test_generate_request_stats_quantization_mode_from_engine(tiny_model_config):
    """Engine stores quantization config and stamps it into every GenerationStats."""
    from tiny_duo_infer.weights.quantizer import LinearWeightStats

    quant = QuantizationConfig(bits=4, group_size=64)
    ws = LinearWeightStats(
        quantized_linear_count=10,
        full_precision_linear_count=2,
        linear_weight_full_precision_bytes=2_000_000,
        linear_weight_runtime_bytes=500_000,
    )
    engine = _make_engine(tiny_model_config)
    engine._quantization = quant
    engine._linear_weight_stats = ws

    req = GenerationRequest(prompt="hello", max_new_tokens=1, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s.quantization_mode == "int4"
    assert s.quantization_bits == 4
    assert s.quantization_group_size == 64
    assert s.quantized_linear_count == 10
    assert s.full_precision_linear_count == 2
    assert s.linear_weight_full_precision_bytes == 2_000_000
    assert s.linear_weight_runtime_bytes == 500_000


# ---------------------------------------------------------------------------
# Slow smoke tests (require local model artifacts)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_llama_generate_request_stats_smoke():
    """Real Llama model: generate_request() produces non-negative coherent stats."""
    import os
    from pathlib import Path

    model_path = Path(os.environ.get("LLAMA_MODEL_PATH", "./models/llama-3.2-1b"))
    engine = Engine.from_model_path(model_path, max_seq_len=128)
    req = GenerationRequest(prompt="The capital of France is", max_new_tokens=3, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s is not None
    assert s.prompt_prepare_ms >= 0
    assert s.prefill_ms >= 0
    assert s.time_to_first_token_ms >= 0
    assert s.total_ms >= 0
    assert s.prompt_tokens == s.accepted_prompt_tokens
    assert s.active_seq_len == s.accepted_prompt_tokens + s.generated_tokens
    assert s.kv_cache_allocated_bytes > 0
    assert s.kv_cache_active_bytes > 0
    assert s.model_type == "llama"


@pytest.mark.slow
def test_qwen3_generate_request_stats_smoke():
    """Real Qwen3 model: generate_request() produces non-negative coherent stats."""
    import os
    from pathlib import Path

    model_path = Path(os.environ.get("QWEN_MODEL_PATH", "./models/qwen3-0.6b"))
    engine = Engine.from_model_path(model_path, max_seq_len=128)
    req = GenerationRequest(prompt="Hello", max_new_tokens=3, temperature=0.0)
    resp = engine.generate_request(req)
    s = resp.stats
    assert s is not None
    assert s.prompt_prepare_ms >= 0
    assert s.prefill_ms >= 0
    assert s.time_to_first_token_ms >= 0
    assert s.total_ms >= 0
    assert s.prompt_tokens == s.accepted_prompt_tokens
    assert s.active_seq_len == s.accepted_prompt_tokens + s.generated_tokens
    assert s.kv_cache_allocated_bytes > 0
    assert s.kv_cache_active_bytes > 0
    assert s.model_type == "qwen3"
