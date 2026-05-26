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

from tiny_duo_infer.engine import Engine


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
# Slow smoke tests (require local model artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_engine_smoke():
    """Load real weights, generate 10 tokens, verify output is non-empty."""
    pytest.skip("not yet implemented")


@pytest.mark.slow
def test_engine_max_new_tokens():
    """Verify generation stops at max_new_tokens even without EOS."""
    pytest.skip("not yet implemented")


@pytest.mark.slow
def test_engine_greedy_deterministic():
    """Two greedy generate() calls with same prompt produce identical output."""
    pytest.skip("not yet implemented")
