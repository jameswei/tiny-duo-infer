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

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        return list(self.token_ids)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(str(token_id) for token_id in token_ids)


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
