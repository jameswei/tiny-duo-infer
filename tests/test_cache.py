"""
Tests for tiny_duo_infer.cache.KVCache.

This is a high-risk component because incorrect current_len tracking or
update/advance semantics silently corrupt attention outputs.

Test categories:
  - Allocation: buffer shapes are (1, n_kv_heads, max_seq_len, head_dim)
  - update(): writes at the correct position, returns valid prefix slice
  - update(): does NOT advance current_len
  - advance(): increments current_len by n_tokens
  - advance(): called once per token step (not once per layer)
  - current_len: reflects latest advance() call
  - Multi-layer: buffers are independent per layer_idx
  - reset(): zeros buffers and resets current_len to 0
  - reset(): allows re-use for a second request
  - Returned slice is the valid prefix, not the full max_seq_len buffer
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from tiny_duo_infer.cache import KVCache


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

N_LAYERS = 2
N_KV_HEADS = 2
MAX_SEQ_LEN = 8
HEAD_DIM = 4


def make_cache() -> KVCache:
    return KVCache(N_LAYERS, N_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)


def filled_kv(new_len: int, fill: float = 1.0) -> tuple[mx.array, mx.array]:
    """Return (new_k, new_v) filled with `fill` of shape (1, N_KV_HEADS, new_len, HEAD_DIM)."""
    shape = (1, N_KV_HEADS, new_len, HEAD_DIM)
    return mx.full(shape, fill), mx.full(shape, fill)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

def test_initial_buffer_shape():
    cache = make_cache()
    expected = (1, N_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM)
    for layer in range(N_LAYERS):
        assert cache._keys[layer].shape == expected
        assert cache._values[layer].shape == expected


def test_initial_current_len_is_zero():
    assert make_cache().current_len == 0


def test_initial_buffers_are_zeros():
    cache = make_cache()
    mx.eval(cache._keys[0], cache._values[0])
    assert mx.all(cache._keys[0] == 0).item()
    assert mx.all(cache._values[0] == 0).item()


# ---------------------------------------------------------------------------
# update() — basic shape and content
# ---------------------------------------------------------------------------

def test_update_prefill_returned_slice_shape():
    """update() returns (1, n_kv_heads, position+new_len, head_dim), not the full buffer."""
    cache = make_cache()
    new_k, new_v = filled_kv(3)
    k_out, v_out = cache.update(0, new_k, new_v, position=0)
    mx.eval(k_out, v_out)
    assert k_out.shape == (1, N_KV_HEADS, 3, HEAD_DIM)
    assert v_out.shape == (1, N_KV_HEADS, 3, HEAD_DIM)


def test_update_prefill_writes_correct_values():
    """update() makes the returned slice equal to the written new_k/new_v."""
    cache = make_cache()
    fill = mx.full((1, N_KV_HEADS, 3, HEAD_DIM), 7.0)
    k_out, v_out = cache.update(0, fill, fill, position=0)
    mx.eval(k_out, v_out)
    assert mx.all(k_out == 7.0).item()
    assert mx.all(v_out == 7.0).item()


def test_update_does_not_advance_current_len():
    """update() must NOT change current_len — that is advance()'s job."""
    cache = make_cache()
    new_k, new_v = filled_kv(3)
    cache.update(0, new_k, new_v, position=0)
    assert cache.current_len == 0


def test_update_returns_prefix_not_full_buffer():
    """The returned tensor width equals position+new_len, not MAX_SEQ_LEN."""
    cache = make_cache()
    new_k, new_v = filled_kv(2)
    k_out, v_out = cache.update(0, new_k, new_v, position=0)
    assert k_out.shape[2] == 2
    assert k_out.shape[2] != MAX_SEQ_LEN


# ---------------------------------------------------------------------------
# advance() and current_len
# ---------------------------------------------------------------------------

def test_advance_increments_current_len():
    cache = make_cache()
    cache.advance(3)
    assert cache.current_len == 3


def test_advance_accumulates_across_multiple_calls():
    """Prefill advance then per-decode advance accumulate correctly."""
    cache = make_cache()
    cache.advance(3)  # prefill
    cache.advance(1)  # decode step 1
    cache.advance(1)  # decode step 2
    assert cache.current_len == 5


def test_advance_called_once_per_step_not_per_layer():
    """Simulates 2 layers calling update(), then engine calls advance() once."""
    cache = make_cache()
    new_k, new_v = filled_kv(3)
    cache.update(0, new_k, new_v, position=0)
    cache.update(1, new_k, new_v, position=0)
    assert cache.current_len == 0  # still 0 — advance not called yet
    cache.advance(3)
    assert cache.current_len == 3  # one advance covers all layers


# ---------------------------------------------------------------------------
# Decode-step update
# ---------------------------------------------------------------------------

def test_update_decode_returned_slice_grows():
    """Each decode step returns a slice one wider than the prefill slice."""
    cache = make_cache()
    new_k, new_v = filled_kv(3)
    cache.update(0, new_k, new_v, position=0)
    cache.advance(3)

    k1, v1 = filled_kv(1)
    k_out, _ = cache.update(0, k1, v1, position=cache.current_len)
    mx.eval(k_out)
    assert k_out.shape == (1, N_KV_HEADS, 4, HEAD_DIM)


def test_update_decode_writes_correct_position():
    """Decode token written at position 3 has the expected value; earlier slots unchanged."""
    cache = make_cache()
    zero_k = mx.zeros((1, N_KV_HEADS, 3, HEAD_DIM))
    cache.update(0, zero_k, zero_k, position=0)
    cache.advance(3)

    token_k = mx.full((1, N_KV_HEADS, 1, HEAD_DIM), 9.0)
    k_out, _ = cache.update(0, token_k, token_k, position=cache.current_len)
    mx.eval(k_out)
    assert mx.all(k_out[:, :, 3:4, :] == 9.0).item()
    assert mx.all(k_out[:, :, :3, :] == 0.0).item()


# ---------------------------------------------------------------------------
# Multi-layer independence
# ---------------------------------------------------------------------------

def test_update_layers_are_independent():
    """Writing to layer 0 does not corrupt layer 1's buffer."""
    cache = make_cache()
    fill = mx.full((1, N_KV_HEADS, 2, HEAD_DIM), 5.0)
    zeros = mx.zeros((1, N_KV_HEADS, 2, HEAD_DIM))
    cache.update(0, fill, fill, position=0)
    k1, _ = cache.update(1, zeros, zeros, position=0)
    mx.eval(k1)
    assert mx.all(k1 == 0.0).item()


def test_update_each_layer_holds_its_own_data():
    """Different layers can hold different values at the same positions."""
    cache = make_cache()
    k0 = mx.full((1, N_KV_HEADS, 2, HEAD_DIM), 1.0)
    k1 = mx.full((1, N_KV_HEADS, 2, HEAD_DIM), 2.0)
    out0, _ = cache.update(0, k0, k0, position=0)
    out1, _ = cache.update(1, k1, k1, position=0)
    mx.eval(out0, out1)
    assert mx.all(out0 == 1.0).item()
    assert mx.all(out1 == 2.0).item()


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_zeros_all_buffers():
    cache = make_cache()
    fill = mx.ones((1, N_KV_HEADS, 4, HEAD_DIM))
    cache.update(0, fill, fill, position=0)
    cache.advance(4)
    cache.reset()
    mx.eval(cache._keys[0], cache._values[0])
    assert mx.all(cache._keys[0] == 0).item()
    assert mx.all(cache._values[0] == 0).item()


def test_reset_resets_current_len():
    cache = make_cache()
    cache.advance(5)
    cache.reset()
    assert cache.current_len == 0


def test_reset_allows_reuse_for_second_request():
    """A second request after reset() sees a clean cache with no residue."""
    cache = make_cache()
    fill = mx.full((1, N_KV_HEADS, 3, HEAD_DIM), 3.0)
    cache.update(0, fill, fill, position=0)
    cache.advance(3)

    cache.reset()

    fill2 = mx.full((1, N_KV_HEADS, 2, HEAD_DIM), 2.0)
    k_out, _ = cache.update(0, fill2, fill2, position=0)
    mx.eval(k_out)
    assert k_out.shape == (1, N_KV_HEADS, 2, HEAD_DIM)
    assert mx.all(k_out == 2.0).item()


def test_reset_zeros_all_layers():
    """reset() must zero every layer's buffer, not just layer 0."""
    cache = make_cache()
    fill = mx.ones((1, N_KV_HEADS, 2, HEAD_DIM))
    for layer in range(N_LAYERS):
        cache.update(layer, fill, fill, position=0)
    cache.reset()
    for layer in range(N_LAYERS):
        mx.eval(cache._keys[layer])
        assert mx.all(cache._keys[layer] == 0).item()


# ---------------------------------------------------------------------------
# Input validation — update() bounds and shape checks
# ---------------------------------------------------------------------------

def test_update_rejects_negative_layer_idx():
    cache = make_cache()
    k, v = filled_kv(1)
    with pytest.raises(ValueError, match="layer_idx"):
        cache.update(-1, k, v, position=0)


def test_update_rejects_layer_idx_too_large():
    cache = make_cache()
    k, v = filled_kv(1)
    with pytest.raises(ValueError, match="layer_idx"):
        cache.update(N_LAYERS, k, v, position=0)


def test_update_rejects_negative_position():
    cache = make_cache()
    k, v = filled_kv(1)
    with pytest.raises(ValueError, match="position"):
        cache.update(0, k, v, position=-1)


def test_update_rejects_write_past_max_seq_len():
    cache = make_cache()
    k, v = filled_kv(1)
    with pytest.raises(ValueError, match="max_seq_len"):
        cache.update(0, k, v, position=MAX_SEQ_LEN)  # position+1 > MAX_SEQ_LEN


def test_update_rejects_mismatched_new_k_new_v_lengths():
    """new_k and new_v must have the same sequence length (shape[2])."""
    cache = make_cache()
    shape_k = (1, N_KV_HEADS, 2, HEAD_DIM)
    shape_v = (1, N_KV_HEADS, 1, HEAD_DIM)
    with pytest.raises(ValueError, match="same sequence length"):
        cache.update(0, mx.zeros(shape_k), mx.zeros(shape_v), position=0)


def test_update_rejects_wrong_n_kv_heads():
    cache = make_cache()
    bad_shape = (1, N_KV_HEADS + 1, 1, HEAD_DIM)
    with pytest.raises(ValueError, match="n_kv_heads"):
        cache.update(0, mx.zeros(bad_shape), mx.zeros(bad_shape), position=0)


def test_update_rejects_wrong_head_dim():
    cache = make_cache()
    bad_shape = (1, N_KV_HEADS, 1, HEAD_DIM + 1)
    with pytest.raises(ValueError, match="head_dim"):
        cache.update(0, mx.zeros(bad_shape), mx.zeros(bad_shape), position=0)


def test_update_rejects_non_rank4_input():
    cache = make_cache()
    bad = mx.zeros((N_KV_HEADS, 1, HEAD_DIM))  # rank 3
    with pytest.raises(ValueError, match="rank-4"):
        cache.update(0, bad, bad, position=0)


def test_update_rejects_batch_size_not_one():
    cache = make_cache()
    bad = mx.zeros((2, N_KV_HEADS, 1, HEAD_DIM))  # batch=2
    with pytest.raises(ValueError, match="batch dimension"):
        cache.update(0, bad, bad, position=0)


# ---------------------------------------------------------------------------
# Input validation — advance() bounds checks
# ---------------------------------------------------------------------------

def test_advance_rejects_zero():
    with pytest.raises(ValueError, match="n_tokens"):
        make_cache().advance(0)


def test_advance_rejects_negative():
    with pytest.raises(ValueError, match="n_tokens"):
        make_cache().advance(-5)


def test_advance_rejects_exceeding_max_seq_len():
    cache = make_cache()
    with pytest.raises(ValueError, match="max_seq_len"):
        cache.advance(MAX_SEQ_LEN + 1)


def test_advance_rejects_cumulative_overflow():
    """Advancing to exactly max_seq_len is fine; going one more must fail."""
    cache = make_cache()
    cache.advance(MAX_SEQ_LEN - 1)
    cache.advance(1)  # fills the cache exactly — should succeed
    assert cache.current_len == MAX_SEQ_LEN
    with pytest.raises(ValueError, match="max_seq_len"):
        cache.advance(1)  # one beyond capacity
