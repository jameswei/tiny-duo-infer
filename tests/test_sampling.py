"""
Tests for tiny_duo_infer.sampling.

All tests use fixed synthetic logits. No model artifacts required.

Test categories:
  - greedy(): returns token with highest logit
  - greedy(): deterministic on same input
  - greedy(): returns a Python int
  - sample(temperature=0.0): equivalent to greedy          (M1.8)
  - sample(top_k=1): equivalent to greedy                  (M1.8)
  - sample(temperature=1.0, top_k=0, top_p=1.0): unconstrained multinomial (M1.8)
  - sample(top_k=k): only samples from top-k tokens        (M1.8)
  - sample(top_p=p): nucleus sampling threshold             (M1.8)
"""

import mlx.core as mx
import pytest

from tiny_duo_infer.sampling import greedy, sample


# ---------------------------------------------------------------------------
# greedy() — M1.6
# ---------------------------------------------------------------------------

def test_greedy_returns_argmax():
    """greedy picks the token with the highest logit value."""
    logits = mx.array([0.1, 0.9, 0.3, 0.7])
    mx.eval(logits)
    assert greedy(logits) == 1


def test_greedy_returns_correct_index_at_end():
    """greedy handles the case where the highest logit is at the last position."""
    logits = mx.array([0.1, 0.2, 0.3, 0.9])
    mx.eval(logits)
    assert greedy(logits) == 3


def test_greedy_returns_correct_index_at_start():
    """greedy handles the case where the highest logit is at position 0."""
    logits = mx.array([5.0, 1.0, 2.0, 3.0])
    mx.eval(logits)
    assert greedy(logits) == 0


def test_greedy_returns_python_int():
    """greedy returns a plain Python int, not an MLX array."""
    logits = mx.array([0.0, 1.0, 0.5])
    mx.eval(logits)
    result = greedy(logits)
    assert isinstance(result, int)


def test_greedy_is_deterministic():
    """greedy always returns the same token for the same logits."""
    logits = mx.array([0.3, 0.7, 0.1, 0.9, 0.2])
    mx.eval(logits)
    assert greedy(logits) == greedy(logits)


def test_greedy_larger_vocab():
    """greedy works correctly across a larger vocabulary."""
    vocab_size = 256
    logits = (mx.arange(vocab_size) == 100).astype(mx.float32) * 10.0
    mx.eval(logits)
    assert greedy(logits) == 100


# ---------------------------------------------------------------------------
# sample() — M1.8
# ---------------------------------------------------------------------------

def test_sample_temperature_zero_matches_greedy():
    """sample(temperature=0.0) always picks the highest logit token."""
    logits = mx.array([0.1, 0.9, 0.3, 0.7])
    mx.eval(logits)
    assert sample(logits, temperature=0.0) == greedy(logits)


def test_sample_top_k_one_matches_greedy():
    """sample(top_k=1) keeps only the top token, equivalent to greedy."""
    logits = mx.array([1.0, 5.0, 2.0, 3.0])
    mx.eval(logits)
    assert sample(logits, top_k=1) == greedy(logits)


def test_sample_seeded_is_deterministic():
    """sample() with the same seed returns the same token twice."""
    logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mx.eval(logits)
    mx.random.seed(42)
    token_a = sample(logits, temperature=1.0)
    mx.random.seed(42)
    token_b = sample(logits, temperature=1.0)
    assert token_a == token_b


def test_sample_unconstrained_produces_varied_tokens():
    """sample() with no constraints eventually samples more than one distinct token."""
    # Uniform logits: all tokens equally likely. With 50 draws from vocab_size=5,
    # the probability of drawing only one distinct token is (1/5)^49 ≈ negligible.
    logits = mx.zeros(5)
    mx.eval(logits)
    seen = {sample(logits, temperature=1.0) for _ in range(50)}
    assert len(seen) > 1


def test_sample_top_k_restricts_candidates():
    """sample(top_k=k) only returns tokens among the top-k highest-logit tokens."""
    # Vocab: tokens 0–9. Tokens 7, 8, 9 have high logits; 0–6 have very low logits.
    # With top_k=3, only tokens 7, 8, 9 should ever be sampled.
    logits = mx.array([-10.0] * 7 + [1.0, 2.0, 3.0])
    mx.eval(logits)
    sampled = {sample(logits, temperature=1.0, top_k=3) for _ in range(100)}
    assert sampled.issubset({7, 8, 9})


def test_sample_top_p_restricts_candidates():
    """sample(top_p=p) only samples from the nucleus that covers probability >= p."""
    # Three tokens: token 2 has logit 10, tokens 0 and 1 have logit -10.
    # After softmax, token 2 has probability ≈ 1.0. With top_p=0.99, only token 2
    # crosses the threshold, so all samples must be token 2.
    logits = mx.array([-10.0, -10.0, 10.0])
    mx.eval(logits)
    sampled = {sample(logits, temperature=1.0, top_p=0.99) for _ in range(20)}
    assert sampled == {2}


def test_sample_returns_python_int():
    """sample() returns a plain Python int, not an MLX array."""
    logits = mx.array([0.0, 1.0, 0.5])
    mx.eval(logits)
    result = sample(logits, temperature=1.0)
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# sample() — parameter validation
# ---------------------------------------------------------------------------

def test_sample_rejects_negative_temperature():
    """sample() raises ValueError for temperature < 0."""
    logits = mx.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="temperature"):
        sample(logits, temperature=-0.1)


def test_sample_rejects_negative_top_k():
    """sample() raises ValueError for top_k < 0."""
    logits = mx.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="top_k"):
        sample(logits, top_k=-1)


def test_sample_rejects_zero_top_p():
    """sample() raises ValueError for top_p=0.0, which would mask all tokens."""
    logits = mx.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="top_p"):
        sample(logits, top_p=0.0)


def test_sample_rejects_top_p_above_one():
    """sample() raises ValueError for top_p > 1.0."""
    logits = mx.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="top_p"):
        sample(logits, top_p=1.1)
