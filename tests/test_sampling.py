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

from tiny_duo_infer.sampling import greedy


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
