"""
Sampling strategies for token selection.

Operates on a single (vocab_size,) logits vector — one position at a time.
All functions are stateless.

Sampling order (must be applied in this order):
  1. Temperature:  logits = logits / max(temperature, 1e-6)
  2. Top-k:        zero out logits outside top-k  (skip if top_k == 0)
  3. Top-p:        zero out logits outside nucleus (skip if top_p == 1.0)
  4. Softmax:      convert to probabilities
  5. Sample:       draw one token from the distribution

Phase 1 milestones:
  M1.6 — greedy() only
  M1.8 — full sample() with temperature, top-k, top-p
"""

from __future__ import annotations

import mlx.core as mx


def greedy(logits: mx.array) -> int:
    """
    Return the token ID with the highest logit.

    Greedy decoding is deterministic: the same logits always produce the same
    token. It is the M1.6 baseline and equivalent to temperature→0 or top_k=1
    in M1.8.

    The engine calls mx.eval(logits) before invoking this function, so the
    array is fully materialised and .item() is a CPU read with no GPU sync.

    Args:
        logits: (vocab_size,) unnormalized log-probabilities for one position.

    Returns:
        int token ID of the highest-scoring token.
    """
    return mx.argmax(logits).item()


def sample(
    logits: any,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """
    Sample one token ID from logits.

    Order of operations (must be applied in this order):
      1. Temperature:  logits = logits / max(temperature, 1e-6)
      2. Top-k:        set logits outside top-k to -inf  (skip if top_k == 0)
      3. Top-p:        set logits outside nucleus to -inf (skip if top_p == 1.0)
      4. Softmax:      convert to probabilities
      5. Sample:       draw one token from the distribution

    Special cases:
      temperature=0.0  →  equivalent to greedy (argmax)
      top_k=1          →  equivalent to greedy
      top_k=0          →  top-k disabled
      top_p=1.0        →  top-p disabled

    Top-p implementation note: keep the smallest prefix of tokens (sorted by
    descending probability) whose cumulative probability >= top_p, INCLUDING
    the token that crosses the threshold.

    Args:
        logits:      (vocab_size,) — single position only.
        temperature: divide logits before softmax. 1.0 = unchanged.
        top_k:       keep only top-k tokens. 0 = disabled.
        top_p:       nucleus threshold. 1.0 = disabled.

    Returns:
        int token ID.
    """
    raise NotImplementedError
