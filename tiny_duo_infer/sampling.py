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


def greedy(logits: any) -> int:
    """
    Return the token ID with the highest logit.

    The simplest possible sampling strategy: always pick the most probable
    token. Produces deterministic, repeatable output given the same model
    and prompt. Good for testing correctness before probabilistic sampling
    is implemented.

    Args:
        logits: (vocab_size,) — single position only.

    Returns:
        int token ID.
    """
    raise NotImplementedError


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
