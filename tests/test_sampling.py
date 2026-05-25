"""
Tests for tiny_duo_infer.sampling.

All tests use fixed synthetic logits. No model artifacts required.

Test categories:
  - greedy(): returns token with highest logit
  - greedy(): deterministic on same input
  - sample(temperature=0.0): equivalent to greedy
  - sample(top_k=1): equivalent to greedy
  - sample(temperature=1.0, top_k=0, top_p=1.0): unconstrained multinomial
  - sample(top_k=k): only samples from top-k tokens
  - sample(top_p=p): nucleus sampling respects probability threshold
  - sample(top_p=p): includes the token that crosses the threshold (no off-by-one)
  - sample(): temperature scaling changes distribution shape
"""

import pytest


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

# TODO M1.6 (greedy), M1.8 (sample): implement once sampling.py is implemented.
