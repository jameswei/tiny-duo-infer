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


# ---------------------------------------------------------------------------
# Unit tests (no model artifacts required)
# ---------------------------------------------------------------------------

# TODO M1.6: implement engine tests once Engine is implemented.


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
