"""
Tests for tiny_duo_infer.tokenizer.loader.Tokenizer.

Unit tests use a lightweight fixture (no real model artifacts).
Slow tests (marked @pytest.mark.slow) require local Llama-3.2-1B artifacts
and are skipped unless --run-slow is passed.

Test categories:
  - encode/decode round-trip
  - BOS/EOS token IDs
  - vocab_size
  - add_special_tokens=False
  - skip_special_tokens=False
  - optional: parity with transformers.AutoTokenizer
"""

import pytest


# ---------------------------------------------------------------------------
# Unit tests (no model artifacts required)
# ---------------------------------------------------------------------------

# TODO M1.1: implement tokenizer tests once Tokenizer is implemented.


# ---------------------------------------------------------------------------
# Slow tests (require local model artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_tokenizer_smoke(tmp_path):
    """Load real tokenizer artifacts and verify encode/decode round-trip."""
    pytest.skip("not yet implemented")
