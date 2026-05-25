"""
Tests for LlamaBlock and LlamaModel assembly.

All tests use TINY_CONFIG with randomly initialised weights. No model artifacts required.

Test categories:
  - LlamaModel forward: output shape (B, S, V) for a batch-1 prompt
  - LlamaBlock forward: output shape (B, S, D)
  - Residual connections: output differs from input (sanity check)
  - Slow: forward pass on real Llama-3.2-1B weights (requires model artifacts)
  - Optional slow: logit parity with transformers reference (tolerance 1e-3)
"""

import pytest


# ---------------------------------------------------------------------------
# Unit tests (no model artifacts required)
# ---------------------------------------------------------------------------

# TODO M1.4: implement model forward tests once LlamaModel is implemented.


# ---------------------------------------------------------------------------
# Slow tests (require local model artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_model_forward_smoke():
    """Load real weights and run a forward pass; verify logits shape."""
    pytest.skip("not yet implemented")
