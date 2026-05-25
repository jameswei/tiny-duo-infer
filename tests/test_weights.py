"""
Tests for tiny_duo_infer.weights.loader and llama_converter.

Test categories:
  - HF key mapping: all 12 Llama-3.2-1B key patterns are translated correctly
  - Shape validation: converted tensors match config-derived shapes
  - Tied embeddings: lm_head.weight equals embed_tokens.weight
  - Missing key reporting: clear error when required key is absent
  - Unexpected key reporting: warning when an unknown key is present
  - Slow: load real safetensors shards and convert (requires model artifacts)
"""

import pytest


# ---------------------------------------------------------------------------
# Unit tests (no model artifacts required)
# ---------------------------------------------------------------------------

# TODO M1.2: implement weight loading and conversion tests once implemented.


# ---------------------------------------------------------------------------
# Slow tests (require local model artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_weights_smoke(tmp_path):
    """Load real safetensors shards and verify key mapping and shapes."""
    pytest.skip("not yet implemented")
