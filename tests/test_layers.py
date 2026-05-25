"""
Tests for individual Llama layer implementations.

All tests use TINY_CONFIG (from conftest.py) with randomly initialised weights.
No real model artifacts are required.

Test categories:
  - RMSNorm: output shape, normalisation formula, weight scaling
  - RoPE: precompute_freqs output shape, apply_rope output shape, offset handling
  - LlamaAttention: output shape, KV cache writes, causal mask (prefill), no mask (decode)
  - SwiGLUFFN: output shape, gate × up element-wise, down projection
"""

import pytest


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

# TODO M1.3: implement RMSNorm tests once RMSNorm is implemented.


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

# TODO M1.3: implement RoPE tests once rope.py is implemented.


# ---------------------------------------------------------------------------
# LlamaAttention
# ---------------------------------------------------------------------------

# TODO M1.3: implement attention tests once attention.py is implemented.


# ---------------------------------------------------------------------------
# SwiGLUFFN
# ---------------------------------------------------------------------------

# TODO M1.3: implement FFN tests once feedforward.py is implemented.
