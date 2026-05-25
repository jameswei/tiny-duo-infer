"""
Tests for tiny_duo_infer.cache.KVCache.

All tests use TINY_CONFIG dimensions. No model artifacts required.

Test categories:
  - Allocation: buffer shapes are (1, n_kv_heads, max_seq_len, head_dim)
  - update(): writes at the correct position, returns valid slice
  - update(): does NOT advance current_len
  - advance(): increments current_len by n_tokens
  - advance(): called once per token step (not once per layer)
  - current_len: reflects latest advance() call
  - reset(): zeros buffers and resets current_len to 0
  - Prefill sequence: update(position=0, new_len=S) + advance(S) → current_len == S
  - Decode sequence: update(position=current_len, new_len=1) + advance(1) per step
"""

import pytest


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

# TODO M1.5: implement KVCache tests once cache.py is implemented.
