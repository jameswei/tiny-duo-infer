"""
Rotary Positional Embeddings (RoPE).

RoPE encodes absolute position information into Q and K tensors before the
attention dot-product. Unlike learned positional embeddings, RoPE uses a fixed
mathematical rotation that naturally preserves relative position relationships
in the attention scores.

How it works:
  Each head vector of dimension Dh is split into Dh/2 consecutive pairs (x0, x1).
  Each pair is rotated by an angle that depends on the pair index and the
  absolute token position:
      x0' = x0 * cos(pos * freq_i) - x1 * sin(pos * freq_i)
      x1' = x0 * sin(pos * freq_i) + x1 * cos(pos * freq_i)

  Frequency for pair i: freq_i = 1 / (theta ^ (2i / head_dim))
  For Llama-3.2-1B: theta = 500_000.0 (large theta = longer effective context)

Usage:
  1. At model init: cos_table, sin_table = precompute_freqs(head_dim, max_seq_len, theta)
  2. Each forward pass: q = apply_rope(q, cos_table, sin_table, offset=position_offset)
                        k = apply_rope(k, cos_table, sin_table, offset=position_offset)

  The offset distinguishes prefill (offset=0) from decode (offset=tokens_so_far).
"""

from __future__ import annotations


def precompute_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float,
) -> tuple[any, any]:
    """
    Precompute RoPE cosine and sine tables.

    Returns (cos_table, sin_table), each of shape (max_seq_len, head_dim // 2).
    Called once at model init; the tables are stored and reused every forward pass.

    Frequency formula: freq_i = 1 / (theta ^ (2i / head_dim))
    for i in 0 .. head_dim // 2

    The result is indexed by absolute position during apply_rope.

    Args:
        head_dim:    head dimension (Dh); must be even.
        max_seq_len: maximum sequence length; determines table height.
        theta:       RoPE base frequency (500_000.0 for Llama-3.2-1B).

    Returns:
        (cos_table, sin_table): each (max_seq_len, head_dim // 2).
    """
    raise NotImplementedError


def apply_rope(
    x: any,
    cos: any,
    sin: any,
    offset: int = 0,
) -> any:
    """
    Compute and return rotary positional embeddings applied to Q or K.

    Splits each head vector into consecutive pairs (x0, x1), then rotates:
        x0' = x0 * cos[pos] - x1 * sin[pos]
        x1' = x0 * sin[pos] + x1 * cos[pos]

    The `offset` ensures that decode steps use the correct absolute position.
    During prefill, positions are 0..S-1. During decode step t, the single
    new token sits at position (prompt_len + t), so offset = prompt_len + t.

    Args:
        x:      (B, S, H, Dh) — Q or K, BEFORE head transpose.
        cos:    (max_seq_len, Dh // 2) cosine table from precompute_freqs.
        sin:    (max_seq_len, Dh // 2) sine table from precompute_freqs.
        offset: starting absolute position; 0 for prefill, current_len for decode.

    Returns:
        (B, S, H, Dh) tensor with RoPE applied.
    """
    raise NotImplementedError
