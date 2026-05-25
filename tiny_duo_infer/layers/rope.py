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

import mlx.core as mx


def precompute_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float,
) -> tuple[mx.array, mx.array]:
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
    # Even indices 0, 2, 4, ..., head_dim-2 → one per pair
    i = mx.arange(0, head_dim, 2, dtype=mx.float32)  # (head_dim // 2,)
    # freq_i = 1 / (theta ^ (2i / head_dim)) — lower frequencies for later pairs
    freqs = 1.0 / (theta ** (i / head_dim))           # (head_dim // 2,)

    # Absolute positions 0, 1, ..., max_seq_len-1
    positions = mx.arange(max_seq_len, dtype=mx.float32)  # (max_seq_len,)

    # angles[pos, i] = pos * freq_i — outer product via broadcasting
    angles = positions[:, None] * freqs[None, :]          # (max_seq_len, head_dim // 2)

    cos_table = mx.cos(angles)  # (max_seq_len, head_dim // 2)
    sin_table = mx.sin(angles)  # (max_seq_len, head_dim // 2)

    return cos_table, sin_table


def apply_rope(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    offset: int = 0,
) -> mx.array:
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
    B, S, H, Dh = x.shape

    # Select the cos/sin rows for positions [offset, offset+S) and broadcast
    # to (1, S, 1, Dh//2) so they align with (B, S, H, Dh//2)
    cos_s = cos[offset : offset + S].reshape(1, S, 1, Dh // 2)
    sin_s = sin[offset : offset + S].reshape(1, S, 1, Dh // 2)

    # Split each head vector into consecutive pairs along the last dimension.
    # x[..., 0::2]: elements at indices 0, 2, 4, ... (the "x0" of each pair)
    # x[..., 1::2]: elements at indices 1, 3, 5, ... (the "x1" of each pair)
    # Both have shape (B, S, H, Dh // 2)
    x0 = x[..., 0::2]  # (B, S, H, Dh // 2)
    x1 = x[..., 1::2]  # (B, S, H, Dh // 2)

    # Apply the 2D rotation to each pair:
    #   x0' = x0 * cos - x1 * sin
    #   x1' = x0 * sin + x1 * cos
    x0_rot = x0 * cos_s - x1 * sin_s  # (B, S, H, Dh // 2)
    x1_rot = x0 * sin_s + x1 * cos_s  # (B, S, H, Dh // 2)

    # Interleave x0_rot and x1_rot back to (B, S, H, Dh).
    # mx.stack(..., axis=-1) → (B, S, H, Dh // 2, 2)
    # reshape → (B, S, H, Dh), with pairs restored to consecutive positions
    rotated = mx.stack([x0_rot, x1_rot], axis=-1).reshape(B, S, H, Dh)

    return rotated
