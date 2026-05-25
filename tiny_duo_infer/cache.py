"""
KV cache: pre-allocated static buffers for single-request inference.

The KV cache holds the key and value tensors computed during prefill and each
decode step, so that the model does not need to recompute them on every step.

Design (Phase 1):
  One pair of (K, V) buffers per transformer layer, allocated at request start.
  Buffer shape per layer:
      keys:   (1, n_kv_heads, max_seq_len, head_dim)
      values: (1, n_kv_heads, max_seq_len, head_dim)

  Only the slice [:, :, :current_len, :] is valid at any point.

Write/commit protocol:
  update(layer_idx, new_k, new_v, position) — writes K/V for one layer at
    `position`; does NOT advance current_len.
  advance(n_tokens) — called ONCE per token step by the engine after all
    layers have written; increments current_len by n_tokens.

This separation ensures current_len is consistent across all layers: the
attention layer receives position_offset as a parameter from the model forward
call, never by reading cache.current_len mid-forward-pass.

Phase 3 will replace this with a PagedAttention block manager that allocates
KV memory in fixed-size pages rather than one contiguous pre-allocated buffer.
"""

from __future__ import annotations


class KVCache:
    """
    Pre-allocated static KV cache for single-request Phase 1 inference.

    Allocates one fixed-size (K, V) buffer pair per layer at construction.
    During prefill, positions [0, prompt_len) are written. During each decode
    step, one new position is written at index current_len.

    Pre-allocation avoids the O(seq_len) copy overhead of growing by
    concatenation each step. The tradeoff is that max_seq_len must be known
    upfront (passed from Engine.from_model_path).

    Buffer shape per layer:
        keys:   (1, n_kv_heads, max_seq_len, head_dim)  — pre-allocated zeros
        values: (1, n_kv_heads, max_seq_len, head_dim)  — pre-allocated zeros

    Only the slice [:, :, :current_len, :] is valid at any point.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
    ) -> None:
        """Allocate zeroed K/V buffers for all layers."""
        raise NotImplementedError

    def update(
        self,
        layer_idx: int,
        new_k: any,
        new_v: any,
        position: int,
    ) -> tuple[any, any]:
        """
        Write new_k/new_v into the pre-allocated buffer starting at `position`.
        Returns the valid K/V slice: [:, :, :position + new_len, :].

        Does NOT advance current_len. Call advance() once after all layers have
        processed the same token step.

        During prefill: position=0, new_len=prompt_len.
        During decode:  position=current_len, new_len=1.

        Args:
            layer_idx: which transformer layer is writing (0-indexed).
            new_k:     (1, n_kv_heads, new_len, head_dim) new key tensor.
            new_v:     (1, n_kv_heads, new_len, head_dim) new value tensor.
            position:  first cache index to write; always equals current_len at
                       the start of the current token step, passed in by the
                       caller (LlamaModel). NOT read from current_len inside
                       update() to avoid mid-forward-pass ambiguity.

        Returns:
            (k_cache, v_cache): valid slices (1, n_kv_heads, position+new_len, head_dim).
        """
        raise NotImplementedError

    def advance(self, n_tokens: int) -> None:
        """
        Advance current_len by n_tokens after all layers have written their
        K/V for the current token step.

        Called once per token step by the engine, NOT once per layer:
            model.forward(...)    # all 16 layers call update() at position p
            cache.advance(n)      # current_len += n, now equals p + n

        During prefill: advance(prompt_len).
        During decode:  advance(1).
        """
        raise NotImplementedError

    @property
    def current_len(self) -> int:
        """
        Number of valid token positions in the cache.
        Reflects the state after the last advance() call.
        All layers share this single value — it does not increment per layer.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Zero out all buffers and reset current_len to 0. Call between requests."""
        raise NotImplementedError
