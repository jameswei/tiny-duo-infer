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

import mlx.core as mx


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
        self._n_layers = n_layers
        self._n_kv_heads = n_kv_heads
        self._max_seq_len = max_seq_len
        self._head_dim = head_dim
        self._current_len = 0
        shape = (1, n_kv_heads, max_seq_len, head_dim)
        self._keys: list[mx.array] = [mx.zeros(shape) for _ in range(n_layers)]
        self._values: list[mx.array] = [mx.zeros(shape) for _ in range(n_layers)]

    def update(
        self,
        layer_idx: int,
        new_k: mx.array,
        new_v: mx.array,
        position: int,
    ) -> tuple[mx.array, mx.array]:
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

        Raises:
            ValueError: if layer_idx or position are out of bounds, or if
                        new_k/new_v have incompatible shapes.
        """
        if not (0 <= layer_idx < self._n_layers):
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self._n_layers})"
            )
        if position < 0:
            raise ValueError(f"position must be >= 0, got {position}")
        if new_k.ndim != 4 or new_v.ndim != 4:
            raise ValueError(
                f"new_k and new_v must be rank-4, got {new_k.ndim} and {new_v.ndim}"
            )
        if new_k.shape[0] != 1 or new_v.shape[0] != 1:
            raise ValueError(
                f"batch dimension must be 1, got new_k.shape[0]={new_k.shape[0]}, "
                f"new_v.shape[0]={new_v.shape[0]}"
            )
        if new_k.shape[1] != self._n_kv_heads or new_v.shape[1] != self._n_kv_heads:
            raise ValueError(
                f"expected n_kv_heads={self._n_kv_heads}, "
                f"got new_k.shape[1]={new_k.shape[1]}, new_v.shape[1]={new_v.shape[1]}"
            )
        if new_k.shape[2] != new_v.shape[2]:
            raise ValueError(
                f"new_k and new_v must have the same sequence length, "
                f"got {new_k.shape[2]} and {new_v.shape[2]}"
            )
        if new_k.shape[3] != self._head_dim or new_v.shape[3] != self._head_dim:
            raise ValueError(
                f"expected head_dim={self._head_dim}, "
                f"got new_k.shape[3]={new_k.shape[3]}, new_v.shape[3]={new_v.shape[3]}"
            )
        new_len = new_k.shape[2]
        if position + new_len > self._max_seq_len:
            raise ValueError(
                f"write would exceed max_seq_len={self._max_seq_len}: "
                f"position={position}, new_len={new_len}"
            )
        self._keys[layer_idx][:, :, position : position + new_len, :] = new_k
        self._values[layer_idx][:, :, position : position + new_len, :] = new_v
        valid_end = position + new_len
        return (
            self._keys[layer_idx][:, :, :valid_end, :],
            self._values[layer_idx][:, :, :valid_end, :],
        )

    def advance(self, n_tokens: int) -> None:
        """
        Advance current_len by n_tokens after all layers have written their
        K/V for the current token step.

        Called once per token step by the engine, NOT once per layer:
            model.forward(...)    # all 16 layers call update() at position p
            cache.advance(n)      # current_len += n, now equals p + n

        During prefill: advance(prompt_len).
        During decode:  advance(1).

        Raises:
            ValueError: if n_tokens <= 0 or advancing would exceed max_seq_len.
        """
        if n_tokens <= 0:
            raise ValueError(f"n_tokens must be > 0, got {n_tokens}")
        if self._current_len + n_tokens > self._max_seq_len:
            raise ValueError(
                f"advance({n_tokens}) would exceed max_seq_len={self._max_seq_len}: "
                f"current_len={self._current_len}"
            )
        self._current_len += n_tokens

    @property
    def current_len(self) -> int:
        """
        Number of valid token positions in the cache.
        Reflects the state after the last advance() call.
        All layers share this single value — it does not increment per layer.
        """
        return self._current_len

    def reset(self) -> None:
        """Zero out all buffers and reset current_len to 0. Call between requests."""
        shape = (1, self._n_kv_heads, self._max_seq_len, self._head_dim)
        for i in range(self._n_layers):
            self._keys[i] = mx.zeros(shape)
            self._values[i] = mx.zeros(shape)
        self._current_len = 0

    def eval(self) -> None:
        """
        Materialise all K/V cache buffers on the MLX backend.

        MLX uses lazy evaluation. After prefill, the engine must ensure that
        every layer's cache writes are committed before decode starts reading
        those buffers at the next token position. P1-T15 will refine broader
        evaluation placement, but prefill needs this explicit synchronization
        point to make the cache state concrete.
        """
        mx.eval(*self._keys, *self._values)
