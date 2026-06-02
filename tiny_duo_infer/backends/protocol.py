"""
Backend Protocol: defines the tensor operations a backend must provide.

This is a Phase 1 draft written to capture the interface shape early. MLX code
is not required to formally conform until Phase 2, when torch_backend.py is
added and parity tests are introduced.

Tier-1 ops (APIs differ across backends — must go through this protocol):
  softmax, silu, array, eval, to_numpy

Tier-2 candidate ops (used directly in Phase 1 MLX code; portability across
backends must be validated in M2.0 — NumPy, MLX, and PyTorch differ on split/
concat naming, .T behaviour on rank > 2 tensors, dtype promotion, and device
handling):
  @, .T, reshape, split, concatenate, sqrt, exp, cos, sin, arange, zeros, tril
"""

from typing import Protocol

import numpy as np


class Backend(Protocol):
    """
    Minimal interface a backend must implement to run the inference engine.

    Phase 1: MLX is used directly and does not formally implement this Protocol.
    Phase 2: mlx_backend.py, torch_backend.py, and numpy_backend.py will each
    implement this Protocol, and model code will call through it exclusively.
    """

    def softmax(self, x: any, axis: int = -1) -> any:
        """Numerically stable softmax along `axis`."""

    def silu(self, x: any) -> any:
        """SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))."""

    def array(self, data: any, dtype: any = None) -> any:
        """Create a backend tensor from Python data or a NumPy array."""

    def eval(self, *arrays: any) -> None:
        """
        Materialise deferred computation (MLX lazy eval).
        No-op for eager backends (PyTorch, NumPy).
        """

    def to_numpy(self, x: any) -> np.ndarray:
        """
        Convert a backend tensor to a NumPy array for CPU-side processing.
        For example, sampling.
        """
