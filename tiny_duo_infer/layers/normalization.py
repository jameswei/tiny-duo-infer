"""
RMSNorm: Root Mean Square Layer Normalization.

Used in Llama as a pre-norm (applied BEFORE attention and FFN, not after).
The residual connection bypasses the norm entirely:
    x = x + attn(norm1(x))
    x = x + ffn(norm2(x))

Unlike LayerNorm, RMSNorm has no mean subtraction and no bias term.
This makes it cheaper to compute while retaining re-scaling ability.

Formula:
    rms(x) = sqrt(mean(x^2) + eps)
    y = x / rms(x) * weight

Where:
    weight: (D,) learnable scale, initialised to ones in HF checkpoint.
    eps:    small constant (1e-5 for Llama-3.2-1B) for numerical stability.
"""

from __future__ import annotations

import mlx.core as mx

from tiny_duo_infer.models.base import Module


class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization.

    Formula: y = x / sqrt(mean(x^2) + eps) * weight

    Unlike LayerNorm, RMSNorm has no mean subtraction and no bias term.
    Llama uses pre-norm: RMSNorm is applied BEFORE attention and FFN,
    not after. Residual connections bypass the norm entirely.

    Attributes:
        weight: (D,) scale parameter, initialised to ones in HF checkpoint.
        eps:    small constant for numerical stability (default 1e-5).
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        """
        Args:
            d_model: hidden dimension size, determines weight shape (d_model,).
            eps:     numerical stability constant.
        """
        self.d_model = d_model
        self.eps = eps
        self.weight: mx.array | None = None  # (D,)

    def forward(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, S, D) input tensor.
        Returns:
            (B, S, D) normalised and scaled tensor.
        """
        # RMSNorm rescales each token independently across the hidden dimension
        # only. There is no mean subtraction and bias, unlike LayerNorm.
        mean_square = mx.mean(x * x, axis=-1, keepdims=True)  # (B, S, 1)
        # avoid using `x * 1 / sqrt(...)` for better numerical stability:
        # if sqrt(...) is very small, 1 / sqrt(...) can overflow to INF,
        # and better performance:
        # rsqrt(...) is computed into 1 instruction about 4 cycles on GPU,
        # while 1 / sqrt(...) is computed into 2 instructions (sqrt + div) about 16+20 cycles, almost 9x faster
        # also `* rsqrt(...)` can be kernel fused.
        normalized = x * mx.rsqrt(mean_square + self.eps)  # (B, S, D)
        return normalized * self.weight  # (B, S, D)
