"""
SwiGLU Feed-Forward Network.

Llama uses a gated FFN variant called SwiGLU. Unlike a standard two-layer FFN
(W2 * silu(W1 * x)), SwiGLU uses three weight matrices and a multiplicative
gate:
    gate = silu(gate_proj(x))   — activation path, shape (B, S, I)
    up   = up_proj(x)           — gate path,       shape (B, S, I)
    output = down_proj(gate * up)  — (B, S, D)

Where:
    I = intermediate_size = 8192 for Llama-3.2-1B
    silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

The gate controls how much of `up` passes through, acting as a learned filter.
This architecture gives better task performance than vanilla ReLU/SiLU FFNs
at the same parameter count.

HF weight key names → project names:
    model.layers.{i}.mlp.gate_proj.weight  → layers.{i}.mlp.gate_proj.weight
    model.layers.{i}.mlp.up_proj.weight    → layers.{i}.mlp.up_proj.weight
    model.layers.{i}.mlp.down_proj.weight  → layers.{i}.mlp.down_proj.weight
"""

from __future__ import annotations

from tiny_duo_infer.config import ModelConfig
from tiny_duo_infer.models.base import Module


class SwiGLUFFN(Module):
    """
    SwiGLU Feed-Forward Network used in Llama.

    Three linear projections: gate_proj and up_proj expand to intermediate_size,
    their element-wise product (after silu on gate) is projected back down.

    Attributes:
        gate_proj: Linear (d_model → intermediate_size)
        up_proj:   Linear (d_model → intermediate_size)
        down_proj: Linear (intermediate_size → d_model)
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Args:
            config: model config (d_model, intermediate_size).
        """
        raise NotImplementedError

    def forward(self, x: any) -> any:
        """
        Args:
            x: (B, S, D) input hidden states.
        Returns:
            (B, S, D) FFN output.
        """
        raise NotImplementedError
