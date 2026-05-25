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
    model.layers.{i}.mlp.gate_proj.weight  → layers.{i}.ffn.gate_proj.weight
    model.layers.{i}.mlp.up_proj.weight    → layers.{i}.ffn.up_proj.weight
    model.layers.{i}.mlp.down_proj.weight  → layers.{i}.ffn.down_proj.weight
"""

from __future__ import annotations

import mlx.core as mx

from tiny_duo_infer.config import ModelConfig
from tiny_duo_infer.models.base import Linear, Module


class SwiGLUFFN(Module):
    """
    SwiGLU Feed-Forward Network used in Llama.

    Three independent linear projections: gate_proj and up_proj expand the
    hidden state from D to I; their element-wise product (with SiLU applied
    to gate only) is projected back to D by down_proj.

    Forward pass:
        gate = gate_proj(x)                       # (B, S, I)
        up   = up_proj(x)                         # (B, S, I)
        out  = down_proj(silu(gate) * up)         # (B, S, D)

    SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

    gate_proj and up_proj are separate projections — they do NOT share weights.
    The gate is what makes this SwiGLU rather than a plain gated FFN: the SiLU
    activation on the gate path allows it to smoothly pass or suppress features.

    Weights (stored as (out_dim, in_dim), applied as x @ weight.T):
        gate_proj: (intermediate_size, d_model)
        up_proj:   (intermediate_size, d_model)
        down_proj: (d_model, intermediate_size)

    Attributes:
        gate_proj: Linear(d_model → intermediate_size)
        up_proj:   Linear(d_model → intermediate_size)
        down_proj: Linear(intermediate_size → d_model)
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Args:
            config: model config supplying d_model and intermediate_size.
        """
        self.gate_proj = Linear(config.d_model, config.intermediate_size)
        self.up_proj   = Linear(config.d_model, config.intermediate_size)
        self.down_proj = Linear(config.intermediate_size, config.d_model)

    def forward(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, S, D) input hidden states.
        Returns:
            (B, S, D) FFN output.
        """
        # Both gate and up project from the same input — separate weight matrices
        gate = self.gate_proj(x)  # (B, S, I)
        up   = self.up_proj(x)    # (B, S, I)

        # SiLU on gate only: x * sigmoid(x) = x / (1 + exp(-x))
        # Do NOT use mlx.nn.SiLU or any high-level activation
        activated = gate * (1.0 / (1.0 + mx.exp(-gate)))  # (B, S, I)

        # Gate controls which features from up pass through
        return self.down_proj(activated * up)  # (B, S, D)
