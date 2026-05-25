"""
Model configuration: parses config.json into a typed dataclass.

Reads the HuggingFace-format config.json from the model directory and
extracts the fields needed to construct the model, cache, and RoPE tables.

The ModelConfig dataclass is passed to every layer constructor so that
layer-specific values (n_heads, head_dim, etc.) are never hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Typed model configuration parsed from config.json.

    All architectural dimensions needed to construct the Llama model and
    its KV cache. Derived values (head_dim, n_groups) are computed from
    the raw config fields.

    Llama-3.2-1B values shown as defaults for reference:
        d_model=2048, n_layers=16, n_heads=32, n_kv_heads=8,
        intermediate_size=8192, vocab_size=128256, max_seq_len=131072,
        rope_theta=500000.0, rms_norm_eps=1e-5
    """

    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_seq_len: int
    rope_theta: float
    rms_norm_eps: float

    @property
    def head_dim(self) -> int:
        """Head dimension: d_model // n_heads."""
        return self.d_model // self.n_heads

    @property
    def n_groups(self) -> int:
        """GQA group count: n_heads // n_kv_heads (how many Q heads share each KV head)."""
        return self.n_heads // self.n_kv_heads


def load_config(model_path: Path | str) -> ModelConfig:
    """
    Parse config.json from a local HuggingFace-compatible model directory.

    Args:
        model_path: path to the directory containing config.json.

    Returns:
        ModelConfig with all fields populated.
    """
    raise NotImplementedError
