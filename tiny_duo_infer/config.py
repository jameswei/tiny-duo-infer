"""
Model configuration: parses config.json into a typed dataclass.

Reads the HuggingFace-format config.json from the model directory and
extracts the fields needed to construct the model, cache, and RoPE tables.

The ModelConfig dataclass is passed to every layer constructor so that
layer-specific values (n_heads, head_dim, etc.) are never hard-coded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    model_dir = Path(model_path)
    config_path = model_dir / "config.json"

    with config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)

    if not isinstance(raw_config, dict):
        raise ValueError("config.json must contain a JSON object")

    _require_model_type(raw_config)

    config = ModelConfig(
        d_model=_read_positive_int(raw_config, "hidden_size"),
        n_layers=_read_positive_int(raw_config, "num_hidden_layers"),
        n_heads=_read_positive_int(raw_config, "num_attention_heads"),
        n_kv_heads=_read_positive_int(raw_config, "num_key_value_heads"),
        intermediate_size=_read_positive_int(raw_config, "intermediate_size"),
        vocab_size=_read_positive_int(raw_config, "vocab_size"),
        max_seq_len=_read_positive_int(raw_config, "max_position_embeddings"),
        rope_theta=_read_positive_float(raw_config, "rope_theta"),
        rms_norm_eps=_read_positive_float(raw_config, "rms_norm_eps"),
    )
    _validate_config(config)
    return config


def _require_model_type(raw_config: dict[str, Any]) -> None:
    """
    Reject non-Llama configs before interpreting architecture fields.

    Phase 1 implements the Llama block structure only. Failing early gives a
    clearer error than loading the numbers and later failing during weight
    conversion or model construction.
    """
    model_type = raw_config.get("model_type")
    if model_type != "llama":
        raise ValueError(f"unsupported model_type {model_type!r}; expected 'llama'")


def _read_positive_int(raw_config: dict[str, Any], key: str) -> int:
    """
    Read a required positive integer from a HuggingFace config dict.

    Keeping this strict catches incomplete or accidentally edited config.json
    files before they can produce confusing tensor shape errors.
    """
    value = _read_required(raw_config, key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config field {key!r} must be a positive integer")
    return value


def _read_positive_float(raw_config: dict[str, Any], key: str) -> float:
    """
    Read a required positive numeric field and normalize it to float.

    JSON may decode values like 1e-5 as float and values like 500000 as int;
    both are acceptable for numeric hyperparameters.
    """
    value = _read_required(raw_config, key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config field {key!r} must be a positive number")
    return float(value)


def _read_required(raw_config: dict[str, Any], key: str) -> Any:
    """Return a required config value or raise a field-specific error."""
    if key not in raw_config:
        raise ValueError(f"missing required config field {key!r}")
    return raw_config[key]


def _validate_config(config: ModelConfig) -> None:
    """
    Validate cross-field relationships used by attention and KV cache code.

    These checks encode the assumptions behind `head_dim` and `n_groups`.
    Without them, invalid configs would silently floor-divide and create wrong
    tensor shapes later in the model.
    """
    if config.d_model % config.n_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if config.n_heads % config.n_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
