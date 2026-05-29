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

    All architectural dimensions needed to construct supported decoder-only
    models and their KV cache. `head_dim` is stored because Qwen3 may define it
    explicitly instead of deriving it from hidden size. `n_groups` and
    `qk_norm` remain derived from the model family fields.

    Reference values:
        Llama-3.2-1B: d_model=2048, n_layers=16, n_heads=32,
            n_kv_heads=8, head_dim=64, intermediate_size=8192,
            vocab_size=128256, max_seq_len=131072, rope_theta=500000.0,
            rms_norm_eps=1e-5
        Qwen3-0.6B: d_model=1024, n_layers=28, n_heads=16,
            n_kv_heads=8, head_dim=128, intermediate_size=3072,
            vocab_size=151936, max_seq_len=40960, rope_theta=1000000.0,
            rms_norm_eps=1e-6
    """

    model_type: str
    # mapping to `hidden_size`, `D`
    d_model: int
    # mapping to `num_hidden_layers`, `L`
    n_layers: int
    # mapping to `num_attention_heads`, `H`
    n_heads: int
    # mapping to `num_key_value_heads`, `Hkv`
    n_kv_heads: int
    # for Llama3, this can be derived from `hidden_size/num_attention_heads`, but Qwen3 defines it explicitly.
    # `Dh`
    head_dim: int
    # dimension of FFN layer, `I`
    intermediate_size: int
    # `V`
    vocab_size: int
    # `T`
    max_seq_len: int
    rope_theta: float
    rms_norm_eps: float

    # derived value
    @property
    def n_groups(self) -> int:
        """GQA group count: n_heads // n_kv_heads (how many Q heads share each KV head)."""
        return self.n_heads // self.n_kv_heads

    @property
    def qk_norm(self) -> bool:
        """Whether this model family applies per-head Q/K RMSNorm before RoPE."""
        return self.model_type == "qwen3"


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

    model_type = _read_model_type(raw_config)
    n_heads = _read_positive_int(raw_config, "num_attention_heads")

    config = ModelConfig(
        model_type=model_type,
        d_model=_read_positive_int(raw_config, "hidden_size"),
        n_layers=_read_positive_int(raw_config, "num_hidden_layers"),
        n_heads=n_heads,
        n_kv_heads=_read_positive_int(raw_config, "num_key_value_heads"),
        head_dim=_read_head_dim(raw_config, model_type, n_heads),
        intermediate_size=_read_positive_int(raw_config, "intermediate_size"),
        vocab_size=_read_positive_int(raw_config, "vocab_size"),
        max_seq_len=_read_positive_int(raw_config, "max_position_embeddings"),
        rope_theta=_read_positive_float(raw_config, "rope_theta"),
        rms_norm_eps=_read_positive_float(raw_config, "rms_norm_eps"),
    )
    _validate_config(config)
    return config


def _read_model_type(raw_config: dict[str, Any]) -> str:
    """
    Read and validate the supported HuggingFace model_type value.

    Phase 1.5 supports Llama and Qwen3 on the MLX path. Failing early gives a
    clearer error than loading dimensions and later failing in converter or
    model dispatch code.
    """
    model_type = raw_config.get("model_type")
    if model_type not in {"llama", "qwen3"}:
        raise ValueError(
            f"unsupported model_type {model_type!r}; expected 'llama' or 'qwen3'"
        )
    return model_type


def _read_head_dim(raw_config: dict[str, Any], model_type: str, n_heads: int) -> int:
    """
    Read explicit head_dim when present, otherwise derive it for Llama configs.

    Llama-3.2-1B omits `head_dim`, and the correct value is
    `hidden_size // num_attention_heads`. Qwen3-0.6B includes an explicit
    `head_dim` and intentionally has `n_heads * head_dim != hidden_size`.
    """
    if "head_dim" in raw_config:
        return _read_positive_int(raw_config, "head_dim")

    if model_type == "qwen3":
        raise ValueError("missing required config field 'head_dim'")

    hidden_size = _read_positive_int(raw_config, "hidden_size")
    if hidden_size % n_heads != 0:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads when "
            "config field 'head_dim' is absent"
        )
    return hidden_size // n_heads


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
    Qwen3 uses an attention width (`n_heads * head_dim`) that differs from
    hidden size, so projection shape validation belongs in the model-family
    converter rather than in a global config invariant.
    """
    if config.n_heads % config.n_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    attention_width = config.n_heads * config.head_dim
    if config.model_type == "llama" and attention_width != config.d_model:
        raise ValueError(
            "for model_type 'llama', num_attention_heads * head_dim must equal hidden_size"
        )
