"""
Tests for tiny_duo_infer.config.

These tests create tiny HuggingFace-style config.json files on disk. They do
not require real model weights, tokenizer files, MLX, or network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiny_duo_infer.config import ModelConfig, load_config


def write_config(model_dir: Path, **overrides: object) -> None:
    """
    Write a minimal Llama config.json, with optional field overrides.

    The keys use HuggingFace names because `load_config()` is responsible for
    translating from HF config.json names to the internal ModelConfig names.
    """
    raw_config = {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 256,
        "max_position_embeddings": 64,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
    }
    raw_config.update(overrides)
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")


def test_load_config_maps_huggingface_llama_fields(tmp_path: Path) -> None:
    """Parse HF config keys into the internal teaching-friendly dataclass."""
    write_config(tmp_path)

    config = load_config(tmp_path)

    assert config == ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=128,
        vocab_size=256,
        max_seq_len=64,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
    )
    assert config.head_dim == 16
    assert config.n_groups == 2


def test_load_config_accepts_string_path(tmp_path: Path) -> None:
    """The public API accepts either pathlib.Path or string model paths."""
    write_config(tmp_path)

    config = load_config(str(tmp_path))

    assert config.d_model == 64


def test_load_config_reports_missing_required_field(tmp_path: Path) -> None:
    """Missing HF fields fail at load time with a field-specific message."""
    write_config(tmp_path, hidden_size=None)
    raw_config_path = tmp_path / "config.json"
    raw_config = json.loads(raw_config_path.read_text(encoding="utf-8"))
    del raw_config["hidden_size"]
    raw_config_path.write_text(json.dumps(raw_config), encoding="utf-8")

    with pytest.raises(ValueError, match="hidden_size"):
        load_config(tmp_path)


def test_load_config_rejects_non_llama_model_type(tmp_path: Path) -> None:
    """Phase 1 is intentionally scoped to Llama architecture configs."""
    write_config(tmp_path, model_type="gpt2")

    with pytest.raises(ValueError, match="unsupported model_type"):
        load_config(tmp_path)


def test_load_config_requires_json_object(tmp_path: Path) -> None:
    """HF config.json must be an object with named architecture fields."""
    (tmp_path / "config.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_config(tmp_path)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("hidden_size", 0),
        ("num_hidden_layers", -1),
        ("num_attention_heads", 4.5),
        ("num_key_value_heads", True),
        ("rope_theta", 0.0),
        ("rms_norm_eps", False),
    ],
)
def test_load_config_rejects_invalid_field_values(
    tmp_path: Path,
    field_name: str,
    bad_value: object,
) -> None:
    """Invalid scalar types or non-positive values are rejected early."""
    write_config(tmp_path, **{field_name: bad_value})

    with pytest.raises(ValueError, match=field_name):
        load_config(tmp_path)


def test_load_config_requires_heads_to_divide_hidden_size(tmp_path: Path) -> None:
    """Attention head_dim must be an integer for tensor reshapes to work."""
    write_config(tmp_path, hidden_size=66, num_attention_heads=4)

    with pytest.raises(ValueError, match="hidden_size"):
        load_config(tmp_path)


def test_load_config_requires_kv_heads_to_group_evenly(tmp_path: Path) -> None:
    """GQA repeats each KV head across an integer number of query heads."""
    write_config(tmp_path, num_attention_heads=6, num_key_value_heads=4)

    with pytest.raises(ValueError, match="num_attention_heads"):
        load_config(tmp_path)


def test_load_config_raises_file_not_found_for_missing_config(tmp_path: Path) -> None:
    """A model directory without config.json should fail with Python's normal IO error."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path)
