"""
Tests for tiny_duo_infer.weights.loader and llama_converter.

Test categories:
  - HF key mapping: all 12 Llama-3.2-1B key patterns are translated correctly
  - Shape validation: converted tensors match config-derived shapes
  - Tied embeddings: lm_head.weight equals embed_tokens.weight
  - Missing key reporting: clear error when required key is absent
  - Unexpected key reporting: warning when an unknown key is present
  - Slow: load real safetensors shards and convert (requires model artifacts)
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
from safetensors.mlx import save_file

from tiny_duo_infer.weights.loader import load_weights


# ---------------------------------------------------------------------------
# Safetensors loader tests (no model artifacts required)
# ---------------------------------------------------------------------------

def test_load_weights_single_safetensors_file(tmp_path: Path) -> None:
    """Single-file checkpoints use model.safetensors without an index file."""
    save_file(
        {
            "model.embed_tokens.weight": mx.array([[1, 2], [3, 4]], dtype=mx.float32),
            "model.norm.weight": mx.array([1, 1], dtype=mx.float32),
        },
        str(tmp_path / "model.safetensors"),
    )

    weights = load_weights(tmp_path)

    assert set(weights) == {"model.embed_tokens.weight", "model.norm.weight"}
    assert weights["model.embed_tokens.weight"].shape == (2, 2)
    assert weights["model.embed_tokens.weight"].dtype == mx.float32


def test_load_weights_sharded_safetensors_index(tmp_path: Path) -> None:
    """Sharded checkpoints load each shard listed by model.safetensors.index.json."""
    save_file(
        {"model.embed_tokens.weight": mx.array([[1, 2]], dtype=mx.float32)},
        str(tmp_path / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {"model.norm.weight": mx.array([1, 1], dtype=mx.float32)},
        str(tmp_path / "model-00002-of-00002.safetensors"),
    )
    write_index(
        tmp_path,
        {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
        },
    )

    weights = load_weights(tmp_path)

    assert set(weights) == {"model.embed_tokens.weight", "model.norm.weight"}
    assert weights["model.embed_tokens.weight"].shape == (1, 2)
    assert weights["model.norm.weight"].shape == (2,)


def test_load_weights_index_takes_precedence_over_single_file(tmp_path: Path) -> None:
    """When an index exists, load only shards named by the index."""
    save_file(
        {"ignored.weight": mx.array([0], dtype=mx.float32)},
        str(tmp_path / "model.safetensors"),
    )
    save_file(
        {"used.weight": mx.array([1], dtype=mx.float32)},
        str(tmp_path / "model-00001-of-00001.safetensors"),
    )
    write_index(tmp_path, {"used.weight": "model-00001-of-00001.safetensors"})

    weights = load_weights(tmp_path)

    assert set(weights) == {"used.weight"}


def test_load_weights_preserves_raw_huggingface_key_names(tmp_path: Path) -> None:
    """P1-T03 loads raw keys; P1-T04 is responsible for key conversion."""
    hf_key = "model.layers.0.self_attn.q_proj.weight"
    save_file({hf_key: mx.array([[1]], dtype=mx.float32)}, str(tmp_path / "model.safetensors"))

    weights = load_weights(tmp_path)

    assert hf_key in weights


def test_load_weights_raises_when_no_safetensors_files(tmp_path: Path) -> None:
    """A model directory must contain either a single shard or an index."""
    with pytest.raises(FileNotFoundError, match="no safetensors weights"):
        load_weights(tmp_path)


def test_load_weights_raises_when_index_has_no_weight_map(tmp_path: Path) -> None:
    """HF sharded indexes must contain a non-empty weight_map object."""
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="weight_map"):
        load_weights(tmp_path)


def test_load_weights_raises_when_index_points_to_missing_shard(tmp_path: Path) -> None:
    """Missing shard files should fail before partially loading weights."""
    write_index(tmp_path, {"model.norm.weight": "missing.safetensors"})

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        load_weights(tmp_path)


def test_load_weights_raises_on_duplicate_tensor_key_across_shards(tmp_path: Path) -> None:
    """A repeated key across shards indicates a malformed checkpoint."""
    save_file({"shared.weight": mx.array([1], dtype=mx.float32)}, str(tmp_path / "a.safetensors"))
    save_file({"shared.weight": mx.array([2], dtype=mx.float32)}, str(tmp_path / "b.safetensors"))
    write_index(
        tmp_path,
        {
            "first.reference": "a.safetensors",
            "second.reference": "b.safetensors",
        },
    )

    with pytest.raises(ValueError, match="duplicate tensor key"):
        load_weights(tmp_path)


def write_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    """Write a minimal HF sharded safetensors index file."""
    index = {
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Slow tests (require local model artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_weights_smoke(tmp_path):
    """Load real safetensors shards and verify key mapping and shapes."""
    pytest.skip("not yet implemented")
