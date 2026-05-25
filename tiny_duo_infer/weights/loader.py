"""
Safetensors weight loader.

Reads one or more .safetensors shards from a local HuggingFace-compatible
model directory and returns a flat dict mapping weight key names to backend
arrays (mx.array in Phase 1).

Sharded models include a model.safetensors.index.json that maps each weight
key to the shard filename containing it. Single-shard models just have
model.safetensors. Both layouts are supported.

The raw HuggingFace key names (e.g. "model.layers.0.self_attn.q_proj.weight")
are passed as-is to llama_converter.py for renaming and validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from safetensors.mlx import load_file


def load_weights(model_path: Path | str) -> dict[str, mx.array]:
    """
    Load all safetensors shards from model_path into a flat key→array dict.

    Handles both single-file (model.safetensors) and sharded layouts
    (model.safetensors.index.json + multiple shard files).

    Returns HuggingFace key names unchanged. Callers should pass the result
    to llama_converter.convert() to get project-namespaced keys.

    Args:
        model_path: path to the model directory.

    Returns:
        dict mapping raw HF key name → mx.array (bfloat16, as stored in the file).
    """
    model_dir = Path(model_path)
    shard_paths = _discover_safetensor_files(model_dir)

    weights: dict[str, mx.array] = {}
    for shard_path in shard_paths:
        shard_weights = load_file(str(shard_path))
        _merge_shard(weights, shard_weights, shard_path)

    return weights


def _discover_safetensor_files(model_dir: Path) -> list[Path]:
    """
    Return safetensors files that should be loaded for a model directory.

    HuggingFace checkpoints use one of two layouts:
      - single-file: `model.safetensors`
      - sharded: `model.safetensors.index.json` plus shard files listed in
        the index's `weight_map`

    The index is treated as authoritative when present. That mirrors HF model
    directories and avoids accidentally loading unrelated `.safetensors` files.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        return _read_shard_paths_from_index(model_dir, index_path)

    single_shard_path = model_dir / "model.safetensors"
    if single_shard_path.exists():
        return [single_shard_path]

    raise FileNotFoundError(
        f"no safetensors weights found in {model_dir}; expected "
        "model.safetensors or model.safetensors.index.json"
    )


def _read_shard_paths_from_index(model_dir: Path, index_path: Path) -> list[Path]:
    """
    Parse a HuggingFace safetensors index and return unique shard paths.

    The `weight_map` maps every tensor key to the shard filename containing it.
    Multiple keys usually point to the same shard, so this function preserves
    first-seen shard order while removing duplicates.
    """
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise ValueError(f"{index_path} must contain a JSON object")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path} must contain a non-empty 'weight_map' object")

    shard_filenames: list[str] = []
    seen_filenames: set[str] = set()
    for tensor_name, shard_filename in weight_map.items():
        if not isinstance(tensor_name, str):
            raise ValueError(f"{index_path} contains a non-string tensor key")
        if not isinstance(shard_filename, str):
            raise ValueError(
                f"{index_path} maps tensor {tensor_name!r} to a non-string shard filename"
            )
        if shard_filename not in seen_filenames:
            seen_filenames.add(shard_filename)
            shard_filenames.append(shard_filename)

    shard_paths = [model_dir / shard_filename for shard_filename in shard_filenames]
    missing_paths = [path for path in shard_paths if not path.exists()]
    if missing_paths:
        missing_list = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"missing safetensors shard file(s): {missing_list}")

    return shard_paths


def _merge_shard(
    weights: dict[str, mx.array],
    shard_weights: dict[str, Any],
    shard_path: Path,
) -> None:
    """
    Merge one shard into the full weight dict, rejecting duplicate tensor keys.

    Duplicate keys indicate a malformed checkpoint or accidental double-loading
    of the same shard. Detecting that here prevents later conversion code from
    silently using whichever value happened to be loaded last.
    """
    for key, value in shard_weights.items():
        if key in weights:
            raise ValueError(f"duplicate tensor key {key!r} while loading {shard_path}")
        weights[key] = value
