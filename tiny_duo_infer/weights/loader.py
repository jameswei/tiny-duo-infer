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

from pathlib import Path


def load_weights(model_path: Path | str) -> dict[str, any]:
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
    raise NotImplementedError
