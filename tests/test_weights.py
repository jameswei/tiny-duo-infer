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

from tiny_duo_infer.config import ModelConfig
from tiny_duo_infer.weights.llama_converter import convert
from tiny_duo_infer.weights.qwen3_converter import convert as convert_qwen3
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
# Llama converter tests (no model artifacts required)
# ---------------------------------------------------------------------------

def test_convert_maps_hf_keys_to_project_keys() -> None:
    """All required Llama HF key patterns are translated to project keys."""
    config = tiny_model_config()
    converted = convert(make_hf_weights(config), config)

    assert set(converted) == expected_project_keys(config)
    assert "layers.0.input_norm.weight" in converted
    assert "layers.0.attn.q_proj.weight" in converted
    assert "layers.0.attn.k_proj.weight" in converted
    assert "layers.0.attn.v_proj.weight" in converted
    assert "layers.0.attn.o_proj.weight" in converted
    assert "layers.0.post_attn_norm.weight" in converted
    assert "layers.0.ffn.gate_proj.weight" in converted
    assert "layers.0.ffn.up_proj.weight" in converted
    assert "layers.0.ffn.down_proj.weight" in converted
    assert "final_norm.weight" in converted


def test_convert_validates_project_shapes_from_config() -> None:
    """Converted tensors keep HF layout and match config-derived dimensions."""
    config = tiny_model_config()
    converted = convert(make_hf_weights(config), config)

    assert converted["embed_tokens.weight"].shape == (config.vocab_size, config.d_model)
    assert converted["layers.0.attn.q_proj.weight"].shape == (config.d_model, config.d_model)
    assert converted["layers.0.attn.k_proj.weight"].shape == (
        config.n_kv_heads * config.head_dim,
        config.d_model,
    )
    assert converted["layers.0.ffn.gate_proj.weight"].shape == (
        config.intermediate_size,
        config.d_model,
    )
    assert converted["layers.0.ffn.down_proj.weight"].shape == (
        config.d_model,
        config.intermediate_size,
    )
    assert converted["final_norm.weight"].shape == (config.d_model,)


def test_convert_reuses_embed_tokens_for_missing_tied_lm_head() -> None:
    """When HF omits lm_head.weight, converter ties it to embed_tokens.weight."""
    config = tiny_model_config()
    hf_weights = make_hf_weights(config, include_lm_head=False)

    converted = convert(hf_weights, config)

    assert converted["lm_head.weight"] is converted["embed_tokens.weight"]


def test_convert_uses_checkpoint_lm_head_when_present() -> None:
    """If HF provides lm_head.weight explicitly, keep that tensor."""
    config = tiny_model_config()
    hf_weights = make_hf_weights(config, include_lm_head=True)

    converted = convert(hf_weights, config)

    assert converted["lm_head.weight"] is hf_weights["lm_head.weight"]


def test_convert_reports_missing_required_key() -> None:
    """Missing expected tensors should fail with the project key name."""
    config = tiny_model_config()
    hf_weights = make_hf_weights(config)
    del hf_weights["model.layers.0.self_attn.q_proj.weight"]

    with pytest.raises(ValueError, match="layers.0.attn.q_proj.weight"):
        convert(hf_weights, config)


def test_convert_reports_shape_mismatch() -> None:
    """Wrong tensor shapes should fail at conversion time, not during forward."""
    config = tiny_model_config()
    hf_weights = make_hf_weights(config)
    hf_weights["model.layers.0.self_attn.k_proj.weight"] = mx.zeros(
        (config.d_model, config.d_model),
        dtype=mx.float32,
    )

    with pytest.raises(ValueError, match="self_attn.k_proj.weight"):
        convert(hf_weights, config)


def test_convert_warns_and_ignores_unexpected_hf_key() -> None:
    """Unexpected HF keys are reported but do not block conversion."""
    config = tiny_model_config()
    hf_weights = make_hf_weights(config)
    hf_weights["model.layers.99.self_attn.q_proj.weight"] = mx.zeros(
        (config.d_model, config.d_model),
        dtype=mx.float32,
    )

    with pytest.warns(UserWarning, match="unexpected HF weight key"):
        converted = convert(hf_weights, config)

    assert "model.layers.99.self_attn.q_proj.weight" not in converted


def tiny_model_config() -> ModelConfig:
    """Return a tiny Llama-compatible config for converter unit tests."""
    return ModelConfig(
        model_type="llama",
        d_model=8,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=2,
        intermediate_size=16,
        vocab_size=32,
        max_seq_len=16,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
    )


def make_hf_weights(
    config: ModelConfig,
    *,
    include_lm_head: bool = False,
) -> dict[str, mx.array]:
    """Build a complete synthetic HF Llama weight dict for a tiny config."""
    weights: dict[str, mx.array] = {
        "model.embed_tokens.weight": mx.zeros(
            (config.vocab_size, config.d_model),
            dtype=mx.float32,
        ),
        "model.norm.weight": mx.zeros((config.d_model,), dtype=mx.float32),
    }

    if include_lm_head:
        weights["lm_head.weight"] = mx.ones(
            (config.vocab_size, config.d_model),
            dtype=mx.float32,
        )

    for layer_idx in range(config.n_layers):
        prefix = f"model.layers.{layer_idx}"
        weights.update(
            {
                f"{prefix}.input_layernorm.weight": mx.zeros(
                    (config.d_model,),
                    dtype=mx.float32,
                ),
                f"{prefix}.self_attn.q_proj.weight": mx.zeros(
                    (config.d_model, config.d_model),
                    dtype=mx.float32,
                ),
                f"{prefix}.self_attn.k_proj.weight": mx.zeros(
                    (config.n_kv_heads * config.head_dim, config.d_model),
                    dtype=mx.float32,
                ),
                f"{prefix}.self_attn.v_proj.weight": mx.zeros(
                    (config.n_kv_heads * config.head_dim, config.d_model),
                    dtype=mx.float32,
                ),
                f"{prefix}.self_attn.o_proj.weight": mx.zeros(
                    (config.d_model, config.d_model),
                    dtype=mx.float32,
                ),
                f"{prefix}.post_attention_layernorm.weight": mx.zeros(
                    (config.d_model,),
                    dtype=mx.float32,
                ),
                f"{prefix}.mlp.gate_proj.weight": mx.zeros(
                    (config.intermediate_size, config.d_model),
                    dtype=mx.float32,
                ),
                f"{prefix}.mlp.up_proj.weight": mx.zeros(
                    (config.intermediate_size, config.d_model),
                    dtype=mx.float32,
                ),
                f"{prefix}.mlp.down_proj.weight": mx.zeros(
                    (config.d_model, config.intermediate_size),
                    dtype=mx.float32,
                ),
            }
        )

    return weights


def expected_project_keys(config: ModelConfig) -> set[str]:
    """Return the complete project-key set expected after conversion."""
    keys = {"embed_tokens.weight", "final_norm.weight", "lm_head.weight"}
    for layer_idx in range(config.n_layers):
        prefix = f"layers.{layer_idx}"
        keys.update(
            {
                f"{prefix}.input_norm.weight",
                f"{prefix}.attn.q_proj.weight",
                f"{prefix}.attn.k_proj.weight",
                f"{prefix}.attn.v_proj.weight",
                f"{prefix}.attn.o_proj.weight",
                f"{prefix}.post_attn_norm.weight",
                f"{prefix}.ffn.gate_proj.weight",
                f"{prefix}.ffn.up_proj.weight",
                f"{prefix}.ffn.down_proj.weight",
            }
        )
    return keys


# ---------------------------------------------------------------------------
# Qwen3 converter tests (no model artifacts required)
# ---------------------------------------------------------------------------

def test_qwen3_convert_maps_hf_keys_to_project_keys() -> None:
    """All required Qwen3 HF key patterns are translated to project keys."""
    config = tiny_qwen3_model_config()
    converted = convert_qwen3(make_qwen3_hf_weights(config), config)

    assert set(converted) == expected_qwen3_project_keys(config)
    assert "embed_tokens.weight" in converted
    assert "lm_head.weight" in converted
    assert "final_norm.weight" in converted
    assert "layers.0.attn.q_proj.weight" in converted
    assert "layers.0.attn.k_proj.weight" in converted
    assert "layers.0.attn.v_proj.weight" in converted
    assert "layers.0.attn.o_proj.weight" in converted
    assert "layers.0.attn.q_norm.weight" in converted
    assert "layers.0.attn.k_norm.weight" in converted
    assert "layers.0.ffn.gate_proj.weight" in converted


def test_qwen3_convert_validates_project_shapes_from_config() -> None:
    """Qwen3 converter uses A=H*Dh, including A != D for q_proj/o_proj."""
    config = tiny_qwen3_model_config()
    converted = convert_qwen3(make_qwen3_hf_weights(config), config)
    attention_width = config.n_heads * config.head_dim
    kv_width = config.n_kv_heads * config.head_dim

    assert attention_width != config.d_model
    assert converted["embed_tokens.weight"].shape == (config.vocab_size, config.d_model)
    assert converted["lm_head.weight"].shape == (config.vocab_size, config.d_model)
    assert converted["layers.0.attn.q_proj.weight"].shape == (
        attention_width,
        config.d_model,
    )
    assert converted["layers.0.attn.k_proj.weight"].shape == (
        kv_width,
        config.d_model,
    )
    assert converted["layers.0.attn.v_proj.weight"].shape == (
        kv_width,
        config.d_model,
    )
    assert converted["layers.0.attn.o_proj.weight"].shape == (
        config.d_model,
        attention_width,
    )
    assert converted["layers.0.attn.q_norm.weight"].shape == (config.head_dim,)
    assert converted["layers.0.attn.k_norm.weight"].shape == (config.head_dim,)
    assert converted["layers.0.ffn.down_proj.weight"].shape == (
        config.d_model,
        config.intermediate_size,
    )


def test_qwen3_convert_requires_lm_head_weight() -> None:
    """Qwen3 support treats lm_head.weight as required, not tied from embeddings."""
    config = tiny_qwen3_model_config()
    hf_weights = make_qwen3_hf_weights(config)
    del hf_weights["lm_head.weight"]

    with pytest.raises(ValueError, match="lm_head.weight"):
        convert_qwen3(hf_weights, config)


def test_qwen3_convert_requires_qk_norm_weights() -> None:
    """Qwen3 checkpoints must include q_norm and k_norm per layer."""
    config = tiny_qwen3_model_config()
    hf_weights = make_qwen3_hf_weights(config)
    del hf_weights["model.layers.0.self_attn.q_norm.weight"]

    with pytest.raises(ValueError, match="layers.0.attn.q_norm.weight"):
        convert_qwen3(hf_weights, config)


def test_qwen3_convert_reports_shape_mismatch() -> None:
    """Wrong Qwen3 tensor shapes fail during conversion with the HF key name."""
    config = tiny_qwen3_model_config()
    hf_weights = make_qwen3_hf_weights(config)
    hf_weights["model.layers.0.self_attn.o_proj.weight"] = mx.zeros(
        (config.d_model, config.d_model),
        dtype=mx.float32,
    )

    with pytest.raises(ValueError, match="self_attn.o_proj.weight"):
        convert_qwen3(hf_weights, config)


def test_qwen3_convert_validates_qk_norm_shape() -> None:
    """Q/K norm weights are shared per-head vectors shaped (Dh,), not (H, Dh)."""
    config = tiny_qwen3_model_config()
    hf_weights = make_qwen3_hf_weights(config)
    hf_weights["model.layers.0.self_attn.k_norm.weight"] = mx.zeros(
        (config.n_kv_heads, config.head_dim),
        dtype=mx.float32,
    )

    with pytest.raises(ValueError, match="self_attn.k_norm.weight"):
        convert_qwen3(hf_weights, config)


def test_qwen3_convert_warns_and_ignores_unexpected_hf_key() -> None:
    """Unexpected Qwen3 HF keys are reported but do not block conversion."""
    config = tiny_qwen3_model_config()
    hf_weights = make_qwen3_hf_weights(config)
    hf_weights["model.layers.99.self_attn.q_norm.weight"] = mx.zeros(
        (config.head_dim,),
        dtype=mx.float32,
    )

    with pytest.warns(UserWarning, match="unexpected HF weight key"):
        converted = convert_qwen3(hf_weights, config)

    assert "model.layers.99.self_attn.q_norm.weight" not in converted


def test_qwen3_convert_rejects_non_qwen3_config() -> None:
    """The Qwen3 converter should not silently accept another model family."""
    config = tiny_model_config()

    with pytest.raises(ValueError, match="model_type 'qwen3'"):
        convert_qwen3({}, config)


def tiny_qwen3_model_config() -> ModelConfig:
    """Return a tiny Qwen3-compatible config for converter unit tests."""
    return ModelConfig(
        model_type="qwen3",
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        intermediate_size=64,
        vocab_size=128,
        max_seq_len=128,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
    )


def make_qwen3_hf_weights(config: ModelConfig) -> dict[str, mx.array]:
    """Build a complete synthetic HF Qwen3 weight dict for a tiny config."""
    attention_width = config.n_heads * config.head_dim
    kv_width = config.n_kv_heads * config.head_dim
    weights: dict[str, mx.array] = {
        "model.embed_tokens.weight": mx.zeros(
            (config.vocab_size, config.d_model),
            dtype=mx.float32,
        ),
        "model.norm.weight": mx.zeros((config.d_model,), dtype=mx.float32),
        "lm_head.weight": mx.ones(
            (config.vocab_size, config.d_model),
            dtype=mx.float32,
        ),
    }

    for layer_idx in range(config.n_layers):
        prefix = f"model.layers.{layer_idx}"
        weights.update(
            {
                f"{prefix}.input_layernorm.weight": mx.zeros(
                    (config.d_model,),
                    dtype=mx.float32,
                ),
                f"{prefix}.self_attn.q_proj.weight": mx.zeros(
                    (attention_width, config.d_model),
                    dtype=mx.float32,
                ),
                f"{prefix}.self_attn.k_proj.weight": mx.zeros(
                    (kv_width, config.d_model),
                    dtype=mx.float32,
                ),
                f"{prefix}.self_attn.v_proj.weight": mx.zeros(
                    (kv_width, config.d_model),
                    dtype=mx.float32,
                ),
                f"{prefix}.self_attn.o_proj.weight": mx.zeros(
                    (config.d_model, attention_width),
                    dtype=mx.float32,
                ),
                f"{prefix}.self_attn.q_norm.weight": mx.zeros(
                    (config.head_dim,),
                    dtype=mx.float32,
                ),
                f"{prefix}.self_attn.k_norm.weight": mx.zeros(
                    (config.head_dim,),
                    dtype=mx.float32,
                ),
                f"{prefix}.post_attention_layernorm.weight": mx.zeros(
                    (config.d_model,),
                    dtype=mx.float32,
                ),
                f"{prefix}.mlp.gate_proj.weight": mx.zeros(
                    (config.intermediate_size, config.d_model),
                    dtype=mx.float32,
                ),
                f"{prefix}.mlp.up_proj.weight": mx.zeros(
                    (config.intermediate_size, config.d_model),
                    dtype=mx.float32,
                ),
                f"{prefix}.mlp.down_proj.weight": mx.zeros(
                    (config.d_model, config.intermediate_size),
                    dtype=mx.float32,
                ),
            }
        )

    return weights


def expected_qwen3_project_keys(config: ModelConfig) -> set[str]:
    """Return the complete project-key set expected after Qwen3 conversion."""
    keys = {"embed_tokens.weight", "final_norm.weight", "lm_head.weight"}
    for layer_idx in range(config.n_layers):
        prefix = f"layers.{layer_idx}"
        keys.update(
            {
                f"{prefix}.input_norm.weight",
                f"{prefix}.attn.q_proj.weight",
                f"{prefix}.attn.k_proj.weight",
                f"{prefix}.attn.v_proj.weight",
                f"{prefix}.attn.o_proj.weight",
                f"{prefix}.attn.q_norm.weight",
                f"{prefix}.attn.k_norm.weight",
                f"{prefix}.post_attn_norm.weight",
                f"{prefix}.ffn.gate_proj.weight",
                f"{prefix}.ffn.up_proj.weight",
                f"{prefix}.ffn.down_proj.weight",
            }
        )
    return keys


# ---------------------------------------------------------------------------
# Slow tests (require local model artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_weights_smoke(tmp_path):
    """Load real safetensors shards and verify key mapping and shapes."""
    pytest.skip("not yet implemented")
