"""
Shared pytest fixtures for tiny-duo-infer tests.

All unit and shape tests use TINY_CONFIG instead of loading real Llama-3.2-1B
weights. This keeps tests fast and hardware-independent.

Tests requiring the real model are marked @pytest.mark.slow and skipped unless
--run-slow is passed to pytest.
"""

import pytest

from tiny_duo_infer.config import ModelConfig


# Tiny Llama-compatible config for fast unit and shape tests.
# Mirrors the Llama-3.2-1B architecture at 1/32 scale.
TINY_CONFIG = {
    "model_type": "llama",
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,            # d_model // n_heads
    "intermediate_size": 128,
    "vocab_size": 256,
    "max_seq_len": 64,
    "rope_theta": 500000.0,
    "rms_norm_eps": 1e-5,
}

# Tiny Qwen3-compatible config for Phase 1.5 config and shape tests.
# The key property is H * Dh != D: 4 * 16 = 64, while D = 32.
TINY_QWEN3_CONFIG = {
    "model_type": "qwen3",
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "intermediate_size": 64,
    "vocab_size": 128,
    "max_seq_len": 128,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that require local model artifacts.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: marks tests that require local model artifacts (deselect with -m 'not slow')",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="pass --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture
def tiny_config() -> dict:
    """Return the shared tiny model config dict for shape and unit tests."""
    return dict(TINY_CONFIG)


@pytest.fixture
def tiny_qwen3_config() -> dict:
    """Return the shared tiny Qwen3 config dict for Phase 1.5 shape tests."""
    return dict(TINY_QWEN3_CONFIG)


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    """Return the shared tiny model config as a typed ModelConfig dataclass."""
    return ModelConfig(
        model_type=TINY_CONFIG["model_type"],
        d_model=TINY_CONFIG["d_model"],
        n_layers=TINY_CONFIG["n_layers"],
        n_heads=TINY_CONFIG["n_heads"],
        n_kv_heads=TINY_CONFIG["n_kv_heads"],
        head_dim=TINY_CONFIG["head_dim"],
        intermediate_size=TINY_CONFIG["intermediate_size"],
        vocab_size=TINY_CONFIG["vocab_size"],
        max_seq_len=TINY_CONFIG["max_seq_len"],
        rope_theta=TINY_CONFIG["rope_theta"],
        rms_norm_eps=TINY_CONFIG["rms_norm_eps"],
    )


@pytest.fixture
def tiny_qwen3_model_config() -> ModelConfig:
    """Return the shared tiny Qwen3 config as a typed ModelConfig dataclass.

    Key property: H * Dh = 4 * 16 = 64 != D = 32. This exercises the A != D
    attention projection shape that is unique to Qwen3.
    """
    return ModelConfig(
        model_type=TINY_QWEN3_CONFIG["model_type"],
        d_model=TINY_QWEN3_CONFIG["d_model"],
        n_layers=TINY_QWEN3_CONFIG["n_layers"],
        n_heads=TINY_QWEN3_CONFIG["n_heads"],
        n_kv_heads=TINY_QWEN3_CONFIG["n_kv_heads"],
        head_dim=TINY_QWEN3_CONFIG["head_dim"],
        intermediate_size=TINY_QWEN3_CONFIG["intermediate_size"],
        vocab_size=TINY_QWEN3_CONFIG["vocab_size"],
        max_seq_len=TINY_QWEN3_CONFIG["max_seq_len"],
        rope_theta=TINY_QWEN3_CONFIG["rope_theta"],
        rms_norm_eps=TINY_QWEN3_CONFIG["rms_norm_eps"],
    )
