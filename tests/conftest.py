"""
Shared pytest fixtures for tiny-duo-infer tests.

All unit and shape tests use TINY_CONFIG instead of loading real Llama-3.2-1B
weights. This keeps tests fast and hardware-independent.

Tests requiring the real model are marked @pytest.mark.slow and skipped unless
--run-slow is passed to pytest.
"""

import pytest


# Tiny Llama-compatible config for fast unit and shape tests.
# Mirrors the Llama-3.2-1B architecture at 1/32 scale.
TINY_CONFIG = {
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
