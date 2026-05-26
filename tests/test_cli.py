"""
Tests for tiny_duo_infer.cli.

The CLI is intentionally a thin wrapper over Engine. Unit tests use a fake
Engine class so they verify argument wiring and output behavior without loading
real model artifacts.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from tiny_duo_infer.cli import main


class _FakeEngine:
    """Engine test double that records construction and generate() arguments."""

    from_model_path_calls: list[tuple[Path, int]] = []
    instances: list["_FakeEngine"] = []

    def __init__(self) -> None:
        self.generate_calls: list[tuple[str, int, float, int, float]] = []

    @classmethod
    def from_model_path(cls, model_path: Path, max_seq_len: int):
        cls.from_model_path_calls.append((model_path, max_seq_len))
        instance = cls()
        cls.instances.append(instance)
        return instance

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ):
        self.generate_calls.append(
            (prompt, max_new_tokens, temperature, top_k, top_p)
        )
        yield "hello"
        yield " "
        yield "world"


@pytest.fixture(autouse=True)
def reset_fake_engine() -> None:
    """Clear fake-engine call records before each test."""
    _FakeEngine.from_model_path_calls.clear()
    _FakeEngine.instances.clear()


def test_main_streams_generated_text_to_stdout():
    """CLI writes generated fragments exactly as Engine.generate() yields them."""
    stdout = StringIO()

    exit_code = main(
        [
            "--model-path",
            "models/tiny",
            "--prompt",
            "Say hi",
        ],
        engine_cls=_FakeEngine,
        stdout=stdout,
    )

    assert exit_code == 0
    assert stdout.getvalue() == "hello world"


def test_main_passes_model_path_and_max_seq_len_to_engine():
    """CLI loads the engine from the requested local model directory."""
    stdout = StringIO()

    main(
        [
            "--model-path",
            "models/tiny",
            "--prompt",
            "Prompt",
            "--max-seq-len",
            "128",
        ],
        engine_cls=_FakeEngine,
        stdout=stdout,
    )

    assert _FakeEngine.from_model_path_calls == [(Path("models/tiny"), 128)]


def test_main_passes_generation_arguments_to_engine():
    """CLI forwards prompt and generation parameters to Engine.generate()."""
    stdout = StringIO()

    main(
        [
            "--model-path",
            "models/tiny",
            "--prompt",
            "The prompt",
            "--max-new-tokens",
            "7",
            "--temperature",
            "0.5",
            "--top-k",
            "4",
            "--top-p",
            "0.9",
        ],
        engine_cls=_FakeEngine,
        stdout=stdout,
    )

    engine = _FakeEngine.instances[0]
    assert engine.generate_calls == [("The prompt", 7, 0.5, 4, 0.9)]


def test_main_uses_documented_defaults():
    """CLI defaults match the Phase-1 Engine API defaults."""
    stdout = StringIO()

    main(
        [
            "--model-path",
            "models/tiny",
            "--prompt",
            "Prompt",
        ],
        engine_cls=_FakeEngine,
        stdout=stdout,
    )

    assert _FakeEngine.from_model_path_calls == [(Path("models/tiny"), 2048)]
    engine = _FakeEngine.instances[0]
    assert engine.generate_calls == [("Prompt", 200, 1.0, 0, 1.0)]


def test_main_rejects_negative_max_new_tokens():
    """Generation length must be non-negative."""
    with pytest.raises(SystemExit):
        main(
            [
                "--model-path",
                "models/tiny",
                "--prompt",
                "Prompt",
                "--max-new-tokens",
                "-1",
            ],
            engine_cls=_FakeEngine,
            stdout=StringIO(),
        )


def test_main_rejects_zero_max_seq_len():
    """Cache sequence length must be positive."""
    with pytest.raises(SystemExit):
        main(
            [
                "--model-path",
                "models/tiny",
                "--prompt",
                "Prompt",
                "--max-seq-len",
                "0",
            ],
            engine_cls=_FakeEngine,
            stdout=StringIO(),
        )
