"""
Tests for tiny_duo_infer.cli.

The CLI is intentionally a thin wrapper over Engine and GenerationRequest.
Unit tests use a fake Engine so they verify argument wiring and output
behavior without loading real model artifacts.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from tiny_duo_infer.cli import main, _parse_message
from tiny_duo_infer.generation import ChatMessage, GenerationRequest, GenerationResponse


class _FakeEngine:
    """Engine test double that records construction and generate_request() arguments."""

    from_model_path_calls: list[tuple[Path, int]] = []
    instances: list["_FakeEngine"] = []

    def __init__(self) -> None:
        self.generate_request_calls: list[GenerationRequest] = []

    @classmethod
    def from_model_path(cls, model_path: Path, max_seq_len: int):
        cls.from_model_path_calls.append((model_path, max_seq_len))
        instance = cls()
        cls.instances.append(instance)
        return instance

    def generate_request(self, request: GenerationRequest) -> GenerationResponse:
        self.generate_request_calls.append(request)
        return GenerationResponse(
            text="hello world",
            prompt_tokens=3,
            generated_tokens=2,
            stop_reason="eos",
        )


@pytest.fixture(autouse=True)
def reset_fake_engine() -> None:
    """Clear fake-engine call records before each test."""
    _FakeEngine.from_model_path_calls.clear()
    _FakeEngine.instances.clear()


# ---------------------------------------------------------------------------
# Core output and argument wiring
# ---------------------------------------------------------------------------


def test_main_writes_generated_text_to_stdout():
    """CLI writes response.text exactly to stdout."""
    stdout = StringIO()

    exit_code = main(
        ["--model-path", "models/tiny", "--prompt", "Say hi"],
        engine_cls=_FakeEngine,
        stdout=stdout,
    )

    assert exit_code == 0
    assert stdout.getvalue() == "hello world"


def test_main_passes_model_path_and_max_seq_len_to_engine():
    """CLI loads the engine from the requested local model directory."""
    main(
        ["--model-path", "models/tiny", "--prompt", "Prompt", "--max-seq-len", "128"],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )

    assert _FakeEngine.from_model_path_calls == [(Path("models/tiny"), 128)]


def test_main_passes_generation_arguments_to_engine():
    """CLI forwards prompt and generation parameters to generate_request()."""
    main(
        [
            "--model-path", "models/tiny",
            "--prompt", "The prompt",
            "--max-new-tokens", "7",
            "--temperature", "0.5",
            "--top-k", "4",
            "--top-p", "0.9",
        ],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )

    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.prompt == "The prompt"
    assert req.max_new_tokens == 7
    assert req.temperature == 0.5
    assert req.top_k == 4
    assert req.top_p == 0.9
    assert req.chat is False


def test_main_accepts_qwen3_model_path_and_sampling_flags():
    """Qwen3 uses the same CLI surface; model family is inferred by Engine."""
    stdout = StringIO()

    main(
        [
            "--model-path", "models/qwen3-0.6b",
            "--prompt", "The capital of France is",
            "--max-new-tokens", "8",
            "--temperature", "0.7",
            "--top-p", "0.8",
        ],
        engine_cls=_FakeEngine,
        stdout=stdout,
    )

    assert _FakeEngine.from_model_path_calls == [(Path("models/qwen3-0.6b"), 2048)]
    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.prompt == "The capital of France is"
    assert req.max_new_tokens == 8
    assert req.temperature == 0.7
    assert req.top_p == 0.8
    assert stdout.getvalue() == "hello world"


def test_main_uses_documented_defaults():
    """CLI defaults match the GenerationRequest API defaults."""
    main(
        ["--model-path", "models/tiny", "--prompt", "Prompt"],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )

    assert _FakeEngine.from_model_path_calls == [(Path("models/tiny"), 2048)]
    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.prompt == "Prompt"
    assert req.max_new_tokens == 200
    assert req.temperature == 1.0
    assert req.top_k == 0
    assert req.top_p == 1.0
    assert req.chat is False
    assert req.stop == []
    assert req.seed is None


# ---------------------------------------------------------------------------
# Chat mode
# ---------------------------------------------------------------------------


def test_main_chat_flag_sets_chat_mode():
    """--chat wraps the prompt as a chat-mode request."""
    main(
        ["--model-path", "models/tiny", "--prompt", "Hello", "--chat"],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )

    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.chat is True
    assert req.prompt == "Hello"
    assert req.messages is None


def test_main_message_flag_builds_messages():
    """--message builds ChatMessage list and implies chat=True."""
    main(
        ["--model-path", "models/tiny", "--message", "user:Hello"],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )

    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.chat is True
    assert req.messages == [ChatMessage(role="user", content="Hello")]
    assert req.prompt is None


def test_main_message_splits_on_first_colon_only():
    """Content containing colons is preserved intact."""
    main(
        ["--model-path", "models/tiny", "--message", "user:Hello:World"],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )

    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.messages == [ChatMessage(role="user", content="Hello:World")]


def test_main_multiple_message_flags_build_ordered_messages():
    """Multiple --message flags accumulate into an ordered message list."""
    main(
        [
            "--model-path", "models/tiny",
            "--message", "system:Be helpful.",
            "--message", "user:Hi",
        ],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )

    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.messages == [
        ChatMessage(role="system", content="Be helpful."),
        ChatMessage(role="user", content="Hi"),
    ]


# ---------------------------------------------------------------------------
# Stop strings and seed
# ---------------------------------------------------------------------------


def test_main_stop_flag_sets_stop_strings():
    """--stop flags are collected into request.stop."""
    main(
        [
            "--model-path", "models/tiny",
            "--prompt", "hi",
            "--stop", "END",
            "--stop", "STOP",
        ],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )

    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.stop == ["END", "STOP"]


def test_main_seed_flag_sets_seed():
    """--seed N sets request.seed to N."""
    main(
        ["--model-path", "models/tiny", "--prompt", "hi", "--seed", "42"],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )

    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.seed == 42


# ---------------------------------------------------------------------------
# Stats output
# ---------------------------------------------------------------------------


def test_main_show_stats_prints_stats_after_text():
    """--show-stats appends a stats line after the generated text."""
    stdout = StringIO()

    main(
        ["--model-path", "models/tiny", "--prompt", "hi", "--show-stats"],
        engine_cls=_FakeEngine,
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert output.startswith("hello world")
    assert "prompt_tokens=3" in output
    assert "generated_tokens=2" in output
    assert "stop_reason=eos" in output


def test_main_no_show_stats_by_default():
    """Stats are not printed unless --show-stats is given."""
    stdout = StringIO()

    main(
        ["--model-path", "models/tiny", "--prompt", "hi"],
        engine_cls=_FakeEngine,
        stdout=stdout,
    )

    assert stdout.getvalue() == "hello world"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_main_rejects_missing_prompt_and_message():
    """Neither --prompt nor --message causes a clean error exit."""
    with pytest.raises(SystemExit):
        main(
            ["--model-path", "models/tiny"],
            engine_cls=_FakeEngine,
            stdout=StringIO(),
        )


def test_main_rejects_prompt_and_message_together():
    """--prompt and --message together cause a clean error exit."""
    with pytest.raises(SystemExit):
        main(
            [
                "--model-path", "models/tiny",
                "--prompt", "hi",
                "--message", "user:hi",
            ],
            engine_cls=_FakeEngine,
            stdout=StringIO(),
        )


def test_main_rejects_negative_max_new_tokens():
    """Generation length must be non-negative."""
    with pytest.raises(SystemExit):
        main(
            ["--model-path", "models/tiny", "--prompt", "Prompt", "--max-new-tokens", "-1"],
            engine_cls=_FakeEngine,
            stdout=StringIO(),
        )


def test_main_rejects_zero_max_seq_len():
    """Cache sequence length must be positive."""
    with pytest.raises(SystemExit):
        main(
            ["--model-path", "models/tiny", "--prompt", "Prompt", "--max-seq-len", "0"],
            engine_cls=_FakeEngine,
            stdout=StringIO(),
        )


def test_main_invalid_message_does_not_load_model():
    """Invalid --message (missing colon) fails before loading the model."""
    with pytest.raises(SystemExit):
        main(
            ["--model-path", "models/tiny", "--message", "nocontent"],
            engine_cls=_FakeEngine,
            stdout=StringIO(),
        )
    assert _FakeEngine.from_model_path_calls == []


def test_main_invalid_top_p_does_not_load_model():
    """top_p=0.0 fails GenerationRequest validation before loading the model."""
    with pytest.raises(SystemExit):
        main(
            ["--model-path", "models/tiny", "--prompt", "hi", "--top-p", "0.0"],
            engine_cls=_FakeEngine,
            stdout=StringIO(),
        )
    assert _FakeEngine.from_model_path_calls == []


# ---------------------------------------------------------------------------
# _parse_message unit tests
# ---------------------------------------------------------------------------


def test_parse_message_returns_chat_message():
    """ROLE:CONTENT is parsed into a ChatMessage with the correct fields."""
    msg = _parse_message("user:Hello")
    assert msg == ChatMessage(role="user", content="Hello")


def test_parse_message_splits_on_first_colon():
    """Content containing colons is preserved from the second colon onward."""
    msg = _parse_message("assistant:a:b:c")
    assert msg == ChatMessage(role="assistant", content="a:b:c")


def test_parse_message_raises_on_missing_colon():
    """Input without a colon raises ValueError."""
    with pytest.raises(ValueError, match="ROLE:CONTENT"):
        _parse_message("nocontent")


# ---------------------------------------------------------------------------
# Slow smoke tests (require local model artifacts)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_qwen3_cli_smoke():
    """Run the real CLI path against local Qwen3 artifacts when available."""
    import os

    stdout = StringIO()
    model_path = os.environ.get("QWEN_MODEL_PATH", "./models/qwen3-0.6b")

    exit_code = main(
        [
            "--model-path", model_path,
            "--prompt", "The capital of France is",
            "--max-new-tokens", "2",
            "--temperature", "0.0",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert isinstance(stdout.getvalue(), str)


@pytest.mark.slow
def test_qwen3_cli_chat_smoke():
    """Run CLI chat mode against local Qwen3 artifacts when available."""
    import os

    stdout = StringIO()
    model_path = os.environ.get("QWEN_MODEL_PATH", "./models/qwen3-0.6b")

    exit_code = main(
        [
            "--model-path", model_path,
            "--message", "user:Say hello.",
            "--max-new-tokens", "4",
            "--temperature", "0.0",
            "--show-stats",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    output = stdout.getvalue()
    assert "prompt_tokens" in output
    assert "stop_reason" in output
