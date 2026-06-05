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

from tiny_duo_infer.cli import main, _parse_message, _build_quantization_config
from tiny_duo_infer.generation import (
    ChatMessage,
    GenerationRequest,
    GenerationResponse,
    GenerationStats,
)
from tiny_duo_infer.quantization import QuantizationConfig


def _make_fake_stats(**overrides) -> GenerationStats:
    """Build a `GenerationStats` populated with placeholder values for CLI tests.

    The values are chosen to satisfy `GenerationStats` invariants
    (`prompt_tokens == accepted_prompt_tokens`,
    `active_seq_len == accepted_prompt_tokens + generated_tokens`) so the
    dataclass `__post_init__` does not raise during test setup.
    """
    defaults: dict = dict(
        context_policy="allow_context_stop",
        original_prompt_tokens=3,
        accepted_prompt_tokens=3,
        truncated_prompt_tokens=0,
        rejected_prompt_tokens=0,
        prompt_tokens=3,
        generated_tokens=2,
        stop_reason="eos",
        prompt_prepare_ms=0.5,
        prefill_ms=12.5,
        time_to_first_token_ms=15.25,
        decode_ms=20.0,
        total_ms=33.0,
        decode_tokens_per_sec=100.0,
        kv_cache_allocated_bytes=67_108_864,
        kv_cache_active_bytes=983_040,
        max_seq_len=2048,
        active_seq_len=5,  # accepted (3) + generated (2)
    )
    defaults.update(overrides)
    return GenerationStats(**defaults)


class _FakeEngine:
    """Engine test double that records construction and generate_request() arguments.

    By default the fake returns a `GenerationResponse` with `stats=None` so
    pre-Phase-1.7 tests stay unchanged. Tests that exercise the `--show-stats`
    output path set `_FakeEngine.next_stats` (or `next_response`) before
    calling `main()` to inject a populated `GenerationStats`.

    from_model_path_calls records (model_path, max_seq_len, quantization) tuples
    so tests can assert both the model path and the quantization config forwarded
    by the CLI.
    """

    from_model_path_calls: list[tuple[Path, int, QuantizationConfig | None]] = []
    instances: list["_FakeEngine"] = []
    next_stats: GenerationStats | None = None

    def __init__(self) -> None:
        self.generate_request_calls: list[GenerationRequest] = []

    @classmethod
    def from_model_path(
        cls,
        model_path: Path,
        max_seq_len: int,
        quantization: QuantizationConfig | None = None,
    ):
        cls.from_model_path_calls.append((model_path, max_seq_len, quantization))
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
            stats=type(self).next_stats,
        )


@pytest.fixture(autouse=True)
def reset_fake_engine() -> None:
    """Clear fake-engine call records and injected stats before each test."""
    _FakeEngine.from_model_path_calls.clear()
    _FakeEngine.instances.clear()
    _FakeEngine.next_stats = None


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

    assert _FakeEngine.from_model_path_calls == [(Path("models/tiny"), 128, None)]


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

    assert _FakeEngine.from_model_path_calls == [(Path("models/qwen3-0.6b"), 2048, None)]
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
        stderr=StringIO(),
    )

    assert _FakeEngine.from_model_path_calls == [(Path("models/tiny"), 2048, None)]
    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.prompt == "Prompt"
    assert req.max_new_tokens == 200
    assert req.temperature == 1.0
    assert req.top_k == 0
    assert req.top_p == 1.0
    assert req.chat is False
    assert req.stop == []
    assert req.seed is None
    assert req.context_policy == "allow_context_stop"


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


def test_main_show_stats_writes_full_block_to_stderr():
    """--show-stats writes the full Phase 1.7 stats block to stderr.

    Generated text remains on stdout; stats include all 14 spec-required
    fields. Stdout must not contain any of the stats fields so it can still
    be piped as plain generated text.
    """
    stdout = StringIO()
    stderr = StringIO()
    _FakeEngine.next_stats = _make_fake_stats(
        prompt_tokens=7,
        accepted_prompt_tokens=7,
        original_prompt_tokens=10,
        truncated_prompt_tokens=3,
        active_seq_len=9,
        generated_tokens=2,
        prefill_ms=11.111,
        time_to_first_token_ms=14.444,
        decode_ms=22.5,
        total_ms=35.0,
        decode_tokens_per_sec=42.42,
        kv_cache_allocated_bytes=67_108_864,
        kv_cache_active_bytes=983_040,
        context_policy="truncate_left",
        stop_reason="eos",
    )

    exit_code = main(
        ["--model-path", "models/tiny", "--prompt", "hi", "--show-stats"],
        engine_cls=_FakeEngine,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    # Generated text stays on stdout, untouched by stats.
    assert stdout.getvalue() == "hello world"

    err = stderr.getvalue()
    # Every spec-required field appears in the stats block.
    expected_substrings = [
        "prompt_tokens=7",
        "generated_tokens=2",
        "stop_reason=eos",
        "prefill_ms=11.11",
        "time_to_first_token_ms=14.44",
        "decode_ms=22.50",
        "total_ms=35.00",
        "decode_tokens_per_sec=42.42",
        "kv_cache_allocated_bytes=67108864",
        "kv_cache_active_bytes=983040",
        "context_policy=truncate_left",
        "original_prompt_tokens=10",
        "accepted_prompt_tokens=7",
        "truncated_prompt_tokens=3",
    ]
    for substr in expected_substrings:
        assert substr in err, f"missing {substr!r} in stats block: {err!r}"


def test_main_show_stats_one_field_per_line():
    """The stats block writes one key=value per line, in stable spec order."""
    stdout = StringIO()
    stderr = StringIO()
    _FakeEngine.next_stats = _make_fake_stats()

    main(
        ["--model-path", "models/tiny", "--prompt", "hi", "--show-stats"],
        engine_cls=_FakeEngine,
        stdout=stdout,
        stderr=stderr,
    )

    lines = stderr.getvalue().rstrip("\n").splitlines()
    keys = [line.split("=", 1)[0] for line in lines]
    assert keys == [
        "prompt_tokens",
        "generated_tokens",
        "stop_reason",
        "prefill_ms",
        "time_to_first_token_ms",
        "decode_ms",
        "total_ms",
        "decode_tokens_per_sec",
        "kv_cache_allocated_bytes",
        "kv_cache_active_bytes",
        "context_policy",
        "original_prompt_tokens",
        "accepted_prompt_tokens",
        "truncated_prompt_tokens",
        "quantization_mode",
        "quantization_bits",
        "quantization_group_size",
        "quantized_linear_count",
        "full_precision_linear_count",
        "linear_weight_full_precision_bytes",
        "linear_weight_runtime_bytes",
    ]


def test_main_show_stats_falls_back_when_response_has_no_stats():
    """Engines without stats (legacy fakes) still get a basic stats line on stderr."""
    stdout = StringIO()
    stderr = StringIO()
    # _FakeEngine.next_stats stays None (fixture default).

    main(
        ["--model-path", "models/tiny", "--prompt", "hi", "--show-stats"],
        engine_cls=_FakeEngine,
        stdout=stdout,
        stderr=stderr,
    )

    assert stdout.getvalue() == "hello world"
    err = stderr.getvalue()
    assert "prompt_tokens=3" in err
    assert "generated_tokens=2" in err
    assert "stop_reason=eos" in err


def test_main_no_show_stats_by_default():
    """Without --show-stats, neither stdout nor stderr receives a stats block."""
    stdout = StringIO()
    stderr = StringIO()
    _FakeEngine.next_stats = _make_fake_stats()

    main(
        ["--model-path", "models/tiny", "--prompt", "hi"],
        engine_cls=_FakeEngine,
        stdout=stdout,
        stderr=stderr,
    )

    assert stdout.getvalue() == "hello world"
    assert stderr.getvalue() == ""


# ---------------------------------------------------------------------------
# Context policy flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [
        "allow_context_stop",
        "reject",
        "truncate_left",
        "truncate_right",
        "reserve_generation",
    ],
)
def test_main_context_policy_flag_forwards_value(policy):
    """All five policies are accepted and forwarded to the request."""
    main(
        [
            "--model-path", "models/tiny",
            "--prompt", "hi",
            "--context-policy", policy,
        ],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
        stderr=StringIO(),
    )

    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.context_policy == policy


def test_main_context_policy_default_is_allow_context_stop():
    """Without --context-policy, the request defaults to allow_context_stop."""
    main(
        ["--model-path", "models/tiny", "--prompt", "hi"],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
        stderr=StringIO(),
    )

    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.context_policy == "allow_context_stop"


def test_main_context_policy_with_message_mode_forwards_value():
    """--context-policy is also forwarded when --message is used."""
    main(
        [
            "--model-path", "models/tiny",
            "--message", "user:Hi",
            "--context-policy", "truncate_left",
        ],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
        stderr=StringIO(),
    )

    req = _FakeEngine.instances[0].generate_request_calls[0]
    assert req.chat is True
    assert req.context_policy == "truncate_left"


def test_main_invalid_context_policy_does_not_load_model():
    """Unknown context policy fails before the model is loaded."""
    with pytest.raises(SystemExit):
        main(
            [
                "--model-path", "models/tiny",
                "--prompt", "hi",
                "--context-policy", "nope",
            ],
            engine_cls=_FakeEngine,
            stdout=StringIO(),
            stderr=StringIO(),
        )
    assert _FakeEngine.from_model_path_calls == []


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
# Quantization flags — T04
# ---------------------------------------------------------------------------


def test_main_quantization_default_is_none():
    """--quantization defaults to none; Engine receives quantization=None."""
    main(
        ["--model-path", "models/tiny", "--prompt", "hi"],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )
    _, _, quant = _FakeEngine.from_model_path_calls[0]
    assert quant is None


@pytest.mark.parametrize("flag,expected_bits", [("int4", 4), ("int8", 8)])
def test_main_quantization_flag_forwards_config_to_engine(flag, expected_bits):
    """--quantization int4/int8 builds a QuantizationConfig and forwards it."""
    main(
        ["--model-path", "models/tiny", "--prompt", "hi", "--quantization", flag],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )
    _, _, quant = _FakeEngine.from_model_path_calls[0]
    assert isinstance(quant, QuantizationConfig)
    assert quant.bits == expected_bits
    assert quant.group_size == 64  # default


def test_main_quant_group_size_flag_forwarded():
    """--quant-group-size N is passed through to the QuantizationConfig."""
    main(
        [
            "--model-path", "models/tiny",
            "--prompt", "hi",
            "--quantization", "int4",
            "--quant-group-size", "32",
        ],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )
    _, _, quant = _FakeEngine.from_model_path_calls[0]
    assert isinstance(quant, QuantizationConfig)
    assert quant.group_size == 32


def test_main_quantization_none_explicit_passes_none_to_engine():
    """--quantization none explicitly passes None, same as the default."""
    main(
        ["--model-path", "models/tiny", "--prompt", "hi", "--quantization", "none"],
        engine_cls=_FakeEngine,
        stdout=StringIO(),
    )
    _, _, quant = _FakeEngine.from_model_path_calls[0]
    assert quant is None


def test_main_invalid_quantization_does_not_load_model():
    """Unknown --quantization value fails before the model is loaded."""
    with pytest.raises(SystemExit):
        main(
            ["--model-path", "models/tiny", "--prompt", "hi", "--quantization", "fp8"],
            engine_cls=_FakeEngine,
            stdout=StringIO(),
        )
    assert _FakeEngine.from_model_path_calls == []


def test_main_quant_group_size_zero_does_not_load_model():
    """--quant-group-size 0 is rejected by argparse before the model loads."""
    with pytest.raises(SystemExit):
        main(
            [
                "--model-path", "models/tiny",
                "--prompt", "hi",
                "--quantization", "int4",
                "--quant-group-size", "0",
            ],
            engine_cls=_FakeEngine,
            stdout=StringIO(),
        )
    assert _FakeEngine.from_model_path_calls == []


# ---------------------------------------------------------------------------
# _build_quantization_config unit tests
# ---------------------------------------------------------------------------


def test_build_quantization_config_none_returns_none():
    import argparse
    args = argparse.Namespace(quantization="none", quant_group_size=64)
    assert _build_quantization_config(args) is None


def test_build_quantization_config_int4():
    import argparse
    args = argparse.Namespace(quantization="int4", quant_group_size=64)
    cfg = _build_quantization_config(args)
    assert isinstance(cfg, QuantizationConfig)
    assert cfg.bits == 4
    assert cfg.group_size == 64


def test_build_quantization_config_int8_custom_group_size():
    import argparse
    args = argparse.Namespace(quantization="int8", quant_group_size=32)
    cfg = _build_quantization_config(args)
    assert isinstance(cfg, QuantizationConfig)
    assert cfg.bits == 8
    assert cfg.group_size == 32


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
    """Run CLI chat mode against local Qwen3 artifacts when available.

    Stats from --show-stats are written to stderr per Phase 1.7 spec, so the
    smoke check inspects stderr for the basic stats fields and stdout for the
    generated text.
    """
    import os

    stdout = StringIO()
    stderr = StringIO()
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
        stderr=stderr,
    )

    assert exit_code == 0
    err = stderr.getvalue()
    assert "prompt_tokens" in err
    assert "stop_reason" in err
