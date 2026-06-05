"""
Command-line interface for local text generation.

Thin wrapper over Engine and GenerationRequest. Accepts a model path, prompt
or chat messages, and generation parameters; prints generated text to stdout.

Usage (plain completion):
    uv run python -m tiny_duo_infer.cli \\
      --model-path ./models/llama-3.2-1b \\
      --prompt "The capital of France is" \\
      --max-new-tokens 32

Usage (chat mode with implicit user message):
    uv run python -m tiny_duo_infer.cli \\
      --model-path ./models/qwen3-0.6b \\
      --prompt "What is 2+2?" \\
      --chat

Usage (chat mode with explicit messages):
    uv run python -m tiny_duo_infer.cli \\
      --model-path ./models/qwen3-0.6b \\
      --message system:You are a helpful assistant. \\
      --message user:What is 2+2?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from tiny_duo_infer.engine import Engine
from tiny_duo_infer.generation import (
    ChatMessage,
    GenerationRequest,
    GenerationResponse,
    GenerationStats,
)
from tiny_duo_infer.quantization import QuantizationConfig

# Keep these choice lists in lockstep with the spec and with the corresponding
# literals in `tiny_duo_infer.generation` and `tiny_duo_infer.quantization`.
# argparse uses them for both `choices=` validation and `--help` listing.
_CONTEXT_POLICY_CHOICES: tuple[str, ...] = (
    "allow_context_stop",
    "reject",
    "truncate_left",
    "truncate_right",
    "reserve_generation",
)

_QUANTIZATION_CHOICES: tuple[str, ...] = ("none", "int4", "int8")


def main(
    argv: list[str] | None = None,
    *,
    engine_cls: type[Engine] = Engine,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """
    Parse CLI arguments and run local text generation.

    Args:
        argv:       command-line arguments excluding the program name. `None`
                    uses `sys.argv[1:]`, matching normal CLI execution.
        engine_cls: Engine-compatible class, injectable for tests. Runtime uses
                    `Engine.from_model_path()` to load local model artifacts.
        stdout:     text stream that receives generated text only. Stats from
                    `--show-stats` go to `stderr` so stdout stays pipe-friendly.
        stderr:     text stream that receives the `--show-stats` block.

    Returns:
        Process exit code. `0` means generation completed without an exception.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.prompt and not args.message:
        parser.error("one of --prompt or --message is required")
    if args.prompt and args.message:
        parser.error("--prompt and --message are mutually exclusive")

    # Build and validate the request before loading the model so that invalid
    # inputs (bad role, empty content, out-of-range temperature, etc.) fail
    # fast without paying the model-loading cost.
    try:
        if args.message:
            messages = [_parse_message(m) for m in args.message]
            request = GenerationRequest(
                messages=messages,
                chat=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                stop=args.stop or [],
                seed=args.seed,
                context_policy=args.context_policy,
            )
        else:
            request = GenerationRequest(
                prompt=args.prompt,
                chat=args.chat,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                stop=args.stop or [],
                seed=args.seed,
                context_policy=args.context_policy,
            )
    except ValueError as exc:
        parser.error(str(exc))

    engine = engine_cls.from_model_path(
        Path(args.model_path),
        max_seq_len=args.max_seq_len,
        quantization=_build_quantization_config(args),
    )

    response = engine.generate_request(request)
    print(response.text, end="", file=stdout, flush=True)

    if args.show_stats:
        _write_stats_block(response, stderr)

    return 0


def _write_stats_block(response: GenerationResponse, stderr: TextIO) -> None:
    """Render the `--show-stats` block to `stderr` per the Phase 1.7 spec.

    Generated text remains on stdout. Stats go to stderr so callers can pipe
    `tiny_duo_infer.cli ... | downstream` without the stats interleaving.

    When `response.stats` is populated (real engine path after Phase 1.7-T03),
    the full 14-field block is emitted, one `key=value` per line. When stats
    are absent (legacy fakes, unit-test stubs), a short fallback line is
    written so callers can still see prompt/generated/stop fields.
    """
    stats = response.stats
    if stats is None:
        # Backward-compat fallback: no per-request metrics surface available.
        # Match the Phase 1.6 single-line shape so older shell helpers keep
        # parsing the basic fields.
        print(
            f"prompt_tokens={response.prompt_tokens}"
            f" generated_tokens={response.generated_tokens}"
            f" stop_reason={response.stop_reason}",
            file=stderr,
        )
        return

    # Field order follows the Phase 1.7 spec "CLI" section. Each field is on
    # its own line so individual values can be grepped or parsed without
    # depending on whitespace within a long line.
    lines = _format_stats_lines(stats)
    print("\n".join(lines), file=stderr)


def _format_stats_lines(stats: GenerationStats) -> list[str]:
    """Return the ordered `key=value` lines that make up the stats block."""
    return [
        f"prompt_tokens={stats.prompt_tokens}",
        f"generated_tokens={stats.generated_tokens}",
        f"stop_reason={stats.stop_reason}",
        f"prefill_ms={stats.prefill_ms:.2f}",
        f"time_to_first_token_ms={stats.time_to_first_token_ms:.2f}",
        f"decode_ms={stats.decode_ms:.2f}",
        f"total_ms={stats.total_ms:.2f}",
        f"decode_tokens_per_sec={stats.decode_tokens_per_sec:.2f}",
        f"kv_cache_allocated_bytes={stats.kv_cache_allocated_bytes}",
        f"kv_cache_active_bytes={stats.kv_cache_active_bytes}",
        f"context_policy={stats.context_policy}",
        f"original_prompt_tokens={stats.original_prompt_tokens}",
        f"accepted_prompt_tokens={stats.accepted_prompt_tokens}",
        f"truncated_prompt_tokens={stats.truncated_prompt_tokens}",
        f"quantization_mode={stats.quantization_mode}",
        f"quantization_bits={stats.quantization_bits}",
        f"quantization_group_size={stats.quantization_group_size}",
        f"quantized_linear_count={stats.quantized_linear_count}",
        f"full_precision_linear_count={stats.full_precision_linear_count}",
        f"linear_weight_full_precision_bytes={stats.linear_weight_full_precision_bytes}",
        f"linear_weight_runtime_bytes={stats.linear_weight_runtime_bytes}",
    ]


def _build_quantization_config(args: argparse.Namespace) -> QuantizationConfig | None:
    """Translate --quantization / --quant-group-size into a QuantizationConfig.

    Returns None when quantization is disabled (the default), which keeps
    Engine.from_model_path() on the full-precision path.
    """
    if args.quantization == "none":
        return None
    bits = 4 if args.quantization == "int4" else 8
    return QuantizationConfig(bits=bits, group_size=args.quant_group_size)


def _build_parser() -> argparse.ArgumentParser:
    """Build the Phase-1.8 text-generation argument parser."""
    parser = argparse.ArgumentParser(
        prog="tiny-duo-infer",
        description="Generate text locally with tiny-duo-infer.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a local HuggingFace-compatible model directory.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text to complete. Mutually exclusive with --message.",
    )
    parser.add_argument(
        "--message",
        action="append",
        metavar="ROLE:CONTENT",
        help=(
            "Structured chat message in ROLE:CONTENT form. Repeatable. "
            "ROLE:CONTENT is split on the first colon only, so content may "
            "contain colons. Implies --chat; mutually exclusive with --prompt."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=_non_negative_int,
        default=200,
        help="Maximum number of new tokens to generate. Default: 200.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=_positive_int,
        default=2048,
        help="Maximum prompt + generated sequence length. Default: 2048.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. 1.0 = unchanged; 0.0 = greedy.",
    )
    parser.add_argument(
        "--top-k",
        type=_non_negative_int,
        default=0,
        help="Top-k sampling limit. 0 disables it.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus threshold. 1.0 disables it.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help=(
            "Format --prompt as a user chat message and apply the model's "
            "chat template before tokenization."
        ),
    )
    parser.add_argument(
        "--stop",
        action="append",
        metavar="TEXT",
        help=(
            "Stop generation when TEXT appears in the output. Repeatable. "
            "The stop marker is not included in returned text."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic probabilistic sampling.",
    )
    parser.add_argument(
        "--context-policy",
        choices=_CONTEXT_POLICY_CHOICES,
        default="allow_context_stop",
        help=(
            "Per-request context-budget policy applied before prefill. "
            "allow_context_stop preserves the Phase 1.6 behavior; "
            "reject fails when prompt + max_new_tokens > max_seq_len; "
            "truncate_left/right drop earliest/latest prompt tokens to fit; "
            "reserve_generation is truncate_left framed for chat prompts. "
            "Default: allow_context_stop."
        ),
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help=(
            "After generation, write a stats block to stderr (so stdout "
            "still contains only the generated text). The block reports "
            "timing, token accounting, KV-cache memory, and the applied "
            "context policy, one key=value per line."
        ),
    )
    parser.add_argument(
        "--quantization",
        choices=_QUANTIZATION_CHOICES,
        default="none",
        help=(
            "Weight-only quantization mode. "
            "none = full precision (default); "
            "int4 = INT4 affine quantization; "
            "int8 = INT8 affine quantization. "
            "Invalid group sizes or non-divisible matrix shapes fail before generation."
        ),
    )
    parser.add_argument(
        "--quant-group-size",
        type=_positive_int,
        default=64,
        help=(
            "Quantization group size: elements per group along the input dimension. "
            "Each group gets its own scale and bias. "
            "Must evenly divide every quantized weight's input dimension. "
            "Default: 64."
        ),
    )
    return parser


def _parse_message(raw: str) -> ChatMessage:
    """Parse ROLE:CONTENT into a ChatMessage, splitting on the first colon only."""
    if ":" not in raw:
        raise ValueError(f"--message must be ROLE:CONTENT, got {raw!r}")
    role, content = raw.split(":", 1)
    return ChatMessage(role=role, content=content)


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _non_negative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
