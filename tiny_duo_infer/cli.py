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
from tiny_duo_infer.generation import ChatMessage, GenerationRequest


def main(
    argv: list[str] | None = None,
    *,
    engine_cls: type[Engine] = Engine,
    stdout: TextIO = sys.stdout,
) -> int:
    """
    Parse CLI arguments and run local text generation.

    Args:
        argv:       command-line arguments excluding the program name. `None`
                    uses `sys.argv[1:]`, matching normal CLI execution.
        engine_cls: Engine-compatible class, injectable for tests. Runtime uses
                    `Engine.from_model_path()` to load local model artifacts.
        stdout:     text stream that receives generated text and optional stats.

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
            )
    except ValueError as exc:
        parser.error(str(exc))

    engine = engine_cls.from_model_path(
        Path(args.model_path),
        max_seq_len=args.max_seq_len,
    )

    response = engine.generate_request(request)
    print(response.text, end="", file=stdout, flush=True)

    if args.show_stats:
        print(
            f"\nprompt_tokens={response.prompt_tokens}"
            f" generated_tokens={response.generated_tokens}"
            f" stop_reason={response.stop_reason}",
            file=stdout,
        )

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the Phase-1.6 text-generation argument parser."""
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
        "--show-stats",
        action="store_true",
        help=(
            "After generation, print a stats line: "
            "prompt_tokens=N generated_tokens=N stop_reason=REASON."
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
