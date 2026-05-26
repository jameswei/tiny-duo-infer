"""
Command-line interface for local text generation.

Thin wrapper over the Engine class. Accepts a model path, prompt, and
generation parameters; prints generated text to stdout.

Usage:
    uv run python -m tiny_duo_infer.cli \\
      --model-path ./models/llama-3.2-1b \\
      --prompt "The capital of France is" \\
      --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from tiny_duo_infer.engine import Engine


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
        stdout:     text stream that receives generated fragments.

    Returns:
        Process exit code. `0` means generation completed without an exception.
    """
    args = _build_parser().parse_args(argv)
    engine = engine_cls.from_model_path(
        Path(args.model_path),
        max_seq_len=args.max_seq_len,
    )

    for fragment in engine.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    ):
        print(fragment, end="", file=stdout, flush=True)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the Phase-1 text-generation argument parser.

    The CLI intentionally stays thin: all model loading and token generation
    goes through `Engine`. Sampling flags are accepted because the public engine
    API already exposes them; M1.6 generation is greedy until M1.8 wires
    probabilistic sampling.
    """
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
        required=True,
        help="Prompt text to complete.",
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
    return parser


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
