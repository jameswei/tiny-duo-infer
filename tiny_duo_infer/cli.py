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


def main() -> None:
    """Parse CLI arguments and run generation via Engine."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
