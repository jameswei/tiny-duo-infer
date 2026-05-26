"""
Benchmark script: tokens/sec and KV cache memory for M1.7 baseline metrics.

Measures end-to-end generation throughput by timing a full generate() call
over a fixed number of tokens with greedy decoding. Also prints the peak KV
cache memory for several reference sequence lengths.

The timer wraps the entire generate() iterator, which includes both prefill
and decode. For a more granular split, see the learning notes below.

Usage:
    uv run python scripts/benchmark.py \\
      --model-path ./models/llama-3.2-1b \\
      --prompt "The capital of France is" \\
      --n-tokens 100

Example output:
    Loading model from ./models/llama-3.2-1b ...
    Prompt  : "The capital of France is"
    Tokens  : 100 (greedy)

    --- throughput ---
    tokens generated : 100
    elapsed          : 8.234 s
    tokens/sec       : 12.1

    --- kv cache memory (bfloat16, L=16, Hkv=8, Dh=64) ---
    formula: 2 × L × Hkv × T × Dh × 2 bytes
      T=  100:        3,276,800 bytes  (  3.1 MB)
      T=  256:        8,388,608 bytes  (  8.0 MB)
      T= 1024:       33,554,432 bytes  ( 32.0 MB)
      T= 2048:       67,108,864 bytes  ( 64.0 MB)

Learning notes:
  - The timer includes prefill because generate() runs prefill on the first
    next() call. On large prompts, prefill dominates early; decode dominates
    for long outputs. Separating the two requires calling engine.prefill()
    and engine.generate() separately, which is a Phase 2 refinement.
  - tokens/sec here is decode throughput averaged over the full run.
    The first decode step is typically slower than subsequent ones because
    the MLX computation graph is larger before the first mx.eval() warms it.
  - KV cache memory grows linearly with sequence length T. At T=1024 for
    Llama-3.2-1B (bfloat16), the formula gives exactly 32 MB — a useful
    reference point for Apple Silicon unified memory budgeting.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tiny_duo_infer.engine import Engine


def kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
    bytes_per_element: int = 2,
) -> int:
    """
    Compute the peak KV cache memory in bytes for one generation request.

    Formula: 2 × n_layers × n_kv_heads × seq_len × head_dim × bytes_per_element

    The leading factor of 2 accounts for both the K buffer and the V buffer.
    Each layer maintains one pair. All positions up to seq_len are pre-allocated
    even if fewer tokens are actually generated.

    For Llama-3.2-1B in bfloat16:
        2 × 16 × 8 × T × 64 × 2 = 32,768 × T bytes
        At T=1024: 33,554,432 bytes ≈ 32 MB

    Args:
        n_layers:          number of transformer layers (L).
        n_kv_heads:        number of KV attention heads (Hkv).
        seq_len:           total sequence length T (prompt + generated).
        head_dim:          per-head dimension (Dh = d_model // n_heads).
        bytes_per_element: bytes per scalar. 2 for bfloat16, 4 for float32.

    Returns:
        Total KV cache size in bytes.
    """
    return 2 * n_layers * n_kv_heads * seq_len * head_dim * bytes_per_element


def main() -> None:
    """Parse CLI args, run timed generation, print throughput and memory stats."""
    args = _build_parser().parse_args()

    print(f"Loading model from {args.model_path} ...")
    engine = Engine.from_model_path(args.model_path, max_seq_len=args.max_seq_len)

    print(f'Prompt  : "{args.prompt}"')
    print(f"Tokens  : {args.n_tokens} (greedy)\n")

    # Time the full generate() call.  The iterator is drained by list(), which
    # drives both prefill (on the first next()) and every decode step.
    t_start = time.perf_counter()
    fragments = list(engine.generate(args.prompt, max_new_tokens=args.n_tokens))
    t_end = time.perf_counter()

    n_generated = len(fragments)
    elapsed = t_end - t_start
    tps = n_generated / elapsed if elapsed > 0 else 0.0

    print("--- throughput ---")
    print(f"tokens generated : {n_generated}")
    print(f"elapsed          : {elapsed:.3f} s")
    print(f"tokens/sec       : {tps:.1f}")

    # KV cache memory for several reference sequence lengths.
    # The pre-allocated buffer covers the full max_seq_len, but the formula
    # shows the cost at each T so the growth rate is visible.
    cfg = engine.config
    ref_lengths = sorted({n_generated, 256, 1024, 2048})

    print(
        f"\n--- kv cache memory"
        f" (bfloat16, L={cfg.n_layers}, Hkv={cfg.n_kv_heads}, Dh={cfg.head_dim}) ---"
    )
    print(f"formula: 2 × L × Hkv × T × Dh × 2 bytes")
    for seq_len in ref_lengths:
        nbytes = kv_cache_bytes(cfg.n_layers, cfg.n_kv_heads, seq_len, cfg.head_dim)
        print(f"  T={seq_len:5d}: {nbytes:>15,} bytes  ({nbytes / 1024 ** 2:6.1f} MB)")

    if args.show_output:
        print("\n--- generated text ---")
        print("".join(fragments))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Measure tiny-duo-infer tokens/sec and KV cache memory.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a local HuggingFace-compatible model directory.",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help='Prompt text. Default: "The capital of France is".',
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate. Default: 100.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum total sequence length for the KV cache. Default: 2048.",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Print the generated text after the benchmark.",
    )
    return parser


if __name__ == "__main__":
    main()
