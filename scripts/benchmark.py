"""
Benchmark script: tokens/sec and KV cache memory measurement.

Measures end-to-end generation throughput and peak KV cache memory usage
for M1.7 baseline metrics. Run after MLX lazy eval placement is finalized.

Usage:
    uv run python scripts/benchmark.py \\
      --model-path ./models/llama-3.2-1b \\
      --prompt "The capital of France is" \\
      --n-tokens 100

Output:
    tokens/sec: <value>
    kv_cache_bytes (n_tokens=100): <value>
    kv_cache_bytes (n_tokens=1024): <value>
"""

from __future__ import annotations


def main() -> None:
    """Parse CLI args, run timed generation, print throughput and memory stats."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
