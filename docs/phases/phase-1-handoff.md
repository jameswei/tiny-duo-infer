# Phase 1 Handoff

**Status:** Ready for review  
**Date:** 2026-05-26  
**Owner:** codex  
**Task:** P1-T18

This document closes Phase 1 from the implementation-owner side. A separate
reviewing agent must independently verify the commands below before marking
P1-T18 `done`.

## Summary

Phase 1 implements single-user local inference on Apple Silicon through MLX.
The engine can load local HuggingFace-compatible Llama artifacts, tokenize a
plain prompt, run prefill, decode one token at a time using the static KV
cache, sample tokens, and expose generation through both Python API and CLI.

Completed scope:

- Config, tokenizer, safetensors, and Llama weight conversion
- Base module helpers, RMSNorm, RoPE, GQA attention, SwiGLU FFN
- Llama model assembly
- Static per-request KV cache
- Prefill path and decode loop
- Greedy, temperature, top-k, and top-p sampling
- CLI for local text generation
- MLX eval placement documentation
- Benchmark script for tokens/sec and KV cache memory estimates

## Verification

Environment:

- macOS 26.5, build 25F71
- Apple M3 Pro, 36 GB unified memory
- Python 3.12.13 (via uv)
- MLX 0.31.2
- Model: meta-llama/Llama-3.2-1B (bfloat16, HuggingFace cache)

### Unit tests

```bash
uv run pytest
```

Result: `172 passed, 7 skipped`. The 7 skips are `@pytest.mark.slow` tests
that require real model artifacts and are gated by `--run-slow`.

### Import

```bash
uv run python -c "import tiny_duo_infer; print('import ok')"
```

Result: `import ok`.

### CLI help

```bash
uv run python -m tiny_duo_infer.cli --help
uv run python scripts/benchmark.py --help
```

Result: both printed help without error.

### CLI smoke test (greedy, real weights)

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08 \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --temperature 0.0
```

Result: ` the capital of France is Paris. It is the capital of France. It is
the capital of France. It is the capital of France. It is the capital`

Generation completed without error. Output is coherent (repetition is
expected with greedy decoding at max_new_tokens=32).

### Benchmark (real weights)

```bash
uv run python scripts/benchmark.py \
  --model-path ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08 \
  --n-tokens 100 --show-output
```

Result:

```
tokens generated : 100
elapsed          : 10.919 s
tokens/sec       : 9.2

--- kv cache memory (bfloat16, L=16, Hkv=8, Dh=64) ---
formula: 2 × L × Hkv × T × Dh × 2 bytes
  T=  100:       3,276,800 bytes  (   3.1 MB)
  T=  256:       8,388,608 bytes  (   8.0 MB)
  T= 1024:      33,554,432 bytes  (  32.0 MB)
  T= 2048:      67,108,864 bytes  (  64.0 MB)
```

Baseline: **9.2 tokens/sec** on Apple M3 Pro, 36 GB unified memory.

## Current Usage

Python API:

```python
from tiny_duo_infer.engine import Engine

engine = Engine.from_model_path("./models/llama-3.2-1b")
text = "".join(
    engine.generate(
        "The capital of France is",
        max_new_tokens=32,
        temperature=0.0,
    )
)
print(text)
```

CLI:

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/llama-3.2-1b \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --temperature 0.0
```

Benchmark:

```bash
uv run python scripts/benchmark.py \
  --model-path ./models/llama-3.2-1b \
  --n-tokens 100
```

## Known Limitations

- Batch size is fixed at 1.
- MLX is the only runtime backend.
- No HTTP server, batching, scheduler, streaming API, quantization, speculative
  decoding, or PagedAttention.
- No instruct/chat-template support.
- Real-model verification still needs local Llama artifacts.
- Backend protocol conformance is deferred to Phase 2.

## Reviewer Checklist

Before marking P1-T18 `done`, the reviewing agent should independently run:

```bash
uv run pytest
uv run python -c "import tiny_duo_infer; print('import ok')"
uv run python -m tiny_duo_infer.cli --help
uv run python scripts/benchmark.py --help
```

If real model artifacts are available, also run one short CLI or Python
generation smoke test and one benchmark run. If artifacts are unavailable,
record the skip reason in the taskboard.
