# tiny-duo-infer

A tiny inference engine implementation for learning purposes. Phase 1 runs on
Apple Silicon with MLX; later phases add model-family portability, generation
UX, local serving, observability, weight-only quantization, then a PyTorch/CUDA
backend.

## What It Is

- A learning-first LLM inference engine inspired by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), [MinivLLM](https://github.com/Wenyueh/MinivLLM), and [tiny-llm](https://github.com/skyzh/tiny-llm), implemented in pure Python. The goal is to understand how an inference engine works by building each piece from scratch: model loading, tokenization, forward pass, KV cache, sampling, and scheduling/serving.

- Key design principle: readable, teachable code over optimized code. Every concept (prefill, decode, KV cache updates, GQA, RoPE) must be visible in the implementation rather than hidden behind transformers, mlx-lm, or vLLM.

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| Phase 1 | Single-user inference on Apple Silicon using MLX | Done |
| Phase 1.5 | Add Qwen3-0.6B support on the same MLX backend | Done |
| Phase 1.6 | Refine CLI and support HTTP serving | Done |
| Phase 1.7 | Engine observability: timing, KV-cache memory, and per-request context-budget policy | Done |
| Phase 1.8 | MLX-native weight-only quantization | Active |
| Phase 2 | Add NVIDIA/PyTorch/CUDA backend | Deferred |
| Phase 3 | Multi-user serving: scheduling, batching, streaming, PagedAttention | Not started |

## Model Targets

- [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

## Local CLI

Plain completion (Llama or Qwen3):

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/llama-3.2-1b \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --temperature 0.0
```

Qwen3 chat mode — wrap a plain prompt as a user message and apply the ChatML
template:

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --prompt "What is the capital of France?" \
  --chat \
  --max-new-tokens 64 \
  --temperature 0.7
```

Qwen3 chat mode with explicit system and user messages:

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --message system:"You are a concise assistant." \
  --message user:"What is the capital of France?" \
  --max-new-tokens 64 \
  --temperature 0.7
```

Additional flags (all models):

| Flag | Description |
|---|---|
| `--stop TEXT` | Stop when TEXT appears in output (repeatable). |
| `--seed N` | Seed for deterministic probabilistic sampling. |
| `--show-stats` | Write a 21-field timing/memory/context/quantization stats block to stderr after generation. Generated text stays on stdout. |
| `--context-policy POLICY` | How to handle prompts that exceed the context budget. Choices: `allow_context_stop` (default), `reject`, `truncate_left`, `truncate_right`, `reserve_generation`. |
| `--quantization MODE` | Weight-only quantization mode applied at model load. Choices: `none` (default, full precision), `int4`, `int8`. |
| `--quant-group-size N` | Quantization group size along the input dimension. Default `64`. Must evenly divide every quantized weight's `in_features`. |

Llama-3.2-1B is a base completion model. Chat mode (`--chat` or `--message`)
raises an error for Llama because it has no chat template.

Run Llama with INT4 weight-only quantization (full-precision generation
remains the default when `--quantization` is omitted):

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/llama-3.2-1b \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --temperature 0.0 \
  --quantization int4 \
  --quant-group-size 64 \
  --show-stats
```

`--show-stats` reports the quantization mode and linear-weight memory
(`linear_weight_full_precision_bytes` vs `linear_weight_runtime_bytes`)
alongside timing, KV-cache memory, and context-budget accounting.

## HTTP Server

Start the single-request inference server:

```bash
uv run python -m tiny_duo_infer.serving.api \
  --model-path ./models/qwen3-0.6b \
  --max-seq-len 2048
```

The HTTP entrypoint also accepts `--quantization {none,int4,int8}` and
`--quant-group-size N`. The engine is loaded inside the worker thread so
quantized weights stay on the MLX GPU stream that owns generation:

```bash
uv run python -m tiny_duo_infer.serving.api \
  --model-path ./models/qwen3-0.6b \
  --max-seq-len 2048 \
  --quantization int8
```

Full-response generation (JSON):

```bash
curl -s http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 16, "temperature": 0.0, "context_policy": "allow_context_stop"}'
```

The response JSON includes a `stats` field with the full 26-field stats
object (timing, token-budget, KV-cache memory, context policy, and
weight-only quantization metadata). The CLI `--show-stats` prints a
21-field summary to stderr; HTTP exposes all fields.

Streaming generation (NDJSON, one JSON object per line):

```bash
curl -s http://127.0.0.1:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 32}'
```

Fragment chunks carry `{"done": false, "text": "..."}`. The final chunk is
`{"done": true, "text": "...", "prompt_tokens": N, "generated_tokens": N,
"stop_reason": "...", "stats": {...}}`.

Server status:

```bash
curl http://127.0.0.1:8000/health
```

The server handles one request at a time. Concurrent requests receive a 503
"server busy" response.

## Profiling

Measure latency, throughput, and KV-cache memory across prompt sets:

```bash
uv run python scripts/profile_generation.py \
  --model-path ./models/qwen3-0.6b \
  --max-seq-len 512 \
  --max-new-tokens 64 \
  --runs 5 \
  --warmup-runs 1
```

Key flags:

| Flag | Description |
|---|---|
| `--prompt TEXT` | Add a prompt to the set (repeatable). Uses built-in prompts if omitted. |
| `--prompt-file PATH` | Load prompts from a file (one per line). |
| `--runs N` | Number of timed runs per prompt (default 3). |
| `--warmup-runs N` | Warmup runs to exclude from summary (default 1). |
| `--context-policy POLICY` | Context-budget policy applied to every request. |
| `--quantization MODE` | Weight-only quantization mode (`none`, `int4`, `int8`). Same flag as the CLI/HTTP entrypoints; lets one prompt set compare full-precision against INT8/INT4 runs. |
| `--quant-group-size N` | Group size for quantized runs. Default `64`. |
| `--json` | Emit a machine-readable JSON report to stdout (silences progress output). |

The summary reports `min`, `p50`, `p95`, and `max` for TTFT, decode
throughput, and KV-cache active memory. JSON output is versioned with
`schema_version=2`; v2 adds quantization mode and linear-weight memory
totals (`linear_weight_full_precision_bytes`,
`linear_weight_runtime_bytes`) to `engine_info` so full-precision and
quantized runs can be diffed directly.

## Development Checks

Install development dependencies:

```bash
uv sync --group dev
```

Run the fast local test suite manually:

```bash
uv run pytest -q
git diff --check
```

GitHub Actions runs the same fast regression gate on pushes and pull requests.
Slow real-model smoke tests remain manual phase-close checks because they
require local model artifacts and hardware.
