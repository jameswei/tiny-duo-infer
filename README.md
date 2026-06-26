# tiny-duo-infer

**Project site:** https://jameswei.github.io/tiny-duo-infer/

A learning-first LLM inference engine built from scratch in pure Python on Apple Silicon (MLX).
Every inference concept — prefill, decode, KV cache, GQA, RoPE, SwiGLU, weight-only quantization —
is explicitly implemented in readable code rather than hidden behind `transformers`, `mlx-lm`, or vLLM.

Inspired by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), [MinivLLM](https://github.com/Wenyueh/MinivLLM), and [tiny-llm](https://github.com/skyzh/tiny-llm).

## What's Implemented

- **Prefill & decode loop** — full generation lifecycle with EOS detection and stop-string support
- **Grouped-query attention (GQA)** — explicit head expansion, causal masking, KV cache update/advance protocol
- **Rotary position embeddings (RoPE)** — frequency precomputation and pair-wise rotation
- **SwiGLU feed-forward networks** — gate/up/down projections with explicit SiLU activation
- **KV cache** — pre-allocated per-layer buffers with position-consistent write/commit semantics
- **Sampling** — greedy, temperature scaling, top-k, top-p nucleus
- **Weight-only quantization** — INT4/INT8 via MLX-native `quantized_matmul`; per-run memory accounting
- **HTTP serving** — single-request FastAPI server with JSON and NDJSON streaming endpoints
- **Observability** — per-request TTFT, decode throughput, KV-cache memory, context-budget policy
- **Profiling** — repeatable latency/throughput benchmarks across prompts and quantization modes
- **Multi-model support** — Llama-3.2-1B and Qwen3-0.6B on the same engine

## Models

| Model | HuggingFace |
|---|---|
| Llama-3.2-1B (base) | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| Qwen3-0.6B | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |

## Getting Started

**Requirements:** Python 3.12, [uv](https://docs.astral.sh/uv/), Apple Silicon Mac (MLX)

```bash
# Clone and install
git clone https://github.com/jameswei/tiny-duo-infer.git
cd tiny-duo-infer
uv sync

# Download a model (example: Qwen3-0.6B)
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/qwen3-0.6b

# Run your first generation
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --prompt "What is attention in transformers?" \
  --chat --max-new-tokens 64
```

## CLI

Plain completion (Llama-3.2-1B):

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/llama-3.2-1b \
  --prompt "The capital of France is" \
  --max-new-tokens 32 --temperature 0.0
```

Qwen3 chat with explicit messages:

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --message system:"You are a concise assistant." \
  --message user:"Explain KV cache in one paragraph." \
  --max-new-tokens 128 --temperature 0.7
```

INT4 weight-only quantization with stats:

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/llama-3.2-1b \
  --prompt "The capital of France is" \
  --max-new-tokens 32 --temperature 0.0 \
  --quantization int4 --show-stats
```

**Key flags:**

| Flag | Description |
|---|---|
| `--chat` | Wrap prompt as a user message and apply ChatML template (Qwen3 only). |
| `--message ROLE:TEXT` | Explicit system/user messages (repeatable). |
| `--quantization MODE` | `none` (default), `int4`, or `int8` weight-only quantization. |
| `--quant-group-size N` | Group size along the input dimension. Default `64`. |
| `--show-stats` | Print timing, KV-cache memory, and quantization stats to stderr. |
| `--context-policy POLICY` | `allow_context_stop` (default), `reject`, `truncate_left`, `truncate_right`, `reserve_generation`. |
| `--stop TEXT` | Stop when TEXT appears in output (repeatable). |
| `--seed N` | Seed for deterministic sampling. |

## HTTP Server

Start the server:

```bash
uv run python -m tiny_duo_infer.serving.api \
  --model-path ./models/qwen3-0.6b \
  --max-seq-len 2048
```

Also accepts `--quantization {none,int4,int8}` and `--quant-group-size N`.

Full-response generation:

```bash
curl -s http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 16, "temperature": 0.0}'
```

Streaming (NDJSON, one object per line):

```bash
curl -s http://127.0.0.1:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 32}'
```

Health check: `curl http://127.0.0.1:8000/health`

The server handles one request at a time; concurrent requests receive a `503` response.

## Profiling

Measure latency, throughput, and KV-cache memory across prompt sets:

```bash
uv run python scripts/profile_generation.py \
  --model-path ./models/qwen3-0.6b \
  --max-seq-len 512 --max-new-tokens 64 \
  --runs 5 --warmup-runs 1
```

Add `--quantization int8` to compare quantized vs full-precision runs side by side.
Use `--json` for machine-readable output.

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest -q

# Check for whitespace issues
git diff --check
```

GitHub Actions runs the same test suite on every push and pull request.
