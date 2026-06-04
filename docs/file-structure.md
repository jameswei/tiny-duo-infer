# File Structure

---

## Root

| File | Purpose |
|---|---|
| `pyproject.toml` | Runtime deps: `mlx`, `tokenizers`, `safetensors`, `huggingface-hub`. Dev: `pytest`, `transformers` |
| `.github/workflows/test.yml` | CI regression gate for pushes and pull requests |
| `README.md` | Project overview, supported models, roadmap |
| `AGENTS.md` | Entry point for AI agents — doc reading order |

---

## `tiny_duo_infer/` — Source

### Top-level

| File | Purpose |
|---|---|
| `engine.py` | `Engine`: `from_model_path()`, `prefill()`, `generate()` — full generation pipeline; instruments timing and KV bytes; attaches `GenerationStats` to every response |
| `cache.py` | `KVCache`: pre-allocated buffers, `update()`/`advance()` write-commit protocol |
| `sampling.py` | `greedy()`, `sample()` — temperature, top-k, top-p token selection |
| `cli.py` | argparse CLI wrapper over `Engine`; `--show-stats` writes 14-field block to stderr; `--context-policy` forwarded to `GenerationRequest` |
| `config.py` | `ModelConfig` dataclass, `load_config()` from `config.json` |
| `generation.py` | `ChatMessage`, `GenerationRequest`, `GenerationResponse`, `GenerationStats` — validated request/response types, `ContextPolicy` literal, `kv_cache_bytes()` formula |
| `context_policy.py` | `apply_context_policy()` — enforces accept/truncate/reject before prefill; `ContextBudgetError`, `ContextPolicyOutcome` |
| `prompt.py` | `format_chat_prompt()` — ChatML template for Qwen3; raises `ValueError` for Llama (base model) |
| `profiling.py` | `percentile()`, `aggregate_runs()`, `run_profile()` — importable profiling logic used by `scripts/profile_generation.py` |

### `models/`

| File | Purpose |
|---|---|
| `base.py` | `Module` ABC, `Linear`, `Embedding`: `load_weights()` dot-path routing |
| `llama.py` | `LlamaBlock`: pre-norm attention + FFN + residuals. `LlamaModel`: embed → 16 blocks → final_norm → lm_head |
| `qwen3.py` | `Qwen3Block` / `Qwen3Model`: Qwen3 assembly using Q/K-normalized attention |

### `layers/`

| File | Purpose |
|---|---|
| `normalization.py` | `RMSNorm`: `x * rsqrt(mean(x²) + eps) * weight` |
| `rope.py` | `precompute_freqs()`, `apply_rope()` — rotary embeddings with absolute position offset |
| `attention.py` | `LlamaAttention`: GQA, Q/K/V projections, RoPE, KV cache, causal mask, `mx.repeat(axis=1)` head expansion |
| `feedforward.py` | `SwiGLUFFN`: gate/up/down projections, `silu(gate) * up` |

### `weights/`

| File | Purpose |
|---|---|
| `loader.py` | `load_weights()`: safetensors shard discovery → `mx.array` dict |
| `llama_converter.py` | `convert()`: HF key mapping, shape validation, tied embeddings |
| `qwen3_converter.py` | `convert()`: Qwen3 HF key mapping, Q/K norm weights, `H * Dh != D` shape validation |

### `tokenizer/`

| File | Purpose |
|---|---|
| `loader.py` | `Tokenizer.from_pretrained()`: wraps `tokenizers`, exposes `encode()`/`decode()`/`bos_token_id`/`eos_token_id` |

### `serving/`

| File | Purpose |
|---|---|
| `worker.py` | `InferenceWorker`: runs engine on a dedicated thread for MLX GPU stream affinity; `submit_generate()` / `submit_stream()` |
| `api.py` | FastAPI app: `GET /health`, `POST /generate` (JSON + stats), `POST /generate/stream` (NDJSON + final stats); `context_policy` in request body; `create_app(engine)` factory; CLI entrypoint via `__main__` |

### `backends/`

| File | Purpose |
|---|---|
| `protocol.py` | `Backend` typing Protocol: softmax, silu, array, eval, to_numpy |
| `mlx_backend.py` | MLX lazy-eval notes; Phase 2 extraction target |

---

## `tests/`

All use `TINY_CONFIG` (2 layers, d_model=64). `@pytest.mark.slow` tests skipped unless `--run-slow`.

| File | What it tests |
|---|---|
| `conftest.py` | `TINY_CONFIG` fixture |
| `test_config.py` | Config parsing, validation, divisibility |
| `test_tokenizer.py` | Encode/decode round-trip, BOS/EOS |
| `test_weights.py` | Safetensors loader, HF→project key mapping, shapes |
| `test_cache.py` | `update()`/`advance()`/`reset()`, position tracking |
| `test_layers.py` | RMSNorm, RoPE, GQA attention, SwiGLU FFN |
| `test_model.py` | Module/Linear/Embedding, LlamaBlock, LlamaModel forward |
| `test_sampling.py` | `greedy()`, `sample()` edge cases |
| `test_engine.py` | Prefill/decode state transitions, eval placement, `max_new_tokens` |
| `test_cli.py` | CLI args, fake-engine integration, `--show-stats` stderr block, `--context-policy` forwarding |
| `test_generation.py` | `ChatMessage`, `GenerationRequest`, `GenerationResponse`, `GenerationStats` validation; `kv_cache_bytes()` formula |
| `test_context_policy.py` | `apply_context_policy()` — all five policies, both spec preconditions, outcome accounting invariants |
| `test_prompt.py` | `format_chat_prompt()` ChatML output, Llama rejection, unsupported model, empty messages |
| `test_serving.py` | HTTP server endpoints, NDJSON streaming, busy response, validation; stats fields, `context_policy` forwarding |
| `test_profiling.py` | `percentile()`, `aggregate_runs()`, prompt loading, CLI validation, JSON schema shape |

---

## `scripts/`

| File | Purpose |
|---|---|
| `benchmark.py` | Tokens/sec throughput, KV cache memory |
| `profile_generation.py` | Repeatable generation profiling: latency (TTFT, prefill, decode, total), throughput, KV-cache memory; supports `--prompt`, `--prompt-file`, `--runs`, `--warmup-runs`, `--json` |

---

## `docs/` — Project & Engineering documents for human and agents.

| File | Purpose |
|---|---|
| `architecture.md` | Control/data plane split, implementation boundary, backend design |
| `refined-plan.md` | Three-phase roadmap, milestones, testing strategy |
| `agent-guidelines.md` | Agent roles, review gates, handoff format, conflict rules |
| `file-structure.md` | This file |
| `phases/phase-1-mlx-single-user.md` | Phase 1 implementation contract: scope, interfaces, shape conventions |
| `phases/phase-1-taskboard.md` | Task tracking: 19 tasks, ownership, status, review gates |
| `phases/phase-1-handoff.md` | Completion handoff: verification results, smoke tests, known gaps |
| `phases/phase-1.5-qwen3-mlx.md` | Phase 1.5 implementation contract: Qwen3-0.6B support on MLX |
| `phases/phase-1.5-taskboard.md` | Phase 1.5 task tracking, ownership, status, review gates |
| `phases/phase-1.6-generation-serving.md` | Phase 1.6 implementation contract: generation UX and single-request serving |
| `phases/phase-1.6-taskboard.md` | Phase 1.6 task tracking, ownership, status, review gates |
| `phases/phase-1.7-observability.md` | Phase 1.7 implementation contract: observability, timing, KV-cache memory, context-budget policy |
| `phases/phase-1.7-taskboard.md` | Phase 1.7 task tracking, ownership, status, review gates |

---

## `learning_materials/` — Learning documents for human readers.

Suggested order: roadmap → internals → data-flow → deep_dives.

| File | Purpose |
|---|---|
| `roadmap.md` | Guided reading order through 18 components with exercises |
| `internals.md` / `_zh.md` | Deep-dive: prefill vs decode, KV cache lifecycle, GQA + RoPE |
| `data-flow.md` / `_zh.md` | Data-flow diagrams (ASCII + Mermaid): prefill, decode, end-to-end |
| `deep_dives/rope.md` | Rotary positional embeddings — math, intuition, frequency spectrum |
| `deep_dives/gqa_and_kv_cache.md` | Grouped query attention and KV cache mechanics |
| `deep_dives/sampling.md` | Greedy, temperature, top-k, top-p sampling |
| `deep_dives/swiglu_ffn.md` | SwiGLU FFN — gate/up/down architecture |
