# File Structure

---

## Root

| File | Purpose |
|---|---|
| `pyproject.toml` | Runtime deps: `mlx`, `tokenizers`, `safetensors`, `huggingface-hub`. Dev: `pytest`, `transformers` |
| `README.md` | Project overview, supported models, roadmap |
| `AGENTS.md` | Entry point for AI agents — doc reading order |

---

## `tiny_duo_infer/` — Source

### Top-level

| File | Purpose |
|---|---|
| `engine.py` | `Engine`: `from_model_path()`, `prefill()`, `generate()` — full generation pipeline |
| `cache.py` | `KVCache`: pre-allocated buffers, `update()`/`advance()` write-commit protocol |
| `sampling.py` | `greedy()`, `sample()` — temperature, top-k, top-p token selection |
| `cli.py` | argparse CLI wrapper over `Engine` |
| `config.py` | `ModelConfig` dataclass, `load_config()` from `config.json` |

### `models/`

| File | Purpose |
|---|---|
| `base.py` | `Module` ABC, `Linear`, `Embedding`: `load_weights()` dot-path routing |
| `llama.py` | `LlamaBlock`: pre-norm attention + FFN + residuals. `LlamaModel`: embed → 16 blocks → final_norm → lm_head |

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

### `tokenizer/`

| File | Purpose |
|---|---|
| `loader.py` | `Tokenizer.from_pretrained()`: wraps `tokenizers`, exposes `encode()`/`decode()`/`bos_token_id`/`eos_token_id` |

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
| `test_cli.py` | CLI args, fake-engine integration |

---

## `scripts/`

| File | Purpose |
|---|---|
| `benchmark.py` | Tokens/sec throughput, KV cache memory |

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
