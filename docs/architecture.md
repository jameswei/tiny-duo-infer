# tiny-duo-infer: Architecture

**Status:** Active reference  
**Authors:** Claude Code + Codex (merged)  
**Supersedes:** `docs/design_cc.md` (historical)  
**Date:** 2026-05-25

This document is the unified architecture reference for `tiny-duo-infer`. It
describes the system's conceptual structure, subsystem responsibilities, and
design decisions that apply across all three phases. For phase-specific scope,
milestones, and done criteria, see `docs/phases/`.

---

## Purpose and Philosophy

`tiny-duo-infer` is a learning-first LLM inference engine inspired by vLLM.
The goal is to understand how an inference engine works by implementing the
core pieces directly: model loading, tokenization, forward pass, KV cache,
sampling, and — in later phases — scheduling, batching, and memory management.

**What it is not:** a thin wrapper around HuggingFace Transformers, `mlx-lm`,
or vLLM. Those libraries are used only as reference material for validation.

**What it is:** readable, explicitly structured Python code where every
inference-engine concept (prefill, decode, KV cache updates, attention masking,
GQA head expansion, lazy evaluation) is visible in the implementation rather
than hidden behind an opaque library call.

---

## Control Plane and Data Plane

The codebase is divided into two conceptual planes. This boundary is the
central architectural principle of the project.

### Control plane — pure Python, owns the logic

The control plane decides *what* to compute and *when*:

- Load model config, tokenizer, and weights
- Manage the generation request lifecycle (prefill → decode → done)
- Maintain KV cache state: which positions are filled, how much memory is used
- Implement the generation loop: token step sequencing, EOS detection, stopping
- Apply sampling strategies: greedy, temperature, top-k, top-p
- Phase 3: schedule requests, manage the KV page pool, handle preemption

The control plane is always Python. It does not call any framework-specific
neural network APIs.

### Data plane — delegated to the hardware backend

The data plane executes *how* tensors are computed:

- Matrix multiplications, softmax, elementwise ops
- Tensor memory allocation and layout on the accelerator
- Device execution: Apple Silicon unified memory (MLX) or NVIDIA CUDA (PyTorch)

The data plane boundary is explicit: only the operations listed in the
implementation boundary table below are delegated. Everything else stays in
the control plane.

---

## Implementation Boundary

The table below defines exactly what the project implements versus what is
delegated to the backend library. This is the authoritative boundary — any
agent considering using a higher-level framework function should check here
first.

| Component | We implement | We delegate to backend |
|---|---|---|
| RMSNorm | `x / sqrt(mean(x²) + eps) * weight` — full formula | `mx.rsqrt`, `mx.mean` |
| RoPE | frequency precomputation, rotation pairs `(x0·cos − x1·sin, x0·sin + x1·cos)` | `mx.cos`, `mx.sin` |
| GQA Attention | QKV projections, head reshape, KV-head repeat (`axis=1`), causal mask, output projection | `mx.matmul`, `mx.softmax` |
| SwiGLU FFN | gate/up/down projections, `silu(gate) * up` | elementwise `*`, `+` |
| Model assembly | block loop, residual connections, final norm, lm_head | tensor storage |
| KV cache | buffer allocation, position tracking, `update()`/`advance()` protocol | backend array ops |
| Sampling | greedy argmax, temperature scaling, top-k filter, top-p nucleus | `mx.argmax` or NumPy |
| Weight loading | HF key mapping, shape validation, tied-embedding handling | `safetensors` |
| Tokenizer | thin wrapper exposing `encode`/`decode`/`bos_token_id`/`eos_token_id` | `tokenizers` package |

**Explicitly prohibited:**

- `mlx.nn.MultiHeadAttention`, `mlx.nn.RMSNorm`, `mlx.nn.SiLU`, `mlx.nn.Linear`
- `transformers.AutoModelForCausalLM` or any `transformers` generation API
- `mlx-lm` as the core engine
- Any framework layer that hides the inference concept being learned

---

## Package Structure

This is the canonical layout for the full project. Phase 1 creates only the
files needed for M1.0–M1.8; the structure remains compatible with this layout.

```text
tiny_duo_infer/
  __init__.py
  engine.py           — Engine class: public API, generation orchestration
  cache.py            — KVCache: pre-allocated buffers, update/advance protocol
  sampling.py         — greedy, temperature, top-k, top-p
  cli.py              — local text-generation CLI
  config.py           — parse config.json into a model config dataclass

  backends/
    __init__.py
    protocol.py       — Backend Protocol (draft Phase 1; enforced Phase 2)
    mlx_backend.py    — MLX tensor helpers, mx.eval() behavior
    torch_backend.py  — PyTorch/CUDA backend (Phase 2)
    numpy_backend.py  — NumPy CPU reference (Phase 2, for validation)

  models/
    __init__.py
    base.py           — Module ABC, Linear, Embedding
    llama.py          — LlamaBlock, LlamaModel assembly

  layers/
    __init__.py
    normalization.py  — RMSNorm
    rope.py           — frequency precomputation, apply_rope
    attention.py      — LlamaAttention: GQA, RoPE, causal mask, KV cache
    feedforward.py    — SwiGLUFFN

  weights/
    __init__.py
    loader.py         — safetensors shard loading → mx.array / torch.Tensor
    llama_converter.py — HF key mapping, shape validation, tied-embedding

  tokenizer/
    __init__.py
    loader.py         — tokenizers-package wrapper

  serving/            — Phase 3 only
    __init__.py
    request.py        — Request dataclass, state machine
    scheduler.py      — FIFO → continuous batching
    block_manager.py  — PagedAttention KV page pool
    api.py            — FastAPI HTTP server

scripts/
  benchmark.py        — tokens/sec and KV cache memory measurement

tests/
  conftest.py         — TINY_CONFIG shared fixture
  test_tokenizer.py
  test_weights.py
  test_layers.py
  test_model.py
  test_cache.py
  test_sampling.py
  test_engine.py

docs/
  architecture.md     — this file
  refined-plan_codex.md
  agent-guidelines.md
  phases/
    phase-1-mlx-single-user.md
  adr/
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `engine.py` | Public `Engine.from_model_path()` API; orchestrates prefill → decode loop; owns `KVCache` lifecycle per request |
| `cache.py` | Pre-allocated K/V buffers; `update()` writes per layer; `advance()` commits one token step |
| `sampling.py` | Stateless sampling functions; operates on `(vocab_size,)` logits |
| `cli.py` | Thin CLI wrapper over `Engine` |
| `config.py` | Reads `config.json`; returns a typed config object used by model and cache constructors |
| `backends/protocol.py` | `Backend` typing Protocol; defines Tier-1 op signatures |
| `models/base.py` | `Module` ABC with `load_weights(flat_dict)`; `Linear`, `Embedding` |
| `models/llama.py` | `LlamaBlock`, `LlamaModel`; assembles layers into the full model |
| `layers/attention.py` | `LlamaAttention`; contains all GQA and RoPE logic |
| `layers/rope.py` | `precompute_freqs()`, `apply_rope()`; called at model init and each forward |
| `weights/llama_converter.py` | HF key → project key mapping; tied embedding handling; shape assertions |
| `tokenizer/loader.py` | `Tokenizer.from_pretrained()`; wraps `tokenizers` package |

---

## Backend Abstraction

### Phase 1 — MLX direct

Phase 1 has no enforced backend abstraction. MLX is used directly, and code
is structured in clearly bounded helper functions inside `backends/mlx_backend.py`
so it can be extracted cleanly in Phase 2.

A draft `backends/protocol.py` module is written during Phase 1 to capture the
intended interface shape early. The MLX code is not required to formally conform
to it until Phase 2.

### Phase 2 — Backend protocol

When PyTorch/CUDA is added, all non-portable backend-specific operations are
routed through backend implementations conforming to `backends/protocol.py`.
The protocol uses `typing.Protocol` to define the required Tier-1 operation
signatures. Direct native array ops (Tier-2 candidates) are allowed only after
M2.0 validation confirms they behave identically across backends.

**Tier-1 ops — must go through the protocol (APIs differ across backends):**

| Method | Purpose |
|---|---|
| `softmax(x, axis)` | numerically stable softmax |
| `silu(x)` | SiLU: `x / (1 + exp(-x))` |
| `array(data, dtype)` | create a backend tensor |
| `eval(*arrays)` | flush lazy computation (MLX) / no-op (PyTorch) |
| `to_numpy(x)` | move tensor to NumPy for CPU-side sampling |

**Tier-2 candidate ops — used directly in Phase 1; portability validated in M2.0:**

`@`, `.T`, `reshape`, `split`, `concatenate`, `sqrt`, `exp`, `cos`, `sin`,
`arange`, `zeros`, `tril`

These are used as native array operators in Phase 1 MLX code. During M2.0,
each must be verified to behave identically across NumPy, MLX, and PyTorch.
Known differences to check: `split`/`cat` naming, `.T` on rank > 2 tensors,
dtype promotion rules, device handling for `zeros` and `arange`.

---

## Module System

The project defines its own minimal `Module` base class. We do not subclass
`mlx.nn.Module` or `torch.nn.Module`. This keeps model code backend-neutral
and makes the module system itself visible and understandable.

```python
class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load_weights(self, weights: dict[str, any]) -> None:
        """
        Populate self.* attributes from a flat weight dict.
        Keys are dot-separated paths relative to this module
        (e.g. "attn.q_proj.weight").
        """
```

Weights are stored as plain backend-native array attributes — `mx.array` in
Phase 1. No gradient tracking, no parameter registration, no device movement
API: inference only.

---

## KV Cache Design

### Phase 1 — pre-allocated static buffer

One pair of K/V buffers per transformer layer, allocated once per request:

```
shape: (1, n_kv_heads, max_seq_len, head_dim)  per layer, per K and V
```

**Write/commit protocol:**

- `cache.update(layer_idx, new_k, new_v, position)` — writes K/V for one layer
  at `position`; does **not** advance `current_len`
- `cache.advance(n_tokens)` — called **once per token step** by the engine after
  all layers have written; increments `current_len` by `n_tokens`

This separation ensures `current_len` is consistent across all layers: the
attention layer receives `position_offset` as a parameter from the model
forward call, never by reading `cache.current_len` mid-forward-pass.

**Memory estimate:**

```
2 × L × Hkv × T × Dh × 2 bytes
= 2 × 16 × 8 × T × 64 × 2
= 32,768 × T bytes
≈ 32 MB for T = 1024 tokens
```

### Phase 3 — PagedAttention

KV cache divided into fixed-size pages (e.g., 16 tokens per page). A
`BlockManager` maintains a pool of free physical pages and a per-request block
table mapping `(logical_page_idx) → physical_page_idx`. Attention is computed
by gathering from the block table rather than a contiguous buffer.

Benefits over Phase 1 design:
- No memory wasted on pre-allocated but unused sequence positions
- Fine-grained reclamation when a request finishes (free individual pages)
- Supports more concurrent requests with the same total memory

---

## Target Model: Llama-3.2-1B

All three phases use `meta-llama/Llama-3.2-1B` (base model).

| Parameter | Value |
|---|---|
| `d_model` | 2048 |
| `n_layers` | 16 |
| `n_heads` | 32 |
| `n_kv_heads` | 8 (GQA groups = 4) |
| `head_dim` | 64 |
| `intermediate_size` | 8192 |
| `vocab_size` | 128256 |
| `max_seq_len` | 131072 |
| `rope_theta` | 500000.0 |
| `rms_norm_eps` | 1e-5 |
| Weight dtype | bfloat16 |
| Tied embeddings | yes (`lm_head.weight` = `embed_tokens.weight`) |

**Why this model:** modern Llama-3 architecture exercises all key building
blocks (RMSNorm, RoPE, GQA, SwiGLU), and uses tiktoken BPE (loaded via the
`tokenizers` package). Weights are roughly ~2GB in bfloat16; runtime memory
is higher once the KV cache, activations, and MLX overhead are included.

---

## Tokenizer

The `tokenizers` package (HuggingFace, ~10MB) is used at runtime. It is loaded
from the `tokenizer.json` file in the local model directory.

`transformers.AutoTokenizer` is a dev/test-only dependency. It may be used in
`@pytest.mark.slow` tests to verify parity, but must not be imported in any
file under `tiny_duo_infer/`.

The project tokenizer wrapper exposes:

```python
encode(text, add_special_tokens=True) -> list[int]
decode(token_ids, skip_special_tokens=True) -> str
bos_token_id: int
eos_token_id: int
vocab_size: int
```

---

## Phase Responsibilities Summary

| Concern | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Backend | MLX (direct) | + PyTorch/CUDA via protocol | same |
| Concurrency | single request | single request | multiple concurrent |
| KV cache | pre-allocated static | same | PagedAttention |
| Scheduling | none | none | FIFO → continuous batching |
| Serving | CLI only | CLI + benchmarks | + HTTP API (FastAPI) |
| Sampling | greedy → top-k/p/temp | same | same |
| Model | Llama-3.2-1B (base) | same | same |

---

## Decisions Out of Scope

These are explicit non-goals for the current roadmap. Raising them as
implementation proposals requires an ADR.

- C++ or custom CUDA kernels — all three phases
- Instruct/chat-template support — dropped by design; the base model is
  sufficient for learning inference engine mechanics, and the instruct variant
  adds chatbot complexity that is not relevant to this project's goals
- Training or fine-tuning — all three phases
- Quantization (INT8/INT4 weights)
- Speculative decoding
- Distributed inference (tensor parallelism, pipeline parallelism)
- Wrapping `mlx-lm`, vLLM, or `transformers` as the engine core
