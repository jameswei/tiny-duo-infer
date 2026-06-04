# Tiny Duo Infer Refined Plan

This document is the refined Codex plan after discussion and cross-review with
the Claude Code proposal documents.

It consolidates the settled decisions for the project direction, phase-1 scope,
runtime dependencies, package layout, testing strategy, and multi-agent
collaboration expectations.

## Summary

`tiny-duo-infer` is a Python-only, learning-first inference engine inspired by
vLLM.

The purpose is to understand how an inference engine works by implementing the
core pieces directly:

- model forward pass
- prefill
- decode
- KV cache
- sampling
- backend execution
- later, multi-user scheduling and serving concepts

The project is not trying to become a production inference server. It should
prefer readable, teachable code over compact or highly optimized code.

## Roadmap

The project follows three major phases, with Phase 1.5, Phase 1.6, and Phase
1.7 inserted after Phase 1 before backend portability:

1. Phase 1: single-user local inference on Apple Silicon using MLX.
2. Phase 1.5: add `Qwen/Qwen3-0.6B` support on the same MLX backend.
3. Phase 1.6: refine generation UX and add single-request local HTTP serving.
4. Phase 1.7: engine observability — per-request timing, KV-cache memory
   accounting, and context-budget policy enforcement.
5. Phase 2: add Nvidia GPU support through a PyTorch/CUDA backend.
6. Phase 3: add multi-user serving concepts such as queues, scheduling,
   batching, streaming, and PagedAttention-style KV-cache management.

Qwen3 support is useful for learning model-family portability before adding a
second backend. Phase 1.6 is useful for learning request boundaries,
generation controls, and streaming I/O while the CUDA development environment
is unavailable. CUDA support remains useful for industry alignment. Multi-user
serving is useful for learning the concepts that make vLLM-like systems
interesting.

## Settled Phase-1 Decisions

Phase 1 should focus on inference-engine mechanics, not chatbot behavior.

Settled choices:

- Primary model: `meta-llama/Llama-3.2-1B`
- Model variant: base model, not instruct
- Prompt mode: plain prompt-to-completion generation
- Backend: MLX on Apple Silicon
- Execution mode: one request at a time
- Runtime tokenizer dependency: `tokenizers`
- Optional dev/test reference dependency: `transformers`
- Public interface: Python `Engine` class plus CLI
- Core inference logic: implemented in this repository

Phase 1 should not include:

- instruct/chat-template support
- system/user/assistant message formatting
- chatbot UX
- conversation history management
- HTTP serving
- multi-user scheduling
- quantization
- speculative decoding
- PagedAttention
- distributed inference
- C++ or custom kernels

## Model And Tokenizer Scope

The primary phase-1 model target is `meta-llama/Llama-3.2-1B`, the base model.

The base model is preferred because it supports simple prompt-to-completion
generation. This keeps the first implementation focused on the inference engine:
weights, tensor operations, prefill, decode, KV cache, and sampling.

The instruct model, `meta-llama/Llama-3.2-1B-Instruct`, is not a phase-1
requirement. It uses the same broad model family, but it expects chat-formatted
input with role markers and special end-of-turn behavior. That is useful for a
chatbot, but it is not required for learning the engine core.

Phase 1.5 adds `Qwen/Qwen3-0.6B` as the second supported model family. The
source-of-truth contract is `docs/phases/phase-1.5-qwen3-mlx.md`. This phase
keeps MLX, single-request execution, and plain prompt-to-completion generation,
but introduces model-family differences such as explicit `head_dim`, Q/K
normalization before RoPE, Qwen3 weight conversion, and model-type dispatch.

Phase 1.6 adds generation UX and local serving features on top of the same MLX
engine. The source-of-truth contract is
`docs/phases/phase-1.6-generation-serving.md`. This phase keeps one active
request at a time, but adds structured request/response metadata, stop strings,
token accounting, optional chat prompt formatting, refined CLI flags, and a
single-request HTTP API. It does not introduce batching, PagedAttention, or a
PyTorch/CUDA backend.

Phase 1.7 adds observability to the existing MLX engine. The source-of-truth
contract is `docs/phases/phase-1.7-observability.md`. This phase instruments
the engine with per-request timing (TTFT, prefill, decode, total), KV-cache
memory accounting, and a `context_policy` parameter that controls how prompts
exceeding the context budget are handled. Stats are exposed in `GenerationStats`,
surfaced in the CLI via `--show-stats` (to stderr), in HTTP JSON responses, and
in the final streaming chunk. A repeatable profiling script
(`scripts/profile_generation.py`) reports min/p50/p95/max summaries.

Tokenizer plan:

- Use the lightweight Hugging Face `tokenizers` package at runtime.
- Load the local `tokenizer.json` from the Hugging Face-compatible model
  directory.
- Wrap it behind a project-owned tokenizer interface.
- Use `transformers.AutoTokenizer` only as an optional dev/test reference if
  parity checks are needed.

The project tokenizer wrapper should expose only what the engine needs:

- `encode(text, add_special_tokens=True) -> list[int]`
- `decode(token_ids, skip_special_tokens=True) -> str`
- `bos_token_id`
- `eos_token_id`

The wrapper may read metadata such as `tokenizer_config.json` or
`special_tokens_map.json` if required to recover BOS/EOS behavior correctly.

## Implementation Boundary

The project should implement inference-engine logic directly while delegating
primitive tensor execution to MLX or PyTorch.

| Component | We implement | We delegate |
|---|---|---|
| RMSNorm | normalization formula and shape handling | primitive math ops |
| RoPE | frequency handling and rotation logic | sin/cos and tensor ops |
| Attention | QKV projections, GQA, masking, KV-cache use | matmul, softmax |
| SwiGLU FFN | gate/up/down projection flow | elementwise ops |
| Model assembly | blocks, residuals, final norm, LM head | tensor storage/execution |
| KV cache | layout, allocation, updates, positions | backend arrays |
| Sampling | greedy first; top-k/top-p/temperature later | primitive array ops |
| Tokenizer | small project wrapper | `tokenizers` runtime |
| Weight loading | HF key mapping and validation | `safetensors` file reading |

The project should not use:

- `transformers.AutoModelForCausalLM`
- `transformers` generation APIs
- `mlx-lm` as the engine implementation
- `vLLM` as the engine implementation
- high-level MLX/PyTorch layers that hide the concepts being learned

## Canonical Package Layout

Use this merged package layout:

```text
tiny_duo_infer/
  __init__.py
  engine.py
  cache.py
  sampling.py
  cli.py
  backends/
    __init__.py
    protocol.py
    mlx_backend.py
    torch_backend.py
    numpy_backend.py
  models/
    __init__.py
    base.py
    llama.py
  layers/
    __init__.py
    attention.py
    feedforward.py
    normalization.py
    rope.py
  weights/
    __init__.py
    loader.py
    llama_converter.py
  tokenizer/
    __init__.py
    loader.py
  serving/
    __init__.py
    request.py
    scheduler.py
    block_manager.py
    api.py
tests/
docs/
```

Phase 1 does not need to fully implement every file above. It should create
only the modules needed for the current milestone, but the layout should remain
compatible with this structure.

Responsibilities:

- `engine.py`: public `Engine` API and generation orchestration
- `cache.py`: static phase-1 KV-cache data structures and update behavior
- `sampling.py`: greedy sampling first; probabilistic sampling later
- `cli.py`: local text-generation command
- `backends/protocol.py`: future backend contract; phase 1 can shape MLX code
  toward it
- `backends/mlx_backend.py`: MLX tensor helpers and evaluation behavior
- `models/base.py`: minimal inference-only module helpers
- `models/llama.py`: Llama model and block assembly
- `layers/attention.py`: GQA attention, masking, and KV-cache interaction
- `layers/feedforward.py`: SwiGLU feed-forward network
- `layers/normalization.py`: RMSNorm
- `layers/rope.py`: rotary positional embedding helpers, including frequency
  precomputation from `rope_theta` and rotation application to Q/K tensors
- `weights/loader.py`: safetensors loading
- `weights/llama_converter.py`: Hugging Face key mapping and validation
- `tokenizer/loader.py`: `tokenizers`-based wrapper

## Phase-1 Milestones

### M1.0 Project Scaffolding

Create the Python package and development tooling.

Done when:

- `pyproject.toml` exists
- Python requirement is `>=3.12,<3.13`
- `uv` can create/sync the environment
- `uv run python -c "import tiny_duo_infer"` succeeds

Use dependency groups rather than confusing extras terminology:

```toml
[project]
dependencies = [
    "mlx",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
]

[dependency-groups]
dev = ["pytest", "transformers"]
```

In Phase 1, `mlx` is a required runtime dependency because MLX is the only
backend. Optional backend extras can be introduced in Phase 2 when PyTorch/CUDA
is added.

### M1.1 Tokenizer And Config Loading

Load tokenizer/config artifacts from a local Hugging Face-compatible model
directory.

Done when:

- `tokenizers` can encode/decode a prompt through the project wrapper
- BOS/EOS IDs are available or explicitly documented as unavailable
- tokenizer round-trip tests pass
- optional dev-only comparison with `AutoTokenizer` passes for selected prompts

### M1.2 Weight Loading

Load safetensors weights and map Hugging Face names to project names.

Prerequisites:

- Accept the Llama 3.2 license for `meta-llama/Llama-3.2-1B` on Hugging Face.
- Authenticate locally with `huggingface-cli login`.
- Download model artifacts into a local directory, for example:

```bash
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir ./models/llama-3.2-1b
```

The exact local model path should be configurable and should not be hard-coded
inside the engine.

Done when:

- all expected Llama 3.2 1B weight keys are loaded or intentionally skipped
- unexpected/missing keys are reported clearly
- loaded tensors have expected shapes and dtypes

### M1.3 Layer Implementations

Implement and test individual Llama components:

- RMSNorm
- RoPE
- GQA attention
- SwiGLU FFN
- embedding and linear helpers

Done when:

- layer-level shape tests pass
- simple deterministic unit tests pass
- comments/docstrings explain the relevant tensor shapes

### M1.4 Model Forward

Assemble the Llama model forward pass.

Done when:

- `model(input_ids)` returns logits with shape `(batch, seq_len, vocab_size)`
- a tiny synthetic Llama-compatible config works in tests
- optional Hugging Face parity checks are recorded as validation, not as the
  only completion gate

Use one shared tiny config for unit and shape tests so all agents test the same
fixture:

```text
n_layers = 2
d_model = 64
n_heads = 4
n_kv_heads = 2
intermediate_size = 128
vocab_size = 256
max_seq_len = 64
rope_theta = 500000.0
```

This fixture should live in `tests/conftest.py` once tests are implemented.

### M1.5 Prefill

Run the full prompt through the model and initialize/fill the KV cache.

Done when:

- prompt tokens produce final-position logits
- KV cache is allocated with documented shape
- cache positions `[0, prompt_len)` are filled
- prefill behavior is covered by tests

### M1.6 Decode Loop

Generate one token at a time using the KV cache.

Done when:

- decode appends K/V at the current position
- attention can read previous K/V from cache
- greedy sampling works
- generation stops at EOS or `max_tokens`
- CLI can produce local text from a prompt

### M1.7 MLX Evaluation And Baseline Metrics

Understand and document MLX lazy evaluation behavior.

Done when:

- `mx.eval()` placement is explicit and documented
- 100-token generation tokens/sec is recorded
- rough KV-cache memory usage is recorded for at least one sequence length

### M1.8 Sampling Strategies

Add probabilistic sampling after greedy decoding is correct.

Done when:

- temperature scaling is implemented and tested
- top-k filtering is implemented and tested
- top-p nucleus sampling is implemented and tested
- temperature `0` or an explicit greedy mode reproduces greedy output exactly
- fixed random seed produces deterministic sampled output where the backend
  supports seeded sampling

## Backend Plan

Phase 1 can use MLX directly where that keeps learning simple.

Phase 2 should introduce or harden a backend protocol before adding PyTorch/CUDA.
Avoid global namespace patching as the core backend design. Prefer an explicit
backend object or protocol that makes required operations clear and testable.

The backend protocol should cover nontrivial operations explicitly instead of
assuming NumPy, MLX, and PyTorch APIs are identical.

Phase 2 should also include:

- NumPy backend or fixtures for reference tests if useful
- PyTorch/CUDA backend
- backend parity tests for deterministic greedy decoding
- benchmark comparison across available backends

Phase 2 milestone names:

- M2.0 Backend protocol: introduce `backends/protocol.py` and refactor MLX code
  to conform.
- M2.1 PyTorch/CUDA backend: add `torch_backend.py` and run the model on Nvidia
  GPU.
- M2.2 Flash attention: use PyTorch scaled dot-product attention on CUDA and
  compare against naive attention.
- M2.3 Benchmark comparison: report tokens/sec and peak memory across available
  NumPy, MLX, and PyTorch/CUDA paths.

## Phase 3 Direction

Phase 3 should introduce multi-user serving progressively:

1. Request objects and request lifecycle.
2. FIFO scheduler.
3. Multi-request decode batching.
4. Streaming output.
5. PagedAttention-style KV block manager.
6. Preemption or recompute policy if memory is exhausted.

PagedAttention should be an explicit learning target in Phase 3, but it does not
need to be the first multi-user milestone.

Throughput acceptance criteria should compare against sequential processing of
the same workload, not require unrealistic speedups such as 4 concurrent
requests exceeding 4x single-request throughput.

Phase 3 milestone names:

- M3.0 Request objects and lifecycle state machine.
- M3.1 FIFO scheduler that still processes one request at a time.
- M3.2 Continuous batching across multiple decode positions.
- M3.3 Streaming token output.
- M3.4 PagedAttention-style KV block manager.
- M3.5 HTTP API with streaming responses.
- M3.6 Preemption or recompute policy when the KV page pool is exhausted.

## Testing Strategy

Phase 1 tests should favor mechanical correctness over semantic output quality.

Required test categories:

- tokenizer encode/decode behavior
- config parsing
- weight key mapping and shape validation
- RMSNorm shape/value behavior
- RoPE shape and position behavior
- attention mask behavior
- KV-cache shape and position updates
- sampling determinism for fixed logits
- engine prefill/decode state transitions

Smoke tests should verify:

- local model artifacts can be loaded when available
- generation completes without crashing
- `max_tokens` is respected
- EOS handling works
- greedy decoding is deterministic
- generated token IDs can be decoded

Smoke tests should not require a specific semantic first token such as
`" Paris"`. That is too brittle for a base model and not central to inference
engine correctness.

Hugging Face logit parity is valuable as a debugging and validation tool, but it
should not be the only early milestone gate. If used, tolerances and known
sources of numerical mismatch should be documented.

## Documentation And Agent Collaboration

The project should keep documentation close to implementation.

Important docs:

- `docs/refined-plan.md`: this consolidated plan
- `docs/agent-guidelines.md`: multi-agent collaboration contract
- `docs/architecture.md`: future merged architecture source of truth
- `docs/phases/phase-1-mlx-single-user.md`: phase-1 implementation contract
- `docs/phases/phase-1.5-qwen3-mlx.md`: phase-1.5 Qwen3 implementation contract
- `docs/phases/phase-1.5-taskboard.md`: phase-1.5 task tracking and review state
- `docs/phases/phase-1.6-generation-serving.md`: phase-1.6 generation UX and serving contract
- `docs/phases/phase-1.6-taskboard.md`: phase-1.6 task tracking and review state
- `docs/phases/phase-1.7-observability.md`: phase-1.7 observability and context-budget policy contract
- `docs/phases/phase-1.7-taskboard.md`: phase-1.7 task tracking and review state
- `docs/adr/*.md`: durable decisions

Recommended next documentation steps before implementation:

1. Create ADRs for durable decisions when a phase changes architecture.
2. Keep the active phase spec and taskboard current before implementation.
3. Update `docs/architecture.md` and `docs/agent-guidelines.md` when the phase
   roadmap or source-of-truth documents change.

## Learning Standards

Every public module, class, and function should have a docstring explaining:

- its role in the inference engine
- input and output meanings
- relevant tensor shapes
- why the implementation is structured that way

Non-obvious internal logic should include comments around:

- prefill versus decode
- KV-cache layout and updates
- attention masking
- RoPE position handling
- backend evaluation behavior
- sampling choices

The code should be easy to read line by line. If a compact implementation hides
the concept being learned, write the clearer version instead.

## Current Decisions

Unless changed by a future ADR, these decisions are settled:

- Python only
- no C++
- `uv` for environment management
- Python `>=3.12,<3.13`
- base `meta-llama/Llama-3.2-1B` for phase 1
- `Qwen/Qwen3-0.6B` as the Phase 1.5 model-portability target
- generation UX and single-request serving as the Phase 1.6 target
- no instruct/chat support in phase 1
- no full chat-template support in phase 1.5
- no full Transformers chat-template parity or conversation memory in phase 1.6
- `tokenizers` runtime dependency for tokenization
- `transformers` only as optional dev/test reference
- MLX first
- Qwen3 model-family support before backend expansion
- Phase 1.6 before backend expansion while the NVIDIA development environment is unavailable
- Phase 1.7 observability before backend expansion: measurement baseline needed for quantization and batching experiments
- PyTorch/CUDA second
- multi-user serving third
- learning clarity over raw performance
- core inference implemented in this repository
