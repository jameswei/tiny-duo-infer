# Tiny Duo Infer Project Strategy

This document records the long-lived project direction and settled roadmap
decisions for `tiny-duo-infer`.

It is not the active implementation contract for a phase. For current phase
routing, read `docs/phases/README.md`. For subsystem boundaries, read
`docs/architecture.md`. For collaboration rules, read
`docs/agent-guidelines.md`.

## Purpose

`tiny-duo-infer` is a Python-only, learning-first inference engine inspired by
vLLM.

The goal is to understand inference-engine mechanics by implementing the core
pieces directly:

- model loading and weight conversion
- tokenization
- model forward pass
- prefill and decode
- KV cache management
- sampling
- request boundaries and serving interfaces
- observability and profiling
- later, backend portability and multi-user scheduling

The project is not trying to become a production inference server. It should
prefer readable, teachable code over compact code, clever abstractions, or raw
performance.

## Roadmap Status

| Area | Focus | Status | Source of truth |
|---|---|---|---|
| Phase 1 | Single-user Llama inference on Apple Silicon MLX | Done | `docs/phases/phase-1-mlx-single-user.md` |
| Phase 1.5 | Qwen3-0.6B support on the same MLX backend | Done | `docs/phases/phase-1.5-qwen3-mlx.md` |
| Phase 1.6 | CLI UX and single-request HTTP serving | Done | `docs/phases/phase-1.6-generation-serving.md` |
| Phase 1.7 | Timing, KV-cache memory, and context-budget observability | Done | `docs/phases/phase-1.7-observability.md` |
| Phase 1.8 | MLX-native weight-only quantization | Done | `docs/phases/phase-1.8-weight-quantization.md` |
| Phase 1.9 | Speculative decoding | Directional | no active implementation contract |
| Phase 1.10 | Minimal continuous batching | Directional | no active implementation contract |
| Phase 2 | PyTorch/CUDA backend for NVIDIA GPUs | Deferred | no active implementation contract |
| Phase 3 | Multi-user serving concepts | Future | no active implementation contract |

Phase routing for agents is tracked in `docs/phases/README.md`. Completed phase
specs remain in `docs/phases/` as historical implementation contracts. Agents
should read only the active phase spec and taskboard by default.

## Current Decisions

Unless changed by a future ADR or active phase spec, these decisions are
settled:

- Python only; no C++ or custom kernels.
- `uv` is the environment-management tool.
- The supported Python range is `>=3.12,<3.13`.
- MLX on Apple Silicon is the first backend.
- `meta-llama/Llama-3.2-1B` base is the Phase 1 Llama target.
- `Qwen/Qwen3-0.6B` is the Phase 1.5 model-portability target.
- Runtime tokenization uses `tokenizers`; `transformers` is optional dev/test
  reference material only.
- Core inference mechanics stay implemented in this repository.
- External libraries may provide primitive tensor operations, tokenizer
  runtime behavior, and safetensors loading, but not the engine itself.
- Plain prompt-to-completion generation is the baseline behavior.
- Full Transformers chat-template parity and conversation memory are out of
  scope until a future phase explicitly accepts them.
- Phase 1.6 HTTP serving remains single-request oriented; no batching or
  scheduler was introduced.
- Phase 1.7 observability established the measurement baseline needed before
  later optimization, quantization, batching, or backend comparison work.
- Phase 1.8 focuses on weight-only quantization using MLX primitives. It should
  preserve the project-owned `Linear` abstraction and use `mx.quantized_matmul`
  as the normal runtime path. `mx.dequantize()` is allowed only as an explicit
  test/debug fallback. Activation quantization, KV-cache quantization, and
  offline quantized artifact formats remain out of scope.
- Phase 1.8 introduces `--quantization {none,int4,int8}` and
  `--quant-group-size N` on the CLI, HTTP server startup, and profiling
  entrypoint. `Engine.from_model_path()` accepts an optional
  `QuantizationConfig`. `GenerationStats` adds 7 quantization fields, and
  the profiling JSON schema is bumped to v2.
- Speculative decoding and minimal continuous batching are directional follow-up
  phases, not part of Phase 1.8.
- PyTorch/CUDA remains useful for industry alignment but is deferred while the
  NVIDIA development environment is unavailable.
- Multi-user serving remains a later learning target after backend and
  single-request mechanics are stable.

## Phase Rationale

Phase 1 focused on the core Llama inference path: config, tokenizer, weight
loading, layers, model assembly, prefill, decode, sampling, and MLX lazy
evaluation.

Phase 1.5 added Qwen3 before CUDA because model-family portability teaches a
different lesson from backend portability. Qwen3 introduced explicit `head_dim`,
Q/K normalization before RoPE, different projection shapes, and model dispatch
while keeping the MLX backend stable.

Phase 1.6 added better user interaction before backend expansion because the
project needed clearer request and response boundaries. The CLI, structured
generation requests, stop conditions, simple chat prompt formatting, and
single-request HTTP API made the engine easier to exercise and review.

Phase 1.7 added observability before optimization. Timing, throughput, KV-cache
memory accounting, and context-budget policy make later changes measurable
instead of anecdotal.

Phase 1.8 adds weight-only quantization before speculative decoding or batching
because it is the most contained next inference-engine feature with immediate
single-request value. It teaches compressed weight storage, quantized linear
matmul, memory accounting, and measured tradeoffs while keeping the existing
single-request MLX engine structure.

Phase 2 CUDA is deferred, not cancelled. It should resume when the NVIDIA
development environment is reliable enough for implementation and verification.

Phase 3 should introduce multi-user serving progressively: request lifecycle,
FIFO scheduling, streaming, batching, KV block management, and memory-pressure
policy. PagedAttention-style ideas are important learning goals, but they should
not be introduced before simpler scheduling and cache mechanics are clear.

## Documentation Map

Use these documents by role:

- `AGENTS.md`: short bootstrap for agents.
- `docs/phases/README.md`: active phase pointer and completed/deferred phase
  index.
- `docs/phases/*.md`: phase specs, taskboards, and handoffs. Completed phase
  docs are historical unless the active phase depends on them.
- `docs/architecture.md`: active subsystem boundaries and implementation
  boundary.
- `docs/agent-guidelines.md`: multi-agent workflow, review gates, sign-off
  rules, and handoff expectations.
- `docs/file-structure.md`: current source, test, script, and docs map.
- `learning_materials/`: guided explanations for human study.
- `docs/adr/*.md`: durable decisions, if ADRs are introduced.

When docs and code disagree, do not silently choose one. Call out the conflict
and either update the stale document as part of the task or ask for
clarification if the correct direction is unclear.

## Revision Policy

Update this document when project-level direction changes, such as:

- a phase is completed, deferred, resumed, or replaced
- a major capability becomes a long-term target
- a durable default changes
- roadmap order changes
- a decision affects multiple future phases

Do not use this document for task-level acceptance criteria, detailed schemas,
exact implementation steps, or active task status. Those belong in the active
phase spec, active taskboard, architecture doc, or code.
