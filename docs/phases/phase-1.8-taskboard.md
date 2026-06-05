# Phase 1.8 Taskboard

This file tracks Phase 1.8 implementation tasks, dependencies, ownership, and
review state.

The active implementation contract is
`docs/phases/phase-1.8-weight-quantization.md`.

## Status Values

- `todo`: not started
- `in_progress`: actively being implemented
- `review`: implementation is ready for review and verification
- `blocked`: cannot proceed; blocker must be written in `Notes`
- `done`: reviewed, tested, and accepted

## Update Rules

- Set `Status` to `in_progress` before starting work.
- Set `Status` to `review` after implementation and local tests.
- Set `Status` to `done` only after review and required tests pass.
- The task owner must not mark their own task `done`; a different reviewing
  agent must sign off and make the `done` update.
- When marking `done`, record the reviewing agent and test result in `Notes`.
- Use `blocked` only with a concrete blocker in `Notes`.
- Keep `Owner` as an agent/person name or `unassigned`.
- Do not change task IDs after creation.
- Update `Notes` with skipped tests, hardware limits, model-artifact limits, or
  follow-up work.
- Keep acceptance criteria short; detailed requirements belong in the phase
  spec.

## Taskboard

| ID | Milestone | Task | Depends On | Status | Owner | Acceptance | Notes |
|---|---|---|---|---|---|---|---|
| P1.8-T00 | Planning | Phase 1.8 source-of-truth docs | Phase 1.7 complete | done | codex | spec and taskboard exist; Phase 1.8 is active in phase index; review confirms scope is MLX weight-only quantization | reviewed and signed off by claudecode; three findings fixed before sign-off: T08 dependency includes T05, tiny Qwen3 quantization tests specify group_size=32 or another divisor of 32, and new GenerationStats quantization fields have explicit Python types; no code tests required for docs-only planning |
| P1.8-T01 | Representation | Quantization config and quantized weight object | P1.8-T00 | todo | unassigned | config validates bits/group/mode; quantized weight stores packed data and metadata |  |
| P1.8-T02 | Linear runtime | Quantized `Linear.forward()` path | P1.8-T01 | todo | unassigned | full-precision path unchanged; quantized path uses `mx.quantized_matmul()` |  |
| P1.8-T03 | Weight conversion | Quantize eligible project weights after HF conversion | P1.8-T01, P1.8-T02 | todo | unassigned | eligible linear matrices quantized; embeddings/norms stay full precision |  |
| P1.8-T04 | Engine and CLI | Engine loading and CLI quantization flags | P1.8-T03 | todo | unassigned | `Engine.from_model_path()` and CLI can select none/INT8/INT4 |  |
| P1.8-T05 | HTTP serving | HTTP server startup and worker quantization forwarding | P1.8-T04 | todo | unassigned | serving loads engine on worker thread with selected quantization config |  |
| P1.8-T06 | Metrics and profiling | Weight memory accounting and profiling comparison | P1.8-T04, P1.8-T05 | todo | unassigned | stats/profiling report quantization mode and linear-weight memory bytes |  |
| P1.8-T07 | Tests | Quantized correctness and regression coverage | P1.8-T02, P1.8-T03, P1.8-T04, P1.8-T05, P1.8-T06 | todo | unassigned | unit/integration tests cover Llama and Qwen3 quantized paths |  |
| P1.8-T08 | Docs | README, architecture, file-structure, strategy, and learning docs updates | P1.8-T04, P1.8-T05, P1.8-T06, P1.8-T07 | todo | unassigned | public usage and learning notes describe weight-only quantization |  |
| P1.8-T09 | Close | Real-model smoke and handoff | P1.8-T08 | todo | unassigned | pytest passes; real-model quantized smoke results are recorded or skipped with reasons |  |

## Review-Sensitive Tasks

These tasks require architecture/code review before being marked `done`:

- `P1.8-T00`: phase scope and roadmap positioning.
- `P1.8-T01`: public quantization config and quantized weight representation.
- `P1.8-T02`: `Linear.forward()` runtime path and MLX primitive usage.
- `P1.8-T03`: weight eligibility rules, tied Llama lm_head handling, and Qwen3
  required lm_head behavior.
- `P1.8-T04`: engine and CLI public interface.
- `P1.8-T05`: HTTP worker lifecycle and MLX stream affinity.
- `P1.8-T06`: memory accounting formulas and profiling interpretation.
- `P1.8-T09`: real-model verification and phase completion.

## Minimum Phase 1.8 Completion

Minimum Phase 1.8 completion requires `P1.8-T00` through `P1.8-T09`.

The phase is not complete until a non-owner reviewing agent signs off on the
handoff and verification results.

## Next Phase

Phase 1.9 is expected to focus on speculative decoding, but it should not be
drafted as an implementation contract until Phase 1.8 quantization behavior,
metrics, and real-model smoke results are reviewed.
