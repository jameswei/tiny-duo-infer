# Phase 1 Taskboard

This file tracks Phase 1 implementation tasks, dependencies, ownership, and
status in one lightweight table.

The active implementation contract is `docs/phases/phase-1-mlx-single-user.md`.
This taskboard should stay aligned with that spec.

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
- Update `Notes` with skipped tests, hardware limits, or follow-up work.
- Keep acceptance criteria short; detailed requirements belong in the phase spec.

## Taskboard

| ID | Milestone | Task | Depends On | Status | Owner | Acceptance | Notes |
|---|---|---|---|---|---|---|---|
| P1-T00 | M1.0 | Project scaffolding | none | done | cc | `uv sync`, import works, pytest runs | scaffold created; reviewed by Codex; all 6 slow tests skip cleanly |
| P1-T01 | M1.1 | Config loader | P1-T00 | done | codex | config fields parsed and tested | reviewed by cc; 14 unit tests pass; no findings |
| P1-T02 | M1.1 | Tokenizer wrapper | P1-T00 | done | cc | encode/decode and BOS/EOS tests pass | reviewed by codex; 25 unit tests pass, 2 slow skipped; no findings |
| P1-T03 | M1.2 | Safetensors loader | P1-T00 | done | codex | single/sharded safetensors load | reviewed by cc; 8 unit tests pass; no findings |
| P1-T04 | M1.2 | Llama weight converter | P1-T01, P1-T03 | done | codex | HF keys map, tied embeddings handled | reviewed by cc; 15 unit tests pass, 1 skipped; no findings |
| P1-T05 | M1.3 | Base module helpers | P1-T01 | done | cc | `Module`, `Linear`, `Embedding` tested | reviewed by codex; 16 unit tests pass; no findings |
| P1-T06 | M1.3 | KV cache | P1-T01 | done | cc | preallocated `update()`/`advance()` tests pass | reviewed by codex; 31 unit tests pass; no findings |
| P1-T07 | M1.3 | RMSNorm | P1-T05 | done | codex | manual formula tests pass | reviewed by cc; 5 unit tests pass; no findings |
| P1-T08 | M1.3 | RoPE | P1-T05 | done | cc | manual rotation tests pass | reviewed by codex; `uv run pytest`: 114 passed, 7 skipped; no findings |
| P1-T09 | M1.3 | SwiGLU FFN | P1-T05 | done | cc | shape and gate/up tests pass | reviewed by codex; `uv run pytest`: 122 passed, 7 skipped; no findings |
| P1-T10 | M1.3 | GQA attention | P1-T06, P1-T08 | done | codex | GQA axis, mask, cache tests pass | reviewed by cc; 6 unit tests pass; no findings |
| P1-T11 | M1.4 | Llama model assembly | P1-T07, P1-T09, P1-T10 | done | cc | tiny model forward shape passes | reviewed by codex; `uv run pytest`: 134 passed, 7 skipped; no findings |
| P1-T12 | M1.5 | Prefill path | P1-T02, P1-T04, P1-T11 | done | codex | cache filled, final logits returned | reviewed by cc; fix applied: `KVCache.eval()` added, prefill flushes cache buffers; `uv run pytest`: 141 passed, 7 skipped; no findings |
| P1-T13 | M1.6 | Greedy decode loop | P1-T12 | done | cc | EOS/max-token stop, deterministic | reviewed by codex; fixes applied: cache.eval() per decode step, no extra decode on last step, top-level import; `uv run pytest`: 154 passed, 7 skipped; no findings |
| P1-T14 | M1.6 | CLI | P1-T13 | done | codex | CLI generates local text | reviewed by cc; 6 unit tests pass; no findings; `uv run pytest`: 160 passed, 7 skipped |
| P1-T15 | M1.7 | MLX eval placement | P1-T13 | todo | unassigned | `mx.eval()` placement documented | |
| P1-T16 | M1.7 | Benchmark script | P1-T13 | todo | unassigned | tokens/sec and KV memory reported | |
| P1-T17 | M1.8 | Sampling extension | P1-T13 | todo | unassigned | temp/top-k/top-p tests pass | optional |
| P1-T18 | Phase close | Handoff and verification | P1-T14, P1-T15, P1-T16 | todo | unassigned | handoff complete, tests recorded | |

## Review-Sensitive Tasks

These tasks require architecture/code review before being marked `done`:

- `P1-T06`: KV cache, especially `current_len`, `update()`, and `advance()`
  semantics.
- `P1-T10`: GQA attention, especially KV-head repeat axis, causal masking, and
  cache interaction.
- `P1-T12`: prefill, especially cache fill positions and final-position logits.
- `P1-T13`: decode loop, especially cache position increments, EOS handling, and
  `max_new_tokens` stopping.

## Minimum Phase 1 Completion

Minimum Phase 1 completion requires `P1-T00` through `P1-T16` plus `P1-T18`.

`P1-T17` is an M1.8 extension. It is useful, but not required for minimum Phase
1 completion unless the phase spec is updated later.
