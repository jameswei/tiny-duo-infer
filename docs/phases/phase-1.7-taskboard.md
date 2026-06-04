# Phase 1.7 Taskboard

This file tracks Phase 1.7 implementation tasks, dependencies, ownership, and
review state.

The proposed implementation contract is
`docs/phases/phase-1.7-observability.md`.

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
| P1.7-T00 | Planning | Phase 1.7 source-of-truth docs | Phase 1.6 complete | done | codex | spec and taskboard exist; scope is observability plus per-request context-budget accounting; review confirms phase boundaries | docs added by codex; reviewed and signed off by Claude Opus 4.7 on 2026-06-04; phase scope and boundaries confirmed against `roadmap_opus_v2.md` v2.2; 3 substantive spec clarifications to land before T01 starts: (a) `allow_context_stop` behavior when `original_prompt_tokens > max_seq_len` is undefined and should fail-fast like `reject`; (b) `decode_step_ms` should not appear in HTTP responses by default (profiling-only or capped); (c) request validation should reject `max_new_tokens > max_seq_len` regardless of policy; 4 minor improvements suggested: drop duplicate `median`/`p50` in profiling summary (line 423), pin CLI stats to stderr instead of leaving stdout/stderr open (line 375-377), document why `prompt_tokens == accepted_prompt_tokens` is required (API stability), add test matrix rows for prompt > `max_seq_len` under each policy. (T07 dep expansion was already addressed in the taskboard before sign-off.) |
| P1.7-T01 | Metrics model | Generation stats types and KV memory formula | P1.7-T00 | done | cc | stats type exists; KV allocated/active bytes are tested | `ContextPolicy` literal + `_VALID_CONTEXT_POLICIES` set; `kv_cache_bytes(n_layers, n_kv_heads, seq_len, head_dim, bytes_per_element=4)` formula function; `GenerationStats` dataclass with all required fields, invariant checks (prompt_tokens==accepted_prompt_tokens, active_seq_len==accepted+generated), optional decode_step_ms/model_type; `GenerationResponse.stats: GenerationStats | None = None`; 14 new tests cover KV formula (Llama/Qwen3 reference values, linear scaling, bfloat16), stats construction/invariants/policy validation, response backward compat; reviewed and signed off by codex on 2026-06-04; `uv run pytest tests/test_generation.py -q`: 38 passed; `uv run pytest -q`: 292 passed, 12 skipped; no regressions |
| P1.7-T02 | Context policy | Per-request context-budget policy | P1.7-T01 | todo | unassigned | request supports context policy; accept/truncate/reject outcomes are tested | |
| P1.7-T03 | Engine metrics | Instrument `Engine.generate_request()` and streaming final response | P1.7-T01, P1.7-T02 | todo | unassigned | real engine responses include coherent stats for all stop reasons and context policies | |
| P1.7-T04 | CLI UX | Show stats and context policy in CLI output | P1.7-T03 | todo | unassigned | `--show-stats` prints stable timing/memory/context fields; `--context-policy` works | |
| P1.7-T05 | HTTP UX | Include stats and context policy in HTTP responses | P1.7-T03 | todo | unassigned | `/generate` and final stream item include stats; HTTP validates `context_policy` | |
| P1.7-T06 | Profiling | Repeatable generation profiling script | P1.7-T03 | todo | unassigned | script supports prompt sets, context policy, warmups, runs, summaries, and `--json` | |
| P1.7-T07 | Docs | README, architecture, file-structure, and learning docs updates | P1.7-T01, P1.7-T02, P1.7-T04, P1.7-T05, P1.7-T06 | todo | unassigned | public behavior and learning notes describe observability and context-budget fields | |
| P1.7-T08 | Close | Verification and handoff | P1.7-T01, P1.7-T02, P1.7-T03, P1.7-T04, P1.7-T05, P1.7-T06, P1.7-T07 | todo | unassigned | pytest passes; real-model stats smoke results are recorded or skipped with reasons | |

## Review-Sensitive Tasks

These tasks require architecture/code review before being marked `done`:

- `P1.7-T00`: phase scope and roadmap positioning.
- `P1.7-T01`: public stats type, KV-cache memory formula, and response shape.
- `P1.7-T02`: context-budget policy semantics and compatibility with existing
  `context_length` stop behavior.
- `P1.7-T03`: timing boundaries, MLX eval semantics, and stop-reason coverage.
- `P1.7-T05`: HTTP response shape and streaming final metadata.
- `P1.7-T06`: profiling script output format and benchmark interpretation.

## Minimum Phase 1.7 Completion

Minimum Phase 1.7 completion requires `P1.7-T00` through `P1.7-T08`.

The phase is not complete until a non-owner reviewing agent signs off on the
handoff and verification results.

## Next Phase

Phase 1.8 is expected to focus on MLX-native weight-only quantization, but it
should not be drafted as an implementation contract until Phase 1.7 metrics are
in place and reviewed.
