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
| P1.7-T02 | Context policy | Per-request context-budget policy | P1.7-T01 | done | Claude Opus 4.7 | request supports context policy; accept/truncate/reject outcomes are tested | implemented 2026-06-04 by Claude Opus 4.7. Changes: `GenerationRequest.context_policy` field with default `"allow_context_stop"` and `__post_init__` validation ([tiny_duo_infer/generation.py](tiny_duo_infer/generation.py)); new pure-Python module [tiny_duo_infer/context_policy.py](tiny_duo_infer/context_policy.py) provides `ContextBudgetError`, `ContextPolicyOutcome`, and `apply_context_policy(token_ids, max_new_tokens, max_seq_len, policy)`; engine integration intentionally deferred to T03. Tests: 14 request-side cases in [tests/test_generation.py](tests/test_generation.py) and 32 policy-module cases in [tests/test_context_policy.py](tests/test_context_policy.py) cover all 5 policies, both spec follow-up clarifications, and outcome accounting invariants. Both spec follow-ups (`phase-1.7-observability.md` "Minimum requirements") are enforced exactly: (a) `allow_context_stop` fail-fasts when `original_prompt_tokens > max_seq_len`; (c) **every** policy fail-fasts when `max_new_tokens > max_seq_len`. Codex review pass 1 (2026-06-04) flagged that an earlier draft exempted `allow_context_stop` from (c), violating the source-of-truth spec; the universal precondition was restored and a parametrized test now verifies all 5 policies reject this case. Reviewed and signed off by codex on 2026-06-04; `uv run pytest tests/test_generation.py tests/test_context_policy.py -q`: 72 passed; `uv run pytest -q`: 326 passed, 12 skipped. |
| P1.7-T03 | Engine metrics | Instrument `Engine.generate_request()` and streaming final response | P1.7-T01, P1.7-T02 | done | cc | real engine responses include coherent stats for all stop reasons and context policies | `_run_generation` now calls `apply_context_policy()` before prefill and instruments timing with `perf_counter()` (prompt_prepare, prefill, TTFT, running decode_ms accumulator, total); KV bytes computed from `cache._keys[0].dtype.size`; `GenerationStats` attached to `GenerationResponse.stats` on every call; `generate_stream()` final item inherits stats; `test_generate_request_stops_on_context_length` updated to `max_new_tokens=4` (<=max_seq_len=4 per T02 policy); 16 new unit tests cover: stats populated for all 4 stop reasons, prompt_tokens==accepted_prompt_tokens invariant, active_seq_len invariant, KV bytes ratio, zero throughput for max_new_tokens=0, non-negative timing, model_type, default/truncate_left/reject policies, stream final stats; 2 slow smoke tests for Llama and Qwen3; `decode_step_ms` left as empty list (default) in all non-profiling paths — per-step timing collection removed; `decode_ms` total tracked via running float accumulator; reviewed and signed off by codex on 2026-06-04 after confirming the `decode_step_ms` default-public-response finding was fixed; `uv run pytest tests/test_engine.py -q`: 48 passed, 2 skipped; `uv run pytest -q`: 341 passed, 11 skipped; no regressions |
| P1.7-T04 | CLI UX | Show stats and context policy in CLI output | P1.7-T03 | in_progress | Claude Opus 4.7 | `--show-stats` prints stable timing/memory/context fields; `--context-policy` works | started 2026-06-04; T03 already attaches `GenerationStats` to every real engine response, so T04 only wires the CLI surface; coordinating with `cc` who is working on T05 (`serving/api.py` + `tests/test_serving.py`) — non-overlapping files |
| P1.7-T05 | HTTP UX | Include stats and context policy in HTTP responses | P1.7-T03 | done | cc | `/generate` and final stream item include stats; HTTP validates `context_policy` | `GenerationStatsBody` Pydantic model added to `serving/api.py` — mirrors all `GenerationStats` fields except `decode_step_ms` (excluded per spec); `context_policy: str = "allow_context_stop"` added to `GenerateRequestBody`; `stats: GenerationStatsBody \| None = None` added to `GenerateResponseBody`; `_stats_to_body()` helper converts engine stats to HTTP body; `_to_generation_request()` forwards `context_policy` to `GenerationRequest` (validation and 422 handled by existing `__post_init__`); `/generate` endpoint populates `stats`; streaming final NDJSON chunk includes `stats` via `model_dump()`; fragment chunks unchanged; module docstring and `generate_stream` docstring updated to reflect new final chunk shape; 13 new tests: stats present/absent in `/generate`, all 19 required fields present, `decode_step_ms` excluded, 5 valid policies accepted (parametrized), `context_policy` forwarded to engine, invalid policy → 422, stream final chunk has stats, stream final chunk stats null when engine returns None, stream fragments omit stats, stream final chunk excludes `decode_step_ms`; slow smoke tests extended to assert `stats` non-null and `decode_step_ms` absent; reviewed and signed off by codex on 2026-06-04; `uv run pytest tests/test_serving.py -q`: 27 passed, 2 skipped; `uv run pytest -q`: 356 passed, 11 skipped; no regressions |
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
