# Phase 1.6 Taskboard

This file tracks Phase 1.6 implementation tasks, dependencies, ownership, and
review state.

The active implementation contract is
`docs/phases/phase-1.6-generation-serving.md`.

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
| P1.6-T00 | Planning | Phase 1.6 source-of-truth docs | Phase 1.5 complete | done | codex | spec and taskboard exist; `AGENTS.md`, `README.md`, `docs/file-structure.md`, `docs/refined-plan.md`, `docs/architecture.md`, and `docs/agent-guidelines.md` reference Phase 1.6 | reviewed by cc; all 6 required docs reference Phase 1.6; spec goal/scope/constraints/acceptance criteria are consistent; taskboard dependencies verified; note for T05: `--message ROLE:CONTENT` must split on first colon only |
| P1.6-T01 | Request UX | Generation request/response types | P1.6-T00 | done | cc | request validation and response metadata tests pass | reviewed by codex; `tiny_duo_infer/generation.py` adds ChatMessage, GenerationRequest, GenerationResponse; 24 new generation tests cover prompt/messages exclusivity, numeric validation, stop strings, defaults, and response metadata; existing engine tests cover empty token prefill and prompt length limit; `uv run pytest tests/test_generation.py tests/test_engine.py -q`: 42 passed, 3 skipped; `uv run pytest -q`: 228 passed, 9 skipped; no findings |
| P1.6-T02 | Request UX | Stop conditions and token accounting | P1.6-T01 | done | cc | EOS, max-token, stop-string, and context-limit stops are tested | reviewed by codex; context_length fix confirmed before decode/counting of out-of-bounds token; stop reasons, stop-string trimming, prompt/generated token counts, max_new_tokens=0, and chat rejection covered; `uv run pytest tests/test_engine.py::test_generate_request_stops_on_context_length -q`: 1 passed; `uv run pytest tests/test_engine.py tests/test_generation.py -q`: 52 passed, 3 skipped; `uv run pytest -q`: 238 passed, 9 skipped; no findings |
| P1.6-T03 | Sampling UX | Seeded sampling support | P1.6-T01 | done | cc | deterministic seed behavior is supported or explicitly documented if limited by MLX | reviewed by codex; `mx.random.seed(request.seed)` is called after prefill and before first sample, making probabilistic sampling deterministic for that request while leaving greedy behavior unchanged; 4 seeded tests cover same-seed determinism, different-seed divergence, `seed=None`, and greedy invariance; `uv run pytest tests/test_engine.py -k seed -q`: 4 passed, 31 deselected; `uv run pytest -q`: 242 passed, 9 skipped; no findings |
| P1.6-T04 | Prompt UX | Chat prompt formatting | P1.6-T01 | done | cc | Qwen3 chat-style prompt formatting works without runtime `transformers` | reviewed by codex; Qwen3 ChatML formatting is explicit and runtime-`transformers` free; Llama/base-model chat raises clearly; engine integration test confirms `generate_request()` sends exact ChatML to tokenizer; `docs/file-structure.md` updated for `prompt.py` and `test_prompt.py`; `uv run pytest tests/test_prompt.py tests/test_engine.py -q`: 41 passed, 3 skipped; `uv run pytest -q`: 251 passed, 9 skipped; no findings |
| P1.6-T05 | CLI UX | Refined CLI flags and stats | P1.6-T02, P1.6-T03, P1.6-T04 | done | cc | CLI supports chat, stop strings, seed, and stats with tests | reviewed by codex; CLI now builds `GenerationRequest` before model loading, supports `--chat`, repeatable `--message ROLE:CONTENT` with first-colon split, repeatable `--stop`, `--seed`, and `--show-stats`; prompt/message exclusivity and fail-fast invalid request cases covered; README updated with plain and Qwen3 chat examples plus flags table; `uv run pytest tests/test_cli.py -q`: 22 passed, 2 skipped; `uv run pytest -q`: 266 passed, 10 skipped; no findings |
| P1.6-T06 | Serving | Single-request HTTP API | P1.6-T02, P1.6-T04 | todo | unassigned | `/generate`, `/generate/stream`, and `/health` work with fake-engine tests |  |
| P1.6-T07 | Close | Real-model verification and handoff | P1.6-T05, P1.6-T06 | todo | unassigned | pytest passes; real Llama/Qwen3 smoke results are recorded or skipped with reasons |  |

## Review-Sensitive Tasks

These tasks require architecture/code review before being marked `done`:

- `P1.6-T01`: public request/response types and validation behavior.
- `P1.6-T02`: generation stop semantics, stop-reason priority, and token
  accounting.
- `P1.6-T04`: chat formatting boundaries and model-family behavior.
- `P1.6-T06`: HTTP serving, streaming semantics, and single-request concurrency
  behavior.

## Minimum Phase 1.6 Completion

Minimum Phase 1.6 completion requires `P1.6-T00` through `P1.6-T07`.

The phase is not complete until a non-owner reviewing agent signs off on the
handoff and verification results.

## Next Phase

Phase 2 adds NVIDIA/PyTorch/CUDA backend support. It remains deferred until the
CUDA development environment is available. Use `docs/refined-plan.md` and
future Phase 2 docs as the implementation contract when that work resumes.
