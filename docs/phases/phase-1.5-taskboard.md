# Phase 1.5 Taskboard

This file tracks Phase 1.5 implementation tasks, dependencies, ownership, and
review state.

The active implementation contract is `docs/phases/phase-1.5-qwen3-mlx.md`.
That spec is the source of truth for Qwen3-0.6B support.

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

## Taskboard

| ID | Milestone | Task | Depends On | Status | Owner | Acceptance | Notes |
|---|---|---|---|---|---|---|---|
| P1.5-T00 | Planning | Phase 1.5 source-of-truth docs | Phase 1 complete | done | codex | spec and taskboard exist; `AGENTS.md`, `docs/refined-plan.md`, `docs/architecture.md`, `docs/agent-guidelines.md`, and `docs/file-structure.md` reference Phase 1.5 | reviewed by cc; spec reviewed across three rounds — all findings resolved; taskboard structure and dependencies verified; all referenced docs confirmed present |
| P1.5-T01 | Config | Config generalization | P1.5-T00 | done | codex | Llama and Qwen3 configs parse; explicit `head_dim`; derived `qk_norm` | reviewed by cc; all acceptance criteria met — explicit head_dim stored, qk_norm derived from model_type, A≠D Qwen3 path validated, Llama invariant scoped correctly, all direct ModelConfig(...) callers updated; 177 passed, 7 skipped |
| P1.5-T02 | Model | Qwen3 attention support | P1.5-T01 | done | cc | Qwen3 Q/K norm before RoPE; `H * Dh != D` tests pass | reviewed by codex; explicit `Qwen3Attention` with q_norm/k_norm before RoPE; A≠D projection shape verified; `uv run pytest tests/test_layers.py`: 33 passed; `uv run pytest`: 183 passed, 7 skipped; no findings |
| P1.5-T03 | Weights | Qwen3 weight conversion | P1.5-T01 | done | codex | Qwen3 HF keys map and validate; absent `lm_head.weight` errors | reviewed by cc; all 14 HF key patterns mapped; A≠D shapes correct for q_proj/o_proj; q/k norm shape (Dh,) validated; lm_head.weight required with no tied fallback; unexpected keys warn; non-qwen3 config rejected; 191 passed, 7 skipped |
| P1.5-T04 | Engine | Model assembly and dispatch | P1.5-T02, P1.5-T03 | done | codex | engine loads Llama or Qwen3 by `model_type`; tiny Qwen3 prefill/decode passes | reviewed by cc; explicit Qwen3Block/Qwen3Model with same forward signature as Llama; dispatch via _model_class_and_converter; q_norm/k_norm weight routing verified end-to-end; all Llama tests preserved; 199 passed, 7 skipped |
| P1.5-T05 | CLI | Tokenizer and CLI smoke | P1.5-T04 | todo | unassigned | Qwen3 tokenizer loads; CLI works with Qwen3 model path | document prompt-mode limitations |
| P1.5-T06 | Close | Real model verification and handoff | P1.5-T05 | todo | unassigned | pytest passes; real Qwen3 smoke/benchmark recorded or skipped with reason | non-owner review required before `done` |

## Review-Sensitive Tasks

These tasks require architecture/code review before being marked `done`:

- `P1.5-T01`: config semantics, especially explicit `head_dim` and derived
  `qk_norm`.
- `P1.5-T02`: Q/K norm placement before RoPE and attention shapes where
  `H * Dh != D`.
- `P1.5-T03`: Qwen3 weight layout, required q/k norm weights, and
  `lm_head.weight` handling.
- `P1.5-T04`: engine model-family dispatch and Llama regression behavior.

## Minimum Phase 1.5 Completion

Phase 1.5 completion requires `P1.5-T00` through `P1.5-T06`.

The phase is not complete until a non-owner reviewing agent signs off on the
handoff and verification results.
