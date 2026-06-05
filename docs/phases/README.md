# Phase Index

This file routes agents to the current implementation contract. It keeps the
default reading path narrow while preserving completed phase docs as historical
records.

## Current Phase

No phase is currently active.

Phase 1.8 closed on 2026-06-06; see the Completed Phases table below for its
spec, taskboard, and handoff. The next implementation contract has not been
opened yet — the directional Phase 1.9 (speculative decoding) and Phase 1.10
(minimal continuous batching) entries below are not active until a dedicated
spec and taskboard are drafted.

Per `AGENTS.md`, agents should not claim or start implementation work until
the next phase scope is confirmed and a phase spec/taskboard exists.

## Agent Reading Rule

For normal work, agents should read:

1. `AGENTS.md`
2. `docs/file-structure.md`
3. `docs/agent-guidelines.md`
4. this file
5. the active phase spec and taskboard listed in `Current Phase`

Completed phase docs are historical references. Read them only when a task
depends on a previous implementation contract, when resolving a docs/code
conflict, or when the active phase explicitly points to them.

## Completed Phases

| Phase | Spec | Taskboard | Status |
|---|---|---|---|
| Phase 1 | `docs/phases/phase-1-mlx-single-user.md` | `docs/phases/phase-1-taskboard.md` | Done |
| Phase 1.5 | `docs/phases/phase-1.5-qwen3-mlx.md` | `docs/phases/phase-1.5-taskboard.md` | Done |
| Phase 1.6 | `docs/phases/phase-1.6-generation-serving.md` | `docs/phases/phase-1.6-taskboard.md` | Done |
| Phase 1.7 | `docs/phases/phase-1.7-observability.md` | `docs/phases/phase-1.7-taskboard.md` | Done |
| Phase 1.8 | `docs/phases/phase-1.8-weight-quantization.md` | `docs/phases/phase-1.8-taskboard.md` | Done |

## Later And Deferred Phases

| Phase | Focus | Status |
|---|---|---|
| Phase 1.9 | Speculative decoding | Directional |
| Phase 1.10 | Minimal continuous batching | Directional |
| Phase 2 | PyTorch/CUDA backend for NVIDIA GPUs | Deferred |

Phase 1.9, Phase 1.10, and Phase 2 should not be treated as active
implementation contracts until dedicated specs and taskboards are created or
reactivated.
