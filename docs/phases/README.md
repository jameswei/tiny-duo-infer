# Phase Index

This file routes agents to the current implementation contract. It keeps the
default reading path narrow while preserving completed phase docs as historical
records.

## Current Phase

No implementation phase is active right now. Phase 1.7 is complete, and Phase 2
CUDA/NVIDIA work is deferred while the NVIDIA development environment is
unavailable.

Before claiming new implementation work, create or confirm the next phase spec
and taskboard.

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

## Deferred Phases

| Phase | Focus | Status |
|---|---|---|
| Phase 2 | PyTorch/CUDA backend for NVIDIA GPUs | Deferred |

Phase 2 should not be treated as the active implementation contract until a
dedicated spec and taskboard are created or reactivated.
