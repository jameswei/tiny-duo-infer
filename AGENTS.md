# Agent Entry Point

If `CURRENT.md` exists at the project root, read it first — it has the live
state of the active task, any open review findings, and the reviewer's last
result. Then continue with the full doc reading order below.

Before changing code, read the project docs in this order:

1. `docs/file-structure.md`
2. `docs/agent-guidelines.md`
3. `docs/phases/README.md`

If `docs/phases/README.md` names an active phase, also read that phase's spec
and taskboard before changing code.

If no phase is active, do not claim or start implementation work until the next
phase scope is confirmed and a phase spec/taskboard exists.

Read `docs/architecture.md` and `docs/refined-plan.md` when a task changes
architecture, roadmap, public interfaces, or phase scope.

Completed phase documents are historical references. Do not read every
completed phase spec by default; use them only when the active phase, taskboard,
or code change depends on that history.

Use the active phase taskboard to claim tasks, update status, record blockers,
and mark review/done state.

`AGENTS.md` is only an entrypoint. Detailed rules, status, scope, review gates, and handoff expectations belong in the docs above.
