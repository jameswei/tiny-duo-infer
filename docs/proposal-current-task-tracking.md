# Proposal: CURRENT.md — Active Task Tracking for Review-Gated Workflow

**Status:** Accepted — adopted for T04 onward  
**Author:** claudecode  
**Reviewed by:** codex (2026-06-05 — five required changes; all incorporated)  
**Date:** 2026-06-05

---

## Problem

The current workflow has one structural friction point: the human owner must
manually relay state between agents.

The relay appears in two places:

1. **Findings relay.** Codex finishes a review and writes findings as chat
   text. The human reads them, copies them into a new session with claudecode,
   who reads and applies the fixes. The spec and the code are authoritative,
   but the findings themselves are ephemeral — they live only in chat history.

2. **Sign-off relay.** Codex signals approval as chat text ("signed off, ready
   to commit"). The human relays this to claudecode. Claudecode proceeds.

The review gate itself (owner cannot mark `done`) is intentional and must stay.
The friction is that all state transitions flow through the human as a relay,
even when the state could be written directly to a file both agents can read.

A secondary friction: at the start of each new session, both agents re-read the
same five files (taskboard, phase spec, base.py, engine.py, ...) just to
re-orient to the active task. This cost compounds across a multi-task phase.

---

## Proposed Solution: CURRENT.md

Add a single mutable file, `CURRENT.md`, at the project root. It holds only
the live state of the active task. It is not a log; it is a whiteboard. When
a task transitions, the file is overwritten — previous content does not matter
because the taskboard already records the permanent history.

### File format

```markdown
# Current Task

Task:   P1.8-T04
Phase:  Phase 1.8 (docs/phases/phase-1.8-weight-quantization.md)
Owner:  cc
Status: in_progress

## Findings from last review
<!-- Reviewer writes here after examining the implementation. -->
<!-- Each finding should be tagged: [Blocking] / [Non-blocking] / [Nit] -->

- none

## Blocker
<!-- If status=blocked, explain what is preventing progress here. -->

- none
```

### Status lifecycle

```
todo → in_progress → review → done
                  ↑        |
                  |  (fixes applied, back to review)
                  └────────┘
```

The same four status values as the taskboard. `CURRENT.md` is always
consistent with the taskboard row for the active task.

### Mutable, not append-only

`CURRENT.md` is overwritten at each transition. It describes *now*, not
history. The taskboard `Notes` column is the permanent record. Do not treat
`CURRENT.md` as a changelog.

---

## What Changes

### 1. New file: `CURRENT.md` at project root

Created when a task moves to `in_progress`. Updated at every status
transition. Deleted (or reset to empty) when the phase is closed.

### 2. `AGENTS.md` gains one line

```markdown
If `CURRENT.md` exists, read it before the taskboard — it has the live
state of the active task including any open review findings.
```

This is the only change to `AGENTS.md`. No restructuring needed.

### 3. `docs/agent-guidelines.md` — Collaboration Flow update

Add step 0 and update steps 2 and 8:

```
0. Read CURRENT.md first if it exists.
...
2. (unchanged) Confirm the task belongs to the current phase.
...
8. Reviewer writes findings directly to CURRENT.md under "Findings from
   last review", tagged [Blocking] / [Non-blocking] / [Nit], and updates
   Status to `review`. Owner reads CURRENT.md to learn what to fix.
```

### 4. Taskboard stays the same

The taskboard structure, the `Notes` column, the review-sensitive task list —
all unchanged. The taskboard is the phase-level historical record. `CURRENT.md`
is just the fast-path view that avoids re-reading it in full every session.

---

## What Does NOT Change

- The review gate. Owner still cannot mark their own task `done`.
- Commit/push still requires explicit sign-off before proceeding.
- The taskboard is still updated at each transition (both files updated together).
- Codex and claudecode roles are unchanged.

The human relay for sign-off ("ready to commit and push") still exists —
neither agent can ping the other directly. But the *findings* relay is
eliminated: codex writes findings to `CURRENT.md`; the human just says
"codex updated CURRENT.md" rather than copying the full finding text.

---

## How the Flow Changes in Practice

### Before (current)

```
cc implements T04
→ cc sets taskboard T04 → review
→ human tells codex "please review T04"
→ codex reviews, writes findings as CHAT TEXT
→ human copies findings text → tells claudecode
→ cc fixes, re-runs tests
→ human tells codex "please re-review"
→ codex signs off as CHAT TEXT
→ human tells claudecode "codex signed off, commit"
→ cc commits
→ codex updates taskboard to done
```

### After (proposed)

```
cc implements T04
→ cc sets taskboard T04 → review, writes CURRENT.md status=review
→ human tells codex "please review T04, read CURRENT.md"
→ codex reviews, writes findings to CURRENT.md (tagged Blocking/Non-blocking)
→ human tells claudecode "codex updated CURRENT.md, please read it and fix"
→ cc reads CURRENT.md, fixes, re-runs tests, updates CURRENT.md
→ human tells codex "cc updated, please re-review"
→ codex signs off, writes CURRENT.md status=done, updates taskboard
→ human tells claudecode "CURRENT.md shows done, commit"
→ cc commits
```

The human still coordinates turns — that is unavoidable in async. But:
- Findings travel as a structured file, not freeform chat relay
- Session orientation is one file read instead of five
- Tags ([Blocking] / [Non-blocking]) make fix priority unambiguous

---

## Open Questions for Codex

1. **Finding tags.** Are `[Blocking]`, `[Non-blocking]`, `[Nit]` the right
   three levels? Should we add a `[Question]` tag for cases where the reviewer
   needs clarification rather than a fix?

2. **Sign-off record.** When codex signs off, should the sign-off be written
   to `CURRENT.md` (e.g. `Signed off by: codex, 2026-06-05`) or is the
   taskboard `Notes` column sufficient?

3. **Phase close.** When T09 is done and Phase 1.8 closes, should `CURRENT.md`
   be deleted, archived (renamed to `docs/phases/phase-1.8-current-final.md`),
   or reset to a blank template for Phase 1.9?

4. **Scope.** This proposal changes process, not code. Implementation is just
   creating `CURRENT.md` and two small doc edits. If you agree in principle,
   we can adopt it immediately for T04 onward. Does anything in the proposal
   conflict with how you work with the taskboard?

---

## Summary

| What | Before | After |
|---|---|---|
| Active task state | Taskboard row + chat history | `CURRENT.md` (always current) |
| Findings delivery | Human relays chat text | Codex writes directly to `CURRENT.md` |
| Session orientation | Re-read 5 files | Read `CURRENT.md` first |
| Finding priority | Inferred from prose | Tagged: Blocking / Non-blocking / Nit |
| Review gate | Unchanged | Unchanged |
| Taskboard | Unchanged | Unchanged |
| Human relay for sign-off | Required | Still required (async limitation) |
