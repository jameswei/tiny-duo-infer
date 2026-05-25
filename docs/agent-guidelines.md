# Agent Collaboration Guidelines

This document defines how multiple agents should collaborate on
`tiny-duo-infer`.

The project is a learning-focused inference engine, so collaboration should
optimize for shared understanding, explicit decisions, and readable
implementation over speed alone.

## Source Of Truth

Agents should read the relevant planning documents before changing code.

Use this source-of-truth order:

1. Active phase spec, currently `docs/phases/phase-1-mlx-single-user.md`:
   implementation contract, milestone scope, and completion criteria.
2. `docs/architecture.md`: active architecture reference and subsystem
   boundaries.
3. `docs/refined-plan.md`: settled project direction, roadmap, and
   cross-review decisions.
4. `docs/adr/*.md`: architecture decisions that should not be silently changed.
5. Code and tests: implementation truth.
6. Earlier proposal and review docs: historical context only, not current
   implementation contracts.

If documents and code disagree, agents should not silently choose one. The agent
should call out the conflict and either update the relevant document as part of
the task or ask for clarification if the correct direction is unclear.

## Role Definitions

Each agent should operate in one clear role for a task. A single agent may fill
multiple roles only when the task is small, but the handoff should still state
which responsibilities were covered.

### Main Developer

The main developer implements scoped changes.

Responsibilities:

- confirm the task belongs to the current phase
- follow the active phase spec and ADRs
- keep changes narrow and intentional
- write teaching-oriented docstrings and comments
- update docs when behavior, interfaces, or architecture decisions change
- run relevant tests before handoff
- produce a concise handoff note

The main developer should not introduce major architectural changes without
updating the phase spec or adding an ADR.

### Architecture Reviewer

The architecture reviewer checks whether a change fits the intended engine
design.

Responsibilities:

- verify control-plane and data-plane separation
- verify backend boundaries remain clean
- check that MLX-specific logic does not leak into generic engine code unless
  explicitly justified
- check that phase scope is respected
- identify decisions that need ADRs
- flag abstractions that are too broad, too narrow, or premature

The architecture reviewer should focus on design risks rather than formatting or
small implementation style issues.

### Code Reviewer

The code reviewer checks implementation quality.

Responsibilities:

- verify correctness and maintainability
- check that public modules, classes, and functions have useful docstrings
- check that non-obvious inference logic has explanatory comments
- look for unclear names, dense code, hidden assumptions, and missing edge cases
- verify tests cover the important behavior introduced by the change
- flag code that hides learning-critical mechanics behind external libraries

The code reviewer should prioritize bugs, behavior regressions, and learning
clarity.

### Test Verifier

The test verifier runs tests and records reproducibility information.

Responsibilities:

- run the relevant unit tests
- run smoke tests when the required hardware and model artifacts are available
- record the operating system, Python version, backend, and hardware used
- report skipped tests with a clear reason
- report failing commands with enough output to diagnose the issue

The test verifier should not mark a phase complete if required tests were
skipped without an explicit documented reason.

### Learning Reviewer

The learning reviewer is optional but useful for this project.

Responsibilities:

- check whether the code can be read line by line by a learner
- identify places where comments or docs should explain the inference concept
- flag overly clever code even if it is technically correct
- suggest documentation improvements that would help future study

The learning reviewer should not request excessive comments for trivial code.
Comments should explain reasoning, invariants, shapes, and inference concepts.

## Collaboration Flow

Use this flow for normal implementation tasks:

1. Read the relevant proposal, phase spec, ADRs, and existing code.
2. Confirm the task belongs to the current phase.
3. Make the smallest change that satisfies the task.
4. Keep code educational and explicit.
5. Update docs if the change affects behavior, public interfaces, architecture,
   or phase scope.
6. Run relevant tests.
7. Produce a handoff note.
8. Reviewer checks the change against the role-specific checklist.
9. Test verifier records test results and environment details.

For large design changes, architecture review should happen before full
implementation.

## Handoff Format

Every substantial implementation task should end with a handoff note.

Use this format:

```markdown
## Handoff

### Task Summary

Briefly describe what changed and why.

### Files Changed

- `path/to/file.py`: short purpose of change

### Design Decisions

- Decision made and reason

### Tests Run

- `command`: pass/fail/skip

### Known Gaps

- Any limitation, skipped test, missing hardware, or incomplete follow-up

### Learning Notes

- Concepts or implementation areas that deserve careful line-by-line reading

### Questions For Next Agent

- Open questions, if any
```

Small documentation-only changes may use a shorter handoff, but they should
still state what changed and whether tests were run.

## Review Gates

Use these gates to avoid unclear or unverified changes.

Architecture review is required when a change:

- introduces or changes backend boundaries
- changes the engine public API
- changes model loading or weight layout assumptions
- changes KV-cache layout
- changes KV-cache position semantics, including `current_len`, `advance()`, or
  per-layer write behavior
- changes the phase roadmap or scope
- adds a new major dependency

Code review is required when a change:

- adds or changes runtime behavior
- adds model, cache, backend, engine, or sampling logic
- changes tests or test strategy
- changes public interfaces

Test verification is required when a change:

- claims a feature works
- changes generation behavior
- changes backend behavior
- changes model loading
- closes a phase completion criterion

Documentation updates are required when a change:

- changes public API usage
- changes architecture or phase scope
- adds a decision future agents should follow
- introduces known limitations or hardware requirements

## Conflict Handling

Agents should handle conflicts explicitly.

If docs and code disagree:

- identify the conflict
- decide whether the code or docs should change
- update the stale source when the correct direction is clear
- ask for clarification when the correct direction is not clear

If phase scope and a requested change disagree:

- do not silently expand the phase
- propose a phase-spec update or record the change as out of scope

If two agents disagree on architecture:

- reduce the disagreement to a concrete decision
- record the final choice in an ADR if it affects future work

If required hardware is unavailable:

- skip only the hardware-dependent tests
- run all hardware-independent tests
- report the skip reason clearly

## Learning-First Implementation Rules

This project values readable, teachable code.

Agents should:

- prefer explicit control flow over compact cleverness
- use descriptive names for tensors, cache positions, token IDs, and shapes
- document tensor shapes in docstrings where relevant
- explain prefill, decode, attention masking, KV-cache updates, and sampling
  choices near the code that implements them
- keep core inference mechanics inside this repository
- use external libraries for backend tensor operations, tokenization, and file
  loading, not to hide the engine itself

Agents should avoid:

- adding C++ or custom kernel code
- wrapping `mlx-lm` or vLLM as the core engine
- optimizing before the baseline behavior is clear
- introducing abstractions that are not needed by the current or next phase
- merging code that works but is too opaque for learning

## Testing Expectations

Agents should run the narrowest useful test set during development and broader
tests before handoff.

Expected phase-1 test categories:

- unit tests for config parsing, sampling, cache behavior, and engine state
- shape tests for attention and KV-cache updates
- smoke test for local MLX generation when model artifacts are available

Phase 1 minimum completion is M1.0 through M1.7 in
`docs/phases/phase-1-mlx-single-user.md`. M1.8 probabilistic sampling is a
Phase 1 extension unless a later phase spec or ADR makes it mandatory.

Phase 1 smoke tests should verify mechanical behavior only:

- local model artifacts load when available
- generation completes without crashing
- `max_new_tokens` is respected
- EOS handling works when EOS is encountered
- greedy decoding is deterministic
- generated token IDs decode to non-empty text

Smoke tests must not require a specific semantic token or phrase, such as
`" Paris"`.

Test reports should include:

- command run
- pass, fail, or skip status
- reason for skip, if skipped
- relevant environment details for backend tests

Suggested environment details:

- operating system
- CPU architecture
- Python version
- backend name and version
- GPU or accelerator availability

## Documentation Expectations

Documentation should stay close to the code and phase roadmap.

Agents should add or update:

- ADRs for durable architecture decisions
- phase specs for phase-level scope and acceptance criteria
- architecture docs for subsystem boundaries
- learning notes when a concept is subtle enough to deserve a guided reading

Docs should be concise but concrete. Avoid vague claims such as "improve
performance" or "clean up architecture" without explaining what changed and why.

## Current Project Defaults

Unless a later ADR changes these defaults, agents should assume:

- Python-only implementation
- learning clarity over raw performance
- MLX backend first
- PyTorch/CUDA backend second
- multi-user serving third
- single-request execution in phase 1
- Hugging Face-compatible local model artifacts
- `meta-llama/Llama-3.2-1B` as the primary phase-1 model target
- base model only in phase 1; no instruct/chat-template support
- `tokenizers` as the phase-1 runtime tokenizer dependency
- `transformers` only as an optional dev/test reference
- TinyLlama only as an optional fallback or debug target, not a phase-1
  acceptance target
- no C++ code
- no HTTP server in phase 1
- no quantization, speculative decoding, paged attention, or distributed
  inference in phase 1
