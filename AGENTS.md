# Agent Instructions

This repository is a learning-first, Python-only tiny LLM inference engine.
Agents must optimize for readable, teachable implementation over raw
performance.

Before changing code, read these documents in order:

1. `docs/phases/phase-1-mlx-single-user.md` — active implementation contract.
2. `docs/architecture.md` — active architecture reference.
3. `docs/refined-plan_codex.md` — roadmap and settled project decisions.
4. `docs/agent-guidelines.md` — collaboration process, review gates, handoff
   format, and testing expectations.

Current defaults:

- Phase 1 targets base `meta-llama/Llama-3.2-1B` on Apple Silicon via MLX.
- No instruct/chat-template support.
- Runtime tokenizer dependency is `tokenizers`; `transformers` is dev/test only.
- Do not use `transformers` model/generation APIs, `mlx-lm`, or vLLM as the
  engine implementation.
- No C++ or custom kernels.
- Phase 1 minimum completion is M1.0 through M1.7; M1.8 sampling is an
  extension unless a later spec changes that.

Follow `docs/agent-guidelines.md` for role responsibilities, source-of-truth
rules, review gates, and handoff format.
