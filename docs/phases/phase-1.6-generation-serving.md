# Phase 1.6 Spec: Generation UX And Single-Request Serving

**Status:** Draft  
**Authors:** Codex  
**Based on:** `docs/refined-plan.md`, `docs/phases/phase-1-mlx-single-user.md`, `docs/phases/phase-1.5-qwen3-mlx.md`  
**Date:** 2026-06-03

---

## Goal

Improve the user-facing generation experience and add a minimal local HTTP
serving layer while staying on the completed Phase 1/1.5 MLX engine.

Phase 1.6 intentionally pauses PyTorch/CUDA backend work. The NVIDIA backend
remains valuable, but it is deferred until the CUDA development machine is
ready. This phase uses the working Apple Silicon/MLX path to teach request
handling, generation controls, streaming boundaries, and service integration.

By the end of this phase, the project should support:

- a refined CLI with better generation controls
- structured generation request/response metadata
- stop strings and token accounting
- optional deterministic sampling through explicit seeds
- simple chat prompt formatting for better Qwen3 interaction
- a local single-request HTTP API with full-response and streaming endpoints

---

## Scope

### In scope

- Preserve existing Llama-3.2-1B and Qwen3-0.6B support on MLX.
- Add project-owned generation request and response types.
- Validate generation parameters before model execution.
- Support stop conditions:
  - EOS token
  - `max_new_tokens`
  - user-provided stop strings
  - total context length limit
- Report prompt token count, generated token count, and stop reason.
- Add optional seeded sampling where MLX supports it cleanly.
- Add simple prompt formatting for chat-style requests, especially Qwen3.
- Refine CLI flags and output stats.
- Add a local HTTP layer that loads one model per process.
- Add non-batched streaming by exposing the existing token iterator over HTTP.
- Add unit tests with fake engines and slow smoke hooks for real local models.
- Update README, architecture docs, and learning docs when behavior changes.

### Out of scope

- PyTorch/CUDA backend support.
- Backend protocol enforcement beyond the current Phase 1 draft.
- Multiple active generation requests.
- Decode batching or continuous batching.
- Scheduler policy beyond a single-request lock or explicit busy response.
- PagedAttention or dynamic KV block management.
- Conversation memory across requests.
- Full Transformers `apply_chat_template()` parity.
- Production HTTP hardening, authentication, TLS, metrics, or deployment.
- Quantization, speculative decoding, MoE support, or distributed inference.

---

## Runtime And Tooling

Phase 1.6 keeps the Phase 1/1.5 runtime model:

- Python `>=3.12,<3.13`
- MLX as the only tensor backend
- `tokenizers` as the runtime tokenizer dependency
- `transformers` only as a dev/test reference dependency

The HTTP server may add one small runtime dependency if needed, such as
`fastapi` plus `uvicorn`. If a server dependency is added, update
`pyproject.toml`, README usage, and tests in the same task.

Runtime code under `tiny_duo_infer/` must not import `transformers`.

---

## Architecture Constraints

1. **Keep model execution unchanged.** Phase 1.6 must not alter core attention,
   model, weight-loading, or KV-cache semantics unless required by a reviewed
   bug fix.

2. **Stay single-request.** The engine still owns one active `KVCache` at a
   time. Serving may serialize requests with a lock or return a clear busy
   response, but it must not attempt batching.

3. **Treat streaming as I/O, not scheduling.** Streaming means yielding decoded
   fragments from the existing generation iterator to CLI or HTTP clients.
   Continuous batching and scheduler-driven decode loops remain Phase 3 work.

4. **Keep stop handling in the control plane.** Stop strings and token
   accounting are request-lifecycle logic. They should not be hidden inside
   model layers or tokenizer internals.

5. **Chat formatting is prompt construction.** Simple chat mode may format the
   current request into model-specific text before tokenization. It must not
   introduce conversation storage, agent behavior, or a chat framework.

6. **Keep code educational.** New request, response, CLI, and serving code must
   be explicit about request state, stop reasons, and streaming behavior.

---

## Generation Request Model

Introduce project-owned request and response types for generation-facing code.
The exact module name can be chosen during implementation, but the types should
live under `tiny_duo_infer/` rather than under tests or scripts.

Required request fields:

```python
prompt: str | None
messages: list[ChatMessage] | None
max_new_tokens: int
temperature: float
top_k: int
top_p: float
stop: list[str]
seed: int | None
chat: bool
```

Required message fields:

```python
role: str  # "system", "user", or "assistant"
content: str
```

Required response metadata:

```python
text: str
prompt_tokens: int
generated_tokens: int
stop_reason: str  # "eos", "max_new_tokens", "stop_string", or "context_length"
```

Validation requirements:

- Exactly one of `prompt` or `messages` must be provided.
- `prompt` and message contents must not be empty after validation.
- `max_new_tokens >= 0`.
- `temperature >= 0.0`.
- `top_k >= 0`.
- `0.0 < top_p <= 1.0`.
- Stop strings must be non-empty strings.
- `messages` are valid only when chat formatting is requested.

---

## Stop Conditions And Accounting

Generation must stop when the first applicable condition is reached:

1. EOS token is sampled.
2. A configured stop string appears in the decoded output.
3. `max_new_tokens` generated tokens have been produced.
4. The request cannot continue without exceeding `max_seq_len`.

The implementation must record a stop reason. If multiple conditions become
true at the same step, prefer the earliest condition in the list above.

Stop strings are matched against decoded text. The stop marker should not be
included in the returned final text. Streaming may emit text fragments before a
later stop string match is discovered; if this creates an unavoidable partial
marker edge case, document it and test the chosen behavior.

Prompt token count should be measured after final prompt formatting and
tokenization. Generated token count counts sampled tokens that are part of the
generation loop, excluding the prompt.

---

## Chat Formatting

Phase 1.6 supports simple chat prompt formatting to improve interaction quality,
especially for Qwen3.

Requirements:

- Plain prompt-to-completion remains the default path.
- Chat mode formats the current request into a prompt string before tokenization.
- Qwen3 chat mode should use a small deterministic template compatible with the
  downloaded tokenizer metadata and documented in tests.
- Llama base-model chat mode may either:
  - use the same simple role-marker template, or
  - reject chat mode with a clear error if the model family is not supported.
- Do not import `transformers` at runtime to call `apply_chat_template()`.
- Do not store conversation history across requests.

If exact Hugging Face chat-template parity is needed later, add it as a
separate reviewed follow-up.

---

## CLI Requirements

The CLI remains a thin wrapper over the engine and generation request layer.

Existing flags remain supported:

- `--model-path`
- `--prompt`
- `--max-new-tokens`
- `--max-seq-len`
- `--temperature`
- `--top-k`
- `--top-p`

New or refined flags:

- `--chat`: format input as a chat-style request.
- `--message ROLE:CONTENT`: optional structured messages for chat mode.
- `--stop TEXT`: repeatable stop string.
- `--seed N`: deterministic sampling seed when supported.
- `--show-stats`: print token counts and stop reason after generation.

CLI output should keep generated text on stdout. If stats are printed, write
them after generation in a stable format that tests can assert. Error messages
should remain clear argparse or validation errors.

---

## HTTP Serving Requirements

Add a local server entrypoint that loads one model once and serves generation
requests.

Required endpoints:

```text
POST /generate
POST /generate/stream
GET /health
```

`POST /generate` returns JSON containing final text and response metadata.

`POST /generate/stream` streams decoded fragments from the same generation path.
The stream format can be plain text chunks or newline-delimited JSON, but the
choice must be documented and covered by tests.

`GET /health` returns a simple status response showing the server is running
and whether a request is currently active.

Concurrency behavior:

- The server supports one active generation request at a time.
- Concurrent requests must either wait behind a lock or return a clear busy
  response. Pick one behavior in the serving task and document it.
- Do not implement a scheduler or request queue in Phase 1.6.

Server entrypoint example:

```bash
uv run python -m tiny_duo_infer.serving.api \
  --model-path ./models/qwen3-0.6b \
  --max-seq-len 2048
```

---

## Testing Requirements

Unit tests should cover:

- request and message validation
- prompt vs messages exclusivity
- stop-string matching and trimming
- stop-reason priority
- prompt and generated token accounting
- seeded sampling behavior when deterministic behavior is supported
- chat formatting for Qwen3 and unsupported-model behavior if applicable
- CLI flag parsing with fake engines
- HTTP full-response endpoint with a fake engine
- HTTP streaming endpoint order and final metadata behavior
- busy or lock behavior for concurrent HTTP requests

Slow smoke tests should be marked with `@pytest.mark.slow` and skipped unless
`--run-slow` is passed. Real-model smoke tests should verify mechanical
behavior only: generation completes, stops correctly, and reports metadata.
They must not require a specific semantic phrase.

---

## Completion Criteria

Phase 1.6 is complete when:

- `P1.6-T00` through `P1.6-T07` are marked `done` by a reviewing agent.
- The normal test suite passes.
- Real-model smoke tests for local Llama and Qwen3 are recorded, or skipped
  with explicit reasons.
- README documents CLI and HTTP usage.
- Architecture docs describe Phase 1.6 as single-request serving, distinct from
  Phase 3 batching and PagedAttention.

---

## Known Tradeoffs

- HTTP serving arrives before batching. This is intentional: it teaches request
  boundaries and streaming without changing tensor shapes or cache layout.
- Chat formatting improves user experience but is still prompt construction,
  not a full chat runtime.
- Stop-string handling on streaming output can have partial-marker edge cases.
  The implementation should choose simple, documented behavior.
- CUDA remains deferred until the NVIDIA development environment is available.
