# Phase 1.7 Spec: Engine Observability And Context-Budget Accounting

**Status:** Draft
**Authors:** Codex
**Based on:** `docs/roadmap_gpt.md`, `docs/roadmap_opus_v2.md`,
`docs/phases/phase-1.6-generation-serving.md`
**Date:** 2026-06-04

---

## Goal

Make the existing MLX single-request engine measurable, and make per-request
context-window decisions explicit.

Phase 1.7 intentionally does not add a new decoding algorithm, scheduler,
backend, model family, or serving mode. Its purpose is to expose the timing,
token-count, throughput, KV-cache memory, and token-budget facts needed to
understand the engine and evaluate later work such as quantization, speculative
decoding, and continuous batching.

By the end of this phase, CLI, HTTP, and profiling flows should be able to
answer:

- how many prompt tokens were processed
- how many new tokens were generated
- why generation stopped
- whether the prompt fit the configured context budget
- how many prompt tokens were accepted, truncated, or rejected
- which context policy was applied
- how long prompt formatting and tokenization took
- how long prefill took
- how long it took to produce the first generated token
- how much time was spent in decode steps
- how many decode tokens per second were produced
- how much KV-cache memory was allocated for the configured context window
- how much KV-cache memory was actively used by the request

These numbers do not need to be perfectly stable across runs. Local MLX timing
can vary with warmup, thermal state, background system load, and lazy execution.
The required behavior is that metrics are captured consistently, internally
coherent, and useful for comparison.

---

## Scope

### In scope

- Preserve existing Llama-3.2-1B and Qwen3-0.6B behavior on MLX.
- Preserve existing stop semantics, CLI behavior, HTTP endpoints, and
  worker-owned MLX lifecycle except where context policy intentionally changes
  request admission before prefill.
- Add project-owned generation metrics types.
- Add explicit per-request context-budget policy.
- Attach metrics to completed `GenerationResponse` values.
- Measure prompt preparation, prefill, first-token latency, decode time, total
  time, and decode throughput.
- Report original, accepted, truncated, and rejected prompt-token counts.
- Report KV-cache allocated memory and active request memory from model/cache
  shapes.
- Surface metrics in CLI `--show-stats` output.
- Surface metrics in HTTP `POST /generate` and final streaming metadata.
- Add a repeatable profiling script that runs multiple prompts and reports
  min/median/max style summaries.
- Add unit tests with fake engines/timers where practical.
- Add slow smoke hooks for real local Llama and Qwen3 metrics.
- Update README, file-structure, architecture, refined-plan, and learning docs
  if public behavior or phase status changes.

### Out of scope

- CUDA/PyTorch backend support.
- Quantization.
- Speculative decoding.
- Continuous batching or multiple active model requests.
- PagedAttention or dynamic KV memory allocation.
- Prefix caching.
- Conversation/session memory.
- Cross-request context reuse.
- Automatic summarization or semantic compression.
- Production observability stacks such as Prometheus, OpenTelemetry, tracing
  systems, dashboards, or long-term metric storage.
- OS-level memory measurement.
- Model-layer instrumentation.
- Changing core attention, model, weight-loading, sampling, or KV-cache
  semantics except for reviewed bug fixes.

---

## Runtime And Tooling

Phase 1.7 keeps the Phase 1.6 runtime model:

- Python `>=3.12,<3.13`
- MLX as the only tensor backend
- `tokenizers` as the runtime tokenizer dependency
- FastAPI and uvicorn for the existing local HTTP layer
- `transformers` only as a dev/test reference dependency

No new runtime dependency is expected for observability. Use the Python standard
library for timing and statistics unless a later review explicitly accepts a
dependency.

Runtime code under `tiny_duo_infer/` must not import `transformers`.

---

## Architecture Constraints

1. **Measure at control-plane boundaries.** Metrics should be collected in the
   engine, CLI, profiling, and serving layers. Do not insert timers inside model
   layers, attention, FFN, or normalization.

2. **Keep model execution unchanged.** Metrics must observe the existing
   prefill/decode path. They must not alter tensor shapes, cache position
   semantics, sampling behavior, or stop-reason priority.

3. **Respect MLX lazy evaluation.** Timing boundaries must include the existing
   `mx.eval()` synchronization points so measured prefill and decode times
   represent completed work, not only queued operations.

4. **Keep context policy per-request.** Context-budget accounting may transform
   or reject the current request's prompt tokens before prefill. It must not
   introduce conversation history, prefix sharing, session memory, or
   cross-request cache reuse.

5. **Keep serving single-request.** The HTTP worker still supports one active
   generation request at a time. Phase 1.7 may report busy/idle state and recent
   request metrics, but it must not add a scheduler or queue.

6. **Keep metrics educational.** Prefer explicit fields with clear definitions
   over opaque generic metric names.

7. **Avoid benchmark pass/fail based on speedup.** Phase 1.7 creates the
   measurement surface. Later phases may compare performance, but this phase
   should not fail because a local machine has noisy timing.

8. **Do not silently truncate.** Any token truncation must be selected through
   an explicit request/CLI/HTTP policy and reflected in the response stats.

---

## Metrics Model

Add a project-owned metrics type, for example in `tiny_duo_infer/generation.py`
or a small sibling module. The exact name can be chosen during implementation,
but this spec uses `GenerationStats`.

Add a project-owned context policy type, for example:

```python
ContextPolicy = Literal[
    "allow_context_stop",
    "reject",
    "truncate_left",
    "truncate_right",
    "reserve_generation",
]
```

`allow_context_stop` preserves the Phase 1.6 generation behavior inside the
existing prompt boundary: prompts may use the full cache, and generation stops
with `context_length` if no room remains for another decode step. This should be
the default to avoid surprising existing callers. If the prompt itself exceeds
`max_seq_len`, `allow_context_stop` must fail before prefill. It must not
truncate.

`reject` fails before prefill when the prompt plus requested `max_new_tokens`
cannot fit in `max_seq_len`.

`truncate_left` removes tokens from the beginning of the prompt until the prompt
plus requested `max_new_tokens` can fit in `max_seq_len`.

`truncate_right` removes tokens from the end of the prompt until the prompt plus
requested `max_new_tokens` can fit in `max_seq_len`.

`reserve_generation` is like `truncate_left`, but its name emphasizes the user
intent: keep the newest prompt suffix while reserving room for the requested
generation budget. It is useful for chat-style prompts where recent turns are
usually more valuable than earlier turns.

All policy decisions happen after final prompt formatting and tokenization, and
before prefill. The model only sees the accepted token IDs.

Minimum requirements:

- never prefill an empty prompt after truncation
- never prefill more than `max_seq_len` tokens
- fail before prefill when `original_prompt_tokens > max_seq_len` and
  `context_policy == "allow_context_stop"`
- fail before prefill when `max_new_tokens > max_seq_len` for any context policy
- never silently ignore rejected or truncated tokens
- record the policy and token counts in final stats
- keep `context_length` stop reason available for `allow_context_stop` and any
  other case where generation fills the cache before another stop condition wins

The implementation may store the context policy on `GenerationRequest`, for
example:

```python
context_policy: ContextPolicy = "allow_context_stop"
```

Required fields:

```python
prompt_tokens: int
generated_tokens: int
stop_reason: str
context_policy: str
original_prompt_tokens: int
accepted_prompt_tokens: int
truncated_prompt_tokens: int
rejected_prompt_tokens: int

prompt_prepare_ms: float
prefill_ms: float
time_to_first_token_ms: float
decode_ms: float
total_ms: float
decode_tokens_per_sec: float

kv_cache_allocated_bytes: int
kv_cache_active_bytes: int
max_seq_len: int
active_seq_len: int
```

Recommended optional fields:

```python
decode_step_ms: list[float]
model_type: str
```

`decode_step_ms` is profiling detail, not default public response metadata. It
must be omitted by default from CLI/HTTP responses and may be populated only by
an explicit profiling/debug path. If a future task exposes per-step timings over
HTTP, it must add an explicit cap and a truncation indicator.

Field definitions:

- `prompt_tokens`: token count after chat formatting and tokenization.
  This must equal `accepted_prompt_tokens`. Keeping both names is intentional:
  `GenerationResponse.prompt_tokens` preserves the Phase 1.6 public contract,
  while `accepted_prompt_tokens` makes the context-budget accounting explicit.
- `original_prompt_tokens`: token count after final prompt formatting and
  tokenization, before context-budget policy is applied.
- `accepted_prompt_tokens`: token count that is actually passed to prefill.
- `truncated_prompt_tokens`: number of prompt tokens removed by policy.
- `rejected_prompt_tokens`: number of prompt tokens rejected by policy. This is
  usually `0` for successful responses; rejected requests should surface clear
  validation/admission errors and may not produce `GenerationResponse`.
- `context_policy`: policy applied to the request.
- `generated_tokens`: number of sampled tokens counted by the existing
  generation loop. This must match `GenerationResponse.generated_tokens`.
- `stop_reason`: existing stop reason. This must match
  `GenerationResponse.stop_reason`.
- `prompt_prepare_ms`: time spent building the final prompt string and
  tokenizing it.
- `prefill_ms`: time from prefill model call start through required `mx.eval()`
  and cache materialization.
- `time_to_first_token_ms`: time from request start until the first generated
  non-EOS token is selected and available to the caller. If no token is yielded,
  use the time until the terminal stop condition is known.
- `decode_ms`: total time spent in decode forward/eval/cache-advance steps
  after the first sampled token. It should not include prompt preparation or
  prefill.
- `total_ms`: wall-clock request time from the start of prompt preparation until
  final `GenerationResponse` construction.
- `decode_tokens_per_sec`: generated token throughput. Use
  `generated_tokens / decode_wall_seconds` when decode time is positive. Return
  `0.0` when no decode time is available or no tokens were generated.
- `kv_cache_allocated_bytes`: full static KV-cache allocation for the configured
  request cache.
- `kv_cache_active_bytes`: KV-cache memory represented by the active sequence
  length, computed with the same formula but using `active_seq_len`.
- `max_seq_len`: configured cache capacity for this engine/request.
- `active_seq_len`: `accepted_prompt_tokens + generated_tokens` after
  generation stops.

KV-cache memory formula:

```text
2 * n_layers * n_kv_heads * seq_len * head_dim * bytes_per_element
```

The leading factor of 2 accounts for K and V. `bytes_per_element` should match
the KV-cache dtype. In the current MLX path this is expected to be 4 bytes if
the cache is float32 and 2 bytes if the cache is bfloat16/float16. If dtype
inspection is awkward in implementation, default to the actual dtype of the
allocated cache buffers rather than hard-coding bfloat16.

---

## Response And API Requirements

### GenerationResponse

Extend the completed generation response to include stats:

```python
stats: GenerationStats | None
```

`stats` may default to `None` for narrow tests or fake engines, but the real
`Engine.generate_request()` path must populate it.

Do not remove existing response fields:

```python
text: str
prompt_tokens: int
generated_tokens: int
stop_reason: StopReason
```

Those fields remain the stable public metadata. The stats object adds timing and
memory detail.

### GenerationRequest

Extend the request type with a context policy field:

```python
context_policy: ContextPolicy = "allow_context_stop"
```

Validation requirements:

- value must be one of the supported context policies
- truncation policies must still leave at least one accepted prompt token
- `max_new_tokens` must be less than or equal to `max_seq_len`
- `allow_context_stop` must fail before prefill when `original_prompt_tokens >
  max_seq_len`
- `reject` must fail before prefill when `accepted_prompt_tokens +
  max_new_tokens > max_seq_len`

The default preserves Phase 1.6 behavior so existing callers do not suddenly
receive admission failures for prompts that previously ended with
`context_length`.

### Streaming

The final item from `Engine.generate_stream()` must include a
`GenerationResponse` with `stats` populated on the real engine path.

Streaming text fragments should not include stats. Stats are final metadata,
because total time and final active KV-cache length are only known when the
request ends.

### HTTP

`POST /generate` response JSON must include a `stats` object when the underlying
engine response has stats.

`POST /generate/stream` final NDJSON item must include the same final stats
object. Intermediate fragment items should remain lightweight and not repeat the
stats.

Existing HTTP status and busy behavior must remain unchanged unless a bug fix is
reviewed.

HTTP request JSON should accept `context_policy` with the same values as
`GenerationRequest`. Invalid values should return the existing validation-error
shape.

### CLI

Existing generated text output remains unchanged by default.

When `--show-stats` is set, print a stable, testable stats block after generated
text. It should include at least:

- prompt tokens
- generated tokens
- stop reason
- prefill latency
- time to first token
- decode latency
- total latency
- decode tokens/sec
- allocated KV-cache memory
- active KV-cache memory
- context policy
- original prompt tokens
- accepted prompt tokens
- truncated prompt tokens

Generated text must remain on stdout. Stats must be written to stderr so stdout
can still be piped as plain generated text.

Add a CLI flag:

```text
--context-policy {allow_context_stop,reject,truncate_left,truncate_right,reserve_generation}
```

The default must be `allow_context_stop`.

---

## Profiling Script Requirements

Add or replace a script under `scripts/`, for example:

```text
scripts/profile_generation.py
```

The script should load one local model once and run a fixed prompt set.

Required CLI flags:

```text
--model-path PATH
--max-seq-len N
--max-new-tokens N
--temperature FLOAT
--top-k N
--top-p FLOAT
--context-policy POLICY
--runs N
--warmup-runs N
--prompt TEXT
--prompt-file PATH
--json
```

Recommended behavior:

- Built-in default prompt set if neither `--prompt` nor `--prompt-file` is
  provided.
- `--prompt` may be repeated.
- `--prompt-file` reads one prompt per non-empty line.
- warmup runs are executed and excluded from summaries.
- summaries report min, p50, p95, and max for latency and throughput fields.
- `--json` emits machine-readable output for future comparison tests.

The existing `scripts/benchmark.py` may remain as a simple baseline script, or
it may become a thin wrapper around the new profiling code. The implementation
should avoid duplicating KV-cache memory formulas in multiple places.

---

## Timing Semantics

Use `time.perf_counter()` or equivalent monotonic timing.

Timing must be measured around completed engine work:

1. Start total timer before prompt formatting/tokenization.
2. Measure prompt preparation around prompt string construction and tokenizer
   encode.
3. Measure prefill around the model prefill call through `mx.eval()` and
   `cache.eval()`.
4. Measure time to first token from total start until the first generated token
   is sampled and available. If EOS or context length stops before a fragment is
   yielded, record time until that stop is known.
5. Measure each subsequent decode step around the one-token model forward,
   `mx.eval(logits)`, `cache.eval()`, `cache.advance(1)`, and sampling.
6. Stop total timer immediately before final response construction.

Do not try to measure raw GPU kernel time. Phase 1.7 measures user-visible
request timing at the engine boundary.

For tests, allow injecting a small fake timer or clock so unit tests can assert
field arithmetic without sleeping or relying on machine speed.

---

## Testing Requirements

Unit tests should cover:

- `GenerationStats` construction and field validation if validation is added.
- KV-cache memory byte formula for Llama and Qwen3-style configs.
- `GenerationResponse` preserves existing fields and can carry stats.
- `Engine.generate_request()` populates stats on normal completion.
- Stats are populated for stop reasons:
  - `eos`
  - `max_new_tokens`
  - `stop_string`
  - `context_length`
- `generated_tokens`, `prompt_tokens`, and `stop_reason` match between
  `GenerationResponse` and `GenerationStats`.
- `prompt_tokens == accepted_prompt_tokens`.
- `active_seq_len == accepted_prompt_tokens + generated_tokens`.
- `kv_cache_active_bytes` uses active sequence length.
- `kv_cache_allocated_bytes` uses configured `max_seq_len`.
- `allow_context_stop` preserves Phase 1.6 context-limit behavior.
- `allow_context_stop` fails before prefill when the prompt itself exceeds
  `max_seq_len`.
- `reject` fails before prefill when prompt plus requested generation budget
  cannot fit.
- `reject` fails before prefill when `original_prompt_tokens > max_seq_len`.
- `truncate_left` removes earliest prompt tokens and records the count.
- `truncate_right` removes latest prompt tokens and records the count.
- `reserve_generation` keeps the newest prompt suffix while reserving requested
  generation budget.
- `truncate_left`, `truncate_right`, and `reserve_generation` handle
  `original_prompt_tokens > max_seq_len` by producing a non-empty accepted
  prompt when `max_new_tokens` is valid and enough room exists.
- all policies fail before prefill when `max_new_tokens > max_seq_len`.
- truncation that would leave no prompt tokens fails clearly.
- No-token generation such as `max_new_tokens=0` reports coherent zero
  throughput.
- CLI `--show-stats` output includes stable field names.
- CLI forwards `--context-policy`.
- HTTP full response includes final stats.
- HTTP accepts and validates `context_policy`.
- HTTP streaming final metadata includes final stats and intermediate chunks do
  not repeat full stats.
- Profiling script argument parsing, context-policy forwarding, and summary
  formatting.

Slow smoke tests should remain optional behind `--run-slow` and should verify
mechanical behavior only:

- real Llama model can produce stats for a short prompt
- real Qwen3 model can produce stats for a short prompt
- metrics fields are non-negative and internally coherent
- smoke tests do not assert exact latency or throughput thresholds

Recommended normal verification:

```bash
uv run pytest -q
uv run python scripts/profile_generation.py --help
```

Recommended slow verification when local model artifacts exist:

```bash
LLAMA_MODEL_PATH=models/llama-3.2-1b \
QWEN_MODEL_PATH=models/qwen3-0.6b \
uv run pytest --run-slow -q
```

---

## Completion Criteria

Phase 1.7 is complete when:

- the Phase 1.7 taskboard is fully `done`
- a non-owner reviewer signs off on the close task
- normal tests pass
- CLI `--show-stats` reports timing and KV-cache memory fields
- CLI and HTTP accept explicit context policy
- HTTP full and streaming responses include final stats
- profiling script runs with fake/unit coverage and prints useful summaries
- real-model smoke metrics are recorded, or skipped with explicit reasons
- README and architecture docs describe Phase 1.7 behavior if this spec becomes
  active source of truth

---

## Known Tradeoffs

- Metrics are measured at engine boundaries, not at GPU-kernel granularity. This
  is intentional because the project is learning request mechanics, not writing
  a profiler.
- Local timing is noisy. Acceptance should check coherence and reporting shape,
  not exact speed.
- `GenerationResponse` keeps existing top-level token metadata even though those
  values are repeated in `GenerationStats`. This preserves simple callers and
  keeps stats optional for tests/fakes.
- KV-cache memory is computed from shapes and dtype, not measured from the OS.
  This makes the result deterministic and easier to reason about.
- Context-budget policy is included, but only per request. Cross-request prefix
  reuse, conversation/session memory, semantic compression, and PagedAttention
  remain deferred.
- Quantization, speculative decoding, and continuous batching are deferred to
  later phases.
