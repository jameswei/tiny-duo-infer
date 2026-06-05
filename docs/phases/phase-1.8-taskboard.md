# Phase 1.8 Taskboard

This file tracks Phase 1.8 implementation tasks, dependencies, ownership, and
review state.

The active implementation contract is
`docs/phases/phase-1.8-weight-quantization.md`.

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
| P1.8-T00 | Planning | Phase 1.8 source-of-truth docs | Phase 1.7 complete | done | codex | spec and taskboard exist; Phase 1.8 is active in phase index; review confirms scope is MLX weight-only quantization | reviewed and signed off by claudecode; three findings fixed before sign-off: T08 dependency includes T05, tiny Qwen3 quantization tests specify group_size=32 or another divisor of 32, and new GenerationStats quantization fields have explicit Python types; no code tests required for docs-only planning |
| P1.8-T01 | Representation | Quantization config and quantized weight object | P1.8-T00 | done | cc | config validates bits/group/mode; quantized weight stores packed data and metadata | new module `tiny_duo_infer/quantization.py`: `QuantizationConfig` (bits/group_size/mode validated), `QuantizedWeight` (stores qweight/scales/biases/bits/group_size/mode/out_features/in_features from mx.quantize()); `generation.py`: `_VALID_QUANTIZATION_MODES`, 7 new `GenerationStats` fields with defaults matching no-quantization path, `quantization_mode` validation + coherence invariants (mode/bits/group_size consistency, non-negative counts and bytes) in `__post_init__`; non-negative checks run before coherence checks to produce clear messages; `linear_weight_runtime_bytes <= linear_weight_full_precision_bytes` intentionally not enforced as a hard invariant (tiny matrices with overhead can exceed it); `tests/test_quantization.py`: 39 tests covering config validation, QuantizedWeight construction and metadata, GenerationStats new field defaults, all valid/invalid quantization_mode values, all coherence invariants (int4/int8 bits mismatch, none+bits/group_size/count), negative counts/bytes; reviewed and signed off by codex after fix; `uv run pytest tests/test_quantization.py tests/test_generation.py -q`: 79 passed; `uv run pytest -q`: 431 passed, 11 skipped |
| P1.8-T02 | Linear runtime | Quantized `Linear.forward()` path | P1.8-T01 | done | cc | full-precision path unchanged; quantized path uses `mx.quantized_matmul()` | `models/base.py`: `Linear` imports `QuantizedWeight`; `weight` type widened to `mx.array | QuantizedWeight | None`; `forward()` dispatches on `isinstance(self.weight, QuantizedWeight)` — raises `ValueError` when `qw.in_features != self.in_features` or `qw.out_features != self.out_features` before any matmul; quantized path calls `mx.quantized_matmul(x, qw.qweight, qw.scales, qw.biases, transpose=True, group_size, bits, mode)`; full-precision path unchanged; `load_weights()` unchanged; `tests/test_quantization.py`: 13 new tests including dimension mismatch rejection for both in/out; reviewed and signed off by codex after dimension-check fix; `uv run pytest tests/test_quantization.py tests/test_model.py -q`: 79 passed, 1 skipped; `uv run pytest -q`: 444 passed, 11 skipped |
| P1.8-T03 | Weight conversion | Quantize eligible project weights after HF conversion | P1.8-T01, P1.8-T02 | done | cc | eligible linear matrices quantized; embeddings/norms stay full precision | new module `tiny_duo_infer/weights/quantizer.py`: `_is_eligible()` (2-D check + suffix/exact match), `quantize_weights(project_weights, config)` iterates project dict, calls `mx.quantize()` on eligible keys, wraps result in `QuantizedWeight`, passes non-eligible through unchanged; eligible: all 7 `*_proj.weight` suffixes + exact `lm_head.weight`; non-eligible: `embed_tokens.weight`, all 1-D tensors (norms, Qwen3 q_norm/k_norm); divisibility guard raises `ValueError` naming key, in_features, and group_size; Llama tied lm_head handled by constructing a new dict so `embed_tokens.weight` array object is never mutated; `models/base.py`: `Module.load_weights()` param widened to `dict[str, mx.array | QuantizedWeight]`; `tests/test_quantization.py`: 13 new T03 tests covering eligible/non-eligible dispatch, all 7 suffixes + lm_head, embed_tokens exclusion, 1-D norm exclusion, Qwen3 q_norm/k_norm exclusion, Llama tied-lm_head independence, non-divisible in_features error (key name + values), INT8, bfloat16 input, metadata fields; reviewed and signed off by codex; `uv run pytest tests/test_quantization.py tests/test_weights.py tests/test_model.py -q`: 115 passed, 2 skipped; `uv run pytest -q`: 457 passed, 11 skipped |
| P1.8-T04 | Engine and CLI | Engine loading and CLI quantization flags | P1.8-T03 | done | cc | `Engine.from_model_path()` and CLI can select none/INT8/INT4 | `engine.py`: `Engine.from_model_path(..., quantization=None)` accepts optional `QuantizationConfig`, calls `quantize_weights()` after model-family conversion and before model construction/loading when enabled; full-precision default skips quantization. `cli.py`: adds `--quantization {none,int4,int8}` and `--quant-group-size N`, builds/forwards `QuantizationConfig`, rejects invalid choices/group sizes before model loading. `tests/test_cli.py`: CLI flag/default/helper coverage. `tests/test_engine.py`: engine-level routing coverage added after review finding, including no-quant skip, quantized dict loaded, and quantizer `ValueError` propagation before model construction. reviewed and signed off by codex after coverage fix; `uv run pytest tests/test_engine.py tests/test_cli.py tests/test_quantization.py -q`: 158 passed, 4 skipped; `uv run pytest -q`: 470 passed, 11 skipped |
| P1.8-T05 | HTTP serving | HTTP server startup and worker quantization forwarding | P1.8-T04 | done | cc | serving loads engine on worker thread with selected quantization config | `serving/worker.py`: `InferenceWorker.from_path(..., quantization=None)` stores optional `QuantizationConfig`, loads `Engine.from_model_path(..., quantization=...)` inside the worker thread to preserve MLX stream affinity, and now captures/re-raises worker-thread load failures so startup cannot hang. `serving/api.py`: `create_app_from_path()` forwards quantization to the worker; `python -m tiny_duo_infer.serving.api` accepts `--quantization {none,int4,int8}` and positive `--quant-group-size N`. `tests/test_serving.py`: forwarding tests for none/INT4/INT8, app factory forwarding/default, and worker load-failure propagation. reviewed and signed off by codex after fixes for worker startup hang and serving positive-int validation; `uv run pytest tests/test_serving.py -q`: 33 passed, 2 skipped; `uv run pytest -q`: 476 passed, 11 skipped |
| P1.8-T06 | Metrics and profiling | Weight memory accounting and profiling comparison | P1.8-T04, P1.8-T05 | done | cc | stats/profiling report quantization mode and linear-weight memory bytes | `QuantizedWeight` now stores `original_nbytes`; `weights/quantizer.py` adds `LinearWeightStats` and `compute_linear_weight_stats()` for eligible linear weights only, excluding embeddings/norms. `Engine.from_model_path()` computes/stores linear-weight stats and `_run_generation()` populates all seven quantization fields in `GenerationStats`. CLI `--show-stats`, HTTP `GenerationStatsBody`/streaming final metadata, and `profile_generation` JSON/human output include quantization mode and linear-weight memory bytes; profiling schema bumped to v2 and accepts quantization flags with positive group-size validation. reviewed and signed off by codex after fixes for HTTP stats field coverage and profiling group-size validation; `uv run pytest tests/test_serving.py tests/test_profiling.py tests/test_quantization.py tests/test_engine.py tests/test_cli.py -q`: 236 passed, 6 skipped; `uv run pytest -q`: 495 passed, 11 skipped |
| P1.8-T07 | Tests | Quantized correctness and regression coverage | P1.8-T02, P1.8-T03, P1.8-T04, P1.8-T05, P1.8-T06 | done | Claude Opus 4.7 | unit/integration tests cover Llama and Qwen3 quantized paths | new `tests/test_quantization_integration.py` covers tiny Llama and Qwen3 generation with full precision, INT8, and INT4; Qwen3 tiny uses `group_size=32` and rejects default `64`; stats/memory assertions cover quantization mode, bits, group size, linear counts, and runtime bytes; slow INT8 smoke tests added for local Llama/Qwen3 model artifacts. reviewed and signed off by codex after stop-string test fix: the quantized stop-string test now wraps the real quantized Llama model, forces inner logits evaluation so `mx.quantized_matmul` executes, then overrides only returned logits to deterministically assert `stop_reason == "stop_string"` and trimmed text; `uv run pytest tests/test_quantization_integration.py -q`: 14 passed, 2 skipped; `uv run pytest -q`: 509 passed, 13 skipped |
| P1.8-T08 | Docs | README, architecture, file-structure, strategy, and learning docs updates | P1.8-T04, P1.8-T05, P1.8-T06, P1.8-T07 | todo | unassigned | public usage and learning notes describe weight-only quantization |  |
| P1.8-T09 | Close | Real-model smoke and handoff | P1.8-T08 | todo | unassigned | pytest passes; real-model quantized smoke results are recorded or skipped with reasons |  |

## Review-Sensitive Tasks

These tasks require architecture/code review before being marked `done`:

- `P1.8-T00`: phase scope and roadmap positioning.
- `P1.8-T01`: public quantization config and quantized weight representation.
- `P1.8-T02`: `Linear.forward()` runtime path and MLX primitive usage.
- `P1.8-T03`: weight eligibility rules, tied Llama lm_head handling, and Qwen3
  required lm_head behavior.
- `P1.8-T04`: engine and CLI public interface.
- `P1.8-T05`: HTTP worker lifecycle and MLX stream affinity.
- `P1.8-T06`: memory accounting formulas and profiling interpretation.
- `P1.8-T09`: real-model verification and phase completion.

## Minimum Phase 1.8 Completion

Minimum Phase 1.8 completion requires `P1.8-T00` through `P1.8-T09`.

The phase is not complete until a non-owner reviewing agent signs off on the
handoff and verification results.

## Next Phase

Phase 1.9 is expected to focus on speculative decoding, but it should not be
drafted as an implementation contract until Phase 1.8 quantization behavior,
metrics, and real-model smoke results are reviewed.
