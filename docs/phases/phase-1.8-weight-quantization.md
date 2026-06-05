# Phase 1.8 Spec: MLX Weight-Only Quantization

**Status:** Draft
**Authors:** Codex
**Based on:** `.tmp/roadmap_opus_v2.md`, `.tmp/roadmap_ds_v2.md`,
`.tmp/roadmap_sonnet.md`, `.tmp/roadmap_gpt.md`,
`docs/phases/phase-1.7-observability.md`
**Date:** 2026-06-05

---

## Goal

Add MLX-native weight-only quantization for the existing Llama and Qwen3 MLX
runtime.

The phase should teach how an inference engine stores compressed linear weights,
how quantized matmul fits into the forward pass, and how memory and decode
performance change compared with the full-precision baseline.

By the end of this phase, a user should be able to load a supported local model
with INT4 or INT8 weight-only quantization enabled, generate text through the
same engine/CLI/HTTP paths, and compare memory and timing against the
full-precision path using Phase 1.7 stats and profiling tools.

---

## Scope

### In scope

- Preserve existing full-precision Llama-3.2-1B and Qwen3-0.6B behavior.
- Add weight-only quantization for `Linear` weights.
- Support INT4 and INT8 affine quantization through MLX primitives.
- Use `mx.quantized_matmul()` as the required normal runtime path for
  quantized `Linear.forward()`.
- Use `mx.quantize()` to convert full-precision matrix weights into quantized
  weight data at model load time.
- Allow `mx.dequantize()` only as an explicit test/debug/fallback path; it must
  not be the normal inference path.
- Add a project-owned quantized-weight representation that stores packed
  weights, scales, biases, bits, group size, mode, and original matrix shape.
- Add a project-owned quantization config surfaced through engine loading, CLI,
  and HTTP server startup.
- Report quantization mode and linear-weight memory fields through
  `GenerationStats`, CLI `--show-stats`, HTTP responses, streaming final
  metadata, and profiling output.
- Measure and report estimated model weight memory for full-precision and
  quantized paths.
- Add unit tests for quantized linear behavior, config validation, conversion,
  routing through model loading, and reporting.
- Add slow real-model smoke tests for Llama and Qwen3 when local model artifacts
  are available.

### Out of scope

- Activation quantization.
- KV-cache quantization.
- GPTQ, AWQ, SmoothQuant, quantization-aware training, or calibration pipelines.
- Offline quantized checkpoint writing or a new persistent artifact format.
- Eager dequantization at load time.
- Quantizing embeddings, RMSNorm weights, RoPE tables, tokenizer data, or KV
  cache buffers.
- Speculative decoding.
- Continuous batching or multiple active model requests.
- CUDA/PyTorch backend support.
- Production-quality benchmark pass/fail criteria based on speedup.

---

## Runtime And Tooling

Phase 1.8 keeps the Phase 1.7 runtime model:

- Python `>=3.12,<3.13`
- MLX as the only tensor backend
- `tokenizers` as the runtime tokenizer dependency
- FastAPI and uvicorn for the existing local HTTP layer
- `transformers` only as a dev/test reference dependency

No new runtime dependency is expected.

The implementation may use these MLX primitives:

- `mx.quantize()`
- `mx.quantized_matmul()`
- `mx.dequantize()` for tests, debugging, and explicit fallback comparison only

Runtime code under `tiny_duo_infer/` must not import `transformers`.

---

## Architecture Constraints

1. **Keep quantization weight-only.** Only matrix weights used by `Linear`
   should be quantized. Activations remain MLX arrays in the existing dtype.

2. **Keep the engine project-owned.** Do not replace project modules with
   `mlx.nn.QuantizedLinear` or `mlx.nn.quantize()`. The project should own the
   `Linear` abstraction and call MLX primitive ops explicitly.

3. **Use fused quantized matmul normally.** Quantized `Linear.forward()` must
   call `mx.quantized_matmul(x, qweight, scales, biases, transpose=True, ...)`.
   This preserves the memory and bandwidth learning target.

4. **Reject eager dequantization.** Loading quantized weights and immediately
   converting them back into full-precision matrices is not an acceptable
   implementation. It hides the memory benefit and defeats the phase purpose.

5. **Avoid a persistent artifact format in Phase 1.8.** The required path is
   in-memory quantization during model load. An offline writer may be proposed in
   a later phase after the runtime representation is stable.

6. **Preserve model-family behavior.** Quantization should work through the
   shared `Linear` abstraction for both Llama and Qwen3. Model-family-specific
   converter shape rules should remain unchanged except where they allow
   quantized values after validation.

7. **Keep quantization explicit and inspectable.** Quantized weights should be
   represented by a small project-owned data object, not an unstructured tuple
   whose meaning must be inferred at each call site.

8. **Do not make throughput a hard pass/fail gate.** Speedup depends on local
   MLX kernels, bit width, group size, prompt shape, and hardware state. The
   required pass/fail benefit is reduced model weight memory. Throughput must be
   measured and reported.

---

## Public Interface Requirements

Add a project-owned quantization config, for example:

```python
QuantizationConfig(
    bits: Literal[4, 8],
    group_size: int = 64,
    mode: Literal["affine"] = "affine",
)
```

Exact module placement is implementation-defined, but it should be importable
from a stable project path such as `tiny_duo_infer.quantization`.

Validation requirements:

- `bits` must be `4` or `8`.
- `group_size` must be positive.
- For every quantized matrix, the input dimension must be divisible by
  `group_size`; otherwise fail clearly before generation.
- Phase 1.8 supports `mode="affine"` only.

`Engine.from_model_path()` should accept an optional quantization config:

```python
Engine.from_model_path(
    model_path,
    max_seq_len=2048,
    quantization=None,
)
```

Full-precision loading remains the default when `quantization is None`.

CLI requirements:

- Add `--quantization {none,int4,int8}`.
- Default must be no quantization.
- Add `--quant-group-size N`, default `64`.
- Invalid bit width, group size, unsupported mode, or incompatible matrix shape
  must fail before generation with a clear error.

HTTP server startup requirements:

- The serving entrypoint and worker path must accept `--quantization
  {none,int4,int8}` and `--quant-group-size N`.
- The engine must still be initialized on the worker thread to preserve MLX GPU
  stream affinity.

Profiling requirements:

- `scripts/profile_generation.py` should accept the quantization option so the
  same prompt set can compare full-precision, INT8, and INT4 runs.
- Human and JSON profiling output should identify the quantization mode used.

---

## Quantized Linear Requirements

`Linear` should support both full-precision and quantized weights.

Full-precision path:

```python
y = x @ weight.T
```

Quantized path:

```python
y = mx.quantized_matmul(
    x,
    qweight,
    scales,
    biases,
    transpose=True,
    group_size=group_size,
    bits=bits,
    mode="affine",
)
```

Required behavior:

- `Linear.forward()` must produce output shaped `(..., out_features)` in both
  full-precision and quantized modes.
- Quantization must preserve `Linear.in_features` and `Linear.out_features`
  checks.
- The quantized representation must record the original `(out_features,
  in_features)` shape.
- The full-precision path must not pay quantization overhead.
- Tests should compare quantized output against dequantized or full-precision
  reference output with realistic tolerances, not exact equality.

---

## Weight Conversion And Loading Requirements

The current loader reads Hugging Face safetensors into full-precision MLX arrays,
and converters validate model-family key names and shapes.

Phase 1.8 should keep that validation flow:

1. load safetensors
2. convert and validate HF keys/shapes for Llama or Qwen3
3. quantize eligible project weights if quantization is enabled
4. load values into the model tree

Eligible weights:

- `*.q_proj.weight`
- `*.k_proj.weight`
- `*.v_proj.weight`
- `*.o_proj.weight`
- `*.gate_proj.weight`
- `*.up_proj.weight`
- `*.down_proj.weight`
- `lm_head.weight`

Non-eligible weights:

- `embed_tokens.weight`
- RMSNorm weights
- Qwen3 `q_norm.weight` and `k_norm.weight`
- any one-dimensional tensor
- any non-matrix tensor

Llama tied embeddings:

- If `lm_head.weight` is tied to `embed_tokens.weight`, quantizing
  `lm_head.weight` must not mutate or replace `embed_tokens.weight`.
- It is acceptable for the tied relationship to become a logical relationship
  rather than the same object identity after quantization, because embeddings
  remain full precision while `lm_head` is a quantized linear projection.

Qwen3:

- `lm_head.weight` remains a required checkpoint tensor.
- Qwen3 Q/K norm weights are never quantized.

---

## Metrics And Reporting Requirements

Extend `GenerationStats` with model-weight quantization metadata alongside the
existing Phase 1.7 KV-cache memory stats.

Required new fields:

| Field | Type | Values / meaning |
|---|---|---|
| `quantization_mode` | `str` | `"none"`, `"int8"`, or `"int4"` |
| `quantization_bits` | `int | None` | `None`, `8`, or `4` |
| `quantization_group_size` | `int | None` | `None` or the configured group size |
| `quantized_linear_count` | `int` | number of `Linear` modules using quantized weights |
| `full_precision_linear_count` | `int` | number of `Linear` modules using full-precision weights |
| `linear_weight_full_precision_bytes` | `int` | estimated bytes if counted linear weights stayed full precision |
| `linear_weight_runtime_bytes` | `int` | estimated runtime bytes for counted linear weights after quantization choice |

Memory accounting should be computed from tensor shapes and dtypes, not OS
process memory.

Counting rules:

- Full-precision matrix bytes use `num_elements * dtype.size`.
- Quantized bytes include packed weight storage plus scales and biases.
- Embeddings and norm weights may be reported separately or excluded from the
  linear-weight comparison, but the choice must be documented in tests and
  profiling output.

When quantization is disabled, the new fields should still be populated:
`quantization_mode="none"`, `quantization_bits=None`,
`quantization_group_size=None`, `quantized_linear_count=0`, and
`linear_weight_runtime_bytes == linear_weight_full_precision_bytes` for the
linear weights included in the accounting.

CLI, HTTP, streaming final metadata, and profiling tests must be updated
together so public surfaces stay consistent.

---

## Testing Requirements

Unit tests:

- Quantization config validation.
- Quantized-weight object construction and shape metadata.
- `Linear.forward()` full-precision path remains unchanged.
- `Linear.forward()` quantized path calls/uses quantized matmul and returns the
  expected shape.
- Quantized linear output is numerically close to a dequantized/full-precision
  reference within a documented tolerance.
- Invalid `group_size` and non-divisible matrix shapes fail clearly.
- Only eligible matrix weights are quantized.
- Embeddings, norm weights, Qwen3 Q/K norm weights, and non-matrix tensors stay
  full precision.
- Llama tied lm_head behavior does not quantize embeddings by accident.
- Qwen3 required lm_head behavior is preserved.
- CLI quantization flags parse and forward to engine loading.
- HTTP serving startup forwards quantization config through worker initialization.
- Profiling forwards quantization config and reports mode.

Integration tests:

- Tiny Llama model can run generation with no quantization, INT8, and INT4.
- Tiny Qwen3 model can run generation with no quantization, INT8, and INT4.
- Tiny Qwen3 quantization tests must use `group_size=32` or another divisor of
  the tiny fixture's 32-wide input dimensions; they must not rely on the default
  `group_size=64`.
- Full-precision outputs remain unchanged when quantization is disabled.
- Quantized generation completes without crashing and respects existing stop
  semantics.

Slow real-model smoke tests:

- Llama and Qwen3 load with INT8 and generate at least one token when local
  model links are available.
- INT4 smoke may be included if local MLX supports all required shapes.
- Smoke tests must report memory and timing; they must not require a specific
  semantic phrase.

Profiling:

- Compare full-precision, INT8, and INT4 on the same prompt set.
- Memory reduction is pass/fail according to expected accounting.
- Throughput and latency deltas are reported, not pass/fail.

---

## Completion Criteria

Phase 1.8 is complete when:

- the Phase 1.8 taskboard is fully `done`
- full-precision model loading and generation remain backward compatible
- INT8 weight-only quantized generation works for supported tiny Llama and
  Qwen3 tests
- INT4 weight-only quantized generation works where MLX shape constraints allow,
  or limitations are documented with clear tests
- quantized `Linear.forward()` uses `mx.quantized_matmul()` in the normal path
- no eager dequantization-at-load path exists
- model-weight memory accounting shows the expected reduction for quantized
  linear weights
- CLI and HTTP server startup can select quantization mode
- profiling can compare full-precision and quantized runs
- real-model smoke results are recorded or skipped with reasons
- README, architecture, file-structure, strategy, and learning docs are updated
  if public behavior or roadmap status changes

---

## Known Risks

- MLX quantization shape constraints may reject some small synthetic fixture
  dimensions unless the test fixture uses a compatible group size.
- Quantized matmul speed may be flat or slower on some local setups even when
  memory accounting improves.
- Runtime in-memory quantization reduces steady-state model weight memory but
  may still require full-precision weights briefly during loading.
- Quantizing `lm_head.weight` while leaving tied embeddings full precision needs
  careful handling for Llama.
- Tolerances for quantized output sanity tests should be loose enough to avoid
  brittle semantic expectations but strict enough to catch broken routing.

---

## Suggested Reading Order

1. `docs/phases/phase-1.7-observability.md`
2. `tiny_duo_infer/models/base.py`
3. `tiny_duo_infer/weights/loader.py`
4. `tiny_duo_infer/weights/llama_converter.py`
5. `tiny_duo_infer/weights/qwen3_converter.py`
6. `tiny_duo_infer/engine.py`
7. `tiny_duo_infer/cli.py`
8. `tiny_duo_infer/serving/worker.py`
9. `tiny_duo_infer/serving/api.py`
10. `tiny_duo_infer/profiling.py`
