# Phase 1.8 Handoff

**Status:** Ready for review
**Date:** 2026-06-05
**Owner:** codex
**Task:** P1.8-T09

This document closes Phase 1.8 from the implementation-owner side. A separate
reviewing agent must independently verify the commands below before marking
P1.8-T09 `done` and closing the phase.

## Summary

Phase 1.8 adds MLX-native, in-memory, weight-only quantization for the existing
Llama and Qwen3 MLX runtime.

Completed scope:

- `QuantizationConfig` and `QuantizedWeight` project-owned representations
- Eligible-weight conversion after Hugging Face key conversion
- Quantized `Linear.forward()` through `mx.quantized_matmul()`
- Engine, CLI, HTTP worker, and profiling quantization forwarding
- Linear-weight memory accounting in `GenerationStats`
- CLI, HTTP, streaming final metadata, and profiling output updates
- Tiny Llama and Qwen3 integration coverage for full precision, INT8, and INT4
- Documentation and learning-material updates for Phase 1.8

Out of scope remains unchanged: activation quantization, KV-cache quantization,
GPTQ/AWQ/SmoothQuant, calibration, quantization-aware training, offline
quantized artifact writing, eager dequantization at load time, speculative
decoding, batching, and CUDA/PyTorch backend work.

## Environment

- OS: macOS 26.5.1, build 25F80
- Platform: macOS-26.5.1-arm64-arm-64bit
- Python: 3.12.13 via `uv`
- MLX: 0.31.2
- MLX default device: `Device(gpu, 0)`
- Models:
  - `./models/llama-3.2-1b` →
    `/Users/wjia/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08`
  - `./models/qwen3-0.6b` →
    `/Users/wjia/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`

The model links contain `config.json`, `tokenizer.json`, and `model.safetensors`
and are usable by the project loader.

## Verification

### Unit and integration tests

```bash
uv run pytest -q
```

Result: `509 passed, 13 skipped`.

The skipped tests are slow real-model tests gated by `--run-slow` in the normal
test run.

### Real-model slow smoke tests

```bash
uv run pytest tests/test_quantization_integration.py --run-slow -q
```

Result: `16 passed`.

This includes the real-model INT8 smoke tests for Llama-3.2-1B and Qwen3-0.6B.

### Import check

```bash
uv run python -c "import tiny_duo_infer; print('import ok')"
```

Result: `import ok`.

### Llama INT8 CLI smoke

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/llama-3.2-1b \
  --prompt "The capital of France is" \
  --max-new-tokens 1 \
  --temperature 0.0 \
  --quantization int8 \
  --show-stats
```

Result:

- generated text fragment: ` Paris`
- `quantization_mode=int8`
- `quantization_bits=8`
- `quantization_group_size=64`
- `quantized_linear_count=113`
- `full_precision_linear_count=0`
- `linear_weight_full_precision_bytes=2471493632`
- `linear_weight_runtime_bytes=1312980992`

The linear-weight runtime bytes are lower than the full-precision bytes.

### Llama INT4 CLI smoke

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/llama-3.2-1b \
  --prompt "The capital of France is" \
  --max-new-tokens 1 \
  --temperature 0.0 \
  --quantization int4 \
  --show-stats
```

Result:

- generated text fragment: ` the`
- `quantization_mode=int4`
- `quantization_bits=4`
- `quantization_group_size=64`
- `quantized_linear_count=113`
- `full_precision_linear_count=0`
- `linear_weight_full_precision_bytes=2471493632`
- `linear_weight_runtime_bytes=695107584`

The linear-weight runtime bytes are lower than the full-precision bytes.

### Qwen3 INT8 CLI smoke

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --temperature 0.0 \
  --quantization int8 \
  --show-stats
```

Result:

- generated text fragment: ` Instructions`
- `quantization_mode=int8`
- `quantization_bits=8`
- `quantization_group_size=64`
- `quantized_linear_count=197`
- `full_precision_linear_count=0`
- `linear_weight_full_precision_bytes=1191968768`
- `linear_weight_runtime_bytes=633233408`

The linear-weight runtime bytes are lower than the full-precision bytes.

### Qwen3 INT4 CLI smoke

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --prompt "Hello" \
  --max-new-tokens 1 \
  --temperature 0.0 \
  --quantization int4 \
  --show-stats
```

Result:

- generated text fragment: empty string for the first decoded token
  (`generated_tokens=1`; some BPE token IDs decode to no visible text at a
  leading position, so this still counts as a successful generation smoke)
- `quantization_mode=int4`
- `quantization_bits=4`
- `quantization_group_size=64`
- `quantized_linear_count=197`
- `full_precision_linear_count=0`
- `linear_weight_full_precision_bytes=1191968768`
- `linear_weight_runtime_bytes=335241216`

The linear-weight runtime bytes are lower than the full-precision bytes.

## Notes

- Semantic quality is not a smoke-test gate for Phase 1.8. The smoke
  requirement is that supported real models load, quantized generation completes,
  and memory accounting reports the expected linear-weight reduction.
- INT4 real-model smoke passed for both local model artifacts on this machine.
- The task owner must not mark P1.8-T09 `done`; another agent should review this
  handoff, rerun the verification they consider necessary, then update the
  taskboard and phase index.

## Recommended Reviewer Close Steps

After sign-off:

1. Mark `P1.8-T09` `done` in `docs/phases/phase-1.8-taskboard.md`.
2. Move Phase 1.8 from active to completed in `docs/phases/README.md`.
3. Update roadmap status references such as `README.md` and
   `docs/refined-plan.md` from `Active` to `Done`.
4. Delete local `CURRENT.md`, because no active task remains after the phase
   close.
