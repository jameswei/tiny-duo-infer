# Phase 1.5 Spec: Qwen3-0.6B Support on MLX

**Status:** Draft  
**Authors:** Codex  
**Based on:** `docs/refined-plan.md`, `docs/phases/phase-1-mlx-single-user.md`  
**Date:** 2026-05-27

---

## Goal

Extend the Phase 1 MLX inference engine from one hard-coded Llama-family target
to two local HuggingFace-compatible dense decoder models:

- `meta-llama/Llama-3.2-1B`
- `Qwen/Qwen3-0.6B`

The purpose of Phase 1.5 is model portability, not backend portability. By the
end of this phase, the engine should load Qwen3-0.6B weights, tokenize a prompt,
run prefill, decode through the existing KV cache, sample tokens, and generate
text through the same Python API and CLI used for Llama.

Phase 1.5 should preserve Llama-3.2-1B behavior and tests. Qwen3 support must
not regress the completed Phase 1 milestone.

---

## Why Qwen3-0.6B

Qwen3-0.6B is a good second model target because it is close enough to Llama to
reuse most of the engine while different enough to expose real model-portability
issues:

- It is a decoder-only causal LM with RMSNorm, RoPE, GQA, SwiGLU-style MLP, and
  a KV cache.
- It uses an explicit `head_dim` that is not derived from
  `hidden_size // num_attention_heads`.
- It uses Q/K normalization inside attention.
- It uses a different tokenizer vocabulary and chat-template behavior.
- It is Apache-2.0 licensed and does not require the same license-gated download
  flow as Llama.

This phase teaches how an inference engine separates shared transformer
mechanics from model-family-specific details.

---

## Scope

### In scope

- Loading `Qwen/Qwen3-0.6B` from a local HuggingFace-compatible model directory
- Preserving existing `meta-llama/Llama-3.2-1B` support
- Config parsing for `model_type == "qwen3"`
- Explicit `head_dim` support when present in `config.json`
- Qwen3 attention projection shapes:
  - `q_proj`: `n_heads * head_dim`
  - `k_proj`: `n_kv_heads * head_dim`
  - `v_proj`: `n_kv_heads * head_dim`
  - `o_proj`: `n_heads * head_dim -> d_model`
- Optional Q/K RMSNorm in attention
- Qwen3 HuggingFace weight-key conversion and shape validation
- Qwen3 tokenizer loading through the existing tokenizer wrapper, if compatible
- CLI and Python API smoke tests with a local Qwen3-0.6B checkpoint
- Unit tests using tiny synthetic Qwen3-style configs
- Documentation updates for supported models and model-family differences

### Out of scope

- PyTorch/CUDA backend support
- Backend protocol enforcement
- HTTP serving
- Multi-user scheduling, batching, or PagedAttention
- Quantization
- Speculative decoding
- MoE models
- Vision, audio, embedding, reranker, or encoder-only Qwen variants
- High-level `transformers` model or generation APIs at runtime
- Wrapping `mlx-lm`
- Full chat UX or conversation memory
- Full `tokenizer.apply_chat_template()` parity with Transformers

Chat-template support is intentionally limited in this phase. Qwen3-0.6B can be
smoke-tested with plain prompt-to-completion generation. If chat-template
support becomes necessary for meaningful output, it should be added as a small
explicit follow-up and documented as prompt formatting, not as model execution
logic.

---

## Runtime And Tooling

Phase 1.5 keeps the Phase 1 runtime model:

- Python `>=3.12,<3.13`
- MLX as the only tensor backend
- `tokenizers` as the runtime tokenizer dependency
- `safetensors` / `mx.load()` for weight files
- `transformers` only as a dev/test reference dependency

Runtime code under `tiny_duo_infer/` must not import `transformers`.

Install:

```bash
uv sync --group dev
```

---

## Model Artifacts

The engine loads Qwen3 from a local HuggingFace-compatible model directory.

Expected files:

```text
models/qwen3-0.6b/
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json        (optional but common)
  generation_config.json         (optional; may document recommended sampling)
  model.safetensors              (single shard, or)
  model-00001-of-0000N.safetensors
  model.safetensors.index.json   (present when sharded)
```

Download example:

```bash
uv run huggingface-cli download Qwen/Qwen3-0.6B \
    --local-dir ./models/qwen3-0.6b
```

The model path must remain configurable through the Python API and CLI. No model
path may be hard-coded in engine code.

---

## Architecture Constraints

1. **Keep MLX direct in Phase 1.5.** This phase adds a second model family on
   the existing MLX path. It must not introduce the Phase 2 backend abstraction
   work.

2. **Do not hide model logic behind high-level libraries.** Do not use
   `transformers.AutoModelForCausalLM`, `transformers` generation APIs,
   `mlx-lm`, or high-level MLX neural network layers.

3. **Make model-family differences explicit.** Qwen3-specific behavior should
   be visible in config parsing, attention construction, and weight conversion.
   Avoid silent assumptions such as `head_dim = d_model // n_heads`.

4. **Preserve Llama behavior.** Existing Llama tests and real-model smoke tests
   must continue to pass.

5. **Batch size remains 1.** All tensors keep the Phase 1 batch dimension, but
   only `B=1` is supported.

6. **Keep code educational.** Public classes/functions need useful docstrings,
   and non-obvious Qwen3 differences need comments explaining why the code path
   differs from Llama.

---

## Model Configs

### Llama-3.2-1B reference

```python
model_type        = "llama"
d_model           = 2048
n_layers          = 16
n_heads           = 32
n_kv_heads        = 8
head_dim          = 64        # derived: d_model // n_heads
intermediate_size = 8192
vocab_size        = 128256
max_seq_len       = 131072
rope_theta        = 500000.0
rms_norm_eps      = 1e-5
qk_norm           = False      # derived from model_type
```

### Qwen3-0.6B reference

```python
model_type        = "qwen3"
d_model           = 1024
n_layers          = 28
n_heads           = 16
n_kv_heads        = 8
head_dim          = 128       # explicit config field, not d_model // n_heads
intermediate_size = 3072
vocab_size        = 151936
max_seq_len       = 40960
rope_theta        = 1000000.0
rms_norm_eps      = 1e-6
qk_norm           = True       # derived from model_type
```

The exact values should be read from the downloaded model's `config.json` and
validated in tests. The reference values above document the expected shape
relationships for Qwen3-0.6B.

---

## Config Requirements

`ModelConfig` should become capable of representing both model families.

Required fields:

```python
model_type: str
d_model: int
n_layers: int
n_heads: int
n_kv_heads: int
head_dim: int
intermediate_size: int
vocab_size: int
max_seq_len: int
rope_theta: float
rms_norm_eps: float
```

Derived properties:

```python
qk_norm: bool  # True for model_type == "qwen3"; False for "llama"
```

Parsing rules:

- Accept `model_type == "llama"` and `model_type == "qwen3"`.
- For Llama configs without `head_dim`, derive `head_dim` as
  `hidden_size // num_attention_heads`.
- For Qwen3 configs with `head_dim`, use the explicit value.
- `head_dim` becomes a stored `ModelConfig` field. `n_groups` remains a derived
  property computed as `n_heads // n_kv_heads`.
- Validate `n_heads % n_kv_heads == 0`.
- Validate `n_heads * head_dim` is positive.
- Do not require `d_model == n_heads * head_dim`; Qwen3-0.6B intentionally
  violates that assumption.
- Validate all projection shapes through the converter, not through an
  over-specific config invariant.
- Do not read `qk_norm` as a required HF config field. Derive it from
  `model_type`: Qwen3 uses Q/K norm, Llama does not. The converter must still
  validate this by requiring Qwen3 `q_norm.weight` and `k_norm.weight` keys.
  If a later phase adds a third model family with Q/K norm, promote this into
  an explicit model-family capability rather than growing ad hoc checks.

This is a breaking dataclass change for tests and fixtures that construct
`ModelConfig` directly. As part of the config task, update the existing
`TINY_CONFIG` fixture and any direct `ModelConfig(...)` construction to include
`model_type` and stored `head_dim`. Do this deliberately in P1.5-T01 rather
than discovering it through cascading test failures.

---

## Tensor Shape Conventions

Phase 1 shape symbols remain active, with one important correction:

| Symbol | Meaning |
|---|---|
| `B` | batch size, always 1 |
| `S` | sequence length for the current step |
| `T` | total valid KV-cache length |
| `D` | model hidden dimension |
| `H` | number of query attention heads |
| `Hkv` | number of key/value heads |
| `Dh` | attention head dimension from config |
| `A` | attention projection width, `H * Dh` |
| `V` | vocabulary size |
| `L` | number of transformer layers |
| `I` | FFN intermediate size |

For Llama-3.2-1B, `A == D`.

For Qwen3-0.6B, `A != D`:

```text
D = 1024
H = 16
Dh = 128
A = H * Dh = 2048
```

Qwen3 attention shape trace:

```text
input hidden states       (B, S, D)
Q after q_proj            (B, S, A)
Q reshaped                (B, S, H, Dh)
K after k_proj            (B, S, Hkv * Dh)
K reshaped                (B, S, Hkv, Dh)
V after v_proj            (B, S, Hkv * Dh)
V reshaped                (B, S, Hkv, Dh)
Q after optional q_norm   (B, S, H, Dh)
K after optional k_norm   (B, S, Hkv, Dh)
Q/K after RoPE            same shape as before RoPE
K/V cache layout          (B, Hkv, T, Dh)
GQA-expanded K/V          (B, H, T, Dh)
attention output heads    (B, H, S, Dh)
merged attention output   (B, S, A)
o_proj output             (B, S, D)
```

The current Llama implementation already has most of this shape structure, but
it must stop assuming that `A == D`.

---

## Attention Requirements

Attention should support both:

- Llama-style attention without Q/K norm
- Qwen3-style attention with Q/K RMSNorm

Prefer an explicit `Qwen3Attention` class over folding Qwen3 behavior into
`LlamaAttention` with a runtime `if qk_norm` branch. The shared math may be
factored into small helpers if useful, but the teaching surface should make
Qwen3's extra Q/K norm step visible and keep the completed Llama path easy to
compare against.

Required attention construction:

```text
q_proj: D -> H * Dh
k_proj: D -> Hkv * Dh
v_proj: D -> Hkv * Dh
o_proj: H * Dh -> D
```

Required Qwen3 flow:

```text
x
  -> q_proj/k_proj/v_proj
  -> reshape into heads
  -> q_norm on Q, k_norm on K
  -> RoPE on Q and K
  -> KV cache update/read
  -> GQA head expansion
  -> scaled dot-product attention
  -> merge heads
  -> o_proj
```

Q/K norm should use the existing `RMSNorm` implementation with dimension
`head_dim`. It should be applied independently to each head's `Dh` vector.
The placement is part of the architecture contract: Q/K norm is applied after
projection and head reshape, before RoPE rotation. Applying Q/K norm after RoPE
changes the attention scores and is incorrect for Qwen3.

Scaling remains:

```python
scores = q @ k.transpose(...) / sqrt(head_dim)
```

---

## Model Assembly

Prefer an explicit `Qwen3Block` and `Qwen3Model` alongside the existing
`LlamaBlock` and `LlamaModel`.

`LlamaBlock` should continue to instantiate `LlamaAttention`. `Qwen3Block`
should instantiate `Qwen3Attention`. Both model classes should expose the same
forward signature:

```python
model(input_ids, cache, position_offset) -> logits
```

`Engine.from_model_path()` should dispatch on `config.model_type` after loading
`config.json`:

- `model_type == "llama"`: construct `LlamaModel` and use the Llama converter.
- `model_type == "qwen3"`: construct `Qwen3Model` and use the Qwen3 converter.

This keeps the completed Llama path easy to read and gives learners a direct
side-by-side comparison of the one architectural difference: Qwen3 attention
adds Q/K norm before RoPE.

---

## Weight Conversion

Add a separate `weights/qwen3_converter.py` and keep model-family dispatch
explicit:

```python
convert_weights(hf_weights, config)
  if config.model_type == "llama": use Llama mapping
  if config.model_type == "qwen3": use Qwen3 mapping
```

This keeps model-specific key layouts easy to read and avoids hiding important
architecture differences.

Qwen3 expected HF keys include:

```text
model.embed_tokens.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.k_proj.weight
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.self_attn.q_norm.weight
model.layers.{i}.self_attn.k_norm.weight
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight
model.norm.weight
lm_head.weight
```

Qwen3 expected shapes:

```text
embed_tokens.weight                    (V, D)
q_proj.weight                          (H * Dh, D)
k_proj.weight                          (Hkv * Dh, D)
v_proj.weight                          (Hkv * Dh, D)
o_proj.weight                          (D, H * Dh)
q_norm.weight                          (Dh,)
k_norm.weight                          (Dh,)
input_layernorm.weight                 (D,)
post_attention_layernorm.weight        (D,)
gate_proj.weight                       (I, D)
up_proj.weight                         (I, D)
down_proj.weight                       (D, I)
model.norm.weight                      (D,)
lm_head.weight                         (V, D)
```

`q_norm.weight` has shape `(Dh,)` and is applied identically to every query
head. `k_norm.weight` has shape `(Dh,)` and is applied identically to every key
head. There are not separate `(H, Dh)` or `(Hkv, Dh)` norm weights; this matches
the HuggingFace checkpoint layout.

Although the Qwen3-0.6B config advertises `tie_word_embeddings: true`, the
official HuggingFace checkpoint includes `lm_head.weight`. Phase 1.5 should
treat `lm_head.weight` as required for Qwen3-0.6B and raise `ValueError` if it
is absent. Do not silently synthesize it from `embed_tokens.weight`; that would
hide incomplete or malformed Qwen3 artifacts.

Unexpected keys should produce warnings. Missing required keys should produce a
clear `ValueError` with a preview of missing project keys.

---

## Tokenizer And Prompting

The existing tokenizer wrapper should be used first. It should load
`tokenizer.json` and resolve BOS/EOS IDs from `tokenizer_config.json` or special
token strings.

Phase 1.5 does not require full chat-template support. However, docs and smoke
tests must be explicit about the prompt mode:

- Plain prompt-to-completion generation is supported by the engine.
- Qwen3 chat/instruct behavior may require prompt formatting for best output.
- Thinking/non-thinking behavior is prompt/template controlled and is not part
  of the tensor forward pass.

If the tokenizer wrapper cannot resolve Qwen3 BOS/EOS IDs, add the smallest
general metadata parser needed and cover it with tests.

---

## Engine And CLI Requirements

The public usage should remain stable:

```python
from tiny_duo_infer.engine import Engine

engine = Engine.from_model_path("./models/qwen3-0.6b")
text = "".join(
    engine.generate(
        "The capital of France is",
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.8,
    )
)
print(text)
```

CLI:

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --temperature 0.7 \
  --top-p 0.8
```

The engine should infer model family from `config.json`; users should not need
to pass `--model-type`.

---

## Testing Strategy

### Unit tests

Add synthetic tiny Qwen3-style fixtures. The important feature is not exact
Qwen3 size; it is the shape relationship `H * Dh != D`.

Example tiny config:

```python
TINY_QWEN3_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,              # H * Dh = 64, not D = 32
    "intermediate_size": 64,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
}
```

In this tiny fixture, `q_proj` and `o_proj` exercise the `H * Dh != D` path.
`k_proj` and `v_proj` coincidentally have width `Hkv * Dh = 32`, equal to `D`.
That is acceptable for unit tests because the main Qwen3-only shape risk is the
query/merged attention width.

Required tests:

- Config accepts `model_type == "qwen3"`.
- Config uses explicit `head_dim`.
- Config rejects unsupported `model_type`.
- Llama config still derives `head_dim`.
- Qwen3 converter maps all expected keys.
- Qwen3 converter validates projection and q/k norm shapes.
- Attention handles `H * Dh != D`.
- Q/K norm is applied when enabled.
- Q/K norm is absent for Llama.
- Tiny Qwen3-style model forward returns logits `(B, S, V)`.
- Prefill/decode work with a tiny Qwen3-style model.
- Existing Llama tests still pass.

### Slow real-model tests

Real Qwen3 tests should be marked `@pytest.mark.slow` and skipped unless
`--run-slow` is passed.

Minimum real-model smoke:

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --prompt "The capital of France is" \
  --max-new-tokens 16 \
  --temperature 0.7 \
  --top-p 0.8
```

Benchmark:

```bash
uv run python scripts/benchmark.py \
  --model-path ./models/qwen3-0.6b \
  --n-tokens 100
```

Record:

- hardware
- Python version
- MLX version
- model path
- dtype
- tokens/sec
- output sample
- skipped tests and reasons

---

## Milestones

### P1.5-T00: Phase 1.5 planning docs

Done when:

- this spec exists
- taskboard entries are added or a Phase 1.5 taskboard is created
- review/sign-off rules are clear

### P1.5-T01: Config generalization

Done when:

- `ModelConfig` includes `model_type`, explicit `head_dim`, and derived
  `qk_norm`
- existing `TINY_CONFIG` and direct `ModelConfig(...)` test construction are
  updated for the new dataclass fields
- Llama configs still parse and validate
- Qwen3 configs parse and validate
- config tests cover both families

### P1.5-T02: Qwen3 attention support

Done when:

- attention projection dimensions use `H * Dh`, not `D`
- `o_proj` accepts `H * Dh`
- optional Q/K RMSNorm is implemented
- attention tests cover `H * Dh != D`
- Llama attention tests still pass

### P1.5-T03: Qwen3 weight conversion

Done when:

- Qwen3 HF keys map to project keys
- Q/K norm weights are loaded
- all Qwen3 shapes are validated
- absent `lm_head.weight` raises `ValueError`
- converter tests cover missing, unexpected, and malformed keys

### P1.5-T04: Model assembly and engine dispatch

Done when:

- model construction uses config-driven attention behavior
- `Engine.from_model_path()` selects the correct weight converter
- tiny Qwen3-style model forward/prefill/decode tests pass
- existing Llama engine tests still pass

### P1.5-T05: Tokenizer and CLI smoke

Done when:

- Qwen3 tokenizer loads through the project wrapper
- BOS/EOS IDs are resolved or documented clearly
- CLI works with Qwen3 model path
- prompt-mode limitations are documented

### P1.5-T06: Real model verification and handoff

Done when:

- `uv run pytest` passes
- slow Qwen3 smoke test passes when local artifacts are available
- benchmark result is recorded
- docs list supported models and known limitations
- another agent reviews and signs off before marking done

---

## Review Gates

Phase 1.5 changes require review by an agent other than the implementation
owner before any task is marked `done`.

Architecture review is required for:

- config model-family changes
- attention shape changes
- Q/K norm integration
- converter dispatch design
- public API or CLI changes

Code review is required for:

- any runtime model, attention, converter, tokenizer, engine, or sampling change
- tests that define expected Qwen3 behavior

Test verification is required for:

- every task claiming runtime behavior
- final Qwen3 support
- any change touching Llama behavior

The owner may move a task to `review` after implementation and local tests. The
owner must not mark their own task `done`.

---

## Phase 1.5 Completion Criteria

Phase 1.5 is complete when:

- Llama-3.2-1B still works through the existing Phase 1 path.
- Qwen3-0.6B loads from local HuggingFace artifacts.
- Qwen3-0.6B can run prefill and decode with the static KV cache.
- Qwen3-0.6B can generate text through Python API and CLI.
- Unit tests cover both Llama and Qwen3 shape assumptions.
- Real Qwen3 smoke test and benchmark are recorded, or a clear artifact/hardware
  skip reason is documented.
- A non-owner reviewer signs off on the implementation and verification.

---

## Known Risks

- Qwen3 output quality may look poor with plain prompts until chat-template
  support is added.
- Recommended Qwen3 sampling settings may differ between thinking and
  non-thinking prompts.
- Qwen3 tokenizer metadata may expose BOS/EOS differently from Llama.
- `H * Dh != D` is the main shape risk; tests must force this case.
- Q/K norm placement is easy to get subtly wrong; it must be reviewed against
  the reference architecture.

---

## Suggested Reading Order

For implementation:

1. `tiny_duo_infer/config.py`
2. `tiny_duo_infer/layers/attention.py`
3. `tiny_duo_infer/models/llama.py`
4. `tiny_duo_infer/weights/llama_converter.py`
5. `tiny_duo_infer/engine.py`
6. `tests/test_config.py`
7. `tests/test_layers.py`
8. `tests/test_weights.py`
9. `tests/test_model.py`
10. `tests/test_engine.py`

For model reference:

1. Qwen3-0.6B `config.json`
2. Qwen3-0.6B safetensors key list
3. Qwen3-0.6B tokenizer metadata
4. Existing Llama-3.2-1B smoke-test notes in `docs/phases/phase-1-handoff.md`
