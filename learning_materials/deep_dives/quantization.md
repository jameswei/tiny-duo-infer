# Deep Dive: MLX Weight-Only Quantization

This document is a focused walkthrough of **weight-only quantization** as
implemented in Phase 1.8 of `tiny-duo-infer`. It pairs with the phase spec
`docs/phases/phase-1.8-weight-quantization.md` and the source files
`tiny_duo_infer/quantization.py`, `tiny_duo_infer/weights/quantizer.py`, and
`tiny_duo_infer/models/base.py`.

The Phase 1.8 goal is not a production quantization toolkit. The goal is to
make every part of an inference-engine quantization path visible: which
weights are eligible, how packed integer weights replace floating-point
matrices, how the fused quantized matmul kernel skips the dequantize step,
and how to account for the memory benefit honestly.

---

## 1. Why Weight-Only Quantization

Modern transformer weights are large. Llama-3.2-1B in bfloat16 is about
2.5 GB on disk, and once loaded the linear projection weights dominate
that footprint:

```
linear weight bytes ≈ Σ_layer 4 × d_model × d_model
                    + Σ_layer 3 × d_model × intermediate_size
                    + lm_head_out × d_model
```

For Llama-3.2-1B that is roughly 1.2 GB just in `Linear` weights at
bfloat16. KV cache, activations, embeddings, RoPE tables, and overhead
account for the rest.

Three families of techniques can shrink this:

| Approach | What gets compressed | Who decides | Phase 1.8? |
|---|---|---|---|
| **Weight-only quantization** | matrix weights of `Linear` modules | offline / load-time | yes — INT4 / INT8 |
| Activation quantization | hidden states between layers | runtime per token | no |
| KV-cache quantization | cached K and V tensors | runtime per layer | no |

Phase 1.8 picks the most contained option: **weight-only**. Activations,
embeddings, RMSNorm/Q-K-norm weights, RoPE tables, and KV cache stay
full precision. This makes the change reviewable layer-by-layer and lets
us measure exactly one variable: linear weight memory.

It also leaves activation precision untouched, so dynamic-range losses
during the forward pass behave the same as the full-precision path. Only
the matrix multiply at each `Linear` projection draws from compressed
storage.

---

## 2. Affine Quantization, Group by Group

The quantization mode in Phase 1.8 is `"affine"`. Each group of
`group_size` consecutive elements along the input dimension shares one
floating-point `scale` and one floating-point `bias` (zero-point):

$$
q = \mathrm{round}\!\left(\frac{x - \mathrm{bias}}{\mathrm{scale}}\right),
\qquad
x \approx q \cdot \mathrm{scale} + \mathrm{bias}
$$

For a weight of shape `(out_features, in_features)`, MLX returns three
arrays from `mx.quantize(weight, group_size, bits)`:

```text
qweight: (out_features, in_features * bits / 32)    packed integers
scales:  (out_features, in_features / group_size)   per-group fp scales
biases:  (out_features, in_features / group_size)   per-group fp biases
```

The packing rule for `qweight` is "as many `bits`-wide integers as fit in
each 32-bit word, along the input dimension." For `bits=4`,
`in_features` collapses by 8×; for `bits=8`, by 4×.

The two reconstruction parameters per group are stored as float, and
their count grows with `in_features / group_size`. This is the trade-off
tuned by `group_size`: smaller groups give finer-grained reconstruction
(better quality) but pay more bytes for `scales` and `biases`.

### Why group_size matters at the input dimension

The fused kernel computes one output coordinate at a time by summing
`in_features` contributions. Within each contribution, dequantizing one
group requires exactly one `(scale, bias)` pair. If `in_features` is not
divisible by `group_size`, some contributions in the last partial group
have no scale/bias defined, and the kernel cannot run.

`tiny_duo_infer.quantization.QuantizationConfig` validates this when
the config is built; `weights/quantizer.py` re-checks per weight and
raises `ValueError` naming the offending key, `in_features`, and
`group_size` so the failure is recoverable without partial loading.

The default group size is `64`. It works for every `Linear` matrix in
real Llama-3.2-1B and Qwen3-0.6B because their relevant input dimensions
(`d_model`, `intermediate_size`, attention widths) are all multiples of
`64`. Tiny synthetic test fixtures with `d_model = 32` cannot use that
default — they must use `group_size = 32` (or another divisor of 32).
The Phase 1.8 spec calls this out, the integration tests verify both
the success and the rejection, and the CLI/profiling entrypoints expose
`--quant-group-size N` so users can tune it deliberately.

---

## 3. The Fused Quantized Matmul

The runtime path through `Linear.forward()` is:

```python
if isinstance(self.weight, QuantizedWeight):
    qw = self.weight
    return mx.quantized_matmul(
        x, qw.qweight, qw.scales, qw.biases,
        transpose=True,
        group_size=qw.group_size,
        bits=qw.bits,
        mode=qw.mode,
    )
return x @ self.weight.T
```

What the fused kernel does conceptually:

1. iterate over output coordinates `o ∈ [0, out_features)`,
2. for each input group along the input dimension, dequantize the
   `group_size` integers into a small register-sized tile using
   `scale[o, g]` and `bias[o, g]`,
3. accumulate `dot(x_tile, dequantized_tile)` into the output,
4. emit `y[..., o]`.

The crucial part for Phase 1.8: **the full-precision weight matrix is
never materialized in DRAM.** Only register-sized tiles are
dequantized, used once, and discarded. That is why the memory benefit
from compressed storage is preserved at runtime, not just at load time.

### Why `mx.dequantize()` is restricted

`mx.dequantize()` reconstructs the full-precision weight matrix from
`(qweight, scales, biases)`. It is genuinely useful for tests and for
debugging numerical accuracy:

- compute reference fp output from a quantized weight,
- compare quantized vs reference output within a numerical tolerance,
- inspect which groups contribute the most reconstruction error.

But if the runtime path were `dequantize → matmul`, every forward pass
would round-trip through full precision and the memory benefit would
vanish. The Phase 1.8 spec explicitly forbids this as the normal path,
and the architecture review gate enforces it. Eager dequantization at
*load time* is rejected for the same reason.

---

## 4. Eligibility: What Gets Quantized

`weights/quantizer.py:_is_eligible()` is the entire eligibility rule
expressed in code:

```python
def _is_eligible(key: str, tensor: mx.array) -> bool:
    if tensor.ndim != 2:
        return False
    if key in _ELIGIBLE_EXACT:        # {"lm_head.weight"}
        return True
    return any(key.endswith(suffix) for suffix in _ELIGIBLE_SUFFIXES)
```

Where `_ELIGIBLE_SUFFIXES` is the seven `Linear` projection suffixes:

```
.q_proj.weight, .k_proj.weight, .v_proj.weight, .o_proj.weight,
.gate_proj.weight, .up_proj.weight, .down_proj.weight
```

Two filters cover everything that should *not* be quantized:

- `tensor.ndim != 2` excludes RMSNorm weights and Qwen3
  `q_norm` / `k_norm` weights, which are 1-D `(head_dim,)`.
- The suffix list is closed: anything outside it stays full precision.
  `embed_tokens.weight` is intentionally absent.

### Llama tied embeddings

Llama-3.2-1B ties `lm_head.weight` and `embed_tokens.weight`: the same
`mx.array` object is referenced by both keys in the converter output.
Quantizing `lm_head.weight` must not propagate back to
`embed_tokens.weight`, because:

- embeddings are looked up with row-indexing; they are not consumed by a
  matmul, so a fused quantized kernel cannot serve them at runtime,
- treating embeddings as quantized would require a different code path
  inside `Embedding.forward()` that Phase 1.8 does not introduce.

`weights/quantizer.py` constructs a new dict so the array under
`embed_tokens.weight` is never mutated. After `quantize_weights()`,
`embed_tokens.weight` remains a full-precision `mx.array`, and
`lm_head.weight` is a `QuantizedWeight`. The two keys no longer share
an object; the *logical* tying becomes a runtime convention rather than
an in-memory aliasing.

### Qwen3 lm_head

Qwen3 stores `lm_head.weight` explicitly in safetensors even though its
config advertises tied embeddings, so converter validation already
expects the key to be present. Phase 1.8 does not change that — Qwen3's
`lm_head.weight` simply becomes a quantized projection like the rest.
Qwen3 `q_norm.weight` and `k_norm.weight` stay 1-D and full precision.

---

## 5. Memory Accounting

`LinearWeightStats` in `weights/quantizer.py` is the four-number summary
of the comparison Phase 1.8 cares about:

```python
@dataclass
class LinearWeightStats:
    quantized_linear_count: int
    full_precision_linear_count: int
    linear_weight_full_precision_bytes: int
    linear_weight_runtime_bytes: int
```

`compute_linear_weight_stats()` walks the weight dict produced by
`quantize_weights()` and counts only eligible linear projections.
Embeddings, RMSNorm, and Qwen3 Q/K-norm weights are excluded so the
comparison answers a single, well-scoped question:

> **For the linear weights this phase is allowed to compress, how many
> bytes does the runtime actually hold, versus how many bytes the
> full-precision version would hold?**

That choice is documented in the dataclass docstring and surfaced
through `GenerationStats`. The seven `GenerationStats` quantization
fields are:

| Field | Meaning |
|---|---|
| `quantization_mode` | `"none"`, `"int8"`, or `"int4"` |
| `quantization_bits` | `None`, `8`, or `4` |
| `quantization_group_size` | `None` or the configured group size |
| `quantized_linear_count` | how many `Linear` modules use a `QuantizedWeight` |
| `full_precision_linear_count` | how many `Linear` modules stayed full precision |
| `linear_weight_full_precision_bytes` | bytes if every counted weight stayed full precision |
| `linear_weight_runtime_bytes` | bytes the runtime actually holds for those weights |

When quantization is disabled, the fields still populate with
`mode="none"`, `bits=None`, `group_size=None`, zero quantized count, and
`runtime_bytes == full_precision_bytes`. That makes "no quantization"
indistinguishable from "an absent stats block" downstream — every
response carries the full schema.

### Why throughput is not a hard gate

The Phase 1.8 acceptance gate is **reduced linear weight memory**, not
faster decode tokens per second. Reasons:

- MLX quantized kernel performance varies by bit width, group size,
  matrix shape, and hardware state.
- Small models can be bandwidth-bound at full precision and become
  compute-bound after quantization, so wall-clock time may not improve
  even when bytes drop.
- A learning-first project should not commit to a moving performance
  target.

So the spec asks: throughput must be *measured* and *reported*, not
*better*. Profiling JSON v2 surfaces the numbers; interpretation is up
to the reader.

---

## 6. The Loading Pipeline

The Phase 1.8 spec frames model loading as four canonical steps:

1. load safetensors,
2. convert and validate HF keys / shapes,
3. quantize eligible project weights (the new step),
4. load values into the model tree.

`Engine.from_model_path()` implements those four steps with one extra
internal pass — memory accounting — that is computed *from the flat
project weight dict* before the model is constructed. This ordering is
deliberate: `compute_linear_weight_stats()` is a dict walk that does not
depend on a constructed model, and running it early keeps the
accounting independent of `Module.load_weights()`.

```text
1. load safetensors            (weights/loader.py)
        │
        ▼
2. convert HF → project keys   (weights/llama_converter.py or qwen3_converter.py)
        │ validates shapes; tied embeddings preserved as object identity
        ▼
3. quantize eligible weights   (weights/quantizer.py)   ← Phase 1.8 step
        │ mx.quantize() on every Linear projection matrix matching the
        │ eligibility rules, leaving everything else unchanged; new dict
        │ so tied lm_head/embed_tokens objects no longer alias.
        ▼
3b. compute_linear_weight_stats(project_weights)
        │ walks the flat dict and counts eligible Linear weights only.
        │ Runs *before* model construction so the accounting is derived
        │ from the dict shape, not by introspecting loaded `Linear`
        │ modules. Result is stored on the Engine and attached to every
        │ GenerationStats response.
        ▼
4. construct model + populate model tree   (Module.load_weights())
        │ model_cls(runtime_config) builds the empty module tree; then
        │ Linear.weight is set per-key from the flat dict. Linear.weight
        │ may be either mx.array or QuantizedWeight; the forward()
        │ dispatch is purely on the type of self.weight.
        ▼
5. return cls(model, tokenizer, ..., quantization, linear_weight_stats)
```

`Engine.from_model_path(model_path, max_seq_len, quantization=None)`
runs the whole pipeline. Passing `quantization=None` short-circuits
step 3 — full-precision loading is the default — and step 3b still runs
on the unmodified dict so the stats fields are populated for the
no-quantization path too. Passing a `QuantizationConfig` flips the
`Linear.forward()` dispatch for every eligible projection without
changing any other module.

The CLI, HTTP server startup, and profiling entrypoint each accept
`--quantization {none,int4,int8}` and `--quant-group-size N`, build a
`QuantizationConfig`, and forward it through the same `from_model_path`
constructor. Validation runs early so an invalid bit width, group size,
or non-divisible matrix shape fails before any tensor is allocated.

---

## 7. What To Verify When You Read The Code

For each component, write down:

- **Inputs:** project weight dict, `QuantizationConfig` fields.
- **Outputs:** `dict[str, mx.array | QuantizedWeight]`, `LinearWeightStats`,
  `GenerationStats` quantization fields.
- **Tensor shapes:** `(out_features, in_features)` for the source weight;
  `qweight: (out, in * bits / 32)`, `scales: (out, in / group_size)`,
  `biases: (out, in / group_size)`.
- **State:** Which dictionary holds full-precision arrays vs
  `QuantizedWeight`; which `Linear` instances accept which weight type.
- **Invariants:** `quantization_mode == "none"` ⇔ `bits is None and group_size is None`;
  when quantization is disabled, `linear_weight_runtime_bytes == linear_weight_full_precision_bytes`
  for the counted linear weights.
- **Failure cases:** non-divisible `in_features`, invalid `bits`, invalid
  `group_size`, unsupported `mode`, eager dequantization (forbidden by
  spec rather than by code).
- **The one thing that would silently corrupt things:** a path where the
  fused kernel is replaced with `dequantize → matmul`. Generation would
  still work and outputs would still look reasonable, but the memory
  benefit would silently disappear, and `linear_weight_runtime_bytes`
  would no longer reflect what is actually held in DRAM.

That last item is exactly why the integration test
`test_llama_quantized_respects_stop_string` wraps the real quantized
model rather than replacing it: it forces the quantized matmul path to
execute and asserts both correct stop semantics and non-zero
`quantized_linear_count`, so a regression that drops either side fails
the test.

---

## Further Reading

- `docs/phases/phase-1.8-weight-quantization.md` — authoritative phase
  spec, including out-of-scope items.
- `tiny_duo_infer/quantization.py` — `QuantizationConfig` and
  `QuantizedWeight` with field-by-field docstrings.
- `tiny_duo_infer/weights/quantizer.py` — `quantize_weights()`,
  `LinearWeightStats`, `compute_linear_weight_stats()`.
- `tiny_duo_infer/models/base.py` — `Linear.forward()` quantized vs
  full-precision dispatch.
- `tests/test_quantization.py` — config validation, packed-weight
  metadata, dispatch correctness.
- `tests/test_quantization_integration.py` — end-to-end tiny Llama and
  Qwen3 generation through full precision, INT8, and INT4.
- `learning_materials/roadmap.md` — guided reading order including the
  Phase 1.8 step.
