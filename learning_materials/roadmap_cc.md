# Learning Roadmap — Phase 1

**Author:** Claude Code  
**Date:** 2026-05-26  
**Covers:** P1-T00 through P1-T18 (all Phase 1 tasks)

This roadmap is for reading the Phase 1 code in a deliberate order. Each
section tells you what to read, what to focus on, and a key question to answer
before moving on. The experiments at the end are for hands-on exploration in
your REPL.

---

## Reading Order

### 1. Config — `tiny_duo_infer/config.py` + `tests/test_config.py`

**What to understand:**  
How a single JSON file drives every dimension in the model. The important
fields are `d_model`, `n_heads`, `n_kv_heads`, and two derived properties:

```
head_dim  = d_model // n_heads
n_groups  = n_heads // n_kv_heads
```

These two derived numbers govern GQA and RoPE. You will see them in every
layer. Internalise them here so they feel obvious later.

**What to skim:** The validation helpers (`_read_positive_int`,
`_read_positive_float`). They exist to catch bad config files early; the logic
is not interesting.

**Key question to answer:**  
For Llama-3.2-1B (`n_heads=32`, `n_kv_heads=8`), what is `n_groups`?  
What does `n_groups > 1` mean for the size of the KV cache relative to a
model with `n_heads == n_kv_heads`?

---

### 2. Tokenizer — `tiny_duo_infer/tokenizer/loader.py` + `tests/test_tokenizer.py`

**What to understand:**  
How `tokenizer.json` drives the `tokenizers` package — vocab, BOS/EOS special
token lookup, `encode()` prepending BOS, `decode()` stripping specials.

Focus on `_read_special_token_ids`. There are two lookup paths: the config
file may give BOS/EOS as integer IDs directly, or as string token names that
must be resolved through the vocab. Both paths end up validated by
`_validate_token_id`.

Read the test fixture carefully (`AddedToken(special=True)`,
`TemplateProcessing`). It shows the exact mechanism that makes
`skip_special_tokens=True` work — tokens must be *registered* as special, not
merely present in the vocabulary.

**Key question to answer:**  
Why does `skip_special_tokens=True` not work for a token registered as a plain
vocabulary entry? What is `AddedToken(special=True)` doing differently?

---

### 3. Safetensors loader — `tiny_duo_infer/weights/loader.py` + `tests/test_weights.py` (lines 1–150)

**What to understand:**  
Single-file vs. sharded checkpoints, the index JSON format, and duplicate key
detection. This is pure I/O — no math.

The interesting decision is in `_discover_safetensor_files`: when both
`model.safetensors` and `model.safetensors.index.json` are present, the index
wins. Read `test_load_weights_index_takes_precedence_over_single_file` to see
why this matters.

Note that `mx.load()` is used rather than `safetensors.mlx.load_file()`.
MLX's own C++ reader handles bfloat16 natively; the Python safetensors MLX
backend had a dtype mapping gap that caused a `TypeError` on real Llama weights.

**Key question to answer:**  
Why does `_merge_shard` check for duplicate tensor keys across shards, and
what kind of real-world checkpoint problem would trigger it?

---

### 4. Weight converter — `tiny_duo_infer/weights/llama_converter.py` + `tests/test_weights.py` (lines 150+)

**What to understand:**  
The HF → project key mapping. Start with the docstring table at the top of the
file — it is the complete specification. Then read `_expected_weight_specs()`
and verify that every row in the table has a corresponding entry.

Next read `_fill_tied_lm_head`. In Llama-3.2-1B the HF checkpoint omits
`lm_head.weight` because it is tied to `embed_tokens.weight`. The converter
handles this by assigning the same array object:

```python
converted["lm_head.weight"] = converted["embed_tokens.weight"]
```

This records the tied relationship without copying memory. No bytes are
duplicated — both keys point at the same underlying storage.

Finally read `_validate_shape`. Shape mismatches are caught here, at load time,
not during a forward pass hours later.

**Key question to answer:**  
After `convert()`, what does
`weights["lm_head.weight"] is weights["embed_tokens.weight"]` return?  
Why does catching shape mismatches at load time (rather than at forward-pass
time) matter in practice?

---

### 5. Base module helpers — `tiny_duo_infer/models/base.py` + `tests/test_model.py`

**What to understand:**  
The `load_weights()` dot-path routing protocol. This is the glue that connects
the flat weight dict produced by the converter to the nested module tree that
the model assembly will build.

Trace through `test_load_weights_routes_recursively` manually. Given a key
`"child.grandchild.weight"` arriving at `RootModule.load_weights()`:

1. Split on the first dot → prefix `"child"`, remainder `"grandchild.weight"`
2. Look up `self.child` → it is a `Module`, so recurse
3. Split again → prefix `"grandchild"`, remainder `"weight"`
4. Look up `child.grandchild` → it is a `Module`, so recurse
5. No dot in `"weight"` → `setattr(self, "weight", value)`

Every layer in the model will be loaded this way. Understanding the protocol
here makes model assembly straightforward.

`Linear` and `Embedding` are thin wrappers. The only thing worth noting in
`Linear.forward` is:

```python
return x @ self.weight.T
```

Weight is stored as `(out_features, in_features)` — the HuggingFace convention.
The transpose is applied on every forward call.

**Key question to answer:**  
What does `load_weights()` raise if a dotted key's prefix names an attribute
that exists but is not a `Module`? Read
`test_load_weights_raises_when_sub_attr_is_not_module` to see the guarded case.

---

### 6. KV cache — `tiny_duo_infer/cache.py` + `tests/test_cache.py`

**What to understand:**  
This is the core inference engine primitive. Everything in attention through the
decode loop depends on getting this right. Read in this exact order:

1. **Module docstring** — the write/commit protocol in plain English
2. **`__init__`** — buffer shape `(1, n_kv_heads, max_seq_len, head_dim)`.
   Note it is `n_kv_heads`, not `n_heads`. With `n_groups=4`, the KV cache is
   4× smaller than full MHA.
3. **`update()`** — `position` is always *passed in* by the caller, never read
   from `current_len` inside `update()`. This is the key invariant. It prevents
   mid-forward-pass ambiguity when all 16 layers write to the same cache at the
   same position during a single token step.
4. **`advance()`** — called once per token step by the engine, *after* all
   layers have written. Not once per layer.
5. **`eval()`** — forces MLX to materialise the K/V buffer writes before the
   next decode step reads them. Without this, the lazy graph can grow unbounded
   across decode steps.

**Trace this sequence manually before moving on:**

```
Prefill: 3 tokens
  cache.update(layer=0, position=0, new_len=3)  → returns slice [:,:,:3,:]
  cache.update(layer=1, position=0, new_len=3)  → returns slice [:,:,:3,:]
  cache.advance(3)                               → current_len = 3

Decode step 1:
  cache.update(layer=0, position=3, new_len=1)  → returns slice [:,:,:4,:]
  cache.update(layer=1, position=3, new_len=1)  → returns slice [:,:,:4,:]
  cache.advance(1)                               → current_len = 4
```

**Key question to answer:**  
Why must `position` be passed in by the caller rather than read from
`current_len` inside `update()`? What would go wrong if `advance()` were
called between layers instead of after all layers?

---

### 7. RMSNorm — `tiny_duo_infer/layers/normalization.py` + `tests/test_layers.py`

**What to understand:**  
RMSNorm normalises by root-mean-square rather than mean + variance (LayerNorm).
The formula:

```
rms  = sqrt(mean(x²) + eps)
x_n  = x / rms * weight
```

There is no learnable bias and no mean subtraction. This is cheaper than
LayerNorm and sufficient for Llama because the residual stream already has
roughly zero mean after random initialisation.

Read the test that computes RMSNorm manually with Python scalars and compares
it against the layer output. This is the simplest form of "test against the
formula directly."

**Key question to answer:**  
Why is `eps` added inside the square root rather than to the RMS after
computing it? What numerical failure would happen without it?

---

### 8. RoPE — `tiny_duo_infer/layers/rope.py` + `tests/test_layers.py`

**What to understand:**  
RoPE encodes position by *rotating* Q and K vectors rather than adding a
positional embedding. Two functions work together:

- `precompute_freqs(head_dim, max_seq_len, theta)` — builds `(cos, sin)` tables
  of shape `(max_seq_len, head_dim//2)` once at model construction time.
- `apply_rope(x, cos, sin, offset)` — applies the rotation to a `(B, S, H, Dh)`
  tensor at a given sequence `offset`.

The rotation pairs up dimensions: `(x0, x1)`, `(x2, x3)`, … Each pair is
rotated by an angle that depends on the pair's index and the token's position.

Focus on how `offset` is used: during prefill it is 0 (tokens 0…S-1), during
decode step k it equals `cache.current_len` (a single token at position k).
This is what allows the same `apply_rope` function to work for both phases.

**Key question to answer:**  
What does "rotation" mean here geometrically? If two tokens have the same
content but different positions, how does RoPE ensure their dot product in
attention decreases as the position gap grows?

---

### 9. SwiGLU FFN — `tiny_duo_infer/layers/feedforward.py` + `tests/test_layers.py`

**What to understand:**  
The feed-forward block uses two independent projections (`gate_proj`,
`up_proj`) and multiplies them elementwise after applying SiLU to the gate:

```python
hidden = mx.sigmoid(gate) * gate * up   # SiLU(gate) * up
output = self.down_proj(hidden)
```

This is the SwiGLU variant. The `intermediate_size` (8192 for Llama-3.2-1B) is
larger than `d_model` (2048) — the FFN expands then contracts.

Read the test that checks `gate_proj` and `up_proj` are independent: changing
one weight should only affect one branch of the product. This verifies the
architecture is wired correctly.

**Key question to answer:**  
Why does SwiGLU use two separate projections of the same input rather than
one projection followed by a split? What would change if you used a single
`(2 * intermediate_size, d_model)` weight and split the output in half?

---

### 10. GQA attention — `tiny_duo_infer/layers/attention.py` + `tests/test_layers.py`

**What to understand:**  
This is the most complex layer. Read in this order:

1. **Head split / reshape** — input `(B, S, d_model)` is reshaped into
   `(B, S, n_heads, head_dim)` for Q and `(B, S, n_kv_heads, head_dim)` for K/V.
2. **RoPE application** — `apply_rope` is called on Q and K before the cache
   write. The offset comes from `position_offset` passed by the model.
3. **Cache write** — `cache.update(layer_idx, new_k, new_v, position_offset)`
   returns the full valid K/V slices including all previous positions.
4. **GQA head repeat** — K and V have `n_kv_heads=8` heads; Q has `n_heads=32`.
   `mx.repeat(k, n_groups, axis=2)` expands K and V so every Q head attends to
   the corresponding KV group. The repeat axis is 2 (the head axis in
   `(B, n_kv_heads, S, Dh)` layout).
5. **Scaled dot-product attention** — `(Q @ K.T) / sqrt(head_dim)` + causal
   mask + softmax + `@ V`.

The causal mask deserves special attention. During prefill (S > 1) a lower
triangular mask is needed. During decode (S = 1) there is nothing to mask —
the single query token can attend to all previous positions freely.

**Key question to answer:**  
Why is the repeat axis for GQA expansion axis 2 rather than axis 1? Draw the
tensor shapes `(B, n_kv_heads, S, Dh)` before and after the repeat and verify
the expanded shape is `(B, n_heads, S, Dh)`.

---

### 11. Llama model assembly — `tiny_duo_infer/models/llama.py` + `tests/test_model.py`

**What to understand:**  
`LlamaBlock` stacks the sub-layers in the standard pre-norm order:

```
x = x + Attention(RMSNorm(x))   # residual around attention
x = x + FFN(RMSNorm(x))         # residual around FFN
```

`LlamaModel` adds the embedding at the front and the final norm + lm_head at
the back. The forward signature is:

```python
def forward(self, input_ids, cache, position_offset) -> mx.array  # (B, S, V)
```

`position_offset` is passed unchanged to every layer's attention so RoPE and
the cache write both use the correct position.

Read `test_llama_model_output_shape` — it confirms the full shape chain from
token IDs to logits using `TINY_CONFIG`. If this test passes, the assembly is
correctly wired end-to-end.

**Key question to answer:**  
Why is `position_offset` a single integer passed from outside rather than
derived inside each layer from `cache.current_len`? What would break if each
layer called `cache.current_len` independently during a forward pass?

---

### 12. Prefill path — `tiny_duo_infer/engine.py` (`prefill_token_ids`)

**What to understand:**  
Prefill processes the entire prompt in one forward pass. Read `prefill_token_ids`
carefully:

1. `_new_cache()` allocates a fresh static KV cache sized for `max_seq_len`.
2. The full prompt `(B=1, S=prompt_len)` is passed to the model at
   `position_offset=0`. All 16 layers write their KV entries in one shot.
3. `logits[0, prompt_len - 1, :]` extracts the final position logits — only
   this position is needed to sample the first generated token.
4. `cache.advance(prompt_len)` commits all written positions at once.
5. `mx.eval(final_logits)` materialises the logits for CPU-side sampling.
6. `cache.eval()` materialises the KV buffers so the first decode step can
   read them.

Note the ordering: `advance` before `eval`. Advancing sets `current_len`
correctly; eval flushes the computation graph.

**Key question to answer:**  
Why is `cache.eval()` called *after* `mx.eval(final_logits)` rather than
combined into a single `mx.eval(final_logits, *cache._keys, *cache._values)`
call? Is there a semantic reason, or is it just style?

---

### 13. Greedy decode loop — `tiny_duo_infer/engine.py` (`generate`)

**What to understand:**  
The decode loop is where all previous components converge. Read it alongside
the prefill code:

```python
first_logits = self.prefill(prompt)
next_token = sample(first_logits, temperature=temperature, ...)

for step in range(max_new_tokens):
    if next_token == eos_token_id: break
    yield self.tokenizer.decode([next_token])
    if step == max_new_tokens - 1: break   # skip last decode forward
    input_ids = mx.array([[next_token]])
    logits = self.model(input_ids, self.cache, self.cache.current_len)
    mx.eval(logits)
    self.cache.eval()
    self.cache.advance(1)
    next_token = sample(logits[0, 0, :], ...)
```

Two subtle correctness points:

- **EOS check before yield** — callers never receive the stop token.
- **Skip last decode** — when `step == max_new_tokens - 1`, the token has
  already been yielded. Running one more forward pass would produce a token
  that is immediately discarded. Skipping it saves one full decode step at
  the cost of a tiny `if` check.

The `mx.eval` + `cache.eval` pattern after each decode step is documented in
`tiny_duo_infer/backends/mlx_backend.py`. Read that file's docstring for the
authoritative statement of the eval boundary policy.

**Key question to answer:**  
Trace what happens when `max_new_tokens=1`. How many model forward passes are
made in total (including prefill)? How many times is `cache.advance` called?

---

### 14. Sampling — `tiny_duo_infer/sampling.py` + `tests/test_sampling.py`

**What to understand:**  
`greedy()` is one line: `mx.argmax(logits).item()`. Understand it first.

`sample()` applies four transforms in a fixed order:

1. **Temperature** — `logits / temperature`. High temperature flattens the
   distribution; low temperature sharpens it. `temperature=0.0` short-circuits
   to `greedy()` directly (division by ~0 is numerically unstable).
2. **Top-k** — keep only the k tokens with the highest logits; set the rest to
   `-inf`. Uses `mx.sort(logits)[-k]` as the threshold.
3. **Top-p (nucleus)** — sort descending, compute `cumsum(softmax(logits))`,
   keep the smallest prefix whose cumulative probability reaches `top_p`.
   The token that *crosses* the threshold is kept; all tokens after it are
   set to `-inf`.
4. **Sample** — `mx.random.categorical(logits)` draws one token. Softmax is
   implicit; the function takes raw logits.

Read the validation block at the top of `sample()`. Each guard exists because
the failure mode without it is silent and wrong: `top_p=0.0` would set every
logit to `-inf`, and `mx.random.categorical` would return an arbitrary token.

**Key question to answer:**  
Why is the top-p keep condition `(cumprobs - probs) < top_p` rather than
`cumprobs < top_p`? Walk through a three-token example to verify that the
threshold-crossing token is included.

---

### 15. MLX eval placement — `tiny_duo_infer/backends/mlx_backend.py`

**What to understand:**  
This short file is the authoritative policy document for `mx.eval()` placement.
Read the module docstring in full — it is four sentences that save hours of
debugging.

The core rule: **eval only at engine boundaries, never inside layers.** There
are exactly two boundaries in Phase 1:

1. After prefill: `mx.eval(final_logits)` then `cache.eval()`.
2. After each decode forward: `mx.eval(logits)` then `cache.eval()`.

Calling `mx.eval()` inside a layer would add a GPU/CPU sync point in the middle
of a forward pass, fragmenting the computation graph and hiding the true cost
boundary between inference steps.

**Key question to answer:**  
`mx.eval(logits)` and `cache.eval()` are separate calls in the decode loop.
Why not collapse them into `mx.eval(logits, *cache._keys, *cache._values)`?
Read the comment in `engine.generate()` at that point for the answer.

---

### 16. CLI — `tiny_duo_infer/cli.py` + `tests/test_cli.py`

**What to understand:**  
The CLI is deliberately thin. Everything interesting is in `Engine`. The only
design choice worth studying is the `main(argv, engine_cls, stdout)` signature:
all three parameters have defaults for production use but are injectable for
tests.

Read `_FakeEngine` in `test_cli.py`. It is a good example of the test-double
pattern: records every call it receives, yields predictable output, never
touches a real model. The `autouse=True` fixture that clears class-level state
before each test is worth noting — class-level mutation is a common source of
test-order dependencies.

**Key question to answer:**  
Why does `main()` return `int` rather than printing an exit message itself?
What does `raise SystemExit(main())` at the bottom of the file accomplish that
a bare `main()` call would not?

---

### 17. Benchmark script — `scripts/benchmark.py`

**What to understand:**  
Read the module docstring first — the "Learning notes" section explains two
non-obvious things:

1. **Why prefill is included in the timer.** `generate()` is a generator;
   prefill runs on the first `next()` call. Draining it with `list()` includes
   both phases.
2. **Why the first decode step is slower.** MLX's computation graph is larger
   before the first `mx.eval()` warms it.

Then read `kv_cache_bytes()`. The formula:

```
2 × n_layers × n_kv_heads × T × head_dim × bytes_per_element
```

The leading `2` is for K and V. For Llama-3.2-1B at T=1024:
`2 × 16 × 8 × 1024 × 64 × 2 = 33,554,432 bytes = 32 MB`.

**Baseline numbers (Apple M3 Pro, 36 GB unified memory, MLX 0.31.2):**

| Metric | Value |
|---|---|
| tokens/sec (greedy, 100 tokens) | 9.2 |
| KV cache @ T=1024 (bfloat16) | 32 MB |
| KV cache @ T=2048 (bfloat16) | 64 MB |

**Key question to answer:**  
The `ref_lengths` set is `{n_generated, 256, 1024, 2048}`. Why include
`n_generated` in this set rather than just always printing the fixed reference
lengths?

---

## Experiments

Run these in order. Each one builds on the previous.

### Experiment 1 — Real Llama-3.2-1B dimensions

```python
from tiny_duo_infer.config import ModelConfig

c = ModelConfig(
    d_model=2048, n_layers=16, n_heads=32, n_kv_heads=8,
    intermediate_size=8192, vocab_size=128256,
    max_seq_len=131072, rope_theta=500000.0, rms_norm_eps=1e-5,
)
print("head_dim :", c.head_dim)    # 64
print("n_groups :", c.n_groups)    # 4

kv_bytes = 2 * c.n_layers * c.n_kv_heads * 1024 * c.head_dim * 2
print(f"KV cache @ 1024 tokens: {kv_bytes / 1024**2:.1f} MB")  # 32.0 MB
```

### Experiment 2 — Tied embeddings

```python
import mlx.core as mx
from tiny_duo_infer.weights.llama_converter import convert
from tiny_duo_infer.config import ModelConfig

c = ModelConfig(d_model=8, n_layers=1, n_heads=2, n_kv_heads=1,
                intermediate_size=16, vocab_size=32, max_seq_len=16,
                rope_theta=500000.0, rms_norm_eps=1e-5)

hf = {
    "model.embed_tokens.weight":             mx.zeros((32, 8)),
    "model.norm.weight":                     mx.zeros((8,)),
    "model.layers.0.input_layernorm.weight": mx.zeros((8,)),
    "model.layers.0.self_attn.q_proj.weight": mx.zeros((8, 8)),
    "model.layers.0.self_attn.k_proj.weight": mx.zeros((4, 8)),
    "model.layers.0.self_attn.v_proj.weight": mx.zeros((4, 8)),
    "model.layers.0.self_attn.o_proj.weight": mx.zeros((8, 8)),
    "model.layers.0.post_attention_layernorm.weight": mx.zeros((8,)),
    "model.layers.0.mlp.gate_proj.weight":   mx.zeros((16, 8)),
    "model.layers.0.mlp.up_proj.weight":     mx.zeros((16, 8)),
    "model.layers.0.mlp.down_proj.weight":   mx.zeros((8, 16)),
}

converted = convert(hf, c)
print("lm_head is embed_tokens:", converted["lm_head.weight"] is converted["embed_tokens.weight"])
```

### Experiment 3 — KV cache trace

```python
import mlx.core as mx
from tiny_duo_infer.cache import KVCache

cache = KVCache(n_layers=2, n_kv_heads=2, max_seq_len=16, head_dim=4)

# Prefill 3 tokens
k3 = mx.zeros((1, 2, 3, 4))
k_out, _ = cache.update(0, k3, k3, position=0)
cache.update(1, k3, k3, position=0)
cache.advance(3)
print(f"after prefill  : current_len={cache.current_len}, shape={k_out.shape}")
# (1, 2, 3, 4)

# Decode steps
for step in range(3):
    k1 = mx.zeros((1, 2, 1, 4))
    pos = cache.current_len
    k_out, _ = cache.update(0, k1, k1, position=pos)
    cache.update(1, k1, k1, position=pos)
    cache.advance(1)
    print(f"after decode {step+1} : current_len={cache.current_len}, shape={k_out.shape}")
# shapes grow: (1,2,4,4), (1,2,5,4), (1,2,6,4)
```

### Experiment 4 — RMSNorm vs LayerNorm

```python
import mlx.core as mx
from tiny_duo_infer.layers.normalization import RMSNorm

norm = RMSNorm(dim=4, eps=1e-5)
norm.weight = mx.ones(4)

x = mx.array([[1.0, 2.0, 3.0, 4.0]])
out = norm(x)
mx.eval(out)

# Manual formula
rms = (x ** 2).mean(axis=-1, keepdims=True) ** 0.5
manual = x / (rms + 1e-5)

print("norm output :", out.tolist())
print("manual      :", manual.tolist())
```

### Experiment 5 — Sampling behaviour

```python
import mlx.core as mx
from tiny_duo_infer.sampling import greedy, sample

logits = mx.array([1.0, 5.0, 2.0, 3.0])
mx.eval(logits)

print("greedy      :", greedy(logits))           # 1

# temperature=0.0 matches greedy
print("temp=0.0    :", sample(logits, temperature=0.0))  # 1

# top_k=1 always returns the top token
print("top_k=1     :", sample(logits, top_k=1))  # 1

# seeded sampling is deterministic
mx.random.seed(42)
a = sample(logits, temperature=1.0)
mx.random.seed(42)
b = sample(logits, temperature=1.0)
print("seeded same :", a == b)  # True
```

### Experiment 6 — End-to-end generation (requires model artifacts)

```python
from tiny_duo_infer.engine import Engine

MODEL = "~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"

engine = Engine.from_model_path(MODEL)

# Greedy — deterministic
text = "".join(engine.generate(
    "The capital of France is",
    max_new_tokens=16,
    temperature=0.0,
))
print("greedy :", text)

# Probabilistic — varies each run
import mlx.core as mx
mx.random.seed(7)
text = "".join(engine.generate(
    "The capital of France is",
    max_new_tokens=16,
    temperature=0.8,
    top_p=0.9,
))
print("sampled:", text)
```

---

## What comes next — Phase 2

Phase 1 established the single-request MLX baseline on Apple Silicon. Phase 2
adds a second compute backend (NVIDIA/CUDA via PyTorch) and validates the
backend abstraction defined in `tiny_duo_infer/backends/protocol.py`.

Key differences to look for as you read Phase 2 code:

| Concern | Phase 1 (MLX) | Phase 2 (adds PyTorch/CUDA) |
|---|---|---|
| Array type | `mx.array` | `torch.Tensor` |
| Eval model | lazy, `mx.eval()` at boundaries | eager by default |
| Memory | Apple unified (CPU+GPU shared) | separate device memory, explicit `.to(device)` |
| KV cache | pre-allocated static buffer | same design, different dtype/device |
| Sampling | `mx.random.categorical` | `torch.multinomial` |

The `backends/protocol.py` file defines the interface both backends must
satisfy. Reading it before Phase 2 starts will make the CUDA backend
implementation easier to follow.
