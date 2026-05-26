# Inference Engine Internals

A deep-dive into the three hardest concepts in Phase 1. Read this after the
[learning roadmap](roadmap.md), or alongside the source code.

---

## 1. Prefill vs Decode — The Two-Pass Engine

Every LLM token generation splits into two distinct phases. Understanding
*why* they're different is the single most important concept in building an
inference engine.

### Prefill: process the prompt in one shot

When the user types "The capital of France is", the engine sees:

```
prompt tokens:  [128000, 791, 14013, 315, 11405, 374]
                 BOS     The   cap    ital  of    France  is
```

All 6 tokens go into the model in **one forward pass** as `input_ids` shaped
`(B=1, S=6)`. The model computes:

1. **Embedding:** `(1, 6, 2048)`
2. **16 transformer blocks** — each attends to all positions 0..5 *simultaneously*
3. **Final norm + lm_head:** `(1, 6, 128256)` logits

The engine only keeps `logits[0, 5, :]` (the last position). Those logits
predict token #7 — the first *generated* token.

**Why one shot?** The GPU can process all prompt tokens in parallel (a single
matrix multiply handles all 6 positions). This is the fast part.

### Decode: one token at a time

Token #7 ("Paris") becomes the input for the *next* model call:

```
decode step 1:  input_ids = [[3663]]           (B=1, S=1)  position_offset=6
decode step 2:  input_ids = [[<next_token>]]   (B=1, S=1)  position_offset=7
...
```

Each step runs the full 16-block transformer, reads the KV cache for all past
positions, and produces one new token. This is the slow part — it's inherently
sequential.

### Why not batch decode?

You might wonder: why not feed all generated tokens back in as a batch, like
prefill? Because each new token depends on the previous token's output. The
engine *must* sample token 7 before it can compute token 8's embedding.
Autoregressive generation is a serial dependency chain — no way around it.

### Position offset — the bridge between phases

```python
# Prefill
model(input_ids, cache, position_offset=0)
# → RoPE rotates positions 0, 1, 2, 3, 4, 5
# → causal mask: position i attends to [0..i]

# Decode step t (where t = prompt_len + tokens_generated_so_far)
model(input_ids, cache, position_offset=t)
# → RoPE rotates the single token at absolute position t
# → causal mask: attends to all T cached positions (all are in the past)
```

The `position_offset` ensures RoPE encodings and causal masks are computed
relative to the *absolute* position in the full sequence, not the small
`(B=1, S=1)` slice the model sees during decode.

### The mx.eval() boundary

MLX uses lazy evaluation: tensor operations build a computation graph but
don't execute until `mx.eval()` forces materialization. Phase 1 evaluates only
at engine boundaries:

```
prefill forward  →  mx.eval(final_logits)  +  cache.eval()  →  sample
                                                                    ↓
decode forward   →  mx.eval(logits)        +  cache.eval()  →  sample
                                                                    ↓
decode forward   →  ...                                            ...
```

Evaluating inside attention or FFN layers would:
- Add unnecessary GPU/CPU sync points
- Hide where inference steps begin and end
- Make the lazy graph harder to reason about

The two `eval()` calls after each forward are *not* redundant:

| Call | What it materializes | Why |
|---|---|---|
| `mx.eval(logits)` | The `(1, 1, V)` output | CPU-side sampling reads token IDs |
| `cache.eval()` | K/V buffers for all 16 layers | Next decode step reads these buffers |

They serve different consumers: `logits` go to the CPU sampler, cached K/V
stay on the accelerator for the next attention pass.

---

## 2. KV Cache Lifecycle — The `update()` / `advance()` Split

The KV cache is the engine's long-term memory. It stores every token's key
and value vectors so the model doesn't recompute them.

### Buffer layout

```
Per layer:  K: (1, n_kv_heads=8, max_seq_len, head_dim=64)
            V: (1, n_kv_heads=8, max_seq_len, head_dim=64)

16 layers × 2 buffers × (1 × 8 × T × 64 × 2 bytes) ≈ 32,768 × T bytes
At T=1024: ~32 MB
```

The `(..., max_seq_len, ...)` dimension is pre-allocated once. Only the slice
`[:, :, :current_len, :]` is valid at any point.

### The two-phase write protocol

This is where most KV cache bugs live. Phase 1 deliberately splits writing
from advancing:

```
update(layer, new_k, new_v, position)  ← called 16 times per token step
advance(n_tokens)                      ← called ONCE per token step (by the engine)
```

**Why not combine them?** During one decode step, all 16 layers write their
K/V for position `p`. If the first layer advanced `current_len` to `p+1`, the
second layer would compute RoPE at the wrong position and mask against stale
cache state. The split ensures every layer sees the same `current_len` for the
entire forward pass.

### Prefill lifecycle

```
Step 0 (prefill, prompt_len=6):
  For each of 16 layers:
    cache.update(layer, new_k, new_v, position=0)  → returns [:, :, :6, :]
  cache.advance(6)  → current_len becomes 6

  Cache state: positions [0..5] valid, current_len = 6
  Logits: (128256,) from position 5
```

### Decode lifecycle

```
Step 1 (decode, token "Paris"):
  position_offset = cache.current_len = 6  ← engine passes this in
  For each of 16 layers:
    new_k, new_v: (1, Hkv, 1, Dh)  — one token
    cache.update(layer, new_k, new_v, position=6)  → returns [:, :, :7, :]
  Forward produces logits (1, 1, V)
  mx.eval(logits)  +  cache.eval()
  cache.advance(1)  → current_len = 7
  Sample next token from logits[0, 0, :]

  Cache state: positions [0..6] valid, current_len = 7
```

### Why pre-allocation instead of growing?

Some implementations append K/V tensors each step (`torch.cat`). That's simpler
conceptually but copies the entire cache buffer every step — O(T) copy per
token, O(T²) total. Pre-allocating avoids this: writes are O(1) index
assignments. The tradeoff is you must know `max_seq_len` upfront.

Phase 3 replaces this with **PagedAttention**: fixed-size KV pages allocated
from a shared pool, eliminating both the copy problem and the pre-allocation
waste.

---

## 3. GQA + RoPE — Why Head Expansion Matters

### GQA (Grouped Query Attention)

Llama-3.2-1B has 32 query heads but only 8 key/value heads. Each KV head is
shared by `n_groups = 4` query heads:

```
Q heads:  [0, 1, 2, 3] → KV head 0
          [4, 5, 6, 7] → KV head 1
          ...and so on
```

This reduces KV cache memory by 4× without significant quality loss. The
attention computation expands KV heads back to the full Q-head count:

```python
k_expanded = mx.repeat(k_full, repeats=n_groups, axis=1)
# (1, 8, T, 64) → (1, 32, T, 64)
```

The `axis=1` choice is deliberate. KV heads are stored along the head
dimension (axis=1 after transpose). Repeating there means:
- `axis=0` (batch): already 1 in Phase 1, would be wrong for multi-batch
- `axis=1` (heads): groups query heads together — Hkv heads [0,0,0,0,1,1,1,1,...]
- `axis=2` (sequence): would repeat tokens, not heads

### RoPE (Rotary Positional Embeddings)

RoPE encodes position by rotating consecutive pairs in each head vector:

```
For each pair (x0, x1) in the head dimension:
  x0' = x0·cos(pos·θ_i) - x1·sin(pos·θ_i)
  x1' = x0·sin(pos·θ_i) + x1·cos(pos·θ_i)
```

The frequency `θ_i` decreases for later pairs, creating a spectrum from
fine-grained (fast-rotating, position-sensitive) to coarse-grained
(slow-rotating, content-focused) encodings.

Key properties:
- **Relative position:** the dot-product between Q and K after RoPE depends
  on `pos_Q - pos_K`, naturally encoding relative distance
- **Applied to Q and K only:** V doesn't get rotated because attention weights
  already encode position relationships; V just stores content
- **No learned parameters:** the rotation is purely a function of position

### The interaction: RoPE before expansion

RoPE is applied to Q and K *before* the KV cache stores them and *before*
GQA head expansion:

```
1. Q projection → reshape → (B, S, H=32, Dh=64)
2. K projection → reshape → (B, S, Hkv=8, Dh=64)
3. apply_rope(q, ...)   ← rotate each head independently
4. apply_rope(k, ...)
5. k = transpose(k, ...) → (B, Hkv=8, S, Dh)     ← ready for cache
6. cache.update(layer, k, v, position)
7. k_full, v_full = ... → (B, Hkv=8, T, Dh)     ← after cache readback
8. k_expanded = repeat(k_full, n_groups, axis=1)  → (B, H=32, T, Dh)
```

Step 8 repeating *after* RoPE is correct because repeated heads share the same
absolute position — they all belong to the same token. Repeating before RoPE
would incorrectly apply different rotations to different copies.

### Causal mask during prefill vs decode

```
Prefill (S > 1):
  query_pos = [0, 1, 2, 3, 4, 5]  (with offset=0)
  key_pos   = [0, 1, 2, 3, 4, 5]  (all keys in cache so far)
  mask      = key_pos > query_pos  → lower-triangular mask
  Token 3 can attend to positions [0,1,2,3] only.

Decode (S = 1):
  query_pos = [6]                  (with offset=6)
  key_pos   = [0, 1, 2, 3, 4, 5]  (all from cache)
  mask      = key_pos > query_pos  → all False, no masking needed
  The single new token attends to everything in the past.
```

During decode, `S=1` means there's only one query position and it sits *after*
all cached keys — it can attend to everything with no masking at all. The code
still computes the mask for correctness (it would be wrong during prefill),
but the mask is a no-op for decode.

---

## Summary: One Complete Generation Request

```
User: "The capital of France is"

PREFILL (1 forward pass)
  tokenize("The capital of France is") → [128000, 791, 14013, 315, 11405, 374]
  allocate KVCache(max_seq_len=2048)
  model(input_ids=(1,6), cache, position_offset=0)
    → 16 layers write K/V at position 0..5
  cache.advance(6)
  mx.eval(final_logits) + cache.eval()
  sample → token "Paris" (3663)

DECODE LOOP (one forward pass per token)
  ┌─ step 0: token "Paris"
  │   model(input_ids=(1,1) [[3663]], cache, position_offset=6)
  │   mx.eval(logits) + cache.eval()
  │   cache.advance(1)
  │   sample → token ","
  │
  ├─ step 1: token ","
  │   model(input_ids=(1,1) [[11]], cache, position_offset=7)
  │   mx.eval(logits) + cache.eval()
  │   cache.advance(1)
  │   sample → token " located"
  │
  ├─ step 2: ...
  │   ...until EOS or max_new_tokens
  │
  └─ yield each token as decoded text fragments
```

The loop stops when:
- The sampled token equals `eos_token_id` (the engine does *not* yield EOS)
- `max_new_tokens` is reached (the engine skips the wasted final decode forward)
