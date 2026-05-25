# Phase 1 Spec: Single-User Local Inference on Apple Silicon

**Status:** Draft  
**Authors:** Claude Code + Codex (merged)  
**Based on:** `docs/refined-plan_codex.md`  
**Date:** 2026-05-25

---

## Goal

Implement a working prefill + decode pipeline for a single request at a time,
running `meta-llama/Llama-3.2-1B` on Apple Silicon via MLX. By the end of
Phase 1, the project can load real weights, tokenize a prompt, run prefill,
decode tokens one at a time using a KV cache, apply sampling strategies, and
produce output via both a Python API and a CLI.

---

## Scope

### In scope

- Loading `meta-llama/Llama-3.2-1B` (base model) from a local
  HuggingFace-compatible model directory
- Tokenizer wrapper using the `tokenizers` package
- Llama 3 architecture: RMSNorm, RoPE, GQA attention, SwiGLU FFN
- Static pre-allocated KV cache (one fixed-size buffer per layer)
- Prefill: full prompt sequence in one forward pass
- Decode: one token per step, attending to the KV cache
- Sampling: greedy (M1.6), then top-k / top-p / temperature (M1.8)
- MLX as the sole tensor execution backend
- Draft `backends/protocol.py` capturing the backend interface shape
- Python `Engine` class as the public API
- CLI for local text generation
- Unit tests for every layer and milestone using a tiny synthetic fixture

### Out of scope for Phase 1

- Instruct/chat-template support
- HTTP serving
- Multi-user scheduling or batching
- PagedAttention or dynamic KV block management
- Quantization
- Speculative decoding
- PyTorch or any non-MLX backend
- Distributed inference
- C++ or custom kernels
- Wrapping `mlx-lm`, vLLM, or `transformers` model/generation APIs

---

## Runtime and Tooling

Use `uv` with Python `>=3.12,<3.13`.

```toml
[project]
name = "tiny-duo-infer"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "mlx>=0.10",
    "tokenizers>=0.19",
    "safetensors>=0.4",
    "huggingface-hub>=0.23",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "transformers>=4.40",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- `mlx` is a required Phase 1 runtime dependency — it is the only backend.
- `transformers` is dev/test only. It must not be imported anywhere under
  `tiny_duo_infer/`.
- Optional backend extras (`mlx` vs `torch`) are introduced in Phase 2.

Install: `uv sync --group dev`

---

## Model Artifacts

The engine loads from a local HuggingFace-compatible model directory.

Expected files:

```
models/llama-3.2-1b/
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  model.safetensors          (single shard, or)
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  model.safetensors.index.json   (present when sharded)
```

The model path must be configurable via the Python API and CLI. Hard-coding
it anywhere in the engine is not allowed.

### Download prerequisites

1. Accept the Llama 3.2 license at `huggingface.co/meta-llama/Llama-3.2-1B`
2. `uv run huggingface-cli login`
3. Download locally:

```bash
uv run huggingface-cli download meta-llama/Llama-3.2-1B \
    --local-dir ./models/llama-3.2-1b
```

---

## Architecture Constraints

1. **MLX used directly in Phase 1** — `backends/protocol.py` is written as a
   draft during Phase 1 but the MLX code is not required to formally conform
   until Phase 2. Phase 1 MLX code should be structured in clearly bounded
   helper functions so extraction into `mlx_backend.py` is straightforward
   later.

2. **No high-level MLX layers** — do not use `mlx.nn.MultiHeadAttention`,
   `mlx.nn.RMSNorm`, `mlx.nn.SiLU`, or `mlx.nn.Linear`. Implement each
   component from MLX primitives: `mx.matmul`, `mx.softmax`, `mx.rsqrt`,
   `mx.cos`, `mx.sin`, etc.

3. **No `transformers` at runtime** — `transformers` is a dev/test dependency
   only. It must not be imported in any file under `tiny_duo_infer/`.

4. **Batch size = 1 throughout Phase 1** — all tensors carry a batch dimension
   for future compatibility, but it is always 1. Do not optimise for batch > 1.

5. **Teaching code standard** — every public class and function must have a
   docstring covering: role in the engine, input/output semantics, and relevant
   tensor shapes. Non-obvious internal steps must have inline comments
   explaining the inference concept (prefill vs decode, KV cache updates,
   attention masking, RoPE rotation, `mx.eval()` placement).

---

## Tensor Shape Conventions

All shapes use these named dimensions consistently across all code, docstrings,
and comments. Never abbreviate differently in different files.

| Symbol | Meaning | Llama-3.2-1B value |
|---|---|---|
| `B` | batch size (always 1 in Phase 1) | 1 |
| `S` | sequence length (prompt length during prefill, 1 during decode) | varies |
| `T` | total tokens in KV cache so far (grows each decode step) | varies |
| `D` | model hidden dimension | 2048 |
| `H` | number of query attention heads | 32 |
| `Hkv` | number of key/value heads (GQA) | 8 |
| `Dh` | head dimension (`D // H`) | 64 |
| `V` | vocabulary size | 128256 |
| `L` | number of transformer layers | 16 |
| `I` | FFN intermediate size | 8192 |

Shape trace through one forward pass:

```
input_ids           (B, S)           integer token IDs
embeddings          (B, S, D)        after token embedding lookup
pre-attn input      (B, S, D)        after input RMSNorm
Q after proj        (B, S, H, Dh)    query heads — NOT transposed yet
K after proj        (B, S, Hkv, Dh)  key heads (fewer than Q in GQA)
V after proj        (B, S, Hkv, Dh)  value heads
Q after RoPE        (B, S, H, Dh)    rotated queries
K after RoPE        (B, S, Hkv, Dh)  rotated keys
Q transposed        (B, H, S, Dh)    ready for matmul
K from cache        (B, Hkv, T, Dh)  all past key positions
K expanded (GQA)    (B, H, T, Dh)    after repeating Hkv heads n_groups times
attn scores         (B, H, S, T)     Q @ K.T / sqrt(Dh)
attn weights        (B, H, S, T)     after causal mask + softmax
attn output         (B, H, S, Dh)    weights @ V
merged output       (B, S, D)        after head transpose + reshape
block output        (B, S, D)        after output projection + residual
logits              (B, S, V)        after final RMSNorm + lm_head
```

---

## Model Config (Llama-3.2-1B)

```python
d_model           = 2048
n_layers          = 16
n_heads           = 32
n_kv_heads        = 8       # GQA: n_groups = n_heads // n_kv_heads = 4
head_dim          = 64      # d_model // n_heads
intermediate_size = 8192
vocab_size        = 128256
max_seq_len       = 131072
rope_theta        = 500000.0
rms_norm_eps      = 1e-5
```

---

## Tiny Synthetic Test Fixture

All unit and shape tests use this config instead of loading real weights.
Tests requiring the real model are marked `@pytest.mark.slow` and skipped
unless `--run-slow` is passed to pytest.

```python
# tests/conftest.py
TINY_CONFIG = {
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,            # d_model // n_heads
    "intermediate_size": 128,
    "vocab_size": 256,
    "max_seq_len": 64,
    "rope_theta": 500000.0,
    "rms_norm_eps": 1e-5,
}
```

Weights are randomly initialised from this config using `mx.random.normal`.

---

## Package Layout

Phase 1 creates only the modules needed for M1.0–M1.8. The layout is kept
compatible with the full end-state structure from `docs/refined-plan_codex.md`.

```text
tiny_duo_infer/
  __init__.py
  engine.py           — Engine class, public API and generation orchestration
  cache.py            — static pre-allocated KV cache
  sampling.py         — greedy first; top-k/top-p/temperature in M1.8
  cli.py              — local text-generation CLI
  backends/
    __init__.py
    protocol.py       — draft backend Protocol (Phase 1 draft, enforced Phase 2)
    mlx_backend.py    — MLX tensor helpers and mx.eval() behavior
  models/
    __init__.py
    base.py           — Module ABC, Linear, Embedding
    llama.py          — LlamaBlock, LlamaModel assembly
  layers/
    __init__.py
    normalization.py  — RMSNorm
    rope.py           — frequency precomputation and Q/K rotation
    attention.py      — GQA attention, causal mask, KV-cache interaction
    feedforward.py    — SwiGLU FFN
  weights/
    __init__.py
    loader.py         — safetensors shard loading → mx.array dict
    llama_converter.py — HF key mapping and shape validation
  tokenizer/
    __init__.py
    loader.py         — tokenizers-based wrapper
scripts/
  benchmark.py        — tokens/sec and KV cache memory measurement
tests/
  conftest.py         — TINY_CONFIG fixture
  test_tokenizer.py
  test_weights.py
  test_layers.py
  test_model.py
  test_cache.py
  test_sampling.py
  test_engine.py
```

---

## Public Interfaces

### `Engine` (`engine.py`)

```python
class Engine:
    """
    Top-level inference engine for single-user local generation.

    Owns the model, tokenizer, and generation loop. All state required
    for one generation request lives here. Phase 1 supports one active
    request at a time.

    Usage:
        engine = Engine.from_model_path(Path("./models/llama-3.2-1b"))
        for token_text in engine.generate("Once upon a time", max_new_tokens=100):
            print(token_text, end="", flush=True)
    """

    @classmethod
    def from_model_path(
        cls,
        model_path: Path | str,
        max_seq_len: int = 2048,
    ) -> "Engine":
        """
        Load model weights and tokenizer from a local HuggingFace-compatible
        model directory.

        Args:
            model_path:  path to a directory containing config.json,
                         tokenizer.json, and safetensors weight shards.
            max_seq_len: maximum total sequence length (prompt + generated).
                         Must not exceed the model's RoPE context length.
        """

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Iterator[str]:
        """
        Tokenize the prompt, run prefill, then decode up to max_new_tokens.

        Yields one decoded text fragment per generated token. Each fragment
        may be a subword (e.g. "▁hel", "lo"). Callers can join fragments
        with "".join(engine.generate(...)) to get the full output string.

        Args:
            prompt:         input text string.
            max_new_tokens: maximum number of NEW tokens to generate
                            (does not count the prompt tokens).
            temperature:    divide logits by this before sampling.
                            1.0 = unchanged. Lower = sharper. 0.0 = greedy.
            top_k:          keep only top-k logits before sampling. 0 = off.
            top_p:          keep tokens summing to probability >= top_p. 1.0 = off.

        Yields:
            str: decoded text fragment for each new token, in order.
        """
```

### `Tokenizer` (`tokenizer/loader.py`)

```python
class Tokenizer:
    """
    Thin wrapper around the HuggingFace `tokenizers` package.

    Loads tokenizer.json and special token metadata from a local model
    directory. Exposes only the operations the engine needs. The `tokenizers`
    package is used at runtime; `transformers.AutoTokenizer` is dev/test only.
    """

    @classmethod
    def from_pretrained(cls, model_path: Path | str) -> "Tokenizer":
        """Load tokenizer.json and special token metadata from model_path."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Convert text to a list of integer token IDs."""

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert a list of token IDs back to a text string."""

    @property
    def bos_token_id(self) -> int:
        """Beginning-of-sequence token ID."""

    @property
    def eos_token_id(self) -> int:
        """End-of-sequence token ID. Generation stops when this is sampled."""

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
```

### `KVCache` (`cache.py`)

```python
class KVCache:
    """
    Pre-allocated static KV cache for single-request Phase 1 inference.

    Allocates one fixed-size (K, V) buffer per layer at construction time.
    During prefill, positions [0, prompt_len) are written. During each decode
    step, one new position is written at index current_len.

    Pre-allocation avoids the O(seq_len) copy overhead of growing by
    concatenation each step. The tradeoff is that max_seq_len must be known
    upfront (passed from Engine.from_model_path).

    Buffer shape per layer:
        keys:   (1, n_kv_heads, max_seq_len, head_dim)  — pre-allocated zeros
        values: (1, n_kv_heads, max_seq_len, head_dim)  — pre-allocated zeros

    Only the slice [:, :, :current_len, :] is valid at any point.
    """

    def __init__(self, n_layers: int, n_kv_heads: int, max_seq_len: int,
                 head_dim: int) -> None:
        """Allocate zeroed K/V buffers for all layers."""

    def update(
        self,
        layer_idx: int,
        new_k: mx.array,   # (1, n_kv_heads, new_len, head_dim)
        new_v: mx.array,   # (1, n_kv_heads, new_len, head_dim)
        position: int,     # first index to write — always equals current_len at
                           # the start of the current token step, passed in by
                           # the caller (LlamaModel). NOT read from current_len
                           # inside update() to avoid mid-forward-pass ambiguity.
    ) -> tuple[mx.array, mx.array]:
        """
        Write new_k/new_v into the pre-allocated buffer starting at `position`.
        Returns the valid K/V slice: [:, :, :position + new_len, :].

        Does NOT advance current_len. Call advance() once after all layers have
        processed the same token step.

        During prefill: position=0, new_len=prompt_len.
        During decode:  position=current_len, new_len=1.
        """

    def advance(self, n_tokens: int) -> None:
        """
        Advance current_len by n_tokens after all layers have written their
        K/V for the current token step.

        Called once per token step by the engine, NOT once per layer:
            model.forward(...)    # all 16 layers call update() at position p
            cache.advance(n)      # current_len += n, now equals p + n

        During prefill: advance(prompt_len).
        During decode:  advance(1).
        """

    @property
    def current_len(self) -> int:
        """
        Number of valid token positions in the cache.
        Reflects the state after the last advance() call.
        All 16 layers share this single value — it does not increment per layer.
        """

    def reset(self) -> None:
        """Zero out all buffers and reset current_len to 0. Call between requests."""
```

### Sampling (`sampling.py`)

```python
def greedy(logits: mx.array) -> int:
    """
    Return the token ID with the highest logit.
    logits shape: (vocab_size,) — single position only.
    """

def sample(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """
    Sample one token ID from logits.

    Order of operations (must be applied in this order):
      1. Temperature:  logits = logits / max(temperature, 1e-6)
      2. Top-k:        set logits outside top-k to -inf  (skip if top_k == 0)
      3. Top-p:        set logits outside nucleus to -inf (skip if top_p == 1.0)
      4. Softmax:      convert to probabilities
      5. Sample:       draw one token from the distribution

    Special cases:
      temperature=0.0  →  equivalent to greedy (argmax)
      top_k=1          →  equivalent to greedy
      top_k=0          →  top-k disabled
      top_p=1.0        →  top-p disabled

    logits shape: (vocab_size,) — single position only.
    Returns: int token ID.
    """
```

### Backend Protocol Draft (`backends/protocol.py`)

```python
from typing import Protocol
import numpy as np

class Backend(Protocol):
    """
    Defines the tensor operations a backend must provide.

    This is a Phase 1 draft written to capture the interface shape early.
    MLX code is not required to formally conform until Phase 2, when
    torch_backend.py is added and parity tests are introduced.

    Tier-1 ops (APIs differ across backends — must go through this protocol):
      softmax, silu, array, eval, to_numpy

    Tier-2 candidate ops (used directly in Phase 1 MLX code; portability across
    backends must be validated in M2.0 — NumPy, MLX, and PyTorch differ on split/
    concat naming, .T behaviour on rank > 2 tensors, dtype promotion, and device
    handling):
      @, .T, reshape, split, concatenate, sqrt, exp, cos, sin, arange, zeros, tril
    """

    def softmax(self, x: any, axis: int = -1) -> any:
        """Numerically stable softmax along `axis`."""

    def silu(self, x: any) -> any:
        """SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))."""

    def array(self, data: any, dtype: any = None) -> any:
        """Create a backend tensor from Python data or a NumPy array."""

    def eval(self, *arrays: any) -> None:
        """
        Materialise deferred computation (MLX lazy eval).
        No-op for eager backends (PyTorch).
        """

    def to_numpy(self, x: any) -> np.ndarray:
        """Convert a backend tensor to a NumPy array for CPU-side processing."""
```

---

## Layer Interfaces

All layers are plain Python classes with a `__call__` method. Weights are
stored as plain `mx.array` attributes, populated by `load_weights(flat_dict)`.
The base `Module` class in `models/base.py` provides the `load_weights` logic.

### `RMSNorm` (`layers/normalization.py`)

```python
class RMSNorm:
    """
    Root Mean Square Layer Normalization.

    Formula: y = x / sqrt(mean(x^2) + eps) * weight

    Unlike LayerNorm, RMSNorm has no mean subtraction and no bias term.
    Llama uses pre-norm: RMSNorm is applied BEFORE attention and FFN,
    not after. Residual connections bypass the norm entirely.

    Attributes:
        weight: (D,) scale parameter, initialised to ones in HF checkpoint.
        eps:    small constant for numerical stability (default 1e-5).
    """

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, S, D) -> (B, S, D)"""
```

### `RoPE` (`layers/rope.py`)

```python
def precompute_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float,
) -> tuple[mx.array, mx.array]:
    """
    Precompute RoPE cosine and sine tables.

    Returns (cos_table, sin_table), each of shape (max_seq_len, head_dim // 2).
    Called once at model init; the tables are stored and reused every forward pass.

    Frequency formula: freq_i = 1 / (theta ^ (2i / head_dim))
    for i in 0 .. head_dim // 2

    The result is indexed by absolute position during apply_rope.
    """

def apply_rope(
    x: mx.array,             # (B, S, H, Dh) — Q or K, before head transpose
    cos: mx.array,           # (max_seq_len, Dh // 2) cosine table
    sin: mx.array,           # (max_seq_len, Dh // 2) sine table
    offset: int = 0,         # position offset:
                             #   prefill → 0
                             #   decode  → prompt_len + tokens_generated_so_far
) -> mx.array:
    """
    Compute and return rotary positional embeddings applied to Q or K.

    Splits each head vector into consecutive pairs (x0, x1), then rotates:
        x0' = x0 * cos[pos] - x1 * sin[pos]
        x1' = x0 * sin[pos] + x1 * cos[pos]

    The `offset` ensures that decode steps use the correct absolute position.
    Without it, every decode step would encode position 0, breaking the model.

    Returns same shape as input: (B, S, H, Dh).
    """
```

### `LlamaAttention` (`layers/attention.py`)

```python
class LlamaAttention:
    """
    Grouped Query Attention (GQA) with RoPE position encoding.

    GQA uses fewer KV heads than Q heads to reduce the KV cache size.
    n_groups = n_heads // n_kv_heads = 4 for Llama-3.2-1B.

    Before the attention matmul, each of the n_kv_heads KV heads is repeated
    n_groups times so the shapes align with Q:
        k_expanded = mx.repeat(k, repeats=n_groups, axis=1)  # (B, H, T, Dh)
    K from cache has shape (B, Hkv, T, Dh). The KV-head axis is axis=1.
    axis=2 would repeat sequence positions — a silent shape bug.

    Weights (all stored as (out_dim, in_dim), applied as x @ weight.T):
        q_proj: (H * Dh, D)    = (2048, 2048)
        k_proj: (Hkv * Dh, D)  = (512, 2048)
        v_proj: (Hkv * Dh, D)  = (512, 2048)
        o_proj: (D, H * Dh)    = (2048, 2048)
    """

    def __call__(
        self,
        x: mx.array,         # (B, S, D)
        cache: KVCache,
        layer_idx: int,
        position_offset: int,
        cos: mx.array,       # (max_seq_len, Dh // 2) from precompute_freqs
        sin: mx.array,       # (max_seq_len, Dh // 2) from precompute_freqs
    ) -> mx.array:
        """
        Compute GQA attention and update the KV cache.

        During prefill: S = prompt_len, causal mask applied to scores.
        During decode:  S = 1, no mask needed (single query attends to all past).

        Returns (B, S, D).
        """
```

### `SwiGLUFFN` (`layers/feedforward.py`)

```python
class SwiGLUFFN:
    """
    SwiGLU feed-forward network used in Llama.

    Uses three linear projections instead of the standard two:
        gate_proj: (I, D)  projects input to gate activations
        up_proj:   (I, D)  projects input to value features
        down_proj: (D, I)  projects gated features back to model dim

    Forward pass:
        gate = x @ gate_proj.T             # (B, S, I)
        up   = x @ up_proj.T               # (B, S, I)
        out  = (silu(gate) * up) @ down_proj.T  # (B, S, D)

    The gate controls which features pass through (learned gating mechanism).
    SiLU is implemented as: x * (1 / (1 + exp(-x))).
    Do NOT use mlx.nn.SiLU, mlx.nn.GELU, or any other high-level activation.
    """

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, S, D) -> (B, S, D)"""
```

---

## Weight Loading

### HuggingFace Key Mapping (`weights/llama_converter.py`)

Complete mapping from HF `state_dict` keys to project flat names.
`{i}` is the zero-indexed layer number (0 to `n_layers - 1`).

| HF key | Project key |
|---|---|
| `model.embed_tokens.weight` | `embed_tokens.weight` |
| `model.layers.{i}.input_layernorm.weight` | `layers.{i}.input_norm.weight` |
| `model.layers.{i}.self_attn.q_proj.weight` | `layers.{i}.attn.q_proj.weight` |
| `model.layers.{i}.self_attn.k_proj.weight` | `layers.{i}.attn.k_proj.weight` |
| `model.layers.{i}.self_attn.v_proj.weight` | `layers.{i}.attn.v_proj.weight` |
| `model.layers.{i}.self_attn.o_proj.weight` | `layers.{i}.attn.o_proj.weight` |
| `model.layers.{i}.post_attention_layernorm.weight` | `layers.{i}.post_attn_norm.weight` |
| `model.layers.{i}.mlp.gate_proj.weight` | `layers.{i}.ffn.gate_proj.weight` |
| `model.layers.{i}.mlp.up_proj.weight` | `layers.{i}.ffn.up_proj.weight` |
| `model.layers.{i}.mlp.down_proj.weight` | `layers.{i}.ffn.down_proj.weight` |
| `model.norm.weight` | `final_norm.weight` |
| `lm_head.weight` | `lm_head.weight` |

**Tied embeddings:** Llama-3.2-1B ties `lm_head.weight` to `embed_tokens.weight`
(they share the same tensor). The HF checkpoint may not include a separate
`lm_head.weight` key. The converter must handle this by reusing
`embed_tokens.weight` for `lm_head.weight` when the key is absent — not raise
a missing-key error.

**Weight layout:** all weights are stored as `(out_dim, in_dim)` in both HF
and our project. The forward pass uses `x @ weight.T`. No transposition is
needed during loading.

**dtype:** load all weights as `mx.array` in their native bfloat16. Do not
convert to float32 — running in bfloat16 matches HF's default behavior and
keeps memory usage low.

### Validation requirements

`llama_converter.py` must:
1. Report any HF keys not mapped (unexpected keys — usually harmless metadata)
2. Report any expected project keys not found (missing keys — likely a bug)
3. Assert each tensor's shape matches the expected shape from config
4. Raise a clear error if the directory does not appear to be a Llama-3.2-1B
   checkpoint (e.g., wrong `model_type` in `config.json`)

---

## MLX-Specific Behavior

### Lazy evaluation and `mx.eval()` placement

MLX defers all computation until an array is explicitly materialised. The
engine must call `mx.eval()` once per decode step at the boundary between
GPU computation and CPU sampling:

```python
# Correct placement — inside the decode loop:
logits = model(input_ids, cache, position_offset)   # schedules computation, no GPU work yet
mx.eval(logits)                                      # flushes the MLX computation graph
next_token = sample(logits[0, 0, :])                 # safe to read on CPU after eval
```

Do **not** call `mx.eval()` inside individual layers — it forces unnecessary
GPU/CPU synchronisation on every layer and drastically slows generation.

Call `mx.eval()` on the KV cache buffers after the prefill step to ensure all
positions are materialised before the first decode step begins.

### dtype

All computations run in bfloat16 (the native weight dtype). Input `token_ids`
are integer arrays. After `mx.eval()`, sampling uses `mx.argmax` (stays on
device) or converts to NumPy via `np.array(logits)` for multinomial sampling.

---

## Milestones

### M1.0 — Project Scaffolding

Files to create: `pyproject.toml`, `tiny_duo_infer/__init__.py`,
`.gitignore` (exclude `__pycache__/`, `.venv/`, `models/`, `*.safetensors`)

Done when:
- `uv sync --group dev` completes without error
- `uv run python -c "import tiny_duo_infer"` succeeds
- `uv run pytest` collects 0 tests and exits cleanly

---

### M1.1 — Tokenizer and Config Loading

Files to create: `tiny_duo_infer/tokenizer/__init__.py`,
`tiny_duo_infer/tokenizer/loader.py`, `tiny_duo_infer/config.py`
(reads `config.json`, returns a dataclass or dict of model hyperparameters)

Done when:
- `Tokenizer.from_pretrained("./models/llama-3.2-1b")` loads without error
- `tokenizer.encode("Hello world")` returns a non-empty `list[int]`
- `tokenizer.decode(tokenizer.encode("Hello world"))` returns `"Hello world"`
- `tokenizer.bos_token_id` and `tokenizer.eos_token_id` return valid ints
- `tests/test_tokenizer.py` passes (no real model needed — mock the file reads)
- Optional `@pytest.mark.slow`: `AutoTokenizer` parity check for 5 prompts

---

### M1.2 — Weight Loading

Files to create: `tiny_duo_infer/weights/__init__.py`,
`tiny_duo_infer/weights/loader.py`, `tiny_duo_infer/weights/llama_converter.py`

Done when:
- All Llama-3.2-1B weight tensors load without error
- Tied `lm_head` / `embed_tokens` handled correctly (no missing-key error)
- Each tensor has the expected shape (verified against the config numbers above)
  and dtype `bfloat16`
- `tests/test_weights.py` passes using a synthetic weight dict (no download needed)

---

### M1.3 — Layer Implementations

Files to create: `tiny_duo_infer/models/__init__.py`,
`tiny_duo_infer/models/base.py`, `tiny_duo_infer/layers/__init__.py`,
`tiny_duo_infer/layers/normalization.py`, `tiny_duo_infer/layers/rope.py`,
`tiny_duo_infer/layers/attention.py`, `tiny_duo_infer/layers/feedforward.py`

Test each layer in isolation using `TINY_CONFIG` before assembling the model.

Done when:
- `RMSNorm`: output shape `(B, S, D)` matches input; manual formula check passes
- `precompute_freqs`: returns `(cos, sin)` each `(max_seq_len, Dh // 2)`
- `apply_rope`: output shape unchanged; applying with `offset=0` and verifying
  against the manual rotation formula passes
- `LlamaAttention`: output shape `(B, S, D)` correct for both prefill (`S > 1`)
  and decode (`S = 1`); KV buffers are written at the expected layer/position;
  `current_len` is advanced by the model/engine once per prefill/decode step,
  not by the attention layer itself
- `SwiGLUFFN`: output shape `(B, S, D)` correct; gate and up projections
  are separate calls (not weight-shared)
- `tests/test_layers.py` passes with `TINY_CONFIG`

---

### M1.4 — Model Forward Pass

Files to create: `tiny_duo_infer/models/llama.py`, `tiny_duo_infer/cache.py`,
`tests/conftest.py` (TINY_CONFIG + random weight initialisation helper)

`LlamaModel.__call__(input_ids, cache, position_offset)` must:
1. Embed `input_ids` → `(B, S, D)`
2. For each of `n_layers` blocks: input RMSNorm → attention → residual →
   post-attention RMSNorm → FFN → residual
3. Apply final RMSNorm
4. Project to vocabulary: `logits = embeddings @ lm_head.weight.T` → `(B, S, V)`

Done when:
- `model(input_ids, cache, position_offset=0)` returns `(1, S, V)` with `TINY_CONFIG`
- All `(B, S, D)` shapes are preserved through residual connections
- KVCache `current_len` equals `S` after one prefill call
- `tests/test_model.py` shape tests pass
- Optional `@pytest.mark.slow`: logits compared against HF `transformers`
  reference; any numerical differences documented (not a hard gate)

---

### M1.5 — Prefill

Location: `tiny_duo_infer/engine.py`

Prefill sequence inside `Engine`:
1. `tokens = tokenizer.encode(prompt)` — encode prompt to token IDs
2. Prepend `bos_token_id` if not already the first token
3. `input_ids = mx.array([tokens])` → `(1, prompt_len)`
4. `cache = KVCache(n_layers, n_kv_heads, max_seq_len, head_dim)` — allocate buffers
5. `logits = model(input_ids, cache, position_offset=0)` → `(1, prompt_len, V)`
6. `mx.eval(logits)` — flush computation graph before sampling
7. Hold `logits[0, -1, :]` as the input for the first decode step

Done when:
- Prefill returns single-position logits of shape `(V,)`
- `cache.current_len == prompt_len` after prefill
- `tests/test_engine.py` prefill state tests pass with `TINY_CONFIG`

---

### M1.6 — Decode Loop

Files to create/update: `tiny_duo_infer/sampling.py` (greedy only),
`tiny_duo_infer/engine.py` (full loop), `tiny_duo_infer/cli.py`

One decode step:
1. `input_ids = mx.array([[next_token_id]])` → `(1, 1)`
2. `position_offset = cache.current_len`
3. `logits = model(input_ids, cache, position_offset)` → `(1, 1, V)`
4. `mx.eval(logits)`
5. `next_token = greedy(logits[0, 0, :])` (until M1.8)
6. Stop if `next_token == eos_token_id` or `len(generated) >= max_new_tokens`
7. `yield tokenizer.decode([next_token])`

CLI:
```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/llama-3.2-1b \
  --prompt "Once upon a time" \
  --max-new-tokens 50
```

Done when:
- `engine.generate("Once upon a time")` yields text fragments without error
- Generation stops at `max_new_tokens`
- Generation stops when EOS token is sampled
- Greedy decoding is deterministic: same prompt → identical output on repeated runs
- CLI produces output on Apple Silicon
- `tests/test_engine.py` decode state tests pass with `TINY_CONFIG`

---

### M1.7 — MLX Evaluation and Baseline Metrics

Files to create: `scripts/benchmark.py`

Done when:
- `mx.eval()` call site in the decode loop has an inline comment explaining
  lazy evaluation and why eval is placed at this boundary
- `mx.eval()` on the KV cache after prefill is documented similarly
- `scripts/benchmark.py` generates 100 tokens with greedy decoding and prints
  tokens/sec
- KV cache memory usage is documented for at least one sequence length using:
  `2 × L × Hkv × T × Dh × bytes_per_element`
  (2 for K and V, 16 layers, 8 KV heads, T tokens, 64 head_dim, 2 bytes for bfloat16)

---

### M1.8 — Sampling Strategies *(Phase 1 extension — M1.0–M1.7 define minimum completion)*

Location: `tiny_duo_infer/sampling.py` — extend `sample()`

**Temperature:** `logits = logits / max(temperature, 1e-6)`. At `temperature=0`
the division returns very large values — use argmax (greedy) directly instead.

**Top-k:** sort logits descending, keep top-k, set the rest to `-inf`.
`top_k=0` disables this step.

**Top-p (nucleus):** sort logits descending, compute cumulative softmax
probabilities, keep the smallest prefix of tokens whose cumulative probability
reaches or exceeds `top_p` — the token that crosses the threshold is kept, all
tokens after it are set to `-inf`. This ensures at least one token is always
kept. `top_p=1.0` disables this step.

**Order:** temperature → top-k → top-p → softmax → sample. Changing this
order produces different distributions; the specified order is the standard.

Done when:
- `temperature=0.0` and `top_k=1` each reproduce greedy output exactly
- `top_p=1.0, temperature=1.0, top_k=0` produces varied output across calls
- Fixed seed test: `mx.random.seed(42)` → same sampled token for same logits
- `tests/test_sampling.py` passes all four cases above

---

## Testing Requirements

### Unit tests (no model download required)

| Test file | Coverage |
|---|---|
| `test_tokenizer.py` | encode/decode round-trips, BOS/EOS IDs, special tokens |
| `test_weights.py` | key mapping correctness, shape validation, tied embedding |
| `test_layers.py` | RMSNorm formula, RoPE shape/invertibility, attention shapes, causal mask, GQA head expansion, FFN gate/up separation |
| `test_cache.py` | pre-allocated buffer shape, update writes correct position, current_len increments, reset clears |
| `test_model.py` | full forward shape `(1, S, V)`, residuals preserve shape |
| `test_sampling.py` | greedy determinism, temperature=0 matches greedy, top_k=1 matches greedy, seeded sampling determinism |
| `test_engine.py` | prefill sets cache length, decode increments cache, EOS stops generation, max_new_tokens respected |

### Smoke tests (`@pytest.mark.slow` — require real model artifacts)

- Local Llama-3.2-1B artifacts load without error
- `engine.generate("...")` produces text on MLX without crashing
- `max_new_tokens` limit is respected
- EOS token stops generation
- Greedy decoding is deterministic (two identical calls produce identical output)
- Generated token IDs decode to a non-empty string

Smoke tests must **not** assert a specific output token (e.g., `" Paris"`) —
base model output is too sensitive to config details to make semantic assertions.

### HF parity checks (optional, `@pytest.mark.slow`)

- Load the same weights into HF `transformers` LlamaForCausalLM
- Run the same prompt through both models
- Record `max(abs(our_logits - hf_logits))` across all positions
- Document known sources of mismatch (bfloat16 accumulation, RoPE
  implementation details, attention mask boundary)
- This is a debugging tool, not a milestone gate

---

## Acceptance Criteria

Phase 1 is complete when all of the following are true:

1. `uv sync --group dev && uv run python -c "import tiny_duo_infer"` succeeds
2. Local `meta-llama/Llama-3.2-1B` weights load via `Engine.from_model_path()`
3. `engine.generate(prompt, max_new_tokens=50)` yields tokens on Apple Silicon
4. The CLI produces text from a prompt
5. Prefill and decode are explicitly separated in code and both have teaching comments
6. The KV cache is pre-allocated and updated position by position during decode
7. Greedy sampling works end-to-end (M1.6); temperature, top-k, and top-p
   work if M1.8 is completed (not required for minimum Phase 1 completion)
8. All unit tests pass with `TINY_CONFIG` (no model download)
9. Smoke test passes: generation completes, `max_new_tokens` respected, EOS stops
   generation, greedy is deterministic
10. All public classes and functions have docstrings including tensor shapes
11. `mx.eval()` placement is documented inline
12. Baseline tokens/sec and KV cache memory estimate are recorded
13. Phase handoff document is complete

---

## Known Limitations (Deferred to Later Phases)

- No formal `backends/protocol.py` conformance — Phase 2 validates this
- No PyTorch/CUDA backend — Phase 2
- KV cache is pre-allocated per request and cleared between requests —
  PagedAttention and shared pool are Phase 3
- No instruct/chat-template support
- No multi-user scheduling or batching — Phase 3
- `mx.eval()` called once per decode step; further optimisation (graph
  compilation, `mx.compile`) is a Phase 2+ concern

---

## Handoff Requirements

The Phase 1 handoff document must include:

- **Implemented milestones** — M1.0 through M1.7 minimum; M1.8 if completed
- **Files changed** — path and one-line purpose for each
- **Public API usage example** — working `Engine.from_model_path` + `generate` snippet
- **Tests run** — command, pass/fail/skip count, reason for any skips
- **Model artifacts** — local path used for smoke testing
- **Environment details** — macOS version, chip, Python version, MLX version,
  available unified memory
- **Known gaps** — anything incomplete, deferred, or behaving unexpectedly
- **Learning notes** — code paths that deserve careful line-by-line reading
