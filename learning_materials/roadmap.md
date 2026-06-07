# Learning Roadmap

This document is a guided reading order for the completed Phase 1, Phase 1.5,
Phase 1.6, and Phase 1.7 engine, plus the Phase 1.8 weight-only quantization
work that builds on top of it. Use it to study how local Llama inference works
first, then how the same engine was extended to Qwen3-0.6B, then how
generation UX and HTTP serving were added, then how the engine was
instrumented with observability, and finally how MLX-native INT4/INT8 weight-only
quantization fits into the same single-request runtime without changing the
prefill/decode flow.

Phase 1 is intentionally learning-first: the goal is not to hide inference
behind a library, but to make the control flow, tensor shapes, KV-cache writes,
and sampling choices visible. Phase 1.5 adds a second model family so the next
lesson is model portability: which pieces are engine-generic, and which pieces
belong to a specific model architecture. Phase 1.6 adds request boundaries and
serving structure. Phase 1.7 adds observability: how to measure what the engine
is actually doing. Phase 1.8 adds weight-only quantization: how compressed
linear weights, fused quantized matmul, and memory accounting plug into the
same `Linear` abstraction.

## How To Read

For each step:

1. Read the implementation files.
2. Read the matching tests.
3. Write down inputs, outputs, invariants, and failure cases.
4. Run the targeted tests.
5. Explain the code back in your own words.

Do not skip tests. The tests are part of the teaching material: they show which
edge cases each component is responsible for.

## Roadmap

### 1. Project Scaffold And Scope

Read:

- `pyproject.toml`
- `docs/phases/phase-1-mlx-single-user.md`
- `docs/phases/phase-1-taskboard.md`
- `docs/phases/phase-1.5-qwen3-mlx.md`
- `docs/phases/phase-1.5-taskboard.md`
- `tests/conftest.py`

Learn:

- Why Phase 1 is MLX-only.
- Why Phase 1.5 adds model-family portability before backend portability.
- Why `transformers` is dev/test only.
- Why tests use tiny Llama and Qwen3 configs instead of real weights.
- Which tasks are required for Phase 1 and Phase 1.5 completion.

Try:

- Run `uv run pytest`.
- Explain why slow tests are skipped by default.

### 2. Config Loader

Read:

- `tiny_duo_infer/config.py`
- `tests/test_config.py`

Learn:

- How Hugging Face `config.json` maps into `ModelConfig`.
- Why the loader accepts only explicitly supported model families.
- Why Llama can derive `head_dim` from `hidden_size // num_attention_heads`.
- Why Qwen3 stores `head_dim` explicitly and can have
  `n_heads * head_dim != d_model`.
- Why `num_attention_heads % num_key_value_heads == 0` matters for GQA.

Try:

- Hand-write a tiny `config.json`.
- Predict `head_dim` and `n_groups`.
- Break one config field and predict the error.

### 3. Tokenizer Wrapper

Read:

- `tiny_duo_infer/tokenizer/loader.py`
- `tests/test_tokenizer.py`

Learn:

- Why runtime uses `tokenizers`, not `AutoTokenizer`.
- How `tokenizer.json`, `tokenizer_config.json`, and sometimes `config.json`
  work together.
- How BOS/EOS IDs are resolved from integer fields, token strings, or
- AddedToken dictionaries.
- Why Qwen3 resolves BOS from `config.json` while EOS comes from tokenizer
  metadata.
- Why registered special tokens are required for `skip_special_tokens=True`.
- Why Qwen3 plain prompt mode does not synthesize a BOS token when
  `add_bos_token=false`.

Try:

- Trace `Tokenizer.from_pretrained()` step by step.
- Explain when `add_special_tokens=True` prepends BOS.
- Explain how generated token IDs become text fragments.

### 4. Safetensors Loader

Read:

- `tiny_duo_infer/weights/loader.py`
- `tests/test_weights.py` loader tests

Learn:

- How single-file and sharded checkpoints are discovered.
- Why `model.safetensors.index.json` is treated as authoritative.
- Why duplicate tensor keys are rejected.
- Why the loader preserves raw Hugging Face key names.
- Why Phase 1 uses `mx.load()` for real safetensors files: the real
  Llama-3.2-1B smoke test exposed that this path preserves MLX bfloat16 tensors.

Try:

- Draw the flow: model directory -> shard file paths -> raw HF weight dict.
- Explain why key conversion is not done in this module.
- Run `uv run pytest tests/test_weights.py`.

### 5. Weight Converter

Read:

- `tiny_duo_infer/weights/llama_converter.py`
- `tiny_duo_infer/weights/qwen3_converter.py`
- `tests/test_weights.py` converter tests

Learn:

- How HF keys map to project module keys.
- Which tensor shapes are expected for Q, K, V, O, FFN, norms, embeddings, and
  LM head.
- How tied embeddings are represented by reusing the same array object.
- Why Qwen3 requires a separate `lm_head.weight` even though its config
  advertises tied embeddings.
- Why Qwen3 converter validates `q_norm.weight` and `k_norm.weight` as
  `(head_dim,)`.
- Why missing keys fail, but unexpected keys warn and are ignored.

Try:

- List all expected project keys for a two-layer tiny config.
- Explain why shape validation belongs before model assembly.

### 6. Minimal Module System

Read:

- `tiny_duo_infer/models/base.py`
- `tests/test_model.py` module, linear, and embedding tests

Learn:

- Why the project does not subclass `mlx.nn.Module`.
- How dotted-path `load_weights()` routing works.
- Why `Linear.forward()` uses `x @ weight.T`.
- Why `Embedding.forward()` is direct matrix row lookup.

Try:

- Trace `load_weights({"attn.q_proj.weight": w})` through nested modules.
- Verify the matrix shape convention by hand.

### 7. KV Cache

Read:

- `tiny_duo_infer/cache.py`
- `tests/test_cache.py`

Learn:

- Why the cache is pre-allocated per layer.
- Why the buffer shape is `(1, n_kv_heads, max_seq_len, head_dim)`.
- Why `update()` writes but does not advance `current_len`.
- Why `advance()` is called once per token step, not once per layer.
- Why `KVCache.eval()` exists: MLX cache writes must be materialized at engine
  boundaries before later decode steps read them.

Try:

- Simulate one prefill of 3 tokens and two decode steps on paper.
- Track `position`, `new_len`, returned slice shape, and `current_len`.

### 8. RMSNorm And RoPE

Read:

- `tiny_duo_infer/layers/normalization.py`
- `tiny_duo_infer/layers/rope.py`
- `tests/test_layers.py` normalization, Q/K norm, and RoPE tests

Learn:

- RMSNorm formula: `x * rsqrt(mean(x^2) + eps) * weight`.
- Why RMSNorm does not subtract the mean.
- How RoPE precomputes frequency tables.
- How even/odd head-dimension pairs are rotated.
- Why decode uses an absolute `position_offset`.
- Why Qwen3 applies Q/K RMSNorm after projection and head reshape, before
  RoPE.

Try:

- Work through a two-dimensional RoPE rotation manually.
- Explain why RoPE is applied to Q and K, not V.

### 9. GQA Attention

Read:

- `tiny_duo_infer/layers/attention.py`
- `tests/test_layers.py` attention tests

Learn:

- How Llama projections reshape directly to `(B, S, H, Dh)` and
  `(B, S, Hkv, Dh)` because `H * Dh == D`.
- How Qwen3 first projects Q to attention width `A = H * Dh`, where `A` can
  differ from hidden size `D`.
- Why Llama-3.2-1B has fewer KV heads than query heads.
- Why Qwen3 still uses the same GQA and KV-cache protocol even though its
  attention width differs from `d_model`.
- How GQA repeats KV heads along the head axis.
- How attention writes the current step into the KV cache and reads back the
  full valid prefix.
- How causal masking works during prefill and decode.

Try:

- For `H=4` and `Hkv=2`, show which query heads share each KV head.
- Explain why cache position is passed into attention instead of read from
  `cache.current_len`.

### 10. SwiGLU Feed-Forward Layer

Read:

- `tiny_duo_infer/layers/feedforward.py`
- `tests/test_layers.py` FFN tests

Learn:

- How Llama's FFN differs from a plain MLP.
- Why there are separate gate, up, and down projections.
- How `silu(gate) * up` controls information flow.

Try:

- Trace shapes through `gate_proj`, `up_proj`, elementwise multiply, and
  `down_proj`.

### 11. Llama Model Assembly

Read:

- `tiny_duo_infer/models/llama.py`
- `tests/test_model.py` LlamaBlock and LlamaModel tests

Learn:

- How embeddings, blocks, final norm, and LM head compose into logits.
- Why Llama uses pre-norm residual blocks.
- Why `LlamaModel.forward()` never calls `cache.advance()`.
- How list-based layer weight routing differs from normal dotted attributes.

Try:

- Trace one forward pass for input IDs shaped `(1, S)`.
- Explain why logits have shape `(B, S, V)`.

### 12. Qwen3 Model Portability

Read:

- `docs/phases/phase-1.5-qwen3-mlx.md`
- `tiny_duo_infer/models/qwen3.py`
- `tiny_duo_infer/layers/attention.py` `Qwen3Attention`
- `tiny_duo_infer/weights/qwen3_converter.py`
- `tests/test_config.py` Qwen3 tests
- `tests/test_layers.py` Qwen3 attention tests
- `tests/test_model.py` Qwen3 model tests
- `tests/test_engine.py` model dispatch tests
- `tests/test_tokenizer.py` Qwen3 tokenizer tests

Learn:

- Which parts stayed engine-generic: `Engine.generate()`, `KVCache`,
  sampling, CLI argument shape, and the model forward signature.
- Which parts are model-family-specific: config validation, attention class,
  model assembly class, weight converter, and tokenizer metadata quirks.
- Why Phase 1.5 uses explicit `Qwen3Block` / `Qwen3Model` instead of hiding
  Qwen3 behavior behind conditionals in `LlamaBlock`.
- Why Qwen3 attention width `A = H * Dh` can be different from hidden size
  `D`.
- Why Q/K RMSNorm belongs between head reshape and RoPE.
- Why `Engine.from_model_path()` dispatches by `config.model_type`, while the
  prefill and decode loops stay unchanged.

Try:

- Compare `LlamaAttention` and `Qwen3Attention` line by line.
- For the tiny Qwen3 fixture, compute `A = H * Dh` and explain why `A != D`.
- Trace how `model.layers.0.self_attn.q_norm.weight` becomes
  `layers.0.attn.q_norm.weight`.
- Run `QWEN_MODEL_PATH=models/qwen3-0.6b uv run pytest --run-slow -k qwen3`
  if local Qwen3 artifacts are available.

### 13. Engine Prefill

Read:

- `tiny_duo_infer/engine.py` prefill methods
- `tests/test_engine.py` prefill tests

Learn:

- How text is tokenized before prefill.
- Why prefill runs the whole prompt in one model forward.
- Why only final-position logits are returned for sampling.
- Why cache length is advanced once after all layers write.
- Why prefill evaluates both final logits and cache buffers.

Try:

- Trace `engine.prefill_token_ids([a, b, c])`.
- Explain what is valid in the cache before and after `cache.advance(3)`.

### 14. Decode Loop

Read:

- `tiny_duo_infer/engine.py` `generate()`
- `tests/test_engine.py` decode tests

Learn:

- How the first generated token comes from prefill logits.
- Why each decode model call receives input IDs shaped `(1, 1)`.
- Why `position_offset` equals `cache.current_len`.
- Why EOS is checked before yielding.
- Why the loop skips an unused decode forward after the final yielded token.
- Why cache buffers are evaluated before the next decode step.

Try:

- Trace `generate(prompt, max_new_tokens=4, temperature=0.0)` on paper.
- Count model calls, cache advances, yielded fragments, and eval calls.

### 15. Sampling

Read:

- `tiny_duo_infer/sampling.py`
- `tests/test_sampling.py`
- `tests/test_engine.py` greedy-generation tests

Learn:

- Why `temperature=0.0` short-circuits to greedy argmax.
- Why invalid sampling parameters raise `ValueError`.
- How top-k masks logits outside the highest-k set.
- How top-p keeps the smallest probability prefix that crosses the threshold.
- Why `mx.random.categorical()` can sample from raw logits.
- How fixed seeds make sampling tests deterministic.

Try:

- For logits `[1, 2, 3]`, compute greedy output.
- For a simple distribution, identify the top-p nucleus by hand.

### 16. CLI And Chat Templating

Read:

- `tiny_duo_infer/cli.py`
- `tiny_duo_infer/prompt.py`
- `tests/test_cli.py`
- `tests/test_prompt.py`

Learn:

- Why the CLI is a thin wrapper over `Engine`.
- How argument parsing maps to `Engine.from_model_path()` and
  `Engine.generate_request()`.
- Why CLI tests use a fake engine instead of real model artifacts.
- Why users do not pass `--model-type`; model family is inferred from
  `config.json`.
- How `--chat`, `--message ROLE:CONTENT`, and the HTTP `messages: [...]`
  field share the same `format_chat_prompt()` formatter in
  `tiny_duo_infer/prompt.py`. Qwen3 uses ChatML
  (`<|im_start|>role\n…<|im_end|>\n` with an open final
  `<|im_start|>assistant\n` suffix); Llama is a base model and
  `format_chat_prompt(..., model_type="llama")` raises a clear `ValueError`
  pointing callers at plain `--prompt` mode.
- The full ChatML protocol — including why the assistant suffix is left open,
  how special tokens are registered, and which validation runs before
  templating — is covered in
  `learning_materials/deep_dives/chat_templating.md`.

Try:

- Run `uv run python -m tiny_duo_infer.cli --help` and read the chat,
  context-policy, and quantization flag groups.
- Explain why `--temperature 0.0` gives deterministic greedy output.
- Run a short Qwen3 smoke if artifacts exist:
  `uv run python -m tiny_duo_infer.cli --model-path models/qwen3-0.6b --prompt "The capital of France is" --max-new-tokens 8 --temperature 0.0`.
- Run a short Qwen3 chat-mode smoke:
  `uv run python -m tiny_duo_infer.cli --model-path models/qwen3-0.6b --message system:Be\ concise. --message user:What\ is\ 2+2? --max-new-tokens 16 --temperature 0.0`.
- Run `uv run pytest tests/test_prompt.py -v` and read the exact ChatML
  string assertions (especially the open-assistant-suffix lock).

See also:

- `learning_materials/deep_dives/inference_worker.md` — the single-thread
  MLX worker (`tiny_duo_infer/serving/worker.py`) that the HTTP server in
  `tiny_duo_infer/serving/api.py` uses to run the same chat-formatted
  prompts behind FastAPI without violating MLX GPU-stream thread affinity.

### 17. MLX Eval Placement

Read:

- `tiny_duo_infer/engine.py` eval comments
- `tiny_duo_infer/cache.py` `KVCache.eval()`
- `tiny_duo_infer/backends/mlx_backend.py`
- `tests/test_engine.py` eval-placement test

Learn:

- MLX is lazy: tensor work is queued until `mx.eval()` materializes arrays.
- Eval happens at engine boundaries, not inside layers.
- Prefill evaluates final logits for sampling and cache buffers for decode.
- Decode evaluates logits for sampling and cache buffers for the next token.

Try:

- Explain why adding `mx.eval()` inside attention would be slower and harder to
  reason about.

### 18. Benchmark And KV Memory

Read:

- `scripts/benchmark.py`
- `docs/phases/phase-1-handoff.md`

Learn:

- How benchmark timing drains the `generate()` iterator.
- Why the reported throughput includes prefill plus decode.
- KV memory formula:
  `2 * n_layers * n_kv_heads * seq_len * head_dim * bytes_per_element`.
- Why Llama-3.2-1B at `T=1024` uses about 32 MB of KV cache in bfloat16.
- Why Qwen3-0.6B uses a different KV memory slope:
  `L=28`, `Hkv=8`, `Dh=128`, so `T=1024` is about 112 MB in bfloat16.
- How `generation.kv_cache_bytes()` is the canonical formula — `benchmark.py`
  imports it rather than duplicating the math.

Try:

- Run `uv run python scripts/benchmark.py --help`.
- Compute KV memory for `T=2048` by hand and compare to the script.
- Run Qwen3 benchmark if artifacts exist:
  `uv run python scripts/benchmark.py --model-path models/qwen3-0.6b --n-tokens 20 --max-seq-len 256 --show-output`.

### 19. Generation Stats And Observability

Read:

- `tiny_duo_infer/generation.py` — `GenerationStats` dataclass, `kv_cache_bytes()`
- `tiny_duo_infer/engine.py` — timing instrumentation with `perf_counter()`
- `tests/test_generation.py` — `GenerationStats` invariant tests
- `tests/test_engine.py` — stats populated on every stop reason

Learn:

- What `GenerationStats` captures: context-policy accounting, token counts,
  timing (prompt_prepare, prefill, TTFT, decode, total), decode throughput,
  KV bytes (allocated and active), and sequence lengths.
- Why `prompt_tokens == accepted_prompt_tokens` is an invariant enforced at
  construction (API stability: callers can rely on this equality).
- Why `active_seq_len == accepted_prompt_tokens + generated_tokens`.
- How timing is measured: `time.perf_counter()` at engine boundaries, not
  inside layers.
- What TTFT means and why it is not the same as prefill latency.
- Why `decode_step_ms` is left empty by default (profiling-only detail that
  should not appear in public responses).
- How `kv_cache_active_bytes` differs from `kv_cache_allocated_bytes`.
- Why KV bytes are computed from `cache._keys[0].dtype.size` at runtime rather
  than assumed to be float32.

Try:

- Run a CLI call with `--show-stats` and read each output line. Phase 1.7
  introduced a 14-line block; Phase 1.8 extends it to 21 lines by adding the
  seven quantization fields covered in section 22. The non-quantization
  lines are unchanged from the Phase 1.7 baseline.
- Compute KV active bytes by hand for your prompt length and compare.
- Explain why TTFT ≥ prefill_ms.

### 20. Context-Budget Policy

Read:

- `tiny_duo_infer/context_policy.py` — `apply_context_policy()`, `ContextPolicyOutcome`
- `tiny_duo_infer/generation.py` — `ContextPolicy` literal, `_VALID_CONTEXT_POLICIES`
- `tests/test_context_policy.py` — all five policies, precondition checks

Learn:

- The five policies: `allow_context_stop`, `reject`, `truncate_left`,
  `truncate_right`, `reserve_generation`.
- What `ContextPolicyOutcome` records: original vs. accepted token counts,
  truncated and rejected counts.
- Why both spec preconditions are universal (all policies):
  (a) `allow_context_stop` fails when `original_prompt_tokens > max_seq_len`
  (would never generate a single token); (b) every policy rejects
  `max_new_tokens > max_seq_len` regardless of policy.
- How `reserve_generation` differs from `truncate_left`: it truncates the
  prompt to `max_seq_len - max_new_tokens` to guarantee generation headroom.
- Why context-policy enforcement happens before prefill, not during decode.

Try:

- For `max_seq_len=16`, `max_new_tokens=4`, and a 20-token prompt, trace each
  policy outcome by hand.
- Run `uv run pytest tests/test_context_policy.py -v` to see all cases.

### 21. Profiling Script

Read:

- `scripts/profile_generation.py`
- `tiny_duo_infer/profiling.py`
- `tests/test_profiling.py`

Learn:

- How `run_profile()` wraps the engine, runs warmup rounds, and
  collects timed `GenerationStats` objects.
- Why warmup rounds are excluded from summary statistics (cold-start bias:
  first run includes MLX lazy-compilation and kernel-cache warm-up effects;
  the model is already loaded before `run_profile()` is called).
- How `percentile()` uses linear interpolation (numpy-default convention).
- What `aggregate_runs()` returns: `min`, `p50`, `p95`, `max` for TTFT,
  decode throughput, and KV-cache active memory.
- Why `--json` emits a stable, versioned schema and silences progress output.
  Phase 1.7 shipped `schema_version=1`; Phase 1.8 bumped it to
  `schema_version=2` to add quantization mode and linear-weight memory totals
  to `engine_info` (see section 22). All other Phase 1.7 fields keep the
  same names and shapes across the bump.
- Why `decode_step_ms` is omitted from per-run JSON (T03 leaves it empty in
  non-profiling paths).
- The stdout/stderr split: `--json` sends JSON to stdout; human mode sends the
  report to stdout and progress lines to stderr.

Try:

- Run `uv run python scripts/profile_generation.py --help`.
- Run with `--json` against a local model and inspect the schema.
- Compare p50 TTFT vs. p50 total latency for a short prompt.

### 22. Weight-Only Quantization

Read:

- `docs/phases/phase-1.8-weight-quantization.md`
- `tiny_duo_infer/quantization.py`
- `tiny_duo_infer/weights/quantizer.py`
- `tiny_duo_infer/models/base.py` `Linear.forward()` quantized branch
- `tiny_duo_infer/engine.py` `Engine.from_model_path(quantization=...)`
- `tests/test_quantization.py`
- `tests/test_quantization_integration.py`
- `learning_materials/deep_dives/quantization.md`

Learn:

- Why Phase 1.8 is *weight-only* quantization: only matrix weights used by
  `Linear` are quantized; activations, embeddings, RMSNorm/Q/K-norm weights,
  and KV-cache buffers stay full precision.
- Why MLX `mx.quantized_matmul()` is the required normal runtime path: the
  fused kernel reads packed integers and per-group scales/biases without
  materializing the full-precision weight, which is the memory and bandwidth
  benefit Phase 1.8 is teaching.
- Why `mx.dequantize()` is allowed only as an explicit test/debug fallback:
  eager dequantization at load time would convert compressed weights back to
  full precision and erase the entire benefit.
- What `QuantizationConfig` validates: `bits ∈ {4, 8}`, positive `group_size`,
  `mode="affine"`, and `in_features % group_size == 0` for every eligible
  matrix. Tiny Qwen3 fixtures with `d_model=32` must use `group_size=32`.
- What `QuantizedWeight` stores: the packed `qweight`, per-group `scales`,
  per-group `biases`, the bit width, group size, mode, original shape
  `(out_features, in_features)`, and `original_nbytes` so memory accounting
  can compare quantized runtime against the original full-precision size
  without assuming a fixed dtype.
- How `Linear.forward()` dispatches purely on `isinstance(self.weight, QuantizedWeight)`:
  the full-precision path stays `x @ weight.T`; the quantized path calls
  `mx.quantized_matmul()` with `transpose=True`.
- Where `weights/quantizer.py` fits in the loading pipeline: step 3 of four
  (load safetensors → HF-key conversion → quantize eligible Linear matrices →
  `model.load_weights(...)`), so HF-key shape validation still runs before
  quantization touches anything.
- Which weights are eligible vs not: eligible includes `*.q_proj.weight`,
  `*.k_proj.weight`, `*.v_proj.weight`, `*.o_proj.weight`,
  `*.gate_proj.weight`, `*.up_proj.weight`, `*.down_proj.weight`, and exact
  `lm_head.weight`; non-eligible includes `embed_tokens.weight`, all 1-D
  tensors (RMSNorm weights, Qwen3 `q_norm`/`k_norm`), and any non-matrix
  tensor.
- Why Llama tied embeddings need careful handling: when `lm_head.weight` is
  tied to `embed_tokens.weight`, quantizing `lm_head` must not mutate
  `embed_tokens`. The new dict shape preserves that — embeddings stay full
  precision while `lm_head` becomes a quantized projection.
- What `LinearWeightStats` and `compute_linear_weight_stats()` count:
  eligible Linear projection weights only. Embeddings and norms are excluded
  by design so the comparison directly answers "how much did quantization
  save on Linear weights?".
- What seven new `GenerationStats` fields encode: `quantization_mode`,
  `quantization_bits`, `quantization_group_size`, `quantized_linear_count`,
  `full_precision_linear_count`, `linear_weight_full_precision_bytes`, and
  `linear_weight_runtime_bytes`. When quantization is disabled, runtime
  bytes equal full-precision bytes for the counted linear weights.
- Why throughput is *measured* but not a hard pass/fail criterion:
  quantized matmul performance depends on MLX kernels, group size, prompt
  shape, and hardware state; the Phase 1.8 acceptance gate is reduced linear
  weight memory, with throughput reported alongside.

Try:

- Build a tiny INT4 Llama engine via `tests/test_quantization_integration.py:_make_llama_engine`
  and call `engine.generate_request(GenerationRequest(prompt="hi", max_new_tokens=2))`.
  Inspect `resp.stats.quantization_*` and verify `linear_weight_runtime_bytes
  < linear_weight_full_precision_bytes`.
- Run `uv run python -m tiny_duo_infer.cli --model-path ./models/llama-3.2-1b
  --prompt "The capital of France is" --max-new-tokens 8 --temperature 0.0
  --quantization int8 --show-stats` and read each quantization line in the
  stats block.
- Compare full-precision and INT4 with the same prompt set:
  `uv run python scripts/profile_generation.py --model-path ./models/llama-3.2-1b
   --runs 5 --warmup-runs 1 --quantization int4 --json` and diff against the
  same command without `--quantization`.
- Predict on paper: for an `(out=2048, in=2048)` weight at `bits=4` with
  `group_size=64`, how many bytes does `qweight` occupy? Compare against
  the full-precision bfloat16 size. Then check the answer with
  `compute_linear_weight_stats()`.

### 23. Phase Handoffs And Real-Model Smoke

Read:

- `docs/phases/phase-1-handoff.md`
- `docs/phases/phase-1-taskboard.md`
- `docs/phases/phase-1.5-qwen3-mlx.md`
- `docs/phases/phase-1.5-taskboard.md`
- `docs/phases/phase-1.7-observability.md`
- `docs/phases/phase-1.7-taskboard.md`
- `docs/phases/phase-1.8-weight-quantization.md`
- `docs/phases/phase-1.8-taskboard.md`

Learn:

- Which verification commands were run.
- What was checked with real Llama-3.2-1B weights.
- What was checked with real Qwen3-0.6B weights.
- Why semantic quality is not the smoke-test gate.
- How the real-model smoke exposed the bfloat16 loader issue.
- How Phase 1.5 verified model-family portability before backend portability.
- What Phase 1.7 added as a measurement baseline for future phases.
- What Phase 1.8 added: in-memory weight-only INT4/INT8 quantization, the
  seven new `GenerationStats` quantization fields, and the v2 profiling
  schema for direct full-precision-vs-quantized comparison.
- What remains out of scope until later phases (activation/KV-cache
  quantization, speculative decoding, continuous batching, CUDA backend).

Try:

- Run `uv run python -c "import tiny_duo_infer; print('import ok')"`.
- If local Llama artifacts are available, run a short CLI smoke with
  `--max-new-tokens 8 --temperature 0.0`.

## Mental Model

The completed Phase 1 pipeline is:

```text
config.json
  -> ModelConfig

tokenizer.json + tokenizer_config.json + optional config.json token metadata
  -> Tokenizer

model.safetensors / shards
  -> raw HF weight dict
  -> project weight dict via llama_converter or qwen3_converter

project weight dict
  -> LlamaModel.load_weights() or Qwen3Model.load_weights()
  -> Embedding / Linear / RMSNorm arrays

prompt text
  -> token IDs
  -> prefill forward(input_ids=(1, S), position_offset=0)
  -> KVCache.update(layer, K/V, position=0)
  -> KVCache.advance(S)
  -> final-position logits
  -> sample first generated token

generated token
  -> decode forward(input_ids=(1, 1), position_offset=cache.current_len)
  -> KVCache.update(layer, K/V, current position)
  -> mx.eval(logits) + KVCache.eval()
  -> KVCache.advance(1)
  -> sample next token
  -> repeat until EOS or max_new_tokens
```

## Suggested Reading Order

1. `pyproject.toml`
2. `tests/conftest.py`
3. `tiny_duo_infer/config.py`
4. `tiny_duo_infer/tokenizer/loader.py`
5. `tiny_duo_infer/weights/loader.py`
6. `tiny_duo_infer/weights/llama_converter.py`
7. `tiny_duo_infer/models/base.py`
8. `tiny_duo_infer/cache.py`
9. `tiny_duo_infer/layers/normalization.py`
10. `tiny_duo_infer/layers/rope.py`
11. `tiny_duo_infer/layers/attention.py`
12. `tiny_duo_infer/layers/feedforward.py`
13. `tiny_duo_infer/models/llama.py`
14. `tiny_duo_infer/models/qwen3.py`
15. `tiny_duo_infer/weights/qwen3_converter.py`
16. `tiny_duo_infer/sampling.py`
17. `tiny_duo_infer/engine.py`
18. `tiny_duo_infer/cli.py`
19. `tiny_duo_infer/prompt.py`
20. `tiny_duo_infer/generation.py`
21. `tiny_duo_infer/context_policy.py`
22. `tiny_duo_infer/profiling.py`
23. `tiny_duo_infer/serving/worker.py`
24. `tiny_duo_infer/serving/api.py`
25. `tiny_duo_infer/quantization.py`
26. `tiny_duo_infer/weights/quantizer.py`
27. `scripts/benchmark.py`
28. `scripts/profile_generation.py`
29. Matching `tests/` files in the same order, plus
    `tests/test_prompt.py`, `tests/test_serving.py`,
    `tests/test_quantization.py`, and `tests/test_quantization_integration.py`
30. `docs/phases/phase-1-handoff.md`
31. `docs/phases/phase-1.5-qwen3-mlx.md`
32. `docs/phases/phase-1.7-observability.md`
33. `docs/phases/phase-1.8-weight-quantization.md`
34. `learning_materials/deep_dives/chat_templating.md`
35. `learning_materials/deep_dives/inference_worker.md`
36. `learning_materials/deep_dives/quantization.md`

## What To Write Down

For each component, write:

- Inputs
- Outputs
- Tensor shapes
- State owned by the component
- Invariants
- Failure cases
- One thing that would silently corrupt generation if implemented incorrectly

That habit matters more than moving quickly.
