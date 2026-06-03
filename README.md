# tiny-duo-infer

A tiny inference engine implementation for learning purposes. Phase 1 runs on
Apple Silicon with MLX; later phases add model-family portability, generation
UX, local serving, then a PyTorch/CUDA backend.

## What It Is

- A learning-first LLM inference engine inspired by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), [MinivLLM](https://github.com/Wenyueh/MinivLLM), and [tiny-llm](https://github.com/skyzh/tiny-llm), implemented in pure Python. The goal is to understand how an inference engine works by building each piece from scratch: model loading, tokenization, forward pass, KV cache, sampling, and scheduling/serving.

- Key design principle: readable, teachable code over optimized code. Every concept (prefill, decode, KV cache updates, GQA, RoPE) must be visible in the implementation rather than hidden behind transformers, mlx-lm, or vLLM.

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| Phase 1 | Single-user inference on Apple Silicon using MLX | Done |
| Phase 1.5 | Add Qwen3-0.6B support on the same MLX backend | Done |
| Phase 1.6 | Refine generation UX and add single-request local HTTP serving | Planned |
| Phase 2 | Add NVIDIA/PyTorch/CUDA backend | Deferred |
| Phase 3 | Multi-user serving: scheduling, batching, streaming, PagedAttention | Not started |

## Model Targets

- [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

## Local CLI

Llama example:

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/llama-3.2-1b \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --temperature 0.0
```

Qwen3 example:

```bash
uv run python -m tiny_duo_infer.cli \
  --model-path ./models/qwen3-0.6b \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --temperature 0.7 \
  --top-p 0.8
```

The CLI uses plain prompt-to-completion mode. It does not apply Qwen3 chat
templates or system/user/assistant message formatting; those are prompt
formatting concerns outside the current engine scope.
