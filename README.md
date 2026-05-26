# tiny-duo-infer
A tiny inference engine implementation for learning purpose. Supports duo backends: Apple Silicon and Nvidia GPU backend.

## What It Is

  A learning-first LLM inference engine inspired by vLLM, implemented in pure Python. The goal is to understand how an
   inference engine works by building each piece from scratch: model loading, tokenization, forward pass, KV cache,
  sampling, and (later) scheduling/serving.

  Key design principle: readable, teachable code over optimized code. Every concept (prefill, decode, KV cache
  updates, GQA, RoPE) must be visible in the implementation rather than hidden behind transformers, mlx-lm, or vLLM.

 ## Three-Phase Roadmap

| Phase | Focus |
|---|---|
| Phase 1 | Single-user inference on Apple Silicon using MLX |
| Phase 2 | Add NVIDIA/PyTorch/CUDA backend |
| Phase 3 | Multi-user serving: scheduling, batching, streaming, PagedAttention |

## Target model
meta-llama/Llama-3.2-1B (base, not instruct)
