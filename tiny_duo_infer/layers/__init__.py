"""
Individual layer implementations for Llama and Qwen3.

Each layer is independently testable and implements the inference concept
explicitly from backend primitives. No mlx.nn.* high-level layers are used.

Layers: RMSNorm, RoPE, LlamaAttention/Qwen3Attention (GQA), SwiGLUFFN.
"""
