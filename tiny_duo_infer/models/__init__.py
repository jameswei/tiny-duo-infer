"""
Model assembly: base module infrastructure plus supported model families.

Models are backend-neutral Python classes. Weights are stored as plain
backend-array attributes (mx.array in Phase 1) populated via load_weights().

llama.py: LlamaBlock and LlamaModel.
qwen3.py: Qwen3Block and Qwen3Model with Q/K-normalized attention.
"""
