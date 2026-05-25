"""
Model assembly: base module infrastructure and Llama model implementation.

Models are backend-neutral Python classes. Weights are stored as plain
backend-array attributes (mx.array in Phase 1) populated via load_weights().
"""
