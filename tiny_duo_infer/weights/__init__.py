"""
Weight loading and HuggingFace key mapping.

loader.py: reads one or more .safetensors shards into a flat dict of mx.array.
llama_converter.py: maps HF checkpoint key names to project model key names
and validates tensor shapes against the model config.
qwen3_converter.py: maps Qwen3 checkpoint keys, including Q/K norm weights.
"""
