"""
Tokenizer wrapper.

Wraps the HuggingFace `tokenizers` package behind a minimal project interface.
The `tokenizers` package is the runtime dependency; `transformers.AutoTokenizer`
is dev/test only and must not be imported from any file under tiny_duo_infer/.
"""
