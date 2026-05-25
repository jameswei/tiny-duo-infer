"""
Engine: top-level public API for single-user local text generation.

The Engine owns the model, tokenizer, and KV cache for one generation request.
It orchestrates the full pipeline:
  1. Tokenize the prompt
  2. Prefill: run the full prompt through the model, populating the KV cache
  3. Decode: generate one token per step, attending to the cached prefix
  4. Sample: choose the next token (greedy or probabilistic)
  5. Stop: on EOS or when max_new_tokens is reached
  6. Yield: decoded text fragments to the caller

Phase 1 supports one active request at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


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
        raise NotImplementedError

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
        raise NotImplementedError
