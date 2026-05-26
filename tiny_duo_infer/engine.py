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

from dataclasses import replace
from pathlib import Path
from typing import Iterator

import mlx.core as mx

from tiny_duo_infer.cache import KVCache
from tiny_duo_infer.config import ModelConfig, load_config
from tiny_duo_infer.models.llama import LlamaModel
from tiny_duo_infer.tokenizer.loader import Tokenizer
from tiny_duo_infer.weights.llama_converter import convert
from tiny_duo_infer.weights.loader import load_weights


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

    def __init__(
        self,
        model: LlamaModel,
        tokenizer: Tokenizer,
        config: ModelConfig,
        max_seq_len: int,
    ) -> None:
        """
        Create an engine around already-constructed model components.

        Args:
            model:       loaded LlamaModel. Forward accepts input IDs shaped
                         (B, S), a KVCache, and a position offset.
            tokenizer:   project tokenizer wrapper used by text prefill and
                         later decode.
            config:      model architecture dimensions used for cache allocation.
            max_seq_len: maximum total sequence length for one request. The
                         per-request KV cache has shape
                         (1, n_kv_heads, max_seq_len, head_dim) per layer.
        """
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")

        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_len = max_seq_len
        self.cache: KVCache | None = None

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
        model_dir = Path(model_path)
        config = load_config(model_dir)
        if max_seq_len > config.max_seq_len:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds model context length "
                f"{config.max_seq_len}"
            )

        runtime_config = replace(config, max_seq_len=max_seq_len)
        tokenizer = Tokenizer.from_pretrained(model_dir)
        hf_weights = load_weights(model_dir)
        project_weights = convert(hf_weights, runtime_config)

        model = LlamaModel(runtime_config)
        model.load_weights(project_weights)

        return cls(
            model=model,
            tokenizer=tokenizer,
            config=runtime_config,
            max_seq_len=max_seq_len,
        )

    def prefill(self, prompt: str) -> mx.array:
        """
        Tokenize `prompt`, run a full-prompt forward pass, and fill the KV cache.

        This is the public Phase-1 prefill API. It performs the first half of
        generation for one request:

        1. Encode text to token IDs.
        2. Allocate a fresh static KV cache.
        3. Run the whole prompt through the model at position_offset=0.
        4. Commit the cache length once, after all layers have written.

        Args:
            prompt: input text. Encoding usually prepends BOS when the tokenizer
                    is configured to do so.

        Returns:
            (V,) logits for the final prompt position. Decode uses this vector
            to sample the first generated token.
        """
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        return self.prefill_token_ids(token_ids)

    def prefill_token_ids(self, token_ids: list[int]) -> mx.array:
        """
        Run prefill for already-tokenized prompt IDs.

        Args:
            token_ids: prompt token IDs with shape semantics (S,) before the
                       engine adds the Phase-1 batch dimension.

        Returns:
            (V,) logits from the final prompt token position.

        Raises:
            ValueError: if the prompt is empty or does not fit in max_seq_len.
        """
        prompt_len = len(token_ids)
        if prompt_len == 0:
            raise ValueError("prefill requires at least one token")
        if prompt_len > self.max_seq_len:
            raise ValueError(
                f"prompt length {prompt_len} exceeds max_seq_len={self.max_seq_len}"
            )

        cache = self._new_cache()
        input_ids = mx.array([token_ids], dtype=mx.int32)  # (B=1, S)

        # Prefill writes positions [0, prompt_len) in every layer. The model
        # does not advance current_len; the engine commits once after all
        # layers have populated the cache for this step.
        logits = self.model(input_ids, cache, position_offset=0)  # (1, S, V)
        final_logits = logits[0, prompt_len - 1, :]  # (V,)
        cache.advance(prompt_len)

        # Materialise both the returned logits and the cache writes that decode
        # will read next. P1-T15 will refine broader eval placement.
        mx.eval(final_logits)
        cache.eval()
        self.cache = cache
        return final_logits

    def _new_cache(self) -> KVCache:
        """
        Allocate the static KV cache for a single generation request.

        Returns:
            KVCache with one K/V buffer pair per layer. Each buffer has shape
            (1, n_kv_heads, max_seq_len, head_dim).
        """
        return KVCache(
            n_layers=self.config.n_layers,
            n_kv_heads=self.config.n_kv_heads,
            max_seq_len=self.max_seq_len,
            head_dim=self.config.head_dim,
        )

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
