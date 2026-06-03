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

Phase 1/1.5 supports one active request at a time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import mlx.core as mx

from tiny_duo_infer.cache import KVCache
from tiny_duo_infer.config import ModelConfig, load_config
from tiny_duo_infer.generation import ChatMessage, GenerationRequest, GenerationResponse
from tiny_duo_infer.prompt import format_chat_prompt
from tiny_duo_infer.models.base import Module
from tiny_duo_infer.models.llama import LlamaModel
from tiny_duo_infer.models.qwen3 import Qwen3Model
from tiny_duo_infer.sampling import sample
from tiny_duo_infer.tokenizer.loader import Tokenizer
from tiny_duo_infer.weights.llama_converter import convert as convert_llama
from tiny_duo_infer.weights.loader import load_weights
from tiny_duo_infer.weights.qwen3_converter import convert as convert_qwen3


class Engine:
    """
    Top-level inference engine for single-user local generation.

    Owns the model, tokenizer, and generation loop. All state required
    for one generation request lives here. Phase 1/1.5 supports one active
    request at a time.

    Usage:
        engine = Engine.from_model_path(Path("./models/llama-3.2-1b"))
        for token_text in engine.generate("Once upon a time", max_new_tokens=100):
            print(token_text, end="", flush=True)
    """

    def __init__(
        self,
        model: Module,
        tokenizer: Tokenizer,
        config: ModelConfig,
        max_seq_len: int,
    ) -> None:
        """
        Create an engine around already-constructed model components.

        Args:
            model:       loaded model. Forward accepts input IDs shaped
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
        model_cls, converter = _model_class_and_converter(runtime_config)
        project_weights = converter(hf_weights, runtime_config)

        model = model_cls(runtime_config)
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

        Raises:
            ValueError: if the prompt is empty or does not fit in max_seq_len.
        """
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_tokens_len = len(token_ids)
        if prompt_tokens_len == 0:
            raise ValueError("prompt must encode to at least one token")
        if prompt_tokens_len > self.max_seq_len:
            raise ValueError(
                f"prompt token length {prompt_tokens_len} exceeds max_seq_len={self.max_seq_len}"
            )
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
            raise ValueError("prefill requires at least one token.")
        if prompt_len > self.max_seq_len:
            raise ValueError(
                f"prompt length {prompt_len} exceeds max_seq_len={self.max_seq_len}."
            )

        cache = self._new_cache()
        input_ids = mx.array([token_ids], dtype=mx.int32)  # (B=1, S)

        # Prefill writes positions [0, prompt_len) in every layer. The model
        # does not advance current_len; the engine commits once after all
        # layers have populated the cache for this step.
        logits = self.model(input_ids, cache, position_offset=0)  # (1, S, V)
        final_logits = logits[0, prompt_len - 1, :]  # (V,)
        cache.advance(prompt_len)

        # MLX is lazy: the prefill forward pass queues both the logits
        # computation and the in-place cache writes. Evaluate the final-position
        # logits before CPU-side greedy sampling, then evaluate the cache
        # buffers before decode reads positions [0, prompt_len).
        mx.eval(final_logits)
        cache.eval()
        # this is a exception-safety pattern, which guarantees that the cache is set
        # only if the prefill forward pass completes successfully.
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

        Generation has two phases:

        Prefill: the full prompt is processed in a single forward pass. The
        model writes KV entries for every prompt position in one shot, and the
        final-position logits give the first generated token. This is the fast
        part — the GPU processes all prompt tokens in parallel.

        Decode: one token is generated per step. Each step runs the model with
        a single input token, reads the KV cache for all past positions, and
        samples the next token. The loop continues until EOS is sampled or
        max_new_tokens is reached. This is the slow part — it is sequential.

        M1.8 wires all sampling parameters through sample(). Pass
        temperature=0.0 for deterministic greedy decoding.

        Args:
            prompt:         input text string.
            max_new_tokens: maximum number of NEW tokens to generate
                            (does not count the prompt tokens).
            temperature:    divide logits by this before sampling. 0.0 = greedy.
            top_k:          keep only top-k logits before sampling. 0 = off.
            top_p:          keep tokens summing to probability >= top_p. 1.0 = off.

        Yields:
            str: decoded text fragment for each new token, in order.
        """
        # Prefill: encode the prompt, run a full forward pass over all prompt
        # tokens, fill the KV cache for positions [0, prompt_len), and return
        # (V,) logits for the last prompt position. self.cache is set here.
        first_logits = self.prefill(prompt)  # (V,)

        # Sample the first generated token from the prefill logits.
        # This token sits at absolute position cache.current_len (= prompt_len).
        next_token = sample(
            first_logits, temperature=temperature, top_k=top_k, top_p=top_p
        )

        for step in range(max_new_tokens):
            # EOS check: stop before yielding the stop token so callers never
            # receive it. This mirrors how generation frameworks handle EOS.
            if next_token == self.tokenizer.eos_token_id:
                break

            yield self.tokenizer.decode([next_token])

            # Skip the decode forward on the last step: the sampled token would
            # never be yielded, so running the model here is pure waste.
            if step == max_new_tokens - 1:
                break

            # Decode step: run the model with next_token as the single input.
            # position_offset tells the attention layer where in the sequence
            # this token sits, so RoPE and the causal mask are correct.
            input_ids = mx.array([[next_token]])  # (B=1, S=1)
            logits = self.model(
                input_ids, self.cache, position_offset=self.cache.current_len
            )  # (1, 1, V)

            # Flush MLX's lazy computation graph at the engine boundary, not
            # inside individual layers. The next line materialises logits before
            # CPU-side greedy sampling reads the selected token ID.
            mx.eval(logits)

            # Materialise the KV cache writes from this token step before the
            # next decode step reads them. This is intentionally separate from
            # mx.eval(logits): logits are sampled on the CPU, while cache buffers
            # are the persisted accelerator state for future attention.
            self.cache.eval()

            # Commit the new KV position. advance() is called once per token
            # step (not once per layer) so current_len stays consistent across
            # all 16 layers during the next decode step.
            self.cache.advance(1)

            next_token = sample(
                logits[0, 0, :], temperature=temperature, top_k=top_k, top_p=top_p
            )

    def generate_request(self, request: GenerationRequest) -> GenerationResponse:
        """
        Run a full generation loop from a validated GenerationRequest.

        Implements all four stop conditions in priority order:
          1. EOS token sampled — stops before yielding or decoding the token.
          2. Stop string found in accumulated decoded text — trims the marker.
          3. max_new_tokens reached — loop exhausted.
          4. Context length full — no room for another decode step.

        Seeded sampling: if request.seed is set, mx.random.seed() is called
        after prefill and before the first sample. This resets MLX's global
        PRNG state so all token draws in the loop are deterministic for the
        given seed. Greedy decoding (temperature=0.0) is already deterministic
        without a seed; seed has no effect in that case.

        Chat mode: if request.chat is True, messages (or a plain prompt wrapped
        as a user message) are formatted into a prompt string via
        format_chat_prompt() before tokenization. Qwen3 uses the ChatML
        template; Llama raises ValueError (base model, no chat template).

        Args:
            request: validated GenerationRequest.

        Returns:
            GenerationResponse with full text, token counts, and stop reason.

        Raises:
            ValueError: if chat=True and the model family does not support it.
        """
        if request.chat:
            if request.messages is not None:
                msgs = request.messages
            else:
                msgs = [ChatMessage(role="user", content=request.prompt)]  # type: ignore[arg-type]
            prompt_str = format_chat_prompt(msgs, self.config.model_type)
        else:
            prompt_str = request.prompt  # type: ignore[assignment]
        token_ids = self.tokenizer.encode(prompt_str, add_special_tokens=True)
        prompt_tokens = len(token_ids)

        first_logits = self.prefill_token_ids(token_ids)

        if request.seed is not None:
            mx.random.seed(request.seed)

        next_token = sample(
            first_logits,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        fragments: list[str] = []
        generated_tokens = 0
        stop_reason = "max_new_tokens"

        for step in range(request.max_new_tokens):
            # Stop condition 1: EOS (before decoding — EOS is never yielded)
            if next_token == self.tokenizer.eos_token_id:
                stop_reason = "eos"
                break

            # Stop condition 4: context full (before decoding — a token sampled
            # for an out-of-bounds position is never yielded or counted)
            if self.cache.current_len >= self.max_seq_len:
                stop_reason = "context_length"
                break

            fragment = self.tokenizer.decode([next_token])
            fragments.append(fragment)
            generated_tokens += 1

            # Stop condition 2: stop string in accumulated text
            current_text = "".join(fragments)
            stop_string_hit = False
            for s in request.stop:
                idx = current_text.find(s)
                if idx != -1:
                    fragments = [current_text[:idx]]
                    stop_reason = "stop_string"
                    stop_string_hit = True
                    break
            if stop_string_hit:
                break

            # Stop condition 3: max_new_tokens exhausted (skip decode on last step)
            if step == request.max_new_tokens - 1:
                break

            input_ids = mx.array([[next_token]])
            logits = self.model(
                input_ids, self.cache, position_offset=self.cache.current_len
            )
            mx.eval(logits)
            self.cache.eval()
            self.cache.advance(1)

            next_token = sample(
                logits[0, 0, :],
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )

        return GenerationResponse(
            text="".join(fragments),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            stop_reason=stop_reason,
        )


ModelClass = type[LlamaModel] | type[Qwen3Model]
Converter = Callable[[dict[str, mx.array], ModelConfig], dict[str, mx.array]]


def _model_class_and_converter(config: ModelConfig) -> tuple[ModelClass, Converter]:
    """
    Select the model assembly and weight converter for a supported model family.

    Keeping dispatch here makes Engine.from_model_path() the single boundary
    between HuggingFace artifacts and the runtime model object.
    """
    if config.model_type == "llama":
        return LlamaModel, convert_llama
    if config.model_type == "qwen3":
        return Qwen3Model, convert_qwen3
    raise ValueError(f"unsupported model_type {config.model_type!r}")
