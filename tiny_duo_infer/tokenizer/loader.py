"""
Tokenizer wrapper using the HuggingFace `tokenizers` package.

Loads tokenizer.json (and optionally tokenizer_config.json / special_tokens_map.json)
from a local HuggingFace-compatible model directory. Exposes only the operations
the engine needs: encode, decode, bos_token_id, eos_token_id, vocab_size.

The `tokenizers` package is a lightweight (~10MB) Rust-backed library. It loads
tokenizer.json directly without pulling in the full `transformers` stack.
`transformers.AutoTokenizer` may be used in dev/test only (parity checks), but
must not be imported from any file under tiny_duo_infer/.

Llama-3.2-1B uses tiktoken BPE with the o200k_base vocabulary. The special
tokens (BOS = 128000, EOS = 128001) are encoded in tokenizer_config.json.

BOS/EOS resolution order:
  1. tokenizer_config.json integer fields: bos_token_id / eos_token_id
  2. tokenizer_config.json string fields: bos_token / eos_token → vocab lookup
     The string value may be a plain str or an AddedToken dict {"content": "..."}.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    """
    Thin wrapper around the HuggingFace `tokenizers` package.

    Loads tokenizer.json and special token metadata from a local model
    directory. Exposes only the operations the engine needs. The `tokenizers`
    package is used at runtime; `transformers.AutoTokenizer` is dev/test only.
    """

    def __init__(
        self,
        _tok: HFTokenizer,
        _bos_token_id: int,
        _eos_token_id: int,
    ) -> None:
        self._tok = _tok
        self._bos_token_id = _bos_token_id
        self._eos_token_id = _eos_token_id

    @classmethod
    def from_pretrained(cls, model_path: Path | str) -> "Tokenizer":
        """
        Load tokenizer.json and special token metadata from model_path.

        Reads:
          - tokenizer.json         (required — full BPE model)
          - tokenizer_config.json  (required — contains bos/eos token info)

        Args:
            model_path: path to the local HuggingFace-compatible model directory.

        Returns:
            Tokenizer instance ready for encode/decode.

        Raises:
            FileNotFoundError: if tokenizer.json or tokenizer_config.json is missing.
            ValueError: if bos/eos token IDs cannot be resolved from the config.
        """
        model_path = Path(model_path)
        tokenizer_json = model_path / "tokenizer.json"
        if not tokenizer_json.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_path}")
        tok = HFTokenizer.from_file(str(tokenizer_json))
        bos_id, eos_id = cls._read_special_token_ids(model_path, tok)
        return cls(tok, bos_id, eos_id)

    @classmethod
    def _read_special_token_ids(
        cls,
        model_path: Path,
        tok: HFTokenizer,
    ) -> tuple[int, int]:
        """
        Resolve BOS and EOS token IDs from tokenizer_config.json.

        Two formats appear in HF checkpoints:
          1. Integer fields: {"bos_token_id": 128000, "eos_token_id": 128001}
          2. Token strings:  {"bos_token": "<|begin_of_text|>", ...}
             where the string value may be a plain str or an AddedToken dict.
        """
        config_path = model_path / "tokenizer_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"tokenizer_config.json not found in {model_path}; "
                "cannot determine bos_token_id / eos_token_id."
            )

        config = json.loads(config_path.read_text(encoding="utf-8"))

        # Case 1: direct integer IDs (newer HF configs sometimes include these)
        bos_id = config.get("bos_token_id")
        eos_id = config.get("eos_token_id")
        if isinstance(bos_id, int) and isinstance(eos_id, int):
            # bool is a subclass of int in Python — reject it explicitly.
            # Out-of-range IDs would silently produce wrong embeddings or miss EOS.
            cls._validate_token_id("bos_token_id", bos_id, tok.get_vocab_size())
            cls._validate_token_id("eos_token_id", eos_id, tok.get_vocab_size())
            return bos_id, eos_id

        # Case 2: token strings — look up their IDs in the tokenizer vocabulary
        bos_str = cls._extract_token_str(config.get("bos_token"))
        eos_str = cls._extract_token_str(config.get("eos_token"))

        if bos_str is None or eos_str is None:
            raise ValueError(
                f"Cannot determine bos_token / eos_token from {config_path}. "
                "Expected 'bos_token_id'/'eos_token_id' integers or "
                "'bos_token'/'eos_token' strings."
            )

        bos_id = tok.token_to_id(bos_str)
        eos_id = tok.token_to_id(eos_str)

        if bos_id is None or eos_id is None:
            raise ValueError(
                f"bos_token '{bos_str}' or eos_token '{eos_str}' not found "
                "in the tokenizer vocabulary."
            )

        return bos_id, eos_id

    @staticmethod
    def _validate_token_id(name: str, token_id: int, vocab_size: int) -> None:
        """
        Reject bool values and out-of-range IDs before they cause silent errors.

        bool is a subclass of int in Python, so isinstance(True, int) is True.
        A bool ID would pass the int check above and later cause wrong embedding
        lookups or missed EOS stops.
        """
        if isinstance(token_id, bool):
            raise ValueError(f"{name} must be an int, got bool: {token_id!r}")
        if not (0 <= token_id < vocab_size):
            raise ValueError(
                f"{name} {token_id} is out of range [0, {vocab_size})"
            )

    @staticmethod
    def _extract_token_str(value: object) -> str | None:
        """
        Extract a token string from a plain str or HF AddedToken dict.

        HF checkpoints represent special tokens in two ways:
          - Plain string:  "bos_token": "<|begin_of_text|>"
          - AddedToken dict: "bos_token": {"content": "<|begin_of_text|>", ...}
        """
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return value.get("content")
        return None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Convert text to a list of integer token IDs.

        Delegates to the underlying tokenizers.Tokenizer, which applies
        the tokenizer.json post-processor (TemplateProcessing in Llama) to
        prepend BOS when add_special_tokens=True.

        Args:
            text:               input string.
            add_special_tokens: if True, the tokenizer's post-processor runs
                                (typically prepends BOS for Llama).

        Returns:
            list of integer token IDs.
        """
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Convert a list of token IDs back to a text string.

        Args:
            token_ids:           list of integer token IDs.
            skip_special_tokens: if True, remove BOS/EOS and other special tokens
                                 before returning the string.

        Returns:
            decoded text string.
        """
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def bos_token_id(self) -> int:
        """Beginning-of-sequence token ID. For Llama-3.2-1B: 128000."""
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        """End-of-sequence token ID. Generation stops when this is sampled. For Llama-3.2-1B: 128001."""
        return self._eos_token_id

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary. For Llama-3.2-1B: 128256."""
        return self._tok.get_vocab_size()
