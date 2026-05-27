"""
Tokenizer wrapper using the HuggingFace `tokenizers` package.

Loads tokenizer.json, tokenizer_config.json, and config.json metadata from a
local HuggingFace-compatible model directory. Exposes only the operations the
engine needs: encode, decode, bos_token_id, eos_token_id, vocab_size.

The `tokenizers` package is a lightweight (~10MB) Rust-backed library. It loads
tokenizer.json directly without pulling in the full `transformers` stack.
`transformers.AutoTokenizer` may be used in dev/test only (parity checks), but
must not be imported from any file under tiny_duo_infer/.

Llama-3.2-1B uses tiktoken BPE with the o200k_base vocabulary. Its special
tokens (BOS = 128000, EOS = 128001) are encoded in tokenizer_config.json.
Qwen3-0.6B sets bos_token=null in tokenizer_config.json and records
bos_token_id in config.json, so BOS and EOS are resolved independently.

BOS/EOS resolution order:
  1. tokenizer_config.json integer field: {bos,eos}_token_id
  2. tokenizer_config.json string field: {bos,eos}_token → vocab lookup
     The string value may be a plain str or an AddedToken dict {"content": "..."}.
  3. config.json integer field: {bos,eos}_token_id
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
          - tokenizer_config.json  (required — tokenizer metadata)
          - config.json            (optional fallback for bos/eos token IDs)

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
        Resolve BOS and EOS token IDs from tokenizer metadata.

        Two formats appear in HF checkpoints:
          1. Integer fields: {"bos_token_id": 128000, "eos_token_id": 128001}
          2. Token strings:  {"bos_token": "<|begin_of_text|>", ...}
             where the string value may be a plain str or an AddedToken dict.

        Qwen3-0.6B combines formats: tokenizer_config.json has no BOS string
        because add_bos_token=false, while config.json still records
        bos_token_id for the model family. Resolve each token independently so
        a missing BOS field does not discard a valid EOS string.
        """
        config_path = model_path / "tokenizer_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"tokenizer_config.json not found in {model_path}; "
                "cannot determine bos_token_id / eos_token_id."
            )

        tokenizer_config = json.loads(config_path.read_text(encoding="utf-8"))
        model_config = cls._read_optional_json(model_path / "config.json")

        bos_id = cls._resolve_special_token_id(
            "bos",
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            tok=tok,
        )
        eos_id = cls._resolve_special_token_id(
            "eos",
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            tok=tok,
        )

        if bos_id is None or eos_id is None:
            missing = []
            if bos_id is None:
                missing.append("bos_token_id")
            if eos_id is None:
                missing.append("eos_token_id")
            raise ValueError(
                f"Cannot determine {' / '.join(missing)} from {model_path}. "
                "Expected integer IDs in tokenizer_config.json or config.json, "
                "or token strings in tokenizer_config.json."
            )

        return bos_id, eos_id

    @classmethod
    def _resolve_special_token_id(
        cls,
        prefix: str,
        *,
        tokenizer_config: dict[str, object],
        model_config: dict[str, object],
        tok: HFTokenizer,
    ) -> int | None:
        """
        Resolve one special token ID from tokenizer metadata.

        `prefix` is "bos" or "eos". Resolving one token at a time handles
        Qwen3-style metadata where EOS is a tokenizer string but BOS is only an
        integer in config.json.
        """
        id_key = f"{prefix}_token_id"
        token_key = f"{prefix}_token"
        vocab_size = tok.get_vocab_size()

        token_id = tokenizer_config.get(id_key)
        if token_id is not None:
            cls._validate_token_id(id_key, token_id, vocab_size)
            return token_id

        token_str = cls._extract_token_str(tokenizer_config.get(token_key))
        if token_str is not None:
            token_id = tok.token_to_id(token_str)
            if token_id is None:
                raise ValueError(
                    f"{token_key} {token_str!r} not found in the tokenizer vocabulary."
                )
            return token_id

        token_id = model_config.get(id_key)
        if token_id is not None:
            cls._validate_token_id(f"config.json {id_key}", token_id, vocab_size)
            return token_id

        return None

    @staticmethod
    def _read_optional_json(path: Path) -> dict[str, object]:
        """Return a JSON object from `path`, or an empty dict if the file is absent."""
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        return data

    @staticmethod
    def _validate_token_id(name: str, token_id: object, vocab_size: int) -> None:
        """
        Reject bool values and out-of-range IDs before they cause silent errors.

        bool is a subclass of int in Python, so isinstance(True, int) is True.
        A bool ID would pass the int check above and later cause wrong embedding
        lookups or missed EOS stops.
        """
        if isinstance(token_id, bool):
            raise ValueError(f"{name} must be an int, got bool: {token_id!r}")
        if not isinstance(token_id, int):
            raise ValueError(f"{name} must be an int, got {token_id!r}")
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
