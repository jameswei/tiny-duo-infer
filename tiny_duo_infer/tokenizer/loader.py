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
"""

from __future__ import annotations

from pathlib import Path


class Tokenizer:
    """
    Thin wrapper around the HuggingFace `tokenizers` package.

    Loads tokenizer.json and special token metadata from a local model
    directory. Exposes only the operations the engine needs. The `tokenizers`
    package is used at runtime; `transformers.AutoTokenizer` is dev/test only.
    """

    @classmethod
    def from_pretrained(cls, model_path: Path | str) -> "Tokenizer":
        """
        Load tokenizer.json and special token metadata from model_path.

        Reads:
          - tokenizer.json     (required — full BPE model)
          - tokenizer_config.json (optional — contains bos_token_id, eos_token_id)
          - special_tokens_map.json (optional — fallback for special token IDs)

        Args:
            model_path: path to the local HuggingFace-compatible model directory.

        Returns:
            Tokenizer instance ready for encode/decode.
        """
        raise NotImplementedError

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Convert text to a list of integer token IDs.

        Args:
            text:               input string.
            add_special_tokens: if True, prepend BOS token (Llama convention).

        Returns:
            list of integer token IDs.
        """
        raise NotImplementedError

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Convert a list of token IDs back to a text string.

        Args:
            token_ids:           list of integer token IDs.
            skip_special_tokens: if True, remove BOS/EOS and other special tokens.

        Returns:
            decoded text string.
        """
        raise NotImplementedError

    @property
    def bos_token_id(self) -> int:
        """Beginning-of-sequence token ID. For Llama-3.2-1B: 128000."""
        raise NotImplementedError

    @property
    def eos_token_id(self) -> int:
        """End-of-sequence token ID. Generation stops when this is sampled. For Llama-3.2-1B: 128001."""
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary. For Llama-3.2-1B: 128256."""
        raise NotImplementedError
