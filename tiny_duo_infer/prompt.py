"""
Chat prompt formatting for supported model families.

Converts a list of ChatMessage objects into a plain string ready for
tokenization. The template is a deterministic subset of each model family's
standard chat format — sufficient for the common case without requiring
transformers.apply_chat_template() parity.

Qwen3 uses the ChatML format recorded in its tokenizer_config.json:

  <|im_start|>system
  {system_content}<|im_end|>
  <|im_start|>user
  {user_content}<|im_end|>
  <|im_start|>assistant
  {assistant_content}<|im_end|>
  <|im_start|>user
  {user_content}<|im_end|>
  <|im_start|>assistant

The final line is the assistant turn prefix that prompts the model to
continue generating. It is not closed with <|im_end|>.

Llama-3.2-1B is a base (completion) model. It has no associated chat template
and is not fine-tuned for instruction following, so chat mode is not supported.
"""

from __future__ import annotations

from tiny_duo_infer.generation import ChatMessage


_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def format_chat_prompt(messages: list[ChatMessage], model_type: str) -> str:
    """
    Format chat messages into a tokenizable prompt string.

    Each message is rendered as:
      <|im_start|>{role}\\n{content}<|im_end|>\\n

    The prompt ends with:
      <|im_start|>assistant\\n

    so the model generates the assistant's next response.

    Args:
        messages:   ordered list of ChatMessage objects. Must not be empty.
        model_type: "qwen3" is supported. "llama" raises ValueError (base model).

    Returns:
        Formatted prompt string ready for Tokenizer.encode().

    Raises:
        ValueError: if messages is empty, model_type is "llama", or model_type
                    is not recognised.
    """
    if not messages:
        raise ValueError("messages must not be empty.")

    if model_type == "qwen3":
        return _format_chatml(messages)
    if model_type == "llama":
        raise ValueError(
            "Llama-3.2-1B is a base model without a chat template. "
            "Use a plain prompt string (chat=False) instead."
        )
    raise ValueError(
        f"unsupported model_type for chat formatting: {model_type!r}"
    )


def _format_chatml(messages: list[ChatMessage]) -> str:
    """
    Render messages using the ChatML format from Qwen3's tokenizer_config.json.

    Each turn:  <|im_start|>{role}\\n{content}<|im_end|>\\n
    Suffix:     <|im_start|>assistant\\n
    """
    parts: list[str] = []
    for msg in messages:
        parts.append(f"{_IM_START}{msg.role}\n{msg.content}{_IM_END}\n")
    parts.append(f"{_IM_START}assistant\n")
    return "".join(parts)
