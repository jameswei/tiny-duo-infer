"""
Tests for tiny_duo_infer.prompt.format_chat_prompt.

Documents the exact ChatML output format for Qwen3 so the template is
unambiguous and verifiable without running a real model.
"""

from __future__ import annotations

import pytest

from tiny_duo_infer.generation import ChatMessage
from tiny_duo_infer.prompt import format_chat_prompt


# ---------------------------------------------------------------------------
# Qwen3 ChatML format
# ---------------------------------------------------------------------------


def test_qwen3_single_user_message():
    """Single user message produces the minimal ChatML prompt."""
    msgs = [ChatMessage(role="user", content="Hello")]
    result = format_chat_prompt(msgs, "qwen3")
    assert result == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"


def test_qwen3_system_then_user():
    """System message followed by a user message is rendered in order."""
    msgs = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="What is 2+2?"),
    ]
    result = format_chat_prompt(msgs, "qwen3")
    expected = (
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert result == expected


def test_qwen3_full_conversation_turn():
    """Multi-turn conversation renders all messages and ends with assistant prefix."""
    msgs = [
        ChatMessage(role="system", content="Be concise."),
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello!"),
        ChatMessage(role="user", content="Bye"),
    ]
    result = format_chat_prompt(msgs, "qwen3")
    expected = (
        "<|im_start|>system\nBe concise.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\nHello!<|im_end|>\n"
        "<|im_start|>user\nBye<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert result == expected


def test_qwen3_assistant_prefix_not_closed():
    """The final assistant turn prefix has no closing <|im_end|>."""
    msgs = [ChatMessage(role="user", content="hi")]
    result = format_chat_prompt(msgs, "qwen3")
    assert result.endswith("<|im_start|>assistant\n")
    assert not result.endswith("<|im_end|>\n")


def test_qwen3_multiline_content_preserved():
    """Newlines inside message content are passed through unchanged."""
    msgs = [ChatMessage(role="user", content="line1\nline2")]
    result = format_chat_prompt(msgs, "qwen3")
    assert "line1\nline2" in result


# ---------------------------------------------------------------------------
# Llama and unsupported models
# ---------------------------------------------------------------------------


def test_llama_chat_raises_value_error():
    """Llama is a base model; chat formatting raises a clear ValueError."""
    msgs = [ChatMessage(role="user", content="hi")]
    with pytest.raises(ValueError, match="[Ll]lama"):
        format_chat_prompt(msgs, "llama")


def test_unsupported_model_type_raises_value_error():
    """Unknown model_type raises ValueError naming the unsupported type."""
    msgs = [ChatMessage(role="user", content="hi")]
    with pytest.raises(ValueError, match="unsupported"):
        format_chat_prompt(msgs, "mistral")


def test_empty_messages_raises_value_error():
    """Empty message list raises ValueError before model dispatch."""
    with pytest.raises(ValueError, match="empty"):
        format_chat_prompt([], "qwen3")
