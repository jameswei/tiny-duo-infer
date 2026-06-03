"""
Tests for tiny_duo_infer.generation.

Covers ChatMessage validation, GenerationRequest validation, and
GenerationResponse field storage.
"""

from __future__ import annotations

import pytest

from tiny_duo_infer.generation import (
    ChatMessage,
    GenerationRequest,
    GenerationResponse,
)


# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------


def test_chat_message_accepts_valid_roles():
    for role in ("system", "user", "assistant"):
        msg = ChatMessage(role=role, content="hello")
        assert msg.role == role
        assert msg.content == "hello"


def test_chat_message_rejects_invalid_role():
    with pytest.raises(ValueError, match="role must be one of"):
        ChatMessage(role="human", content="hello")


def test_chat_message_rejects_empty_content():
    with pytest.raises(ValueError, match="content must not be empty"):
        ChatMessage(role="user", content="")


# ---------------------------------------------------------------------------
# GenerationRequest — prompt / messages exclusivity
# ---------------------------------------------------------------------------


def test_generation_request_with_prompt_only():
    req = GenerationRequest(prompt="hello")
    assert req.prompt == "hello"
    assert req.messages is None


def test_generation_request_with_messages_and_chat():
    msgs = [ChatMessage(role="user", content="hello")]
    req = GenerationRequest(messages=msgs, chat=True)
    assert req.messages == msgs
    assert req.prompt is None


def test_generation_request_rejects_neither_prompt_nor_messages():
    with pytest.raises(ValueError, match="got neither"):
        GenerationRequest()


def test_generation_request_rejects_both_prompt_and_messages():
    msgs = [ChatMessage(role="user", content="hi")]
    with pytest.raises(ValueError, match="got both"):
        GenerationRequest(prompt="hi", messages=msgs, chat=True)


# ---------------------------------------------------------------------------
# GenerationRequest — content validation
# ---------------------------------------------------------------------------


def test_generation_request_rejects_empty_prompt():
    with pytest.raises(ValueError, match="'prompt' must not be empty"):
        GenerationRequest(prompt="")


def test_generation_request_rejects_messages_without_chat():
    msgs = [ChatMessage(role="user", content="hi")]
    with pytest.raises(ValueError, match="requires chat=True"):
        GenerationRequest(messages=msgs, chat=False)


def test_generation_request_rejects_empty_messages_list():
    with pytest.raises(ValueError, match="must not be an empty list"):
        GenerationRequest(messages=[], chat=True)


# ---------------------------------------------------------------------------
# GenerationRequest — numeric validation
# ---------------------------------------------------------------------------


def test_generation_request_rejects_negative_max_new_tokens():
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationRequest(prompt="hi", max_new_tokens=-1)


def test_generation_request_accepts_zero_max_new_tokens():
    req = GenerationRequest(prompt="hi", max_new_tokens=0)
    assert req.max_new_tokens == 0


def test_generation_request_rejects_negative_temperature():
    with pytest.raises(ValueError, match="temperature"):
        GenerationRequest(prompt="hi", temperature=-0.1)


def test_generation_request_accepts_zero_temperature():
    req = GenerationRequest(prompt="hi", temperature=0.0)
    assert req.temperature == 0.0


def test_generation_request_rejects_negative_top_k():
    with pytest.raises(ValueError, match="top_k"):
        GenerationRequest(prompt="hi", top_k=-1)


def test_generation_request_accepts_zero_top_k():
    req = GenerationRequest(prompt="hi", top_k=0)
    assert req.top_k == 0


def test_generation_request_rejects_top_p_zero():
    with pytest.raises(ValueError, match="top_p"):
        GenerationRequest(prompt="hi", top_p=0.0)


def test_generation_request_rejects_top_p_above_one():
    with pytest.raises(ValueError, match="top_p"):
        GenerationRequest(prompt="hi", top_p=1.01)


def test_generation_request_accepts_top_p_one():
    req = GenerationRequest(prompt="hi", top_p=1.0)
    assert req.top_p == 1.0


# ---------------------------------------------------------------------------
# GenerationRequest — stop strings
# ---------------------------------------------------------------------------


def test_generation_request_accepts_non_empty_stop_strings():
    req = GenerationRequest(prompt="hi", stop=[".", "\n", "</s>"])
    assert req.stop == [".", "\n", "</s>"]


def test_generation_request_rejects_empty_stop_string():
    with pytest.raises(ValueError, match="stop string must be non-empty"):
        GenerationRequest(prompt="hi", stop=["valid", ""])


# ---------------------------------------------------------------------------
# GenerationRequest — defaults
# ---------------------------------------------------------------------------


def test_generation_request_default_values():
    req = GenerationRequest(prompt="hi")
    assert req.max_new_tokens == 200
    assert req.temperature == 1.0
    assert req.top_k == 0
    assert req.top_p == 1.0
    assert req.stop == []
    assert req.seed is None
    assert req.chat is False


# ---------------------------------------------------------------------------
# GenerationResponse
# ---------------------------------------------------------------------------


def test_generation_response_stores_all_fields():
    resp = GenerationResponse(
        text="Paris",
        prompt_tokens=5,
        generated_tokens=1,
        stop_reason="eos",
    )
    assert resp.text == "Paris"
    assert resp.prompt_tokens == 5
    assert resp.generated_tokens == 1
    assert resp.stop_reason == "eos"


def test_generation_response_accepts_all_stop_reasons():
    for reason in ("eos", "max_new_tokens", "stop_string", "context_length"):
        resp = GenerationResponse(
            text="", prompt_tokens=0, generated_tokens=0, stop_reason=reason
        )
        assert resp.stop_reason == reason
