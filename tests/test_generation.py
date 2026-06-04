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
    GenerationStats,
    kv_cache_bytes,
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
    assert req.context_policy == "allow_context_stop"


# ---------------------------------------------------------------------------
# GenerationRequest — context policy
# ---------------------------------------------------------------------------


def test_generation_request_accepts_all_context_policies():
    for policy in (
        "allow_context_stop",
        "reject",
        "truncate_left",
        "truncate_right",
        "reserve_generation",
    ):
        req = GenerationRequest(prompt="hi", context_policy=policy)
        assert req.context_policy == policy


def test_generation_request_rejects_unknown_context_policy():
    with pytest.raises(ValueError, match="context_policy must be one of"):
        GenerationRequest(prompt="hi", context_policy="not_a_policy")  # type: ignore[arg-type]


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


def test_generation_response_stats_defaults_to_none():
    resp = GenerationResponse(
        text="hi", prompt_tokens=3, generated_tokens=1, stop_reason="eos"
    )
    assert resp.stats is None


def test_generation_response_accepts_all_stop_reasons():
    for reason in ("eos", "max_new_tokens", "stop_string", "context_length"):
        resp = GenerationResponse(
            text="", prompt_tokens=0, generated_tokens=0, stop_reason=reason
        )
        assert resp.stop_reason == reason


# ---------------------------------------------------------------------------
# kv_cache_bytes
# ---------------------------------------------------------------------------


def test_kv_cache_bytes_llama_config():
    # Llama-3.2-1B: 16 layers, 8 kv_heads, head_dim=64, float32=4 bytes
    # 2 * 16 * 8 * 1024 * 64 * 4 = 67_108_864
    result = kv_cache_bytes(
        n_layers=16, n_kv_heads=8, seq_len=1024, head_dim=64, bytes_per_element=4
    )
    assert result == 67_108_864


def test_kv_cache_bytes_qwen3_config():
    # Qwen3-0.6B: 28 layers, 8 kv_heads, head_dim=128, float32=4 bytes
    # 2 * 28 * 8 * 1024 * 128 * 4 = 234_881_024
    result = kv_cache_bytes(
        n_layers=28, n_kv_heads=8, seq_len=1024, head_dim=128, bytes_per_element=4
    )
    assert result == 234_881_024


def test_kv_cache_bytes_scales_linearly_with_seq_len():
    base = kv_cache_bytes(
        n_layers=4, n_kv_heads=4, seq_len=512, head_dim=64, bytes_per_element=4
    )
    doubled = kv_cache_bytes(
        n_layers=4, n_kv_heads=4, seq_len=1024, head_dim=64, bytes_per_element=4
    )
    assert doubled == 2 * base


def test_kv_cache_bytes_bfloat16():
    # bfloat16 = 2 bytes — result should be half of float32
    f32 = kv_cache_bytes(
        n_layers=8, n_kv_heads=4, seq_len=256, head_dim=64, bytes_per_element=4
    )
    bf16 = kv_cache_bytes(
        n_layers=8, n_kv_heads=4, seq_len=256, head_dim=64, bytes_per_element=2
    )
    assert bf16 == f32 // 2


# ---------------------------------------------------------------------------
# GenerationStats
# ---------------------------------------------------------------------------


def _make_stats(**overrides) -> GenerationStats:
    defaults: dict = dict(
        context_policy="allow_context_stop",
        original_prompt_tokens=10,
        accepted_prompt_tokens=10,
        truncated_prompt_tokens=0,
        rejected_prompt_tokens=0,
        prompt_tokens=10,
        generated_tokens=5,
        stop_reason="eos",
        prompt_prepare_ms=1.0,
        prefill_ms=20.0,
        time_to_first_token_ms=25.0,
        decode_ms=50.0,
        total_ms=75.0,
        decode_tokens_per_sec=100.0,
        kv_cache_allocated_bytes=67_108_864,
        kv_cache_active_bytes=983_040,
        max_seq_len=2048,
        active_seq_len=15,  # accepted_prompt_tokens(10) + generated_tokens(5)
    )
    defaults.update(overrides)
    return GenerationStats(**defaults)


def test_generation_stats_construction():
    stats = _make_stats()
    assert stats.context_policy == "allow_context_stop"
    assert stats.prompt_tokens == 10
    assert stats.accepted_prompt_tokens == 10
    assert stats.generated_tokens == 5
    assert stats.active_seq_len == 15


def test_generation_stats_optional_fields_have_defaults():
    stats = _make_stats()
    assert stats.decode_step_ms == []
    assert stats.model_type == ""


def test_generation_stats_rejects_invalid_context_policy():
    with pytest.raises(ValueError, match="context_policy must be one of"):
        _make_stats(context_policy="unknown_policy")


def test_generation_stats_rejects_prompt_tokens_mismatch():
    with pytest.raises(ValueError, match="prompt_tokens.*must equal.*accepted_prompt_tokens"):
        _make_stats(prompt_tokens=10, accepted_prompt_tokens=8)


def test_generation_stats_rejects_active_seq_len_mismatch():
    with pytest.raises(ValueError, match="active_seq_len.*must equal"):
        _make_stats(accepted_prompt_tokens=10, generated_tokens=5, active_seq_len=14)


def test_generation_stats_accepts_all_context_policies():
    for policy in (
        "allow_context_stop",
        "reject",
        "truncate_left",
        "truncate_right",
        "reserve_generation",
    ):
        stats = _make_stats(context_policy=policy)
        assert stats.context_policy == policy


def test_generation_stats_with_decode_step_ms():
    stats = _make_stats(decode_step_ms=[10.0, 12.5, 11.0])
    assert stats.decode_step_ms == [10.0, 12.5, 11.0]


def test_generation_stats_with_model_type():
    stats = _make_stats(model_type="llama")
    assert stats.model_type == "llama"


def test_generation_response_carries_stats():
    stats = _make_stats()
    resp = GenerationResponse(
        text="hello",
        prompt_tokens=10,
        generated_tokens=5,
        stop_reason="eos",
        stats=stats,
    )
    assert resp.stats is stats
    assert resp.stats.decode_tokens_per_sec == 100.0
