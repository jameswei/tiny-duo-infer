"""
Tests for tiny_duo_infer.context_policy.

Covers each of the five context-budget policies on a tokenized prompt, plus
the spec follow-up clarifications: `allow_context_stop` rejects prompts longer
than `max_seq_len`, and `max_new_tokens > max_seq_len` is rejected by every
policy. Outcome accounting invariants are checked in dedicated cases.
"""

from __future__ import annotations

import pytest

from tiny_duo_infer.context_policy import (
    ContextBudgetError,
    ContextPolicyOutcome,
    apply_context_policy,
)


# ---------------------------------------------------------------------------
# Argument validation (independent of policy semantics)
# ---------------------------------------------------------------------------


def test_apply_rejects_unknown_policy():
    with pytest.raises(ValueError, match="context_policy must be one of"):
        apply_context_policy(
            token_ids=[1, 2, 3],
            max_new_tokens=10,
            max_seq_len=100,
            policy="not_a_policy",  # type: ignore[arg-type]
        )


def test_apply_rejects_negative_max_new_tokens():
    with pytest.raises(ValueError, match="max_new_tokens must be >= 0"):
        apply_context_policy(
            token_ids=[1, 2, 3],
            max_new_tokens=-1,
            max_seq_len=100,
            policy="allow_context_stop",
        )


def test_apply_rejects_zero_max_seq_len():
    with pytest.raises(ValueError, match="max_seq_len must be > 0"):
        apply_context_policy(
            token_ids=[1, 2, 3],
            max_new_tokens=10,
            max_seq_len=0,
            policy="allow_context_stop",
        )


def test_apply_rejects_empty_token_ids():
    with pytest.raises(ValueError, match="token_ids must contain at least one token"):
        apply_context_policy(
            token_ids=[],
            max_new_tokens=10,
            max_seq_len=100,
            policy="allow_context_stop",
        )


# ---------------------------------------------------------------------------
# Spec (Minimum requirements): fail before prefill when max_new_tokens >
# max_seq_len for any context policy.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [
        "allow_context_stop",
        "reject",
        "truncate_left",
        "truncate_right",
        "reserve_generation",
    ],
)
def test_apply_rejects_max_new_tokens_exceeding_max_seq_len(policy):
    with pytest.raises(ContextBudgetError, match="exceeds max_seq_len"):
        apply_context_policy(
            token_ids=[1, 2, 3],
            max_new_tokens=200,
            max_seq_len=100,
            policy=policy,
        )


# ---------------------------------------------------------------------------
# allow_context_stop
# ---------------------------------------------------------------------------


def test_allow_context_stop_accepts_prompt_within_cache():
    out = apply_context_policy(
        token_ids=[1, 2, 3, 4, 5],
        max_new_tokens=10,
        max_seq_len=100,
        policy="allow_context_stop",
    )
    assert out.policy == "allow_context_stop"
    assert out.accepted_token_ids == [1, 2, 3, 4, 5]
    assert out.original_prompt_tokens == 5
    assert out.accepted_prompt_tokens == 5
    assert out.truncated_prompt_tokens == 0
    assert out.rejected_prompt_tokens == 0


def test_allow_context_stop_accepts_prompt_that_overflows_with_generation():
    # Prompt fits in cache but prompt + max_new_tokens does not. This is
    # exactly the case that makes allow_context_stop different from reject:
    # admission succeeds; the engine will later stop with `context_length`.
    # max_new_tokens stays within max_seq_len so the universal precondition
    # (max_new_tokens > max_seq_len rejects everywhere) does not fire.
    out = apply_context_policy(
        token_ids=[1, 2, 3, 4, 5, 6, 7],
        max_new_tokens=10,
        max_seq_len=10,
        policy="allow_context_stop",
    )
    assert out.accepted_token_ids == [1, 2, 3, 4, 5, 6, 7]
    assert out.accepted_prompt_tokens == 7
    assert out.truncated_prompt_tokens == 0


def test_allow_context_stop_accepts_prompt_at_cache_boundary():
    # Prompt of exactly max_seq_len tokens — boundary admit.
    out = apply_context_policy(
        token_ids=list(range(10)),
        max_new_tokens=5,
        max_seq_len=10,
        policy="allow_context_stop",
    )
    assert out.accepted_prompt_tokens == 10
    assert out.truncated_prompt_tokens == 0


def test_allow_context_stop_rejects_prompt_longer_than_cache():
    # Spec follow-up (a): silent truncation is forbidden, so a prompt that
    # cannot fit even with zero generation must fail fast.
    with pytest.raises(ContextBudgetError, match="exceeds max_seq_len"):
        apply_context_policy(
            token_ids=list(range(15)),
            max_new_tokens=2,
            max_seq_len=10,
            policy="allow_context_stop",
        )


# ---------------------------------------------------------------------------
# reject
# ---------------------------------------------------------------------------


def test_reject_accepts_when_prompt_plus_generation_fits():
    out = apply_context_policy(
        token_ids=[1, 2, 3, 4, 5],
        max_new_tokens=5,
        max_seq_len=20,
        policy="reject",
    )
    assert out.policy == "reject"
    assert out.accepted_token_ids == [1, 2, 3, 4, 5]
    assert out.accepted_prompt_tokens == 5
    assert out.truncated_prompt_tokens == 0


def test_reject_accepts_at_exact_budget_boundary():
    # accepted_prompt_tokens + max_new_tokens == max_seq_len is admissible.
    out = apply_context_policy(
        token_ids=[1, 2, 3, 4, 5],
        max_new_tokens=5,
        max_seq_len=10,
        policy="reject",
    )
    assert out.accepted_prompt_tokens == 5


def test_reject_fails_when_budget_exceeded():
    with pytest.raises(ContextBudgetError, match="exceeds max_seq_len"):
        apply_context_policy(
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            max_new_tokens=5,
            max_seq_len=10,
            policy="reject",
        )


# ---------------------------------------------------------------------------
# truncate_left
# ---------------------------------------------------------------------------


def test_truncate_left_passthrough_when_within_budget():
    out = apply_context_policy(
        token_ids=[1, 2, 3, 4, 5],
        max_new_tokens=5,
        max_seq_len=20,
        policy="truncate_left",
    )
    assert out.accepted_token_ids == [1, 2, 3, 4, 5]
    assert out.truncated_prompt_tokens == 0


def test_truncate_left_drops_earliest_tokens_when_overflowing():
    # Prompt = 12 tokens, budget = 10 - 5 = 5, expect last 5 tokens admitted.
    out = apply_context_policy(
        token_ids=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        max_new_tokens=5,
        max_seq_len=10,
        policy="truncate_left",
    )
    assert out.policy == "truncate_left"
    assert out.accepted_token_ids == [17, 18, 19, 20, 21]
    assert out.original_prompt_tokens == 12
    assert out.accepted_prompt_tokens == 5
    assert out.truncated_prompt_tokens == 7


def test_truncate_left_fails_when_budget_zero():
    # max_new_tokens == max_seq_len leaves zero room for any prompt token.
    with pytest.raises(ContextBudgetError, match="leaves no room"):
        apply_context_policy(
            token_ids=[1, 2, 3],
            max_new_tokens=10,
            max_seq_len=10,
            policy="truncate_left",
        )


# ---------------------------------------------------------------------------
# truncate_right
# ---------------------------------------------------------------------------


def test_truncate_right_passthrough_when_within_budget():
    out = apply_context_policy(
        token_ids=[1, 2, 3, 4, 5],
        max_new_tokens=5,
        max_seq_len=20,
        policy="truncate_right",
    )
    assert out.accepted_token_ids == [1, 2, 3, 4, 5]
    assert out.truncated_prompt_tokens == 0


def test_truncate_right_drops_latest_tokens_when_overflowing():
    out = apply_context_policy(
        token_ids=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        max_new_tokens=5,
        max_seq_len=10,
        policy="truncate_right",
    )
    assert out.policy == "truncate_right"
    assert out.accepted_token_ids == [10, 11, 12, 13, 14]
    assert out.original_prompt_tokens == 12
    assert out.accepted_prompt_tokens == 5
    assert out.truncated_prompt_tokens == 7


def test_truncate_right_fails_when_budget_zero():
    with pytest.raises(ContextBudgetError, match="leaves no room"):
        apply_context_policy(
            token_ids=[1, 2, 3],
            max_new_tokens=10,
            max_seq_len=10,
            policy="truncate_right",
        )


# ---------------------------------------------------------------------------
# reserve_generation
# ---------------------------------------------------------------------------


def test_reserve_generation_matches_truncate_left_when_overflowing():
    token_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    left = apply_context_policy(
        token_ids=token_ids,
        max_new_tokens=5,
        max_seq_len=10,
        policy="truncate_left",
    )
    reserve = apply_context_policy(
        token_ids=token_ids,
        max_new_tokens=5,
        max_seq_len=10,
        policy="reserve_generation",
    )
    assert reserve.accepted_token_ids == left.accepted_token_ids
    assert reserve.accepted_prompt_tokens == left.accepted_prompt_tokens
    assert reserve.truncated_prompt_tokens == left.truncated_prompt_tokens
    # Policies report their own name, not the alias.
    assert reserve.policy == "reserve_generation"
    assert left.policy == "truncate_left"


def test_reserve_generation_passthrough_when_within_budget():
    out = apply_context_policy(
        token_ids=[1, 2, 3, 4, 5],
        max_new_tokens=5,
        max_seq_len=20,
        policy="reserve_generation",
    )
    assert out.accepted_token_ids == [1, 2, 3, 4, 5]
    assert out.truncated_prompt_tokens == 0


def test_reserve_generation_fails_when_budget_zero():
    with pytest.raises(ContextBudgetError, match="leaves no room"):
        apply_context_policy(
            token_ids=[1, 2, 3],
            max_new_tokens=10,
            max_seq_len=10,
            policy="reserve_generation",
        )


# ---------------------------------------------------------------------------
# Outcome invariants (cross-policy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [
        "allow_context_stop",
        "reject",
        "truncate_left",
        "truncate_right",
        "reserve_generation",
    ],
)
def test_outcome_token_count_invariants(policy):
    # Pick parameters that succeed for every policy: prompt fits, budget fits.
    out = apply_context_policy(
        token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        max_new_tokens=2,
        max_seq_len=20,
        policy=policy,
    )
    assert isinstance(out, ContextPolicyOutcome)
    assert len(out.accepted_token_ids) == out.accepted_prompt_tokens
    assert (
        out.accepted_prompt_tokens + out.truncated_prompt_tokens
        == out.original_prompt_tokens
    )
    assert out.rejected_prompt_tokens == 0
    assert out.accepted_prompt_tokens >= 1


def test_outcome_after_truncation_satisfies_budget():
    out = apply_context_policy(
        token_ids=list(range(50)),
        max_new_tokens=10,
        max_seq_len=20,
        policy="truncate_left",
    )
    assert out.accepted_prompt_tokens + 10 <= 20


def test_outcome_accepted_tokens_are_a_copy_not_alias():
    # The outcome should not let callers mutate the original token list.
    src = [1, 2, 3]
    out = apply_context_policy(
        token_ids=src,
        max_new_tokens=2,
        max_seq_len=20,
        policy="allow_context_stop",
    )
    out.accepted_token_ids.append(99)
    assert src == [1, 2, 3]
