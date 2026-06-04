"""
Per-request context-budget policy for prompt admission.

Given a tokenized prompt and a desired generation budget, the policy decides
which prompt token IDs are admitted to prefill, optionally truncating earlier
or later tokens, or raising an admission error when no admission satisfies the
configured policy.

Five policies are supported:

  - "allow_context_stop": preserves Phase 1.6 behavior. The prompt is admitted
    as-is when it fits in `max_seq_len`; generation may stop later with
    `context_length` if the cache fills before another stop reason wins.
    Prompts longer than `max_seq_len` are rejected up front (silent truncation
    is forbidden by the spec).
  - "reject": admit only when prompt + `max_new_tokens` fits in `max_seq_len`.
    Otherwise raise `ContextBudgetError` before prefill.
  - "truncate_left": drop earliest prompt tokens until prompt + `max_new_tokens`
    fits in `max_seq_len`. Always leaves at least one accepted token (raises
    otherwise).
  - "truncate_right": drop latest prompt tokens under the same budget.
  - "reserve_generation": semantically identical to "truncate_left", named to
    emphasize the intent of preserving the newest prompt suffix while
    reserving room for the requested generation budget.

This module is the request/admission boundary. It does not call the model,
does not allocate cache, and does not import MLX. Engine integration (using
the outcome to drive prefill and to fill `GenerationStats`) is the
responsibility of the engine instrumentation task (P1.7-T03).

Spec: docs/phases/phase-1.7-observability.md (Metrics Model -> Context Policy)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import get_args

from tiny_duo_infer.generation import ContextPolicy

# Source the literal values directly from the type alias so the module never
# drifts from `generation.ContextPolicy`.
_VALID_POLICIES: frozenset[str] = frozenset(get_args(ContextPolicy))


class ContextBudgetError(ValueError):
    """Raised when a request cannot be admitted under its context-budget policy.

    This is a subclass of `ValueError` so callers that already handle
    validation errors (CLI, HTTP layer) catch it without extra wiring, while
    callers that want to distinguish admission failures from other validation
    errors can catch the more specific type.
    """


@dataclass(frozen=True)
class ContextPolicyOutcome:
    """Result of applying a context-budget policy to a tokenized prompt.

    The outcome carries everything the engine needs to drive prefill plus
    everything `GenerationStats` needs to record what happened. Successful
    outcomes always satisfy:

      - len(accepted_token_ids) == accepted_prompt_tokens
      - accepted_prompt_tokens >= 1
      - accepted_prompt_tokens + truncated_prompt_tokens == original_prompt_tokens
      - rejected_prompt_tokens == 0   (rejection raises rather than returns)
    """

    accepted_token_ids: list[int]
    original_prompt_tokens: int
    accepted_prompt_tokens: int
    truncated_prompt_tokens: int
    rejected_prompt_tokens: int
    policy: ContextPolicy


def apply_context_policy(
    token_ids: Sequence[int],
    max_new_tokens: int,
    max_seq_len: int,
    policy: ContextPolicy,
) -> ContextPolicyOutcome:
    """Apply a context-budget policy to a tokenized prompt.

    Args:
        token_ids:      already-tokenized prompt IDs in their original order.
        max_new_tokens: requested generation budget in tokens. Must be >= 0.
        max_seq_len:    cache capacity for the request. Must be > 0.
        policy:         the named policy to apply.

    Returns:
        ContextPolicyOutcome with the accepted token IDs and accounting fields.

    Raises:
        ValueError:           on invalid arguments (unknown policy, negative
            budgets, empty prompt, max_seq_len <= 0).
        ContextBudgetError:   when no admission satisfies the policy without
            violating the spec's invariants (no empty prompt after truncation,
            no overflow of max_seq_len, no silent rejection).
    """
    if policy not in _VALID_POLICIES:
        raise ValueError(
            f"context_policy must be one of {sorted(_VALID_POLICIES)!r},"
            f" got {policy!r}."
        )
    if max_new_tokens < 0:
        raise ValueError(
            f"max_new_tokens must be >= 0, got {max_new_tokens}."
        )
    if max_seq_len <= 0:
        raise ValueError(
            f"max_seq_len must be > 0, got {max_seq_len}."
        )

    original = len(token_ids)
    if original == 0:
        raise ValueError("token_ids must contain at least one token.")

    # Spec (docs/phases/phase-1.7-observability.md, "Minimum requirements")
    # requires this precondition for *every* policy, including
    # `allow_context_stop`. A request asking for more generation tokens than
    # the cache can hold is treated as malformed input and must fail before
    # prefill rather than silently cap at the cache boundary.
    if max_new_tokens > max_seq_len:
        raise ContextBudgetError(
            f"max_new_tokens ({max_new_tokens}) exceeds max_seq_len"
            f" ({max_seq_len}); no context policy can admit this request."
        )

    if policy == "allow_context_stop":
        # Spec follow-up (a): allow_context_stop preserves Phase 1.6 behavior
        # for prompts that fit (the engine may later stop with context_length
        # mid-decode), but a prompt longer than max_seq_len cannot be admitted
        # under any policy without silent truncation, which is forbidden.
        # Fail fast like `reject` rather than relying on engine-side guards.
        if original > max_seq_len:
            raise ContextBudgetError(
                f"prompt of {original} tokens exceeds max_seq_len"
                f" ({max_seq_len}) under policy 'allow_context_stop';"
                f" choose a truncation policy or shorten the prompt."
            )
        return ContextPolicyOutcome(
            accepted_token_ids=list(token_ids),
            original_prompt_tokens=original,
            accepted_prompt_tokens=original,
            truncated_prompt_tokens=0,
            rejected_prompt_tokens=0,
            policy=policy,
        )

    if policy == "reject":
        if original + max_new_tokens > max_seq_len:
            raise ContextBudgetError(
                f"prompt of {original} tokens plus {max_new_tokens}"
                f" generation tokens exceeds max_seq_len ({max_seq_len})"
                f" under policy 'reject'."
            )
        return ContextPolicyOutcome(
            accepted_token_ids=list(token_ids),
            original_prompt_tokens=original,
            accepted_prompt_tokens=original,
            truncated_prompt_tokens=0,
            rejected_prompt_tokens=0,
            policy=policy,
        )

    # Truncation policies share a single budget calculation. The budget is
    # the maximum number of prompt tokens that can be admitted while still
    # leaving room for max_new_tokens generated tokens within max_seq_len.
    budget = max_seq_len - max_new_tokens
    if budget <= 0:
        # max_new_tokens consumes the entire cache, leaving no room for any
        # prompt token. The spec forbids prefilling an empty prompt, so this
        # has no admissible outcome.
        raise ContextBudgetError(
            f"max_new_tokens ({max_new_tokens}) leaves no room for any"
            f" prompt token within max_seq_len ({max_seq_len}); cannot"
            f" satisfy truncation policy {policy!r}."
        )

    if policy in ("truncate_left", "reserve_generation"):
        if original <= budget:
            accepted = list(token_ids)
            truncated = 0
        else:
            accepted = list(token_ids[-budget:])
            truncated = original - budget
    elif policy == "truncate_right":
        if original <= budget:
            accepted = list(token_ids)
            truncated = 0
        else:
            accepted = list(token_ids[:budget])
            truncated = original - budget
    else:
        # Defensive: _VALID_POLICIES guards the entry point, so this branch is
        # unreachable unless the literal type and the dispatch drift apart.
        raise AssertionError(f"unhandled policy {policy!r}")

    return ContextPolicyOutcome(
        accepted_token_ids=accepted,
        original_prompt_tokens=original,
        accepted_prompt_tokens=len(accepted),
        truncated_prompt_tokens=truncated,
        rejected_prompt_tokens=0,
        policy=policy,
    )
