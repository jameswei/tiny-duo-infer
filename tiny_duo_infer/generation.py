"""
Project-owned generation request and response types.

These types sit between the CLI/HTTP layer and the engine. They validate
parameters before model execution and carry structured metadata back to callers.

Stop reasons:
  "eos"            — EOS token was sampled.
  "max_new_tokens" — max_new_tokens limit was reached.
  "stop_string"    — a configured stop string was matched in the decoded output.
  "context_length" — the request would exceed max_seq_len.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


_VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})
_VALID_QUANTIZATION_MODES: frozenset[str] = frozenset({"none", "int4", "int8"})

StopReason = Literal["eos", "max_new_tokens", "stop_string", "context_length"]

ContextPolicy = Literal[
    "allow_context_stop",
    "reject",
    "truncate_left",
    "truncate_right",
    "reserve_generation",
]

_VALID_CONTEXT_POLICIES: frozenset[str] = frozenset({
    "allow_context_stop",
    "reject",
    "truncate_left",
    "truncate_right",
    "reserve_generation",
})


def kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
    bytes_per_element: int = 4,
) -> int:
    """Compute KV-cache memory in bytes for one sequence length.

    Formula: 2 (K+V) * n_layers * n_kv_heads * seq_len * head_dim * bytes_per_element
    """
    return 2 * n_layers * n_kv_heads * seq_len * head_dim * bytes_per_element


@dataclass
class ChatMessage:
    """A single chat turn with a role and text content."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"ChatMessage role must be one of {sorted(_VALID_ROLES)!r},"
                f" got {self.role!r}."
            )
        if not self.content:
            raise ValueError("ChatMessage content must not be empty.")


@dataclass
class GenerationRequest:
    """
    A single generation request.

    Exactly one of `prompt` or `messages` must be provided.
    `messages` is only valid when `chat=True`.
    """

    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    chat: bool = False
    context_policy: ContextPolicy = "allow_context_stop"

    def __post_init__(self) -> None:
        if self.prompt is None and self.messages is None:
            raise ValueError(
                "Exactly one of 'prompt' or 'messages' must be provided; got neither."
            )
        if self.prompt is not None and self.messages is not None:
            raise ValueError(
                "Exactly one of 'prompt' or 'messages' must be provided; got both."
            )

        if self.prompt is not None and not self.prompt:
            raise ValueError("'prompt' must not be empty.")

        if self.messages is not None:
            if not self.chat:
                raise ValueError("'messages' requires chat=True.")
            if not self.messages:
                raise ValueError("'messages' must not be an empty list.")

        if self.max_new_tokens < 0:
            raise ValueError(
                f"max_new_tokens must be >= 0, got {self.max_new_tokens}."
            )
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be >= 0.0, got {self.temperature}."
            )
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}.")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(
                f"top_p must be in (0.0, 1.0], got {self.top_p}."
            )

        for s in self.stop:
            if not s:
                raise ValueError("Each stop string must be non-empty.")

        if self.context_policy not in _VALID_CONTEXT_POLICIES:
            raise ValueError(
                f"context_policy must be one of {sorted(_VALID_CONTEXT_POLICIES)!r},"
                f" got {self.context_policy!r}."
            )


@dataclass
class GenerationStats:
    """Per-request generation metrics: timing, token accounting, and KV-cache memory.

    Invariants enforced at construction:
      - prompt_tokens == accepted_prompt_tokens
      - active_seq_len == accepted_prompt_tokens + generated_tokens
    """

    # Context policy applied and token-budget accounting
    context_policy: str
    original_prompt_tokens: int
    accepted_prompt_tokens: int
    truncated_prompt_tokens: int
    rejected_prompt_tokens: int

    # Mirrors GenerationResponse fields for standalone readability
    prompt_tokens: int
    generated_tokens: int
    stop_reason: str

    # Timing (milliseconds)
    prompt_prepare_ms: float
    prefill_ms: float
    time_to_first_token_ms: float
    decode_ms: float
    total_ms: float
    decode_tokens_per_sec: float

    # KV-cache memory
    kv_cache_allocated_bytes: int
    kv_cache_active_bytes: int
    max_seq_len: int
    active_seq_len: int

    # Optional profiling detail — omitted from HTTP responses by default
    decode_step_ms: list[float] = field(default_factory=list)
    model_type: str = ""

    # Phase 1.8 — weight quantization metadata.
    # Defaults represent the no-quantization (full-precision) path so existing
    # callers that do not set these fields remain valid.
    quantization_mode: str = "none"           # "none", "int4", or "int8"
    quantization_bits: int | None = None
    quantization_group_size: int | None = None
    quantized_linear_count: int = 0
    full_precision_linear_count: int = 0
    linear_weight_full_precision_bytes: int = 0
    linear_weight_runtime_bytes: int = 0

    def __post_init__(self) -> None:
        if self.quantization_mode not in _VALID_QUANTIZATION_MODES:
            raise ValueError(
                f"quantization_mode must be one of "
                f"{sorted(_VALID_QUANTIZATION_MODES)!r}, "
                f"got {self.quantization_mode!r}."
            )

        # Non-negative counts and byte fields — checked before coherence so that
        # negative values always produce a clear "must be >= 0" message regardless
        # of quantization_mode.
        for _name, _val in (
            ("quantized_linear_count", self.quantized_linear_count),
            ("full_precision_linear_count", self.full_precision_linear_count),
            ("linear_weight_full_precision_bytes", self.linear_weight_full_precision_bytes),
            ("linear_weight_runtime_bytes", self.linear_weight_runtime_bytes),
        ):
            if _val < 0:
                raise ValueError(f"{_name} must be >= 0, got {_val}.")

        # Coherence: bits/group_size/count must be consistent with mode.
        if self.quantization_mode == "none":
            if self.quantization_bits is not None:
                raise ValueError(
                    f"quantization_bits must be None when quantization_mode is 'none',"
                    f" got {self.quantization_bits}."
                )
            if self.quantization_group_size is not None:
                raise ValueError(
                    f"quantization_group_size must be None when quantization_mode is"
                    f" 'none', got {self.quantization_group_size}."
                )
            if self.quantized_linear_count != 0:
                raise ValueError(
                    f"quantized_linear_count must be 0 when quantization_mode is"
                    f" 'none', got {self.quantized_linear_count}."
                )
        else:
            expected_bits = 4 if self.quantization_mode == "int4" else 8
            if self.quantization_bits != expected_bits:
                raise ValueError(
                    f"quantization_bits must be {expected_bits} when"
                    f" quantization_mode is {self.quantization_mode!r},"
                    f" got {self.quantization_bits}."
                )
            if (
                self.quantization_group_size is None
                or self.quantization_group_size <= 0
            ):
                raise ValueError(
                    f"quantization_group_size must be a positive integer when"
                    f" quantization_mode is {self.quantization_mode!r},"
                    f" got {self.quantization_group_size}."
                )

        # Note: linear_weight_runtime_bytes <= linear_weight_full_precision_bytes
        # is NOT enforced as a hard invariant.  For tiny test matrices with small
        # group sizes, scales and biases overhead can exceed weight savings, making
        # runtime_bytes > full_precision_bytes a valid (if unusual) outcome.

        if self.context_policy not in _VALID_CONTEXT_POLICIES:
            raise ValueError(
                f"context_policy must be one of {sorted(_VALID_CONTEXT_POLICIES)!r},"
                f" got {self.context_policy!r}."
            )
        if self.prompt_tokens != self.accepted_prompt_tokens:
            raise ValueError(
                f"prompt_tokens ({self.prompt_tokens}) must equal"
                f" accepted_prompt_tokens ({self.accepted_prompt_tokens})."
            )
        expected_active = self.accepted_prompt_tokens + self.generated_tokens
        if self.active_seq_len != expected_active:
            raise ValueError(
                f"active_seq_len ({self.active_seq_len}) must equal"
                f" accepted_prompt_tokens + generated_tokens ({expected_active})."
            )


@dataclass
class GenerationResponse:
    """Completed generation result with token accounting and stop metadata."""

    text: str
    prompt_tokens: int
    generated_tokens: int
    stop_reason: StopReason
    stats: GenerationStats | None = None
