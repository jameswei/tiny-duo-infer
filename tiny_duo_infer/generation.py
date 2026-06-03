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

StopReason = Literal["eos", "max_new_tokens", "stop_string", "context_length"]


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


@dataclass
class GenerationResponse:
    """Completed generation result with token accounting and stop metadata."""

    text: str
    prompt_tokens: int
    generated_tokens: int
    stop_reason: StopReason
