from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    """A single chat message.

    Args:
        role: The role, such as 'system', 'user', or 'assistant'.
        content: The message text.
    """

    role: str
    content: str


@dataclass(frozen=True)
class Usage:
    """Token usage reported by a provider for one completion.

    Args:
        input_tokens: Prompt (input) tokens consumed.
        output_tokens: Generated (output) tokens.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)

    def __bool__(self) -> bool:
        return bool(self.input_tokens or self.output_tokens)


def _int(value: Any) -> int:
    """Coerce a JSON value to a non-negative token count, treating anything odd as 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


class Provider(ABC):
    """Base class for chat completion providers."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant text.

        Args:
            messages: The conversation messages.
            response_format: Provider response_format object, or None.
            extra_body: Extra fields merged into the request body.

        Returns:
            The assistant message content.
        """
        ...


class AsyncProvider(ABC):
    """Base class for asynchronous chat completion providers."""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant text.

        Args:
            messages: The conversation messages.
            response_format: Provider response_format object, or None.
            extra_body: Extra fields merged into the request body.

        Returns:
            The assistant message content.
        """
        ...
