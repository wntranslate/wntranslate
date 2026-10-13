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
