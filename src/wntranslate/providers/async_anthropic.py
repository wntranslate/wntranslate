from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from .anthropic import (
    _ANTHROPIC_VERSION,
    _build_body,
    _delta,
    _StreamUsage,
    _text,
    _usage,
)
from .base import AsyncProvider, Message, Usage
from .openai import ProviderError


@dataclass
class AsyncAnthropicProvider(AsyncProvider):
    """Async client for the Anthropic Messages API, using aiohttp.

    Args:
        model: Model name to request.
        api_key: Anthropic API key, or None for endpoints that need none.
        base_url: API base URL without the endpoint path.
        timeout: Request timeout in seconds, or None for no limit.
        temperature: Sampling temperature (0-1), or None to use the API default.
        max_tokens: Maximum tokens to generate; required by the API.
        anthropic_version: Value sent as the anthropic-version header.
        headers: Extra headers to send.
        stream: Whether to stream the response (and call on_token per piece).
        on_token: Callback invoked with each streamed text piece.
        on_usage: Callback invoked once per completion with its token Usage.
    """

    model: str
    api_key: str | None = None
    base_url: str = "https://api.anthropic.com/v1"
    timeout: float | None = 1200.0
    temperature: float | None = 0.3
    max_tokens: int = 8192
    anthropic_version: str = _ANTHROPIC_VERSION
    headers: dict[str, str] = field(default_factory=dict)
    stream: bool = False
    on_token: Callable[[str], None] | None = None
    on_usage: Callable[[Usage], None] | None = None

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Send a Messages request and return the assistant text.

        Args:
            messages: The conversation messages.
            response_format: When not None, JSON output is forced via an assistant prefix
                (the Anthropic API has no response_format field).
            extra_body: Extra fields merged into the request body.

        Returns:
            The assistant message content.
        """
        prefill = "{" if response_format is not None else None
        body = _build_body(self.model, messages, self.temperature, self.max_tokens, prefill, extra_body)
        url = f"{self.base_url.rstrip('/')}/messages"
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        if self.stream:
            text = await self._stream(url, body, timeout)
            return prefill + text if prefill is not None else text
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(url, json=body, headers=self._headers()) as response,
            ):
                raw = await response.read()
                if response.status != 200:
                    raise ProviderError(f"{response.status}: {raw.decode('utf-8', 'replace')}")
                decoded = json.loads(raw)
        except aiohttp.ClientError as error:
            raise ProviderError(str(error)) from error
        if not isinstance(decoded, dict):
            raise ProviderError(f"expected a JSON object, got {type(decoded).__name__}")
        self._emit(_usage(decoded))
        text = _text(decoded)
        return prefill + text if prefill is not None else text

    def _emit(self, usage: Usage) -> None:
        if self.on_usage is not None and usage:
            self.on_usage(usage)

    async def _stream(self, url: str, body: dict[str, Any], timeout: aiohttp.ClientTimeout) -> str:
        payload = {**body, "stream": True}
        pieces: list[str] = []
        usage = _StreamUsage()
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(url, json=payload, headers=self._headers()) as response,
            ):
                if response.status != 200:
                    raw = await response.read()
                    raise ProviderError(f"{response.status}: {raw.decode('utf-8', 'replace')}")
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    text, thinking = _delta(data)
                    if thinking and self.on_token is not None:
                        self.on_token(thinking)
                    if text:
                        pieces.append(text)
                        if self.on_token is not None:
                            self.on_token(text)
                    usage.feed(data)
        except aiohttp.ClientError as error:
            raise ProviderError(str(error)) from error
        self._emit(usage.done())
        return "".join(pieces)

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", "anthropic-version": self.anthropic_version, **self.headers}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers
