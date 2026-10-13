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
    _force_tool,
    _StreamUsage,
    _text,
    _tool_input,
    _tool_name,
    _usage,
)
from .base import AsyncProvider, Message, Usage
from .openai import (
    ProviderConnectionError,
    ProviderError,
    ProviderHTTPError,
    _schema_of,
)


@dataclass
class AsyncAnthropicProvider(AsyncProvider):
    """Async client for the Anthropic Messages API, using aiohttp.

    Args:
        model: Model name to request.
        api_key: Anthropic API key, or None for endpoints that need none.
        base_url: API base URL without the endpoint path.
        timeout: Request timeout in seconds, or None for no limit.
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
            response_format: When this carries a JSON schema, it is enforced by forcing a single
                tool whose input_schema is that schema; the tool's input is returned as JSON.
            extra_body: Extra fields merged into the request body.

        Returns:
            The assistant message content, or the forced tool's input as JSON.
        """
        schema = _schema_of(response_format) if response_format is not None else None
        body = _build_body(self.model, messages, self.max_tokens, extra_body)
        if response_format is not None and schema is not None:
            _force_tool(body, _tool_name(response_format), schema)
        url = f"{self.base_url.rstrip('/')}/messages"
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        if self.stream:
            return await self._stream(url, body, timeout, tool=schema is not None)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(url, json=body, headers=self._headers()) as response,
            ):
                raw = await response.read()
                if response.status != 200:
                    raise ProviderHTTPError(response.status, raw.decode("utf-8", "replace"))
                decoded = json.loads(raw)
        except (aiohttp.ClientError, TimeoutError) as error:
            raise ProviderConnectionError(str(error)) from error
        if not isinstance(decoded, dict):
            raise ProviderError(f"expected a JSON object, got {type(decoded).__name__}")
        self._emit(_usage(decoded))
        return _tool_input(decoded) if schema is not None else _text(decoded)

    def _emit(self, usage: Usage) -> None:
        if self.on_usage is not None and usage:
            self.on_usage(usage)

    async def _stream(self, url: str, body: dict[str, Any], timeout: aiohttp.ClientTimeout, tool: bool) -> str:
        payload = {**body, "stream": True}
        text_pieces: list[str] = []
        json_pieces: list[str] = []
        usage = _StreamUsage()
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(url, json=payload, headers=self._headers()) as response,
            ):
                if response.status != 200:
                    raw = await response.read()
                    raise ProviderHTTPError(response.status, raw.decode("utf-8", "replace"))
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    text, thinking, partial_json = _delta(data)
                    if thinking and self.on_token is not None:
                        self.on_token(thinking)
                    for piece, sink in ((text, text_pieces), (partial_json, json_pieces)):
                        if piece:
                            sink.append(piece)
                            if self.on_token is not None:
                                self.on_token(piece)
                    usage.feed(data)
        except (aiohttp.ClientError, TimeoutError) as error:
            raise ProviderConnectionError(str(error)) from error
        self._emit(usage.done())
        return "".join(json_pieces) if tool else "".join(text_pieces)

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", "anthropic-version": self.anthropic_version, **self.headers}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers
