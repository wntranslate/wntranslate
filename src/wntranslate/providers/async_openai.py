from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from .base import AsyncProvider, Message
from .openai import ProviderError, ReasoningEffort, _build_body, _delta


@dataclass
class AsyncOpenAIProvider(AsyncProvider):
    """Async client for any OpenAI-compatible chat completions API, using aiohttp.

    Args:
        model: Model name to request.
        api_key: Bearer token, or None for endpoints that need none.
        base_url: API base URL without the endpoint path.
        timeout: Request timeout in seconds, or None for no limit.
        temperature: Sampling temperature, or None to use the API default.
        supports_schema: Whether the API supports a json_schema response_format;
            when False, the schema is added to the prompt and JSON mode is used.
        reasoning_effort: Reasoning effort to request, or None to omit the field.
        headers: Extra headers to send.
        stream: Whether to stream the response (and call on_token per piece).
        on_token: Callback invoked with each streamed text piece.
    """

    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float | None = 600.0
    temperature: float | None = 0.3
    supports_schema: bool = True
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.LOW
    headers: dict[str, str] = field(default_factory=dict)
    stream: bool = False
    on_token: Callable[[str], None] | None = None

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
        payload = self._payload(messages, response_format, extra_body)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        if self.stream:
            return await self._stream(url, payload, timeout)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(url, json=payload, headers=self._headers()) as response,
            ):
                body = await response.read()
                if response.status != 200:
                    raise ProviderError(f"{response.status}: {body.decode('utf-8', 'replace')}")
                decoded = json.loads(body)
        except aiohttp.ClientError as error:
            raise ProviderError(str(error)) from error
        if not isinstance(decoded, dict):
            raise ProviderError(f"expected a JSON object, got {type(decoded).__name__}")
        try:
            content = decoded["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f"unexpected response: {decoded!r}") from error
        if not isinstance(content, str):
            raise ProviderError(f"non-text content: {content!r}")
        return content

    async def _stream(self, url: str, payload: dict[str, Any], timeout: aiohttp.ClientTimeout) -> str:
        payload = {**payload, "stream": True}
        pieces: list[str] = []
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(url, json=payload, headers=self._headers()) as response,
            ):
                if response.status != 200:
                    body = await response.read()
                    raise ProviderError(f"{response.status}: {body.decode('utf-8', 'replace')}")
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    content, reasoning = _delta(data)
                    if reasoning and self.on_token is not None:
                        self.on_token(reasoning)
                    if content:
                        pieces.append(content)
                        if self.on_token is not None:
                            self.on_token(content)
        except aiohttp.ClientError as error:
            raise ProviderError(str(error)) from error
        return "".join(pieces)

    def _payload(
        self,
        messages: Sequence[Message],
        response_format: dict[str, Any] | None,
        extra_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return _build_body(
            self.model, messages, response_format, self.temperature, self.supports_schema, self.reasoning_effort, extra_body
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


@dataclass
class AsyncDeepSeekProvider(AsyncOpenAIProvider):
    """AsyncOpenAIProvider preset for DeepSeek, which supports only json_object output."""

    base_url: str = "https://api.deepseek.com/v1"
    supports_schema: bool = False
