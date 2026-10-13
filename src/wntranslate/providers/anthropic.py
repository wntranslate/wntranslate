from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .base import Message, Provider, Usage, _int
from .openai import ProviderError

_ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class AnthropicProvider(Provider):
    """Client for the Anthropic Messages API.

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

    def complete(
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
                (the Anthropic API has no response_format field); the schema is taken from the
                prompt the caller already inlined.
            extra_body: Extra fields merged into the request body.

        Returns:
            The assistant message content.
        """
        prefill = "{" if response_format is not None else None
        body = _build_body(self.model, messages, self.temperature, self.max_tokens, prefill, extra_body)
        if self.stream:
            text = self._stream(body)
        else:
            payload = self._post(body)
            self._emit(_usage(payload))
            text = _text(payload)
        return prefill + text if prefill is not None else text

    def _emit(self, usage: Usage) -> None:
        if self.on_usage is not None and usage:
            self.on_usage(usage)

    def _request(self, body: dict[str, Any]) -> urllib.request.Request:
        headers = {"content-type": "application/json", "anthropic-version": self.anthropic_version, **self.headers}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        url = f"{self.base_url.rstrip('/')}/messages"
        return urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(self._request(body), timeout=self.timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise ProviderError(f"{error.code} {error.reason}: {detail}") from error
        except urllib.error.URLError as error:
            raise ProviderError(str(error.reason)) from error
        if not isinstance(decoded, dict):
            raise ProviderError(f"expected a JSON object, got {type(decoded).__name__}")
        return decoded

    def _stream(self, body: dict[str, Any]) -> str:
        request = self._request({**body, "stream": True})
        pieces: list[str] = []
        usage = _StreamUsage()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if line.startswith("data:"):
                        self._consume(line[len("data:") :].strip(), pieces, usage)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise ProviderError(f"{error.code} {error.reason}: {detail}") from error
        except urllib.error.URLError as error:
            raise ProviderError(str(error.reason)) from error
        self._emit(usage.done())
        return "".join(pieces)

    def _consume(self, data: str, pieces: list[str], usage: _StreamUsage) -> None:
        text, thinking = _delta(data)
        if thinking and self.on_token is not None:
            self.on_token(thinking)
        if text:
            pieces.append(text)
            if self.on_token is not None:
                self.on_token(text)
        usage.feed(data)


def _build_body(
    model: str,
    messages: Sequence[Message],
    temperature: float | None,
    max_tokens: int,
    prefill: str | None,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a Messages request body from OpenAI-style messages.

    Leading system messages become the top-level ``system`` prompt; any later system message
    is treated as a user turn. Consecutive same-role turns are merged so the roles alternate,
    and ``prefill`` (when given) is appended as an assistant turn to force its start.
    """
    system_parts: list[str] = []
    turns: list[dict[str, str]] = []
    for message in messages:
        if not turns and message.role == "system":
            system_parts.append(message.content)
            continue
        role = "assistant" if message.role == "assistant" else "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + message.content
        else:
            turns.append({"role": role, "content": message.content})
    if prefill is not None:
        if turns and turns[-1]["role"] == "assistant":
            turns[-1]["content"] += "\n\n" + prefill
        else:
            turns.append({"role": "assistant", "content": prefill})
    body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": turns}
    system = "\n\n".join(system_parts)
    if system:
        body["system"] = system
    if temperature is not None:
        body["temperature"] = temperature
    if extra_body:
        body.update(extra_body)
    return body


def _text(payload: dict[str, Any]) -> str:
    """Concatenate the text of a Messages response's content blocks."""
    content = payload.get("content")
    if not isinstance(content, list):
        raise ProviderError(f"unexpected response: {payload!r}")
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def _delta(data: str) -> tuple[str, str]:
    """Extract the (text, thinking) pieces from one Messages SSE data payload.

    Args:
        data: The JSON text after a 'data:' prefix.

    Returns:
        A (text, thinking) pair; each is an empty string when absent.
    """
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return "", ""
    if not isinstance(event, dict):
        return "", ""
    delta = event.get("delta")
    if not isinstance(delta, dict):
        return "", ""
    if delta.get("type") == "text_delta":
        text = delta.get("text")
        return (text if isinstance(text, str) else ""), ""
    if delta.get("type") == "thinking_delta":
        thinking = delta.get("thinking")
        return "", (thinking if isinstance(thinking, str) else "")
    return "", ""


def _usage(payload: dict[str, Any]) -> Usage:
    """Read token usage from a Messages response body."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return Usage()
    return Usage(_int(usage.get("input_tokens")), _int(usage.get("output_tokens")))


class _StreamUsage:
    """Accumulates token usage across Messages SSE events.

    Input tokens arrive once in message_start; output tokens arrive as a running total in
    message_start and each message_delta, so the last non-zero value is the final count.
    """

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def feed(self, data: str) -> None:
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return
        source = event.get("message") if event.get("type") == "message_start" else event
        usage = source.get("usage") if isinstance(source, dict) else None
        if not isinstance(usage, dict):
            return
        self.input_tokens = _int(usage.get("input_tokens")) or self.input_tokens
        self.output_tokens = _int(usage.get("output_tokens")) or self.output_tokens

    def done(self) -> Usage:
        return Usage(self.input_tokens, self.output_tokens)
