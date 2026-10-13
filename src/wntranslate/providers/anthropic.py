from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .base import Message, Provider, Usage, _int
from .openai import (
    ProviderConnectionError,
    ProviderError,
    ProviderHTTPError,
    _schema_of,
)

_EPHEMERAL = {"type": "ephemeral"}

_ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class AnthropicProvider(Provider):
    """Client for the Anthropic Messages API.

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
            response_format: When this carries a JSON schema, it is enforced by forcing a single
                tool whose input_schema is that schema (the Anthropic way to force structured
                output); the tool's input is returned as a JSON string.
            extra_body: Extra fields merged into the request body.

        Returns:
            The assistant message content, or the forced tool's input as JSON.
        """
        schema = _schema_of(response_format) if response_format is not None else None
        body = _build_body(self.model, messages, self.max_tokens, extra_body)
        if response_format is not None and schema is not None:
            _force_tool(body, _tool_name(response_format), schema)
        if self.stream:
            return self._stream(body, tool=schema is not None)
        payload = self._post(body)
        self._emit(_usage(payload))
        return _tool_input(payload) if schema is not None else _text(payload)

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
            raise ProviderHTTPError(error.code, f"{error.reason}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise ProviderConnectionError(str(getattr(error, "reason", error))) from error
        if not isinstance(decoded, dict):
            raise ProviderError(f"expected a JSON object, got {type(decoded).__name__}")
        return decoded

    def _stream(self, body: dict[str, Any], tool: bool) -> str:
        request = self._request({**body, "stream": True})
        text_pieces: list[str] = []
        json_pieces: list[str] = []
        usage = _StreamUsage()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if line.startswith("data:"):
                        self._consume(line[len("data:") :].strip(), text_pieces, json_pieces, usage)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise ProviderHTTPError(error.code, f"{error.reason}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise ProviderConnectionError(str(getattr(error, "reason", error))) from error
        self._emit(usage.done())
        return "".join(json_pieces) if tool else "".join(text_pieces)

    def _consume(self, data: str, text_pieces: list[str], json_pieces: list[str], usage: _StreamUsage) -> None:
        text, thinking, partial_json = _delta(data)
        if thinking and self.on_token is not None:
            self.on_token(thinking)
        for piece, sink in ((text, text_pieces), (partial_json, json_pieces)):
            if piece:
                sink.append(piece)
                if self.on_token is not None:
                    self.on_token(piece)
        usage.feed(data)


def _build_body(
    model: str,
    messages: Sequence[Message],
    max_tokens: int,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a Messages request body from OpenAI-style messages.

    Leading system messages become the top-level ``system`` prompt; any later system message is
    treated as a user turn. Consecutive same-role turns are merged so the roles alternate, and the
    conversation ends with the last turn as given (a user turn in this library's flows) — no
    assistant prefill is appended, since current models reject it. Temperature is intentionally not
    sent: some current models deprecate it on this endpoint.
    """
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        if not turns and message.role == "system":
            system_parts.append(message.content)
            continue
        role = "assistant" if message.role == "assistant" else "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + message.content
        else:
            turns.append({"role": role, "content": message.content})
    if turns:
        turns[-1]["content"] = [_cached(turns[-1]["content"])]
    body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": turns}
    system = "\n\n".join(system_parts)
    if system:
        body["system"] = [_cached(system)]
    if extra_body:
        body.update(extra_body)
    return body


def _cached(text: str) -> dict[str, Any]:
    """A text content block marked for ephemeral prompt caching, cutting cost across passes."""
    return {"type": "text", "text": text, "cache_control": _EPHEMERAL}


def _tool_name(response_format: dict[str, Any]) -> str:
    """Pick a valid Anthropic tool name from the response_format, defaulting to 'respond'."""
    name = response_format.get("json_schema", {}).get("name")
    return name if isinstance(name, str) and name else "respond"


def _force_tool(body: dict[str, Any], name: str, schema: dict[str, Any]) -> None:
    """Add a single tool with the schema as input_schema and force the model to call it.

    This is Anthropic's way to enforce a JSON schema. A root-level oneOf is wrapped so the
    input_schema is an object schema, as the API expects; the tool is marked for prompt caching.
    """
    input_schema = schema if schema.get("type") == "object" else {"type": "object", **schema}
    body["tools"] = [
        {
            "name": name,
            "description": "Return the result as a single structured object matching the schema.",
            "input_schema": input_schema,
            "cache_control": _EPHEMERAL,
        }
    ]
    body["tool_choice"] = {"type": "tool", "name": name}


def _tool_input(payload: dict[str, Any]) -> str:
    """Return the forced tool's input (the structured result) from a Messages response, as JSON."""
    content = payload.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
                return json.dumps(block["input"], ensure_ascii=False)
    raise ProviderError(f"no tool_use block in response: {payload!r}")


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


def _delta(data: str) -> tuple[str, str, str]:
    """Extract the (text, thinking, tool_json) pieces from one Messages SSE data payload.

    Args:
        data: The JSON text after a 'data:' prefix.

    Returns:
        A (text, thinking, tool_json) triple; each is an empty string when absent. tool_json is a
        fragment of the forced tool's input, concatenated across input_json_delta events.
    """
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return "", "", ""
    if not isinstance(event, dict):
        return "", "", ""
    delta = event.get("delta")
    if not isinstance(delta, dict):
        return "", "", ""
    kind = delta.get("type")
    if kind == "text_delta":
        text = delta.get("text")
        return (text if isinstance(text, str) else ""), "", ""
    if kind == "thinking_delta":
        thinking = delta.get("thinking")
        return "", (thinking if isinstance(thinking, str) else ""), ""
    if kind == "input_json_delta":
        partial = delta.get("partial_json")
        return "", "", (partial if isinstance(partial, str) else "")
    return "", "", ""


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
