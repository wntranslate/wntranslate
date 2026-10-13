from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .base import Message, Provider, Usage, _int


class ProviderError(RuntimeError):
    """Raised when a provider request fails."""


class ReasoningEffort(Enum):
    """Reasoning effort for an OpenAI-compatible model, sent as the `reasoning_effort` field.

    The level is passed through to the API verbatim; each endpoint accepts only some of these
    and may reject or silently ignore the rest, so treat support as a best guess. NONE, LOW,
    HIGH, and MAX tend to be the most broadly accepted; MEDIUM is accepted by OpenAI and
    llama.cpp's server; MINIMAL and XHIGH are OpenAI-specific (MINIMAL is the lowest reasoning
    level, just below LOW). Set a provider's reasoning_effort to None instead of a member to
    omit the field entirely, for endpoints that do not accept it at all.
    """

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@dataclass
class OpenAIProvider(Provider):
    """Client for any OpenAI-compatible chat completions API.

    Args:
        model: Model name to request.
        api_key: Bearer token, or None for endpoints that need none.
        base_url: API base URL without the endpoint path.
        timeout: Request timeout in seconds, or None for no limit.
        temperature: Sampling temperature, or None to use the API default.
        supports_schema: Whether the API supports a json_schema response_format;
            when False, the schema is added to the prompt and JSON mode is used.
        reasoning_effort: Reasoning effort to request, or None to omit the field.
        frequency_penalty: OpenAI frequency_penalty, or None to omit it; a small positive value
            (e.g. 0.3) curbs runaway repetition where a weak model degenerates into an endless run.
        presence_penalty: OpenAI presence_penalty, or None to omit it.
        headers: Extra headers to send.
        stream: Whether to stream the response (and call on_token per piece).
        on_token: Callback invoked with each streamed text piece.
        on_usage: Callback invoked once per completion with its token Usage.
    """

    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float | None = 1200.0
    temperature: float | None = 0.4
    supports_schema: bool = True
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.NONE
    frequency_penalty: float | None = 0.3
    presence_penalty: float | None = None
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
        """Send a chat completion request and return the assistant text.

        Args:
            messages: The conversation messages.
            response_format: Provider response_format object, or None.
            extra_body: Extra fields merged into the request body.

        Returns:
            The assistant message content.
        """
        body = _build_body(self.model, messages, response_format, self.supports_schema, self._sampling(), extra_body)
        if self.stream:
            return self._stream(body)
        payload = self._post(body)
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f"unexpected response: {payload!r}") from error
        if not isinstance(content, str):
            raise ProviderError(f"non-text content: {content!r}")
        self._emit(_usage(payload))
        return content

    def _emit(self, usage: Usage) -> None:
        if self.on_usage is not None and usage:
            self.on_usage(usage)

    def _sampling(self) -> dict[str, Any]:
        return _sampling_params(self.temperature, self.reasoning_effort, self.frequency_penalty, self.presence_penalty)

    def _request(self, body: dict[str, Any]) -> urllib.request.Request:
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url.rstrip('/')}/chat/completions"
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
        request = self._request({**body, "stream": True, "stream_options": {"include_usage": True}})
        pieces: list[str] = []
        usage = Usage()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    usage = self._consume(data, pieces) or usage
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise ProviderError(f"{error.code} {error.reason}: {detail}") from error
        except urllib.error.URLError as error:
            raise ProviderError(str(error.reason)) from error
        self._emit(usage)
        return "".join(pieces)

    def _consume(self, data: str, pieces: list[str]) -> Usage | None:
        content, reasoning = _delta(data)
        if reasoning and self.on_token is not None:
            self.on_token(reasoning)
        if content:
            pieces.append(content)
            if self.on_token is not None:
                self.on_token(content)
        return _stream_usage(data)


@dataclass
class DeepSeekProvider(OpenAIProvider):
    """OpenAIProvider preset for DeepSeek, which supports only json_object output."""

    base_url: str = "https://api.deepseek.com/v1"
    supports_schema: bool = False


def _schema_of(response_format: dict[str, Any]) -> dict[str, Any] | None:
    """Return the JSON schema inside a json_schema response_format, if any."""
    if response_format.get("type") != "json_schema":
        return None
    schema = response_format.get("json_schema", {}).get("schema")
    return schema if isinstance(schema, dict) else None


def _sampling_params(
    temperature: float | None,
    reasoning_effort: ReasoningEffort | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
) -> dict[str, Any]:
    """Collect the sampling fields to send, omitting any left as None."""
    params: dict[str, Any] = {}
    if temperature is not None:
        params["temperature"] = temperature
    if reasoning_effort is not None:
        params["reasoning_effort"] = reasoning_effort.value
    if frequency_penalty is not None:
        params["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        params["presence_penalty"] = presence_penalty
    return params


def _build_body(
    model: str,
    messages: Sequence[Message],
    response_format: dict[str, Any] | None,
    supports_schema: bool,
    sampling: dict[str, Any],
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the chat completion request body.

    A json_schema response_format is downgraded to json_object when the endpoint cannot
    enforce a schema; callers that need the schema in the prompt must include it themselves.
    """
    chat = [{"role": message.role, "content": message.content} for message in messages]
    body: dict[str, Any] = {"model": model, "messages": chat}
    if response_format is not None:
        if not supports_schema and _schema_of(response_format) is not None:
            body["response_format"] = {"type": "json_object"}
        else:
            body["response_format"] = response_format
    body.update(sampling)
    if extra_body:
        body.update(extra_body)
    return body


def _delta(data: str) -> tuple[str, str]:
    """Extract the (content, reasoning_content) pieces from one SSE data payload.

    Args:
        data: The JSON text after a 'data:' prefix.

    Returns:
        A (content, reasoning) pair; each is an empty string when absent.
    """
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return "", ""
    if not isinstance(event, dict) or not event.get("choices"):
        return "", ""
    delta = event["choices"][0].get("delta") or {}
    content = delta.get("content")
    reasoning = delta.get("reasoning_content")
    return (content if isinstance(content, str) else ""), (reasoning if isinstance(reasoning, str) else "")


def _usage(payload: dict[str, Any]) -> Usage:
    """Read token usage from a chat completions response (or streamed chunk)."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return Usage()
    return Usage(_int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens")))


def _stream_usage(data: str) -> Usage | None:
    """Read token usage from one SSE data payload, or None when it carries none.

    OpenAI-compatible endpoints send usage in a final chunk (with include_usage) whose
    choices list is empty; non-usage chunks yield None so the caller keeps the last total.
    """
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    usage = _usage(event)
    return usage or None


def json_schema_format(name: str, schema: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """Build a response_format for OpenAI-style json_schema output.

    Args:
        name: Schema name.
        schema: The JSON schema.
        strict: Whether to request strict adherence.

    Returns:
        A response_format object.
    """
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": strict}}
