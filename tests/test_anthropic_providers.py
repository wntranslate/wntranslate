import asyncio
import json
import urllib.request
from collections.abc import Iterator
from typing import Any, Self

import aiohttp
import pytest

from wntranslate import AnthropicProvider, Message, ProviderError, Usage
from wntranslate.providers.anthropic import _build_body
from wntranslate.providers.async_anthropic import AsyncAnthropicProvider


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class _FakeStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def __iter__(self) -> Iterator[bytes]:
        return iter(f"{line}\n".encode() for line in self._lines)


def _reply(text: str) -> bytes:
    return json.dumps({"content": [{"type": "text", "text": text}]}).encode("utf-8")


def test_anthropic_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["body"] = json.loads(request.data.decode("utf-8")) if request.data else None
        return _FakeResponse(_reply('"ok": true}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = AnthropicProvider(model="claude-x", api_key="sk", temperature=0.4, max_tokens=1234)
    out = provider.complete(
        [Message(role="system", content="sys one"), Message(role="system", content="sys two"), Message(role="user", content="hi")],
        response_format={"type": "json_schema"},
    )
    assert out == '{"ok": true}'  # the forced-JSON prefix plus the model's continuation
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk"
    assert captured["headers"]["anthropic-version"]
    assert captured["body"]["model"] == "claude-x"
    assert captured["body"]["max_tokens"] == 1234
    assert captured["body"]["temperature"] == 0.4
    assert captured["body"]["system"] == "sys one\n\nsys two"
    assert captured["body"]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "{"},
    ]


def test_anthropic_no_key_omits_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        return _FakeResponse(_reply("}"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    AnthropicProvider(model="local").complete([Message(role="user", content="hi")], response_format={"x": 1})
    assert "x-api-key" not in captured["headers"]


def test_anthropic_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ProviderError, match="400"):
        AnthropicProvider(model="claude-x").complete([Message(role="user", content="hi")])


def test_anthropic_no_prefill_without_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        return _FakeResponse(_reply("plain text"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = AnthropicProvider(model="claude-x").complete([Message(role="user", content="hi")])
    assert out == "plain text"


def test_anthropic_usage_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        payload = {"content": [{"type": "text", "text": "ok}"}], "usage": {"input_tokens": 200, "output_tokens": 45}}
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    seen: list[Usage] = []
    AnthropicProvider(model="claude-x", on_usage=seen.append).complete(
        [Message(role="user", content="hi")], response_format={"x": 1}
    )
    assert seen == [Usage(200, 45)]


def test_anthropic_stream_accumulates_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeStream:
        return _FakeStream(
            [
                'data: {"type": "message_start", "message": {"usage": {"input_tokens": 60, "output_tokens": 1}}}',
                'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hel"}}',
                'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo"}}',
                'data: {"type": "message_delta", "usage": {"output_tokens": 48}}',
                'data: {"type": "message_stop"}',
            ]
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    seen: list[Usage] = []
    out = AnthropicProvider(model="claude-x", stream=True, on_usage=seen.append).complete([Message(role="user", content="hi")])
    assert out == "hello"
    assert seen == [Usage(60, 48)]


def test_build_body_collapses_system_and_merges_roles() -> None:
    messages = [
        Message(role="system", content="base"),
        Message(role="user", content="u1"),
        Message(role="assistant", content="a1"),
        Message(role="system", content="data"),
        Message(role="user", content="u2"),
    ]
    body = _build_body("m", messages, 0.3, 8192, "{", None)
    assert body["system"] == "base"
    assert body["messages"] == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "data\n\nu2"},
        {"role": "assistant", "content": "{"},
    ]


class _FakeResp:
    def __init__(self, status: int, payload: bytes) -> None:
        self.status = status
        self._payload = payload

    async def read(self) -> bytes:
        return self._payload

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeSession:
    def __init__(self, resp: _FakeResp, captured: dict[str, Any]) -> None:
        self._resp = resp
        self._captured = captured

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> _FakeResp:
        self._captured["url"] = url
        self._captured["json"] = json
        self._captured["headers"] = headers
        return self._resp

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def test_async_anthropic_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    resp = _FakeResp(200, _reply('"ok": true}'))
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: _FakeSession(resp, captured))
    provider = AsyncAnthropicProvider(model="claude-x", api_key="sk")
    out = asyncio.run(provider.complete([Message(role="user", content="hi")], response_format={"x": 1}))
    assert out == '{"ok": true}'
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk"
    assert captured["json"]["messages"][-1] == {"role": "assistant", "content": "{"}


def test_async_anthropic_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: _FakeSession(_FakeResp(429, b"slow"), {}))
    provider = AsyncAnthropicProvider(model="claude-x")
    with pytest.raises(ProviderError, match="429"):
        asyncio.run(provider.complete([Message(role="user", content="hi")]))
