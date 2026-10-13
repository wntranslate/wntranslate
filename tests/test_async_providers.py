import asyncio
import json
from typing import Any, Self

import aiohttp
import pytest

from wntranslate import Message, ProviderError, Usage
from wntranslate.providers.async_openai import AsyncOpenAIProvider


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


def test_async_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    resp = _FakeResp(200, json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode("utf-8"))

    def factory(*args: object, **kwargs: object) -> _FakeSession:
        captured["timeout"] = kwargs.get("timeout")
        return _FakeSession(resp, captured)

    monkeypatch.setattr(aiohttp, "ClientSession", factory)
    provider = AsyncOpenAIProvider(model="gpt-x", api_key="secret", base_url="https://api.example/v1")
    out = asyncio.run(
        provider.complete(
            [Message(role="user", content="hi")],
            response_format={"type": "json_object"},
        )
    )
    assert out == "hello"
    assert captured["url"] == "https://api.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["model"] == "gpt-x"
    assert captured["json"]["temperature"] == 0.3
    assert captured["json"]["response_format"] == {"type": "json_object"}


def test_async_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _FakeResp(429, b"slow down")
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: _FakeSession(resp, {}))
    provider = AsyncOpenAIProvider(model="gpt-x")
    with pytest.raises(ProviderError, match="429"):
        asyncio.run(provider.complete([Message(role="user", content="hi")]))


def test_async_payload_and_headers() -> None:
    provider = AsyncOpenAIProvider(model="m", api_key="k", temperature=0.5)
    payload = provider._payload([Message(role="user", content="x")], None, None)
    assert payload["temperature"] == 0.5
    assert provider._headers()["Authorization"] == "Bearer k"
    assert "Authorization" not in AsyncOpenAIProvider(model="m")._headers()


def test_async_usage_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 90, "completion_tokens": 12}}
    resp = _FakeResp(200, json.dumps(payload).encode("utf-8"))
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: _FakeSession(resp, {}))
    seen: list[Usage] = []
    provider = AsyncOpenAIProvider(model="gpt-x", on_usage=seen.append)
    asyncio.run(provider.complete([Message(role="user", content="hi")]))
    assert seen == [Usage(90, 12)]
