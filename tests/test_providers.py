import io
import json
import urllib.error
import urllib.request
from typing import Any, Self

import pytest

from wntranslate import (
    DeepSeekProvider,
    Message,
    OpenAIProvider,
    ProviderError,
    ReasoningEffort,
    json_schema_format,
)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def test_builds_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.data.decode("utf-8")) if request.data else None
        captured["timeout"] = timeout
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "hi there"}}]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAIProvider(model="gpt-x", api_key="secret", base_url="https://api.example/v1", timeout=12.0)
    out = provider.complete(
        [Message(role="system", content="sys"), Message(role="user", content="hi")],
        response_format={"type": "json_object"},
    )

    assert out == "hi there"
    assert captured["url"] == "https://api.example/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["timeout"] == 12.0
    assert captured["body"]["model"] == "gpt-x"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["temperature"] == 0.3


def test_deepseek_preset() -> None:
    provider = DeepSeekProvider(model="deepseek-chat")
    assert provider.supports_schema is False
    assert provider.base_url == "https://api.deepseek.com/v1"
    assert provider.reasoning_effort is ReasoningEffort.NONE


def _capture_body(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["body"] = json.loads(request.data.decode("utf-8")) if request.data else None
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_reasoning_effort_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _capture_body(monkeypatch, captured)
    OpenAIProvider(model="m").complete([Message(role="user", content="hi")])
    assert captured["body"]["reasoning_effort"] == "none"


def test_reasoning_effort_custom_level(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _capture_body(monkeypatch, captured)
    OpenAIProvider(model="m", reasoning_effort=ReasoningEffort.HIGH).complete([Message(role="user", content="hi")])
    assert captured["body"]["reasoning_effort"] == "high"


def test_reasoning_effort_omitted_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _capture_body(monkeypatch, captured)
    OpenAIProvider(model="m", reasoning_effort=None).complete([Message(role="user", content="hi")])
    assert "reasoning_effort" not in captured["body"]


def test_schema_passed_through_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["body"] = json.loads(request.data.decode("utf-8")) if request.data else None
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    fmt = json_schema_format("out", {"type": "object", "title": "wn"})
    OpenAIProvider(model="m").complete([Message(role="user", content="hi")], response_format=fmt)
    assert captured["body"]["response_format"] == fmt
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_schema_downgraded_to_json_object_when_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["body"] = json.loads(request.data.decode("utf-8")) if request.data else None
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAIProvider(model="m", supports_schema=False)
    provider.complete(
        [Message(role="user", content="hi")],
        response_format=json_schema_format("out", {"type": "object", "title": "wn"}),
    )
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_custom_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["body"] = json.loads(request.data.decode("utf-8")) if request.data else None
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    OpenAIProvider(model="m", temperature=None).complete([Message(role="user", content="hi")])
    assert "temperature" not in captured["body"]


def test_no_api_key_omits_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["headers"] = request.headers
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    OpenAIProvider(model="local").complete([Message(role="user", content="hi")])
    assert "Authorization" not in captured["headers"]


def test_http_error_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"slow down"))  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ProviderError, match="429"):
        OpenAIProvider(model="gpt-x").complete([Message(role="user", content="hi")])


def test_unexpected_payload_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        return _FakeResponse(json.dumps({"nope": True}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError):
        OpenAIProvider(model="gpt-x").complete([Message(role="user", content="hi")])


def test_json_schema_format() -> None:
    fmt = json_schema_format("out", {"type": "object"}, strict=True)
    assert fmt == {"type": "json_schema", "json_schema": {"name": "out", "schema": {"type": "object"}, "strict": True}}
