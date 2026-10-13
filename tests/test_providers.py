import io
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any, Self

import pytest

from wntranslate import (
    DeepSeekProvider,
    Message,
    OpenAIProvider,
    ProviderConnectionError,
    ProviderError,
    ProviderHTTPError,
    ReasoningEffort,
    Usage,
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


class _FakeStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def __iter__(self) -> Iterator[bytes]:
        return iter(f"{line}\n".encode() for line in self._lines)


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
    assert captured["body"]["temperature"] == 0.4


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


def test_frequency_penalty_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["body"] = json.loads(request.data.decode("utf-8")) if request.data else None
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    OpenAIProvider(model="m").complete([Message(role="user", content="hi")])
    assert captured["body"]["frequency_penalty"] == 0.3
    assert "presence_penalty" not in captured["body"]

    OpenAIProvider(model="m", frequency_penalty=None, presence_penalty=0.5).complete([Message(role="user", content="hi")])
    assert "frequency_penalty" not in captured["body"]
    assert captured["body"]["presence_penalty"] == 0.5


def test_no_api_key_omits_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        captured["headers"] = request.headers
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    OpenAIProvider(model="local").complete([Message(role="user", content="hi")])
    assert "Authorization" not in captured["headers"]


def test_http_error_raises_provider_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"slow down"))  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ProviderHTTPError, match="429") as info:
        OpenAIProvider(model="gpt-x").complete([Message(role="user", content="hi")])
    assert info.value.status == 429


def test_connection_error_raises_provider_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ProviderConnectionError):
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


def test_usage_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        payload = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 120, "completion_tokens": 34}}
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    seen: list[Usage] = []
    OpenAIProvider(model="m", on_usage=seen.append).complete([Message(role="user", content="hi")])
    assert seen == [Usage(120, 34)]


def test_usage_reports_cached_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        usage = {"prompt_tokens": 1000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 900}}
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": usage}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    seen: list[Usage] = []
    OpenAIProvider(model="m", on_usage=seen.append).complete([Message(role="user", content="hi")])
    assert seen == [Usage(1000, 50, cached_tokens=900)]


def test_usage_absent_no_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    seen: list[Usage] = []
    OpenAIProvider(model="m", on_usage=seen.append).complete([Message(role="user", content="hi")])
    assert seen == []


def test_stream_reports_usage_and_requests_it(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _FakeStream:
        captured["body"] = json.loads(request.data.decode("utf-8")) if request.data else None
        return _FakeStream(
            [
                'data: {"choices": [{"delta": {"content": "hel"}}]}',
                'data: {"choices": [{"delta": {"content": "lo"}}]}',
                'data: {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}',
                "data: [DONE]",
            ]
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    seen: list[Usage] = []
    out = OpenAIProvider(model="m", stream=True, on_usage=seen.append).complete([Message(role="user", content="hi")])
    assert out == "hello"
    assert seen == [Usage(7, 2)]
    assert captured["body"]["stream_options"] == {"include_usage": True}


def test_usage_arithmetic() -> None:
    total = Usage(10, 3) + Usage(5, 20)
    assert total == Usage(15, 23)
    assert bool(Usage(0, 0)) is False
    assert bool(Usage(1, 0)) is True
