import asyncio
import json
from collections.abc import Sequence
from typing import Any

import jsonschema
import pytest

from wntranslate import (
    AsyncProvider,
    AsyncWebNovel,
    ChapterTranslation,
    Content,
    ContextUpdate,
    Creativity,
    DataRequest,
    GlossaryOp,
    GlossaryUpdate,
    Message,
    NameType,
    PersonEntry,
    Provider,
    RequestKind,
    TextNode,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
    TranslationSettings,
    WebNovel,
    response_schema,
)


class FakeProvider(Provider):
    def __init__(self, reply: str | list[str]) -> None:
        self.replies = [reply] if isinstance(reply, str) else list(reply)
        self.calls = 0
        self.last_messages: list[Message] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        self.last_messages = list(messages)
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


class FakeAsyncProvider(AsyncProvider):
    def __init__(self, reply: str | list[str]) -> None:
        self.replies = [reply] if isinstance(reply, str) else list(reply)
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def _dump(obj: TranslationResponse | TranslationRequest) -> str:
    return json.dumps(obj.to_dict())


def test_no_provider() -> None:
    with pytest.raises(TranslationError, match="no provider configured"):
        WebNovel("src", "").translate_new_chapter("t", "c")


def test_provider_type_mismatch() -> None:
    with pytest.raises(TypeError, match="synchronous Provider"):
        WebNovel("src", "", provider=FakeAsyncProvider("{}"))
    with pytest.raises(TypeError, match="AsyncProvider"):
        AsyncWebNovel("src", "", provider=FakeProvider("{}"))


def test_translate_new_chapter_applies() -> None:
    reply = TranslationResponse(
        chapters=[
            ChapterTranslation(
                original_title="x",
                original_content="y",
                title="Ch1",
                content=Content(nodes=[TextNode(text="hi")]),
                summary="s",
            )
        ],
        context=ContextUpdate(summary="recap"),
    )
    wn = WebNovel("src", "", provider=FakeProvider(_dump(reply)))
    result = wn.translate_new_chapter("原題", "原文")
    assert isinstance(result, TranslationResponse)
    assert len(wn.chapters) == 1
    assert wn.chapters[0].title == "Ch1"
    assert wn.chapters[0].original_title == "原題"
    assert wn.chapters[0].original_content == "原文"
    assert wn.context.summary == "recap"


def test_translate_new_chapter_auto_fulfills_requests() -> None:
    request = TranslationRequest(requests=[DataRequest(kind=RequestKind.CHAPTER_LIST)])
    response = TranslationResponse(
        chapters=[
            ChapterTranslation(
                original_title="x",
                original_content="y",
                title="Ch1",
                content=Content(nodes=[TextNode(text="hi")]),
                summary="s",
            )
        ]
    )
    provider = FakeProvider([_dump(request), _dump(response)])
    wn = WebNovel("src", "", provider=provider)
    result = wn.translate_new_chapter("t", "c")
    assert isinstance(result, TranslationResponse)
    assert provider.calls == 2
    assert len(wn.chapters) == 1
    requested = next(m.content for m in provider.last_messages if m.content.startswith("Data you requested"))
    assert "chapter_list" in requested


def test_translate_new_chapter_gives_up_after_too_many_requests() -> None:
    request = TranslationRequest(requests=[DataRequest(kind=RequestKind.CHAPTER_LIST)])
    wn = WebNovel("src", "", provider=FakeProvider(_dump(request)))
    with pytest.raises(TranslationError, match="kept requesting data"):
        wn.translate_new_chapter("t", "c")


def test_request_change_glossary_edit() -> None:
    wn = WebNovel("src", "")
    person = PersonEntry(name="Ayami")
    wn.glossary.add(person)
    reply = TranslationResponse(
        glossary_updates=[GlossaryUpdate(op=GlossaryOp.EDIT, target_id=person.id, name="Ayame", type=NameType.PERSON)]
    )
    wn.provider = FakeProvider(_dump(reply))
    result = wn.request_change("rename Ayami to Ayame")
    assert isinstance(result, TranslationResponse)
    edited = wn.glossary.get(person.id)
    assert edited is not None
    assert edited.name == "Ayame"


def test_schema_invalid_structure() -> None:
    wn = WebNovel("src", "", provider=FakeProvider(json.dumps({"foo": "bar"})))
    with pytest.raises(TranslationError, match="response schema invalid"):
        wn.request_change("x")


def test_schema_invalid_bad_json() -> None:
    wn = WebNovel("src", "", provider=FakeProvider("not json"))
    with pytest.raises(TranslationError, match="response schema invalid"):
        wn.request_change("x")


def test_schema_retry_then_succeeds() -> None:
    response = TranslationResponse(
        chapters=[
            ChapterTranslation(
                original_title="x",
                original_content="y",
                title="Ch1",
                content=Content(nodes=[TextNode(text="hi")]),
                summary="s",
            )
        ]
    )
    provider = FakeProvider(["not json", _dump(response)])
    wn = WebNovel("src", "", provider=provider)
    result = wn.translate_new_chapter("t", "c")
    assert isinstance(result, TranslationResponse)
    assert provider.calls == 2
    assert len(wn.chapters) == 1
    assert any(m.role == "assistant" and m.content == "not json" for m in provider.last_messages)
    assert any("did not match" in m.content for m in provider.last_messages)


def test_schema_retry_gives_up_after_limit() -> None:
    provider = FakeProvider("not json")
    wn = WebNovel("src", "", provider=provider)
    with pytest.raises(TranslationError, match="response schema invalid"):
        wn.request_change("x")
    assert provider.calls == 4


def _one_chapter_response() -> TranslationResponse:
    return TranslationResponse(
        chapters=[
            ChapterTranslation(
                original_title="x",
                original_content="y",
                title="Ch1",
                content=Content(nodes=[TextNode(text="hi")]),
                summary="s",
            )
        ]
    )


def test_full_schema_is_sent_to_the_model() -> None:
    provider = FakeProvider(_dump(_one_chapter_response()))
    wn = WebNovel("src", "", provider=provider)
    wn.translate_new_chapter("t", "c")
    assert any(
        m.role == "system" and "JSON Schema" in m.content and "response_type" in m.content and "glossary_updates" in m.content
        for m in provider.last_messages
    )


def test_schema_error_names_the_real_problem_not_the_wrong_branch() -> None:
    bad = json.loads(_dump(_one_chapter_response()))
    bad["type"] = "json_object"
    provider = FakeProvider(json.dumps(bad))
    wn = WebNovel("src", "", provider=provider)
    with pytest.raises(TranslationError, match="response schema invalid"):
        wn.translate_new_chapter("t", "c")
    correction = provider.last_messages[-1].content
    assert "not allowed" in correction
    assert "is not one of ['request']" not in correction


def test_glossary_add_limit() -> None:
    reply = TranslationResponse(
        glossary_updates=[
            GlossaryUpdate(op=GlossaryOp.NEW, name="A", type=NameType.PERSON),
            GlossaryUpdate(op=GlossaryOp.NEW, name="B", type=NameType.PERSON),
        ]
    )
    wn = WebNovel("src", "", settings=TranslationSettings(glossary_add_limit=1), provider=FakeProvider(_dump(reply)))
    with pytest.raises(TranslationError, match="exceeds glossary add limit: 1"):
        wn.request_change("x")
    assert len(wn.glossary) == 0


def test_glossary_add_not_allowed() -> None:
    reply = TranslationResponse(glossary_updates=[GlossaryUpdate(op=GlossaryOp.NEW, name="A", type=NameType.PERSON)])
    wn = WebNovel("src", "", settings=TranslationSettings(allow_glossary_add=False), provider=FakeProvider(_dump(reply)))
    with pytest.raises(TranslationError, match="glossary add not allowed"):
        wn.request_change("x")


def test_chapter_reorder_not_allowed() -> None:
    reply = TranslationResponse(
        chapters=[
            ChapterTranslation(
                original_title="o",
                original_content="c",
                title="t",
                content=Content(nodes=[TextNode(text="x")]),
                summary="",
                position=0,
            )
        ]
    )
    wn = WebNovel("src", "", settings=TranslationSettings(allow_chapter_reorder=False), provider=FakeProvider(_dump(reply)))
    with pytest.raises(TranslationError, match="chapter reorder not allowed"):
        wn.request_change("x")


def test_notes_edited_via_response() -> None:
    wn = WebNovel("src", "", provider=FakeProvider(_dump(TranslationResponse(context=ContextUpdate(notes=["keep X as Y"])))))
    wn.request_change("note that X is always Y")
    assert wn.context.notes == ["keep X as Y"]
    wn.provider = FakeProvider(_dump(TranslationResponse(context=ContextUpdate(summary="recap"))))
    wn.request_change("just summarize")
    assert wn.context.notes == ["keep X as Y"]


def test_state_passed_to_model() -> None:
    wn = WebNovel("src", "", provider=FakeProvider(_dump(TranslationResponse())))
    wn.glossary.add(PersonEntry(name="Ayami"))
    wn.context.notes.append("stylistic rule")
    wn.request_change("x")
    provider = wn.provider
    assert isinstance(provider, FakeProvider)
    state = next(m.content for m in provider.last_messages if m.content.startswith("Project state"))
    assert "Ayami" in state
    assert "stylistic rule" in state


def test_resolve_webnovel_fields() -> None:
    wn = WebNovel("原題", synopsis="A tale.", original_synopsis="物語")
    request = TranslationRequest(
        requests=[
            DataRequest(kind=RequestKind.SYNOPSIS),
            DataRequest(kind=RequestKind.ORIGINAL_TITLE),
            DataRequest(kind=RequestKind.ORIGINAL_SYNOPSIS),
        ]
    )
    results = wn.resolve_request(request)
    assert [r.value for r in results] == ["A tale.", "原題", "物語"]


def test_async_webnovel_translate_and_auto_fulfill() -> None:
    request = TranslationRequest(requests=[DataRequest(kind=RequestKind.CHAPTER_LIST)])
    response = TranslationResponse(
        chapters=[
            ChapterTranslation(
                original_title="x",
                original_content="y",
                title="Ch1",
                content=Content(nodes=[TextNode(text="hi")]),
                summary="s",
            )
        ]
    )
    provider = FakeAsyncProvider([_dump(request), _dump(response)])
    wn = AsyncWebNovel("src", "", provider=provider)
    result = asyncio.run(wn.translate_new_chapter("t", "c"))
    assert isinstance(result, TranslationResponse)
    assert provider.calls == 2
    assert len(wn.chapters) == 1
    assert wn.chapters[0].original_title == "t"


def test_settings_change_applies_next_translation() -> None:
    response = TranslationResponse(
        chapters=[
            ChapterTranslation(
                original_title="x",
                original_content="y",
                title="t",
                content=Content(nodes=[TextNode(text="hi")]),
                summary="s",
            )
        ]
    )
    provider = FakeProvider([_dump(response), _dump(response)])
    wn = WebNovel("src", "", provider=provider)

    wn.settings.creativity = Creativity.VERY_LOW
    wn.translate_new_chapter("a", "b")
    low_prompt = provider.last_messages[0].content

    wn.settings.creativity = Creativity.VERY_HIGH
    wn.translate_new_chapter("c", "d")
    high_prompt = provider.last_messages[0].content

    assert "as literally as fidelity allows" in low_prompt
    assert "Freely polish" in high_prompt


def test_translate_metadata() -> None:
    reply = json.dumps({"title": "The Title", "synopsis": "A synopsis."})
    wn = WebNovel("原題", "元のあらすじ", provider=FakeProvider(reply))
    assert wn.title is None
    assert wn.synopsis is None
    metadata = wn.translate_metadata()
    assert metadata.title == "The Title"
    assert wn.title == "The Title"
    assert wn.synopsis == "A synopsis."


def test_title_synopsis_save_as_null() -> None:
    data = json.loads(WebNovel("原題", "元").dumps())
    assert data["title"] is None
    assert data["synopsis"] is None


def test_response_to_dict_matches_schema() -> None:
    reply = TranslationResponse(
        chapters=[
            ChapterTranslation(
                original_title="o",
                original_content="c",
                title="t",
                content=Content(nodes=[TextNode(text="x")]),
                summary="s",
            )
        ],
        glossary_updates=[GlossaryUpdate(op=GlossaryOp.NEW, name="A", type=NameType.PERSON)],
        context=ContextUpdate(summary="z", notes=["n"]),
    )
    jsonschema.validate(reply.to_dict(), response_schema(TranslationSettings()))
