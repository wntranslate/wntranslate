import asyncio
import json
from collections.abc import Sequence
from typing import Any

import jsonschema
import pytest

from wntranslate import (
    Alias,
    AliasSuggestion,
    AsyncProvider,
    AsyncWebNovel,
    Chapter,
    ChapterAnalysis,
    ChapterCleanup,
    ChapterDraft,
    ChapterKind,
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
    TermNode,
    TextNode,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
    TranslationSettings,
    WebNovel,
    response_schema,
)
from wntranslate.webnovel import _cap_runs


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


def _dump(obj: TranslationResponse | TranslationRequest | ChapterAnalysis | ChapterDraft) -> str:
    return json.dumps(obj.to_dict())


def _analysis(
    summary: str | None = "s",
    kind: ChapterKind = ChapterKind.CHAPTER,
    glossary_updates: list[GlossaryUpdate] | None = None,
    context: ContextUpdate | None = None,
) -> str:
    return _dump(ChapterAnalysis(summary=summary, kind=kind, glossary_updates=glossary_updates or [], context=context))


def _draft(title: str = "Ch1", text: str = "hi", position: int | None = None) -> str:
    return _dump(ChapterDraft(title=title, content=Content(nodes=[TextNode(text=text)]), position=position))


def test_no_provider() -> None:
    with pytest.raises(TranslationError, match="no provider configured"):
        WebNovel("src", "").translate_new_chapter("t", "c")


def test_provider_type_mismatch() -> None:
    with pytest.raises(TypeError, match="synchronous Provider"):
        WebNovel("src", "", provider=FakeAsyncProvider("{}"))
    with pytest.raises(TypeError, match="AsyncProvider"):
        AsyncWebNovel("src", "", provider=FakeProvider("{}"))


def test_translate_new_chapter_applies() -> None:
    provider = FakeProvider([_analysis(summary="s", context=ContextUpdate(summary="recap")), _draft(title="Ch1")])
    wn = WebNovel("src", "", provider=provider)
    result = wn.translate_new_chapter("原題", "原文")
    assert isinstance(result, Chapter)
    assert provider.calls == 4
    assert len(wn.chapters) == 1
    assert wn.chapters[0].title == "Ch1"
    assert wn.chapters[0].original_title == "原題"
    assert wn.chapters[0].original_content == "原文"
    assert wn.chapters[0].summary == "s"
    assert wn.context.summary == "recap"


def test_translate_new_chapter_auto_fulfills_requests() -> None:
    request = TranslationRequest(requests=[DataRequest(kind=RequestKind.CHAPTER_LIST)])
    provider = FakeProvider([_dump(request), _analysis(), _draft(title="Ch1")])
    wn = WebNovel("src", "", provider=provider)
    result = wn.translate_new_chapter("t", "c")
    assert isinstance(result, Chapter)
    assert provider.calls == 5
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


def test_term_node_alias_from_other_entry_is_rejected() -> None:
    wn = WebNovel("src", "")
    iroha = PersonEntry(name="Iroha")
    mai = PersonEntry(name="Mai", aliases=[Alias(original="マイちゃん", translation="Mai-chan")])
    wn.glossary.add(iroha)
    wn.glossary.add(mai)
    bad = TranslationResponse(
        chapters=[
            ChapterTranslation(
                title="Ch",
                summary="",
                content=Content(nodes=[TextNode(text="Hi "), TermNode(term_id=iroha.id, alias_id=mai.aliases[0].id)]),
            )
        ]
    )
    provider = FakeProvider(_dump(bad))
    wn.provider = provider
    with pytest.raises(TranslationError, match="response schema invalid"):
        wn.request_change("x")
    assert any("not an alias of entry" in m.content for m in provider.last_messages)
    assert len(wn.chapters) == 0


def test_request_change_unknown_target_id_errors() -> None:
    wn = WebNovel("src", "")
    wn.chapters.append(
        Chapter(original_title="o", original_content="c", title="old", content=Content(nodes=[TextNode(text="old")]))
    )
    reply = TranslationResponse(
        chapters=[
            ChapterTranslation(title="new", content=Content(nodes=[TextNode(text="new")]), summary="", target_id="chapter_1")
        ]
    )
    wn.provider = FakeProvider(_dump(reply))
    with pytest.raises(TranslationError, match="unknown chapter target_id"):
        wn.request_change("revise chapter 1")
    assert len(wn.chapters) == 1
    assert wn.chapters[0].title == "old"


def test_request_change_valid_target_id_revises() -> None:
    wn = WebNovel("src", "")
    wn.chapters.append(
        Chapter(original_title="o", original_content="c", title="old", content=Content(nodes=[TextNode(text="old")]))
    )
    reply = TranslationResponse(
        chapters=[
            ChapterTranslation(
                title="new", content=Content(nodes=[TextNode(text="new")]), summary="", target_id=wn.chapters[0].id
            )
        ]
    )
    wn.provider = FakeProvider(_dump(reply))
    wn.request_change("revise it")
    assert len(wn.chapters) == 1
    assert wn.chapters[0].title == "new"


def test_request_change_continues_until_done() -> None:
    wn = WebNovel("src", "")
    for i in range(4):
        wn.chapters.append(
            Chapter(
                original_title=f"o{i}",
                original_content=f"c{i}",
                title=f"old{i}",
                content=Content(nodes=[TextNode(text=f"old{i}")]),
                summary="",
            )
        )

    def edit(i: int) -> ChapterTranslation:
        return ChapterTranslation(
            title=f"new{i}", content=Content(nodes=[TextNode(text=f"n{i}")]), summary="", target_id=wn.chapters[i].id
        )

    batch1 = TranslationResponse(chapters=[edit(0), edit(1)], more=True)
    batch2 = TranslationResponse(chapters=[edit(2), edit(3)], more=False)
    provider = FakeProvider([_dump(batch1), _dump(batch2)])
    wn.provider = provider
    result = wn.request_change("update all four chapters")
    assert provider.calls == 2
    assert [c.title for c in wn.chapters] == ["new0", "new1", "new2", "new3"]
    assert len(result.chapters) == 4
    assert any(m.role == "user" and "Continue" in m.content for m in provider.last_messages)


def _chapter_using(person: PersonEntry, *, alias: bool = False) -> Chapter:
    alias_id = person.aliases[0].id if alias else None
    return Chapter(
        original_title="o",
        original_content="c",
        title="C1",
        content=Content(nodes=[TermNode(term_id=person.id, alias_id=alias_id)]),
    )


def test_cannot_delete_referenced_glossary_entry() -> None:
    wn = WebNovel("src", "")
    person = PersonEntry(name="Ayami")
    wn.glossary.add(person)
    wn.chapters.append(_chapter_using(person))
    wn.provider = FakeProvider(_dump(TranslationResponse(glossary_updates=[GlossaryUpdate(op=GlossaryOp.DELETE, target_id=person.id)])))
    with pytest.raises(TranslationError, match="in use by a chapter"):
        wn.request_change("delete Ayami")
    assert len(wn.glossary) == 1


def test_cannot_drop_referenced_alias_via_edit() -> None:
    wn = WebNovel("src", "")
    person = PersonEntry(name="Ayami", aliases=[Alias(original="彩-", translation="Aya-")])
    wn.glossary.add(person)
    wn.chapters.append(_chapter_using(person, alias=True))
    reply = TranslationResponse(
        glossary_updates=[GlossaryUpdate(op=GlossaryOp.EDIT, target_id=person.id, name="Ayame", type=NameType.PERSON)]
    )
    wn.provider = FakeProvider(_dump(reply))
    with pytest.raises(TranslationError, match="drop glossary alias in use"):
        wn.request_change("rename")


def test_edit_keeping_referenced_alias_ok() -> None:
    wn = WebNovel("src", "")
    person = PersonEntry(name="Ayami", aliases=[Alias(original="彩-", translation="Aya-")])
    wn.glossary.add(person)
    wn.chapters.append(_chapter_using(person, alias=True))
    reply = TranslationResponse(
        glossary_updates=[
            GlossaryUpdate(
                op=GlossaryOp.EDIT,
                target_id=person.id,
                name="Ayame",
                type=NameType.PERSON,
                aliases=[AliasSuggestion(original="彩-", translation="Aya-")],
            )
        ]
    )
    wn.provider = FakeProvider(_dump(reply))
    wn.request_change("rename")
    assert wn.chapters[0].render(wn.glossary) == "Aya-"


def test_schema_invalid_structure() -> None:
    wn = WebNovel("src", "", provider=FakeProvider(json.dumps({"foo": "bar"})))
    with pytest.raises(TranslationError, match="response schema invalid"):
        wn.request_change("x")


def test_schema_invalid_bad_json() -> None:
    wn = WebNovel("src", "", provider=FakeProvider("not json"))
    with pytest.raises(TranslationError, match="response schema invalid"):
        wn.request_change("x")


def test_policy_violation_is_fed_back_and_recovers() -> None:
    over = TranslationResponse(
        glossary_updates=[
            GlossaryUpdate(op=GlossaryOp.NEW, name="A", original_name="あ", type=NameType.PERSON),
            GlossaryUpdate(op=GlossaryOp.NEW, name="B", original_name="い", type=NameType.PERSON),
        ]
    )
    ok = TranslationResponse(
        glossary_updates=[GlossaryUpdate(op=GlossaryOp.NEW, name="A", original_name="あ", type=NameType.PERSON)]
    )
    provider = FakeProvider([_dump(over), _dump(ok)])
    wn = WebNovel("src", "", settings=TranslationSettings(glossary_add_limit=1), provider=provider)
    wn.request_change("add names")
    assert provider.calls == 2
    assert len(wn.glossary) == 1
    assert any("exceeds glossary add limit" in m.content for m in provider.last_messages)


def test_schema_retry_then_succeeds() -> None:
    provider = FakeProvider(["not json", _analysis(), _draft(title="Ch1")])
    wn = WebNovel("src", "", provider=provider)
    result = wn.translate_new_chapter("t", "c")
    assert isinstance(result, Chapter)
    assert provider.calls == 5
    assert len(wn.chapters) == 1
    assert any(m.role == "assistant" and m.content == "not json" for m in provider.last_messages)
    assert any("was rejected" in m.content for m in provider.last_messages)


def test_schema_retry_gives_up_after_limit() -> None:
    provider = FakeProvider("not json")
    wn = WebNovel("src", "", provider=provider)
    with pytest.raises(TranslationError, match="response schema invalid"):
        wn.request_change("x")
    assert provider.calls == 6


def test_schema_fail_resends_full_schema() -> None:
    ok = TranslationResponse(
        glossary_updates=[GlossaryUpdate(op=GlossaryOp.NEW, name="A", original_name="あ", type=NameType.PERSON)]
    )
    provider = FakeProvider(["not json", _dump(ok)])
    wn = WebNovel("src", "", provider=provider)
    wn.request_change("x")
    assert provider.calls == 2
    assert sum("JSON Schema" in m.content for m in provider.last_messages) >= 2
    assert any("was rejected" in m.content for m in provider.last_messages)


def test_cap_runs() -> None:
    assert _cap_runs("a" + "ー" * 90 + "b", 24) == "a" + "ー" * 24 + "b"
    assert _cap_runs("hi", 24) == "hi"
    assert _cap_runs("ー" * 90, 0) == "ー" * 90
    assert _cap_runs("x\n\n\n\n\ny", 2) == "x\n\n\n\n\ny"


def test_long_run_capped_in_prompt_but_source_stored_intact() -> None:
    scream = "キャ" + "ー" * 90
    provider = FakeProvider([_analysis(), _draft()])
    wn = WebNovel("src", "", settings=TranslationSettings(max_source_char_run=24), provider=provider)
    chapter = wn.translate_new_chapter("t", scream)
    assert chapter.original_content == scream
    analysis_msg = next(m for m in provider.last_messages if m.role == "user" and "キャ" in m.content)
    assert "ー" * 24 in analysis_msg.content
    assert "ー" * 25 not in analysis_msg.content


def test_overview_recent_chapters_include_original_title() -> None:
    wn = WebNovel("src", "")
    wn.chapters.append(
        Chapter(original_title="元タイトル", original_content="c", title="The Title", content=Content(nodes=[TextNode(text="hi")]))
    )
    recent = wn._overview()["recent_chapters"]
    assert recent[0]["title"] == "The Title"
    assert recent[0]["original_title"] == "元タイトル"


def test_full_schema_is_sent_to_the_model() -> None:
    provider = FakeProvider([_analysis(), _draft()])
    wn = WebNovel("src", "", provider=provider)
    wn.translate_new_chapter("t", "c")
    assert any(
        m.role == "system" and "JSON Schema" in m.content and "response_type" in m.content and "glossary_updates" in m.content
        for m in provider.last_messages
    )


def test_review_and_linking_embed_the_chapter() -> None:
    cleanup = ChapterCleanup(title="Ch1", content=Content(nodes=[TextNode(text="cleaned text")]))
    provider = FakeProvider([_analysis(), _draft(text="draft text"), json.dumps(cleanup.to_dict()), _draft()])
    wn = WebNovel("src", "", provider=provider)
    wn.translate_new_chapter("t", "c")
    convo = provider.last_messages  # the pass-4 (linking) conversation
    assert any(m.role == "user" and "draft text" in m.content for m in convo)  # cleanup pass got the draft to review
    assert any(m.role == "user" and "cleaned text" in m.content for m in convo)  # linking pass got the cleaned chapter
    assert any(m.role == "user" and "in full" in m.content.lower() for m in convo)


def test_cleanup_pass_can_fix_glossary_and_context() -> None:
    cleanup = ChapterCleanup(
        title="Ch1",
        content=Content(nodes=[TextNode(text="hi")]),
        glossary_updates=[GlossaryUpdate(op=GlossaryOp.NEW, name="Iroha", original_name="イロハ", type=NameType.PERSON)],
        context=ContextUpdate(notes=["keep X"]),
    )
    provider = FakeProvider([_analysis(), _draft(), _dump(cleanup), _draft()])
    wn = WebNovel("src", "", provider=provider)
    wn.translate_new_chapter("t", "c")
    assert provider.calls == 4
    assert len(wn.glossary) == 1
    assert wn.context.notes == ["keep X"]
    assert len(wn.chapters) == 1


def test_fourth_pass_links_glossary_terms() -> None:
    wn = WebNovel("src", "")
    person = PersonEntry(name="Iroha")
    wn.glossary.add(person)
    linked = ChapterDraft(title="Ch1", content=Content(nodes=[TextNode(text="Hi "), TermNode(term_id=person.id)]))
    provider = FakeProvider([_analysis(), _draft(), _draft(), _dump(linked)])
    wn.provider = provider
    chapter = wn.translate_new_chapter("t", "c")
    assert provider.calls == 4
    assert any(isinstance(node, TermNode) and node.term_id == person.id for node in chapter.content.nodes)
    assert chapter.render(wn.glossary) == "Hi Iroha"


def test_failed_translation_rolls_back_staged_changes() -> None:
    analysis = _analysis(
        glossary_updates=[GlossaryUpdate(op=GlossaryOp.NEW, name="Iroha", original_name="イロハ", type=NameType.PERSON)],
        context=ContextUpdate(notes=["staged note"]),
    )
    provider = FakeProvider([analysis, "not json"])
    wn = WebNovel("src", "", provider=provider)
    with pytest.raises(TranslationError):
        wn.translate_new_chapter("t", "c")
    assert len(wn.glossary) == 0
    assert wn.context.notes == []
    assert len(wn.chapters) == 0


def test_schema_error_names_the_real_problem_not_the_wrong_branch() -> None:
    bad = json.loads(_analysis())
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
            GlossaryUpdate(op=GlossaryOp.NEW, name="A", original_name="あ", type=NameType.PERSON),
            GlossaryUpdate(op=GlossaryOp.NEW, name="B", original_name="い", type=NameType.PERSON),
        ]
    )
    wn = WebNovel("src", "", settings=TranslationSettings(glossary_add_limit=1), provider=FakeProvider(_dump(reply)))
    with pytest.raises(TranslationError, match="exceeds glossary add limit: 1"):
        wn.request_change("x")
    assert len(wn.glossary) == 0


def test_glossary_add_not_allowed() -> None:
    reply = TranslationResponse(glossary_updates=[GlossaryUpdate(op=GlossaryOp.NEW, name="A", original_name="あ", type=NameType.PERSON)])
    wn = WebNovel("src", "", settings=TranslationSettings(allow_glossary_add=False), provider=FakeProvider(_dump(reply)))
    with pytest.raises(TranslationError, match="glossary add not allowed"):
        wn.request_change("x")


def test_chapter_reorder_not_allowed() -> None:
    reply = TranslationResponse(
        chapters=[
            ChapterTranslation(
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
    provider = FakeAsyncProvider([_dump(request), _analysis(), _draft(title="Ch1")])
    wn = AsyncWebNovel("src", "", provider=provider)
    result = asyncio.run(wn.translate_new_chapter("t", "c"))
    assert isinstance(result, Chapter)
    assert provider.calls == 5
    assert len(wn.chapters) == 1
    assert wn.chapters[0].original_title == "t"


def test_settings_change_applies_next_translation() -> None:
    provider = FakeProvider([_analysis(), _draft(), _draft(), _draft(), _analysis(), _draft(), _draft(), _draft()])
    wn = WebNovel("src", "", provider=provider)

    wn.settings.creativity = Creativity.VERY_LOW
    wn.translate_new_chapter("a", "b")
    low_prompt = "\n".join(m.content for m in provider.last_messages)

    wn.settings.creativity = Creativity.VERY_HIGH
    wn.translate_new_chapter("c", "d")
    high_prompt = "\n".join(m.content for m in provider.last_messages)

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
                title="t",
                content=Content(nodes=[TextNode(text="x")]),
                summary="s",
            )
        ],
        glossary_updates=[GlossaryUpdate(op=GlossaryOp.NEW, name="A", original_name="あ", type=NameType.PERSON)],
        context=ContextUpdate(summary="z", notes=["n"]),
    )
    jsonschema.validate(reply.to_dict(), response_schema(TranslationSettings()))
