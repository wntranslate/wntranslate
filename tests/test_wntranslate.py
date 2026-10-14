import json

import pytest

import wntranslate
from wntranslate import (
    Alias,
    AliasSuggestion,
    Chapter,
    ChapterTranslation,
    Content,
    ContextType,
    ContextUpdate,
    Creativity,
    DataRequest,
    FootnoteNode,
    Glossary,
    GlossaryOp,
    GlossaryUpdate,
    NameType,
    PersonEntry,
    RequestKind,
    ResponseType,
    TermNode,
    TextNode,
    TranslationRequest,
    TranslationResponse,
    TranslationSettings,
    WebNovel,
    generate_analysis_prompt,
    generate_linking_prompt,
    generate_prompt,
    generate_translation_prompt,
    metadata_schema,
    output_schema,
    parse_model_output,
    request_schema,
    response_schema,
)
from wntranslate.webnovel import TranslationError, _flatten


def test_version() -> None:
    assert wntranslate.__version__


def test_defaults() -> None:
    wn = WebNovel("src", "")
    assert wn.settings.source_language is None
    assert wn.settings.target_language == "English"
    assert wn.settings.create_footnotes is True
    assert wn.settings.generate_chapter_summaries is True
    assert wn.settings.creativity is Creativity.LOW
    assert wn.settings.context_type is ContextType.ROLLING_SUMMARY
    assert wn.settings.prompt is None
    assert wn.context.notes == []
    assert isinstance(wn.glossary, Glossary)
    assert wn.chapters == []


def test_build_prompt_default() -> None:
    prompt = WebNovel("src", "").settings.build_prompt()
    assert "the source language" in prompt
    assert "into English" in prompt
    assert "typo" in prompt


def test_creativity_in_prompt() -> None:
    low = generate_prompt(TranslationSettings(creativity=Creativity.VERY_LOW))
    high = generate_prompt(TranslationSettings(creativity=Creativity.VERY_HIGH))
    assert "as literally as fidelity allows" in low
    assert "Freely polish" in high


def test_additional_context_in_prompt() -> None:
    assert "Edo period" in generate_prompt(TranslationSettings(additional_context="Set in the Edo period."))
    assert "Additional context" not in generate_prompt(TranslationSettings())


def test_prompt_uses_languages() -> None:
    prompt = generate_prompt(TranslationSettings(source_language="Japanese", target_language="French"))
    assert "from Japanese into French" in prompt
    assert "typo in Japanese" in prompt


def test_prompt_override() -> None:
    settings = TranslationSettings(prompt="custom")
    assert settings.build_prompt() == "custom"


def test_generate_prompt_varies() -> None:
    assert "footnotes only" in generate_prompt(TranslationSettings())
    assert "runaway" in generate_prompt(TranslationSettings())
    assert "translate the chapter's title" in generate_translation_prompt(TranslationSettings()).lower()
    assert "translate the chapter's title" in generate_linking_prompt(TranslationSettings()).lower()
    assert "Do not produce footnotes" in generate_prompt(TranslationSettings(create_footnotes=False))
    assert "Do not produce chapter summaries" in generate_prompt(
        TranslationSettings(generate_chapter_summaries=False)
    )
    assert "Do not maintain a running story summary" in generate_prompt(
        TranslationSettings(context_type=ContextType.NONE)
    )
    summary_prompt = generate_prompt(TranslationSettings(context_type=ContextType.ROLLING_SUMMARY))
    assert "cumulative digest of the entire story" in summary_prompt
    assert "BUILD ON IT" in summary_prompt
    assert "not a description, backstory, or full sentence" in generate_prompt(TranslationSettings())
    analysis = generate_analysis_prompt(TranslationSettings())
    assert "proper names and catchphrases" in analysis
    assert "duplicate" in analysis
    assert "generic" in analysis
    assert "Notes are always provided" in generate_prompt(TranslationSettings(context_type=ContextType.NONE))
    assert "Notes are always provided" in generate_prompt(TranslationSettings())
    # a request only edits; the model must not reply/answer the user in content
    assert "reply to, answer, or acknowledge the user" in generate_prompt(TranslationSettings())
    assert "A request only EDITS the project" in generate_prompt(TranslationSettings())


def test_ids_generated_and_unique() -> None:
    first = PersonEntry(name="Lin")
    second = PersonEntry(name="Lin")
    assert first.id and second.id
    assert first.id != second.id


def test_id_not_settable() -> None:
    entry = PersonEntry(name="Lin")
    with pytest.raises(AttributeError):
        entry.id = "x"  # type: ignore[misc]


def test_glossary_lookup() -> None:
    wn = WebNovel("src", "")
    person = PersonEntry(name="Ayami", original_name="彩美", aliases=[Alias(original="彩-", translation="Aya-")])
    wn.glossary.add(person)
    assert wn.glossary.lookup("ayami") is person
    assert wn.glossary.lookup("彩美") is person
    assert wn.glossary.lookup("aya-") is person
    assert wn.glossary.get(person.id) is person


def test_render_resolves_ids() -> None:
    person = PersonEntry(name="Ayami", original_name="彩美", aliases=[Alias(original="彩-", translation="Aya-")])
    glossary = Glossary(entries=[person])
    content = Content(
        nodes=[
            TextNode(text="Hello "),
            TermNode(term_id=person.id),
            TextNode(text=", or "),
            TermNode(term_id=person.id, alias_id=person.aliases[0].id),
            FootnoteNode(content="a note"),
        ]
    )
    rendered = Chapter(original_title="元", original_content="本文", title="Ch1", content=content).render(glossary)
    assert "Hello Ayami, or Aya-[^1]" in rendered
    assert "[^1]: a note" in rendered


def test_render_missing_id_raises() -> None:
    content = Content(nodes=[TermNode(term_id="missing")])
    with pytest.raises(KeyError):
        Chapter(original_title="o", original_content="c", content=content).render(Glossary())


def test_round_trip() -> None:
    wn = WebNovel("src", "", title="Demo", settings=TranslationSettings(creativity=Creativity.HIGH))
    person = PersonEntry(
        name="Ayami",
        original_name="彩美",
        pronouns="she/her",
        aliases=[Alias(original="彩-", translation="Aya-", context="cut off during speech")],
    )
    wn.glossary.add(person)
    wn.context.notes.append("chapter 1 intro")
    content = Content(nodes=[TermNode(term_id=person.id, alias_id=person.aliases[0].id)])
    wn.chapters.append(Chapter(original_title="元", original_content="本文", title="Ch1", content=content, summary="intro"))

    restored = WebNovel.loads(wn.dumps())
    assert restored.to_dict() == wn.to_dict()

    restored_entry = restored.glossary.get(person.id)
    assert isinstance(restored_entry, PersonEntry)
    assert restored_entry.id == person.id
    assert restored_entry.original_name == "彩美"
    assert restored_entry.aliases[0].id == person.aliases[0].id
    assert restored_entry.aliases[0].context == "cut off during speech"
    assert restored.chapters[0].id == wn.chapters[0].id
    assert restored.chapters[0].summary == "intro"
    assert restored.chapters[0].render(restored.glossary) == "Aya-"


def test_glossary_update_schema_requires_fields() -> None:
    import jsonschema

    schema = response_schema(TranslationSettings())
    item = schema["properties"]["glossary_updates"]["items"]

    def valid(update: dict[str, object]) -> bool:
        try:
            jsonschema.validate(update, item)
        except jsonschema.ValidationError:
            return False
        return True

    assert valid({"op": "new", "name": "Iroha", "original_name": "イロハ", "type": "person"})
    assert not valid({"op": "new", "name": "Iroha", "type": "person"})  # missing original_name
    assert not valid({"op": "new", "original_name": "イロハ", "type": "person"})  # missing name
    assert not valid({"op": "new", "name": "", "original_name": "イロハ", "type": "person"})  # blank name
    assert not valid({"op": "new", "name": "Iroha", "original_name": "", "type": "person"})  # blank original_name
    assert valid({"op": "delete", "target_id": "abc"})
    assert not valid({"op": "delete", "target_id": None})  # delete needs a real id
    assert not valid({"op": "edit", "name": "x"})  # edit needs target_id
    assert not valid({"op": "new", "name": "A", "original_name": "あ", "type": "person", "aliases": [{"original": "あ"}]})


def test_glossary_update_ops() -> None:
    new = GlossaryUpdate(op=GlossaryOp.NEW, name="Ayami", type=NameType.PERSON, pronouns="she/her")
    edit = GlossaryUpdate(op=GlossaryOp.EDIT, target_id="abc", name="Ayami", definition="updated")
    delete = GlossaryUpdate(op=GlossaryOp.DELETE, target_id="abc")
    for update in (new, edit, delete):
        assert GlossaryUpdate.from_dict(update.to_dict()).to_dict() == update.to_dict()
    entry = new.to_entry()
    assert isinstance(entry, PersonEntry)
    assert entry.pronouns == "she/her"


def test_response_round_trip() -> None:
    response = TranslationResponse(
        chapters=[ChapterTranslation(title="Ch1", content=Content(nodes=[TextNode(text="hi")]), summary="s")],
        glossary_updates=[
            GlossaryUpdate(
                op=GlossaryOp.NEW,
                name="Ayami",
                original_name="彩美",
                type=NameType.PERSON,
                pronouns="she/her",
                aliases=[AliasSuggestion(original="彩-", translation="Aya-", context="speech")],
            )
        ],
        context=ContextUpdate(summary="so far", notes=["note"]),
    )
    restored = TranslationResponse.from_dict(response.to_dict())
    assert restored.to_dict() == response.to_dict()
    assert response.to_dict()["response_type"] == ResponseType.TRANSLATION.value
    assert response.chapters[0].to_chapter().summary == "s"


def test_response_schema_varies() -> None:
    def node_types(settings: TranslationSettings) -> list[str]:
        nodes = response_schema(settings)["properties"]["chapters"]["items"]["properties"]["content"]
        return nodes["properties"]["nodes"]["items"]["properties"]["type"]["enum"]

    def chapter_props(settings: TranslationSettings) -> dict[str, object]:
        return response_schema(settings)["properties"]["chapters"]["items"]["properties"]

    default_schema = response_schema(TranslationSettings())
    assert default_schema["type"] == "object"
    assert "response_type" in default_schema["required"]
    assert "footnote" in node_types(TranslationSettings())
    assert "footnote" not in node_types(TranslationSettings(create_footnotes=False))
    assert "summary" in chapter_props(TranslationSettings())
    assert "summary" not in chapter_props(TranslationSettings(generate_chapter_summaries=False))
    context_props = response_schema(TranslationSettings(context_type=ContextType.NONE))["properties"]["context"]["properties"]
    assert "notes" in context_props


def test_request_schema() -> None:
    schema = request_schema()
    assert set(schema["required"]) == {"response_type", "requests"}
    assert schema["properties"]["response_type"]["enum"] == [ResponseType.REQUEST.value]


def test_output_schema() -> None:
    schema = output_schema(TranslationSettings())
    assert len(schema["oneOf"]) == 2


def test_flatten_wire_schema() -> None:
    flat = _flatten(output_schema(TranslationSettings()))
    assert "oneOf" not in flat
    assert flat["type"] == "object"
    assert flat["required"] == ["response_type"]
    assert set(flat["properties"]["response_type"]["enum"]) == {
        ResponseType.REQUEST.value,
        ResponseType.TRANSLATION.value,
    }
    assert "requests" in flat["properties"]
    assert "chapters" in flat["properties"]
    assert _flatten(metadata_schema()) == metadata_schema()


def test_request_round_trip() -> None:
    request = TranslationRequest(
        requests=[
            DataRequest(kind=RequestKind.CHAPTER_SUMMARY, chapter_id="abc"),
            DataRequest(kind=RequestKind.CHAPTER_TEXT, chapter_id="def"),
            DataRequest(kind=RequestKind.OTHER, detail="map of the city"),
        ],
        reason="need earlier events",
    )
    restored = TranslationRequest.from_dict(request.to_dict())
    assert restored.to_dict() == request.to_dict()
    assert request.to_dict()["response_type"] == ResponseType.REQUEST.value


def test_parse_model_output() -> None:
    request = TranslationRequest(requests=[DataRequest(kind=RequestKind.CHAPTER_TEXT, chapter_id="abc")])
    response = TranslationResponse(context=ContextUpdate(summary="done"))
    assert isinstance(parse_model_output(request.to_dict()), TranslationRequest)
    assert isinstance(parse_model_output(response.to_dict()), TranslationResponse)


def test_dump_is_versioned() -> None:
    data = json.loads(WebNovel("src", "", title="X").dumps())
    assert data["version"] == wntranslate.__version__
    assert WebNovel.loads(json.dumps(data)).title == "X"


def test_load_tolerates_missing_version() -> None:
    data = WebNovel("src", "", title="X").to_dict()
    del data["version"]
    assert WebNovel.from_dict(data).title == "X"


def test_apply_response_new_chapter_glossary_context() -> None:
    wn = WebNovel("src", "")
    response = TranslationResponse(
        chapters=[ChapterTranslation(title="Ch1", content=Content(nodes=[TextNode(text="hi")]), summary="s")],
        glossary_updates=[GlossaryUpdate(op=GlossaryOp.NEW, name="Lin", type=NameType.PERSON, pronouns="she/her")],
        context=ContextUpdate(summary="so far", notes=["n1"]),
    )
    wn.apply_response(response)
    assert [chapter.title for chapter in wn.chapters] == ["Ch1"]
    assert len(wn.glossary) == 1
    assert wn.context.summary == "so far"
    assert wn.context.notes == ["n1"]


def test_apply_glossary_edit_preserves_ids() -> None:
    wn = WebNovel("src", "")
    person = PersonEntry(name="Lin", aliases=[Alias(original="L", translation="Lin")])
    wn.glossary.add(person)
    update = GlossaryUpdate(
        op=GlossaryOp.EDIT,
        target_id=person.id,
        name="Linne",
        type=NameType.PERSON,
        aliases=[AliasSuggestion(original="L", translation="Lin-")],
    )
    wn.apply_response(TranslationResponse(glossary_updates=[update]))
    edited = wn.glossary.get(person.id)
    assert isinstance(edited, PersonEntry)
    assert edited.name == "Linne"
    assert edited.aliases[0].id == person.aliases[0].id
    assert edited.aliases[0].translation == "Lin-"


def test_apply_glossary_edit_partial_keeps_omitted_fields() -> None:
    # Regression: an edit that sends only a changed field (the model's natural behavior) must not
    # wipe the name/original_name/pronouns it omitted. Parsed via from_dict to exercise the real path.
    wn = WebNovel("src", "")
    person = PersonEntry(
        name="Mai",
        original_name="マイ",
        definition="old",
        aliases=[Alias(original="マイちゃん", translation="Mai-chan")],
        pronouns="she/her",
    )
    wn.glossary.add(person)
    response = TranslationResponse.from_dict(
        {
            "chapters": [],
            "glossary_updates": [
                {
                    "op": "edit",
                    "target_id": person.id,
                    "definition": "updated",
                    "aliases": [{"original": "マイちゃん", "translation": "Mai-chan"}],
                }
            ],
            "context": None,
        }
    )
    wn.apply_response(response)
    edited = wn.glossary.get(person.id)
    assert isinstance(edited, PersonEntry)
    assert edited.name == "Mai"
    assert edited.original_name == "マイ"
    assert edited.pronouns == "she/her"
    assert edited.definition == "updated"
    assert edited.aliases[0].id == person.aliases[0].id


def test_glossary_edit_blanking_name_rejected() -> None:
    wn = WebNovel("src", "")
    person = PersonEntry(name="Mai")
    wn.glossary.add(person)
    update = GlossaryUpdate.from_dict({"op": "edit", "target_id": person.id, "name": ""})
    with pytest.raises(TranslationError, match="blank the name"):
        wn._check_glossary_policy([update])


def test_apply_glossary_delete() -> None:
    wn = WebNovel("src", "")
    person = PersonEntry(name="Lin")
    wn.glossary.add(person)
    wn.apply_response(TranslationResponse(glossary_updates=[GlossaryUpdate(op=GlossaryOp.DELETE, target_id=person.id)]))
    assert len(wn.glossary) == 0


def test_apply_chapter_replace_in_place() -> None:
    wn = WebNovel("src", "")
    wn.chapters.append(
        Chapter(original_title="旧", original_content="src", title="old", content=Content(nodes=[TextNode(text="old")]))
    )
    chapter_id = wn.chapters[0].id
    payload = ChapterTranslation(title="new", content=Content(nodes=[TextNode(text="new")]), target_id=chapter_id)
    wn.apply_response(TranslationResponse(chapters=[payload]))
    assert len(wn.chapters) == 1
    assert wn.chapters[0].id == chapter_id
    assert wn.chapters[0].title == "new"
    assert wn.chapters[0].original_content == "src"


def test_apply_chapter_position() -> None:
    wn = WebNovel("src", "")
    for index in range(3):
        wn.chapters.append(Chapter(original_title=str(index), original_content="x", title=f"c{index}"))
    moved_id = wn.chapters[2].id
    payload = ChapterTranslation(title="c2", content=Content(), target_id=moved_id, position=0)
    wn.apply_response(TranslationResponse(chapters=[payload]))
    assert [c.title for c in wn.chapters] == ["c2", "c0", "c1"]


def test_resolve_request() -> None:
    wn = WebNovel("src", "")
    person = PersonEntry(name="Ayami")
    wn.glossary.add(person)
    chapter = Chapter(
        original_title="原題",
        original_content="原文",
        title="C1",
        content=Content(nodes=[TermNode(term_id=person.id)]),
        summary="sum",
    )
    wn.chapters.append(chapter)
    request = TranslationRequest(
        requests=[
            DataRequest(kind=RequestKind.CHAPTER_SUMMARY, chapter_id=chapter.id),
            DataRequest(kind=RequestKind.CHAPTER_TEXT, chapter_id=chapter.id),
            DataRequest(kind=RequestKind.CHAPTER_ORIGINAL_TITLE, chapter_id=chapter.id),
            DataRequest(kind=RequestKind.CHAPTER_ORIGINAL_CONTENT, chapter_id=chapter.id),
            DataRequest(kind=RequestKind.CHAPTER_LIST),
            DataRequest(kind=RequestKind.CHAPTER_SUMMARY, chapter_id="missing"),
            DataRequest(kind=RequestKind.OTHER, detail="map"),
        ]
    )
    results = wn.resolve_request(request)
    assert results[0].found and results[0].value == "sum"
    assert results[1].found and results[1].value == "Ayami"
    assert results[2].value == "原題"
    assert results[3].value == "原文"
    assert results[4].found and results[4].value == [
        {"id": chapter.id, "title": "C1", "kind": "chapter", "summary": "sum"}
    ]
    assert not results[5].found and results[5].value is None
    assert not results[6].found
