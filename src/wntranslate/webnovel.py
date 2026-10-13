from __future__ import annotations

import json
from collections.abc import Callable
from typing import IO, Any, Self, TypeVar

import jsonschema

from ._version import __version__
from .chapter import Chapter
from .content import Content, TermNode
from .context import Context
from .enums import ContextType, GlossaryOp, RequestKind
from .glossary import Glossary
from .prompts import (
    generate_analysis_prompt,
    generate_chapter_prompt,
    generate_cleanup_prompt,
    generate_linking_prompt,
    generate_metadata_prompt,
    generate_translation_prompt,
)
from .providers import AsyncProvider, Message, Provider, json_schema_format
from .render import chapter_document, reader_document, render_content_html
from .request import DataRequest, DataResult, TranslationRequest
from .response import (
    ChapterAnalysis,
    ChapterCleanup,
    ChapterDraft,
    ChapterTranslation,
    ContextUpdate,
    GlossaryUpdate,
    TranslatedMetadata,
    TranslationResponse,
    analysis_output_schema,
    cleanup_output_schema,
    draft_output_schema,
    metadata_schema,
    output_schema,
    parse_analysis,
    parse_cleanup,
    parse_draft,
    parse_model_output,
)
from .settings import TranslationSettings

_MAX_REQUEST_ROUNDS = 8
_MAX_SCHEMA_RETRIES = 3
_MAX_CONTINUATION_ROUNDS = 100

_T = TypeVar("_T")


def _schema_message(schema: dict[str, Any]) -> Message:
    return Message(
        role="system",
        content="Return exactly one JSON object that conforms to this JSON Schema:\n"
        + json.dumps(schema, ensure_ascii=False),
    )


def _select_branch(schema: dict[str, Any], data: Any) -> dict[str, Any]:
    """Pick the oneOf branch matching the data's response_type, for a precise error.

    Args:
        schema: The schema, possibly with a top-level oneOf discriminated by response_type.
        data: The decoded model output.

    Returns:
        The matching branch, or the schema unchanged when no branch applies.
    """
    branches = schema.get("oneOf")
    if not isinstance(branches, list) or not isinstance(data, dict):
        return schema
    discriminator = data.get("response_type")
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        allowed = branch.get("properties", {}).get("response_type", {}).get("enum")
        if allowed and discriminator in allowed:
            return branch
    return schema


def _schema_correction(detail: str) -> str:
    return (
        "Your previous response did not match the required JSON Schema and was rejected: "
        f"{detail}\nReply again with only a JSON object that matches the schema exactly."
    )


class TranslationError(RuntimeError):
    """Raised when a model response violates the schema or configured limits."""


class _SchemaError(TranslationError):
    """Internal: a model response failed schema validation and may be retried.

    Args:
        detail: The validation failure message shown back to the model.
    """

    def __init__(self, detail: str) -> None:
        super().__init__("response schema invalid")
        self.detail = detail


class _BaseWebNovel:
    """Shared webnovel project state and provider-agnostic operations.

    Args:
        original_title: Title of the webnovel in the source language.
        original_synopsis: Synopsis in the source language.
        title: Translated title, or None until translated.
        synopsis: Translated synopsis, or None until translated.
        settings: Translation settings, or None for defaults.
        glossary: An existing glossary, or None for an empty one.
        chapters: Existing chapters in order, or None for none.
    """

    def __init__(
        self,
        original_title: str,
        original_synopsis: str,
        *,
        title: str | None = None,
        synopsis: str | None = None,
        settings: TranslationSettings | None = None,
        glossary: Glossary | None = None,
        chapters: list[Chapter] | None = None,
    ) -> None:
        self.original_title = original_title
        self.original_synopsis = original_synopsis
        self.title = title
        self.synopsis = synopsis
        self.settings = settings if settings is not None else TranslationSettings()
        self.glossary = glossary if glossary is not None else Glossary()
        self.chapters = chapters if chapters is not None else []
        self.context = Context()

    def apply_response(self, response: TranslationResponse, original_title: str = "", original_content: str = "") -> None:
        """Apply a translation response, updating glossary, chapters, and context.

        Args:
            response: The parsed response.
            original_title: Source title to store on any newly added chapter.
            original_content: Source text to store on any newly added chapter.
        """
        for update in response.glossary_updates:
            self._apply_glossary_update(update)
        for payload in response.chapters:
            self._apply_chapter(payload, original_title, original_content)
        if response.context is not None:
            if response.context.summary is not None:
                self.context.summary = response.context.summary
            if response.context.notes is not None:
                self.context.notes = list(response.context.notes)

    def resolve_request(self, request: TranslationRequest) -> list[DataResult]:
        """Resolve each item of a data request against this project.

        Args:
            request: The parsed data request.

        Returns:
            One result per requested item, in order.
        """
        return [self._resolve(item) for item in request.requests]

    def export(self) -> str:
        """Render every chapter to markdown, resolving glossary references.

        Returns:
            The concatenated markdown.
        """
        rendered: list[str] = []
        for chapter in self.chapters:
            body = chapter.render(self.glossary)
            rendered.append(f"# {chapter.title}\n\n{body}" if chapter.title else body)
        return "\n\n".join(rendered)

    def chapter_to_html(self, chapter: Chapter) -> str:
        """Render one chapter as a standalone HTML document.

        Args:
            chapter: The chapter to render.

        Returns:
            A complete HTML document.
        """
        title = chapter.title or self.title or self.original_title
        return chapter_document(title, render_content_html(chapter.content, self.glossary))

    def to_html(self) -> str:
        """Render the whole project as one HTML reader with a table of contents.

        Returns:
            A complete HTML document with per-chapter navigation.
        """
        chapters = [
            (
                chapter.title or f"Chapter {index + 1}",
                render_content_html(chapter.content, self.glossary, id_prefix=f"c{index}-"),
            )
            for index, chapter in enumerate(self.chapters)
        ]
        return reader_document(self.title or "Untitled", chapters)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the project to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {
            "version": __version__,
            "original_title": self.original_title,
            "title": self.title,
            "synopsis": self.synopsis,
            "original_synopsis": self.original_synopsis,
            "settings": self.settings.to_dict(),
            "glossary": self.glossary.to_dict(),
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "context": self.context.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild a project from its dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed project.
        """
        if data.get("version") != __version__:
            # XXX: allow migrating old versions if old version dump output does not match
            pass
        webnovel = cls(
            original_title=data.get("original_title", ""),
            original_synopsis=data.get("original_synopsis", ""),
            title=data.get("title"),
            synopsis=data.get("synopsis"),
            settings=TranslationSettings.from_dict(data["settings"]) if "settings" in data else None,
            glossary=Glossary.from_dict(data["glossary"]) if "glossary" in data else None,
            chapters=[Chapter.from_dict(chapter) for chapter in data["chapters"]] if "chapters" in data else None,
        )
        if "context" in data:
            webnovel.context = Context.from_dict(data["context"])
        return webnovel

    def dumps(self, **kwargs: Any) -> str:
        """Serialize the project to a compact JSON string.

        Args:
            kwargs: Extra arguments forwarded to json.dumps (override the compact defaults).

        Returns:
            The JSON string.
        """
        kwargs.setdefault("separators", (",", ":"))
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kwargs)

    def dump(self, fp: IO[str], **kwargs: Any) -> None:
        """Serialize the project as compact JSON to a file object.

        Args:
            fp: A writable text file object.
            kwargs: Extra arguments forwarded to json.dump (override the compact defaults).
        """
        kwargs.setdefault("separators", (",", ":"))
        kwargs.setdefault("ensure_ascii", False)
        json.dump(self.to_dict(), fp, **kwargs)

    @classmethod
    def loads(cls, s: str | bytes | bytearray) -> Self:
        """Rebuild a project from a JSON string.

        Args:
            s: The JSON string.

        Returns:
            The reconstructed project.
        """
        return cls.from_dict(json.loads(s))

    @classmethod
    def load(cls, fp: IO[str]) -> Self:
        """Rebuild a project from a JSON file object.

        Args:
            fp: A readable text file object.

        Returns:
            The reconstructed project.
        """
        return cls.from_dict(json.load(fp))

    def _state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"glossary": self.glossary.to_dict(), "notes": self.context.notes}
        if self.settings.context_type is ContextType.ROLLING_SUMMARY:
            state["summary"] = self.context.summary
        return state

    def _messages(self, user: str, fulfilled: list[dict[str, Any]]) -> list[Message]:
        messages = [
            Message(role="system", content=self.settings.build_prompt()),
            _schema_message(output_schema(self.settings)),
            Message(role="system", content="Project state (JSON):\n" + json.dumps(self._state(), ensure_ascii=False)),
        ]
        if fulfilled:
            messages.append(
                Message(role="system", content="Data you requested (JSON):\n" + json.dumps(fulfilled, ensure_ascii=False))
            )
        messages.append(Message(role="user", content=user))
        return messages

    def _metadata_messages(self) -> list[Message]:
        user = f"Translate this webnovel's title and synopsis.\n\nTitle:\n{self.original_title}\n\nSynopsis:\n{self.original_synopsis}"
        return [
            Message(role="system", content=generate_metadata_prompt(self.settings)),
            _schema_message(metadata_schema()),
            Message(role="system", content="Project state (JSON):\n" + json.dumps(self._state(), ensure_ascii=False)),
            Message(role="user", content=user),
        ]

    def _chapter_system(self) -> list[Message]:
        return [
            Message(role="system", content=generate_chapter_prompt(self.settings)),
            Message(role="system", content="Project state (JSON):\n" + json.dumps(self._state(), ensure_ascii=False)),
        ]

    def _analysis_user(self, original_title: str, original_content: str) -> str:
        return f"Analyze this new chapter.\n\nTitle:\n{original_title}\n\nContent:\n{original_content}"

    def _draft_user(self) -> str:
        return "Now translate this chapter, using the finalized glossary in the updated project state above."

    def _cleanup_user(self) -> str:
        return "Review the translation you just produced and return the corrected chapter."

    def _linking_user(self) -> str:
        return "Now link the glossary names in the chapter you just produced, changing nothing else."

    def _apply_glossary_and_context(self, glossary_updates: list[GlossaryUpdate], context: ContextUpdate | None) -> None:
        self._check_glossary_policy(glossary_updates)
        for update in glossary_updates:
            self._apply_glossary_update(update)
        if context is not None:
            if context.summary is not None:
                self.context.summary = context.summary
            if context.notes is not None:
                self.context.notes = list(context.notes)

    def _apply_analysis(self, analysis: ChapterAnalysis) -> None:
        self._apply_glossary_and_context(analysis.glossary_updates, analysis.context)

    def _apply_cleanup(self, cleanup: ChapterCleanup) -> None:
        self._apply_glossary_and_context(cleanup.glossary_updates, cleanup.context)

    def _merge_response(self, merged: TranslationResponse, response: TranslationResponse) -> None:
        merged.chapters.extend(response.chapters)
        merged.glossary_updates.extend(response.glossary_updates)
        if response.context is not None:
            merged.context = response.context

    def _apply_draft(
        self, draft: ChapterDraft, original_title: str, original_content: str, analysis: ChapterAnalysis
    ) -> Chapter:
        if not self.settings.allow_chapter_reorder and draft.position is not None:
            raise TranslationError("chapter reorder not allowed")
        chapter = draft.to_chapter(original_title, original_content, analysis.kind, analysis.summary)
        if draft.position is None:
            self.chapters.append(chapter)
        else:
            self.chapters.insert(draft.position, chapter)
        return chapter

    def _parse_metadata(self, raw: str) -> TranslatedMetadata:
        data = self._validate(raw, metadata_schema())
        return TranslatedMetadata.from_dict(data)

    def _validate(self, raw: str, schema: dict[str, Any]) -> Any:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise _SchemaError(f"the response was not valid JSON: {error}") from error
        try:
            jsonschema.validate(data, _select_branch(schema, data))
        except jsonschema.ValidationError as error:
            raise _SchemaError(error.message) from error
        return data

    def _check_glossary_policy(self, updates: list[GlossaryUpdate]) -> None:
        settings = self.settings
        counts = {GlossaryOp.NEW: 0, GlossaryOp.EDIT: 0, GlossaryOp.DELETE: 0}
        for update in updates:
            counts[update.op] += 1
        checks = [
            (GlossaryOp.NEW, "add", settings.allow_glossary_add, settings.glossary_add_limit),
            (GlossaryOp.EDIT, "edit", settings.allow_glossary_edit, settings.glossary_edit_limit),
            (GlossaryOp.DELETE, "delete", settings.allow_glossary_delete, settings.glossary_delete_limit),
        ]
        for op, action, allowed, limit in checks:
            if counts[op] and not allowed:
                raise TranslationError(f"glossary {action} not allowed")
            if counts[op] > limit:
                raise TranslationError(f"exceeds glossary {action} limit: {limit}")
        self._check_glossary_references(updates)

    def _check_term_refs(self, result: object) -> None:
        """Validate that every content term node points at a real entry and one of its own aliases.

        Raises:
            _SchemaError: When a term_id is not a glossary entry, or an alias_id does not belong
                to that entry — so the caller can ask the model to correct it.
        """
        contents: list[Content] = []
        if isinstance(result, ChapterDraft | ChapterCleanup):
            contents.append(result.content)
        elif isinstance(result, TranslationResponse):
            contents.extend(payload.content for payload in result.chapters)
        for content in contents:
            for node in content.nodes:
                if not isinstance(node, TermNode):
                    continue
                entry = self.glossary.get(node.term_id)
                if entry is None:
                    raise _SchemaError(f"term node term_id {node.term_id!r} is not an id in the glossary")
                if node.alias_id is not None and entry.get_alias(node.alias_id) is None:
                    raise _SchemaError(
                        f"term node alias_id {node.alias_id!r} is not an alias of entry {node.term_id!r} "
                        f"({entry.name!r}); an alias_id must be one of that same entry's own aliases"
                    )

    def _referenced_ids(self) -> tuple[set[str], set[str]]:
        """Collect glossary entry ids and alias ids referenced by any chapter's content.

        Returns:
            A (term ids, alias ids) pair of everything chapters currently point at.
        """
        term_ids: set[str] = set()
        alias_ids: set[str] = set()
        for chapter in self.chapters:
            for node in chapter.content.nodes:
                if isinstance(node, TermNode):
                    term_ids.add(node.term_id)
                    if node.alias_id is not None:
                        alias_ids.add(node.alias_id)
        return term_ids, alias_ids

    def _check_glossary_references(self, updates: list[GlossaryUpdate]) -> None:
        term_ids, alias_ids = self._referenced_ids()
        for update in updates:
            if update.target_id is None:
                continue
            entry = self.glossary.get(update.target_id)
            if update.op is GlossaryOp.DELETE and update.target_id in term_ids:
                raise TranslationError(f"cannot delete glossary entry in use by a chapter: {update.target_id}")
            if update.op is GlossaryOp.EDIT and entry is not None:
                kept = {suggestion.original for suggestion in update.aliases}
                if any(alias.id in alias_ids and alias.original not in kept for alias in entry.aliases):
                    raise TranslationError(f"cannot drop glossary alias in use by a chapter: {update.target_id}")

    def _check_policy(self, response: TranslationResponse) -> None:
        self._check_glossary_policy(response.glossary_updates)
        ids = {chapter.id for chapter in self.chapters}
        for payload in response.chapters:
            if payload.target_id is not None and payload.target_id not in ids:
                raise TranslationError(f"unknown chapter target_id: {payload.target_id}")
        if not self.settings.allow_chapter_reorder and any(payload.position is not None for payload in response.chapters):
            raise TranslationError("chapter reorder not allowed")

    def _apply_glossary_update(self, update: GlossaryUpdate) -> None:
        if update.op is GlossaryOp.DELETE:
            if update.target_id is not None:
                self.glossary.remove(update.target_id)
        elif update.op is GlossaryOp.EDIT:
            if update.target_id is not None:
                self.glossary.replace(update.target_id, update.to_entry())
        else:
            self.glossary.add(update.to_entry())

    def _apply_chapter(self, payload: ChapterTranslation, original_title: str = "", original_content: str = "") -> None:
        existing = next((c for c in self.chapters if c.id == payload.target_id), None) if payload.target_id else None
        if existing is None:
            chapter = payload.to_chapter(original_title, original_content)
            if payload.position is None:
                self.chapters.append(chapter)
            else:
                self.chapters.insert(payload.position, chapter)
            return
        existing.title = payload.title
        existing.content = payload.content
        existing.kind = payload.kind
        existing.summary = payload.summary
        if payload.position is not None:
            self.chapters.remove(existing)
            self.chapters.insert(payload.position, existing)

    def _resolve(self, item: DataRequest) -> DataResult:
        if item.kind is RequestKind.SYNOPSIS:
            return DataResult(request=item, found=True, value=self.synopsis)
        if item.kind is RequestKind.ORIGINAL_TITLE:
            return DataResult(request=item, found=True, value=self.original_title)
        if item.kind is RequestKind.ORIGINAL_SYNOPSIS:
            return DataResult(request=item, found=True, value=self.original_synopsis)
        if item.kind is RequestKind.CHAPTER_LIST:
            listed = self.chapters[item.start : item.end]
            overview = [
                {"id": chapter.id, "title": chapter.title, "kind": chapter.kind.value, "summary": chapter.summary}
                for chapter in listed
            ]
            return DataResult(request=item, found=True, value=overview)
        chapter = next((c for c in self.chapters if c.id == item.chapter_id), None) if item.chapter_id else None
        found = chapter is not None
        if item.kind is RequestKind.CHAPTER_SUMMARY:
            return DataResult(request=item, found=found, value=chapter.summary if chapter else None)
        if item.kind is RequestKind.CHAPTER_TEXT:
            return DataResult(request=item, found=found, value=chapter.render(self.glossary) if chapter else None)
        if item.kind is RequestKind.CHAPTER_ORIGINAL_TITLE:
            return DataResult(request=item, found=found, value=chapter.original_title if chapter else None)
        if item.kind is RequestKind.CHAPTER_ORIGINAL_CONTENT:
            return DataResult(request=item, found=found, value=chapter.original_content if chapter else None)
        return DataResult(request=item, found=False, value=None)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(title={self.title!r}, settings={self.settings!r}, "
            f"chapters={len(self.chapters)}, glossary={self.glossary!r})"
        )


class WebNovel(_BaseWebNovel):
    """A webnovel translation project driven by a synchronous provider.

    Args:
        original_title: Title of the webnovel in the source language.
        original_synopsis: Synopsis in the source language.
        title: Translated title, or None until translated.
        synopsis: Translated synopsis, or None until translated.
        settings: Translation settings, or None for defaults.
        glossary: An existing glossary, or None for an empty one.
        chapters: Existing chapters in order, or None for none.
        provider: Model provider used by translate_new_chapter and request_change.
    """

    def __init__(
        self,
        original_title: str,
        original_synopsis: str,
        *,
        title: str | None = None,
        synopsis: str | None = None,
        settings: TranslationSettings | None = None,
        glossary: Glossary | None = None,
        chapters: list[Chapter] | None = None,
        provider: Provider | None = None,
    ) -> None:
        super().__init__(
            original_title,
            original_synopsis,
            title=title,
            synopsis=synopsis,
            settings=settings,
            glossary=glossary,
            chapters=chapters,
        )
        if isinstance(provider, AsyncProvider):
            raise TypeError("WebNovel needs a synchronous Provider; use AsyncWebNovel for an AsyncProvider")
        self.provider = provider

    def translate_new_chapter(self, original_title: str, original_content: str) -> Chapter:
        """Translate a new chapter in four passes within one conversation and apply it.

        Pass 1 stages glossary, notes, and running-summary updates plus the chapter's kind and
        summary; pass 2 produces the translation using that staged state; pass 3 reviews and
        fixes it (and may correct pass-1 glossary or context mistakes); pass 4 links glossary
        names in the content to their entries by id. Data requests in any pass are
        auto-fulfilled. Nothing is committed to the project until every pass succeeds — on any
        failure the project is left unchanged.

        Args:
            original_title: Source-language title.
            original_content: Raw source text.

        Returns:
            The newly added chapter.
        """
        committed = False
        restore_glossary, restore_context = self.glossary, self.context
        self.glossary = Glossary.from_dict(self.glossary.to_dict())
        self.context = Context.from_dict(self.context.to_dict())
        try:
            conversation = self._chapter_system()
            analysis_schema = analysis_output_schema(self.settings)
            conversation.append(Message(role="system", content=generate_analysis_prompt(self.settings)))
            conversation.append(_schema_message(analysis_schema))
            conversation.append(Message(role="user", content=self._analysis_user(original_title, original_content)))
            analysis = self._ask(conversation, analysis_schema, parse_analysis)
            self._apply_analysis(analysis)

            draft_schema = draft_output_schema(self.settings)
            conversation.append(
                Message(role="system", content="Updated project state (JSON):\n" + json.dumps(self._state(), ensure_ascii=False))
            )
            conversation.append(Message(role="system", content=generate_translation_prompt(self.settings)))
            conversation.append(_schema_message(draft_schema))
            conversation.append(Message(role="user", content=self._draft_user()))
            self._ask(conversation, draft_schema, parse_draft)  # the draft stays in the conversation for pass 3 to review

            cleanup_schema = cleanup_output_schema(self.settings)
            conversation.append(Message(role="system", content=generate_cleanup_prompt(self.settings)))
            conversation.append(_schema_message(cleanup_schema))
            conversation.append(Message(role="user", content=self._cleanup_user()))
            cleanup = self._ask(conversation, cleanup_schema, parse_cleanup)
            self._apply_cleanup(cleanup)

            conversation.append(
                Message(role="system", content="Updated project state (JSON):\n" + json.dumps(self._state(), ensure_ascii=False))
            )
            conversation.append(Message(role="system", content=generate_linking_prompt(self.settings)))
            conversation.append(_schema_message(draft_schema))
            conversation.append(Message(role="user", content=self._linking_user()))
            linked = self._ask(conversation, draft_schema, parse_draft)

            chapter = self._apply_draft(linked, original_title, original_content, analysis)
            committed = True
            return chapter
        finally:
            if not committed:
                self.glossary, self.context = restore_glossary, restore_context

    def _ask(
        self, conversation: list[Message], schema: dict[str, Any], parse: Callable[[Any], TranslationRequest | _T]
    ) -> _T:
        if self.provider is None:
            raise TranslationError("no provider configured")
        fmt = json_schema_format("wntranslate", schema)
        rounds = _MAX_REQUEST_ROUNDS
        schema_retries = _MAX_SCHEMA_RETRIES
        while True:
            raw = self.provider.complete(conversation, response_format=fmt)
            conversation.append(Message(role="assistant", content=raw))
            try:
                result = parse(self._validate(raw, schema))
                if not isinstance(result, TranslationRequest):
                    self._check_term_refs(result)
            except _SchemaError as error:
                if schema_retries <= 0:
                    raise TranslationError("response schema invalid") from error
                schema_retries -= 1
                conversation.append(Message(role="system", content=_schema_correction(error.detail)))
                continue
            if isinstance(result, TranslationRequest):
                rounds -= 1
                if rounds <= 0:
                    raise TranslationError("model kept requesting data")
                fulfilled = [item.to_dict() for item in self.resolve_request(result)]
                conversation.append(
                    Message(role="system", content="Data you requested (JSON):\n" + json.dumps(fulfilled, ensure_ascii=False))
                )
                continue
            return result

    def translate_metadata(self) -> TranslatedMetadata:
        """Translate the webnovel's title and synopsis and apply them.

        Returns:
            The translated title and synopsis.
        """
        metadata = self._complete_parsed(
            self._metadata_messages(),
            json_schema_format("wntranslate_metadata", metadata_schema()),
            self._parse_metadata,
        )
        self.title = metadata.title
        self.synopsis = metadata.synopsis
        return metadata

    def request_change(self, request: str) -> TranslationResponse:
        """Ask for a change and apply it, auto-fulfilling data requests and continuing if asked.

        If the model has more changes than fit in one response it sets ``more`` and is prompted
        to continue; each batch is applied as it arrives. The returned response is the union of
        every applied batch.

        Args:
            request: A free-form description of the change, e.g. a glossary edit.

        Returns:
            The applied change(s).
        """
        schema = output_schema(self.settings)
        conversation = self._messages(request, [])
        merged = TranslationResponse()
        for _ in range(_MAX_CONTINUATION_ROUNDS):
            response = self._ask(conversation, schema, parse_model_output)
            self._check_policy(response)
            self.apply_response(response)
            self._merge_response(merged, response)
            if not response.more:
                return merged
            conversation.append(
                Message(role="user", content="Continue with the remaining changes; set more to false once none are left.")
            )
        raise TranslationError("too many continuation rounds")

    def _complete_parsed(self, messages: list[Message], fmt: dict[str, Any], parse: Callable[[str], _T]) -> _T:
        if self.provider is None:
            raise TranslationError("no provider configured")
        conversation = list(messages)
        retries = _MAX_SCHEMA_RETRIES
        while True:
            raw = self.provider.complete(conversation, response_format=fmt)
            try:
                return parse(raw)
            except _SchemaError as error:
                if retries <= 0:
                    raise TranslationError("response schema invalid") from error
                retries -= 1
                conversation = [
                    *conversation,
                    Message(role="assistant", content=raw),
                    Message(role="system", content=_schema_correction(error.detail)),
                ]


class AsyncWebNovel(_BaseWebNovel):
    """A webnovel translation project driven by an asynchronous provider.

    Args:
        original_title: Title of the webnovel in the source language.
        original_synopsis: Synopsis in the source language.
        title: Translated title, or None until translated.
        synopsis: Translated synopsis, or None until translated.
        settings: Translation settings, or None for defaults.
        glossary: An existing glossary, or None for an empty one.
        chapters: Existing chapters in order, or None for none.
        provider: Async model provider used by translate_new_chapter and request_change.
    """

    def __init__(
        self,
        original_title: str,
        original_synopsis: str,
        *,
        title: str | None = None,
        synopsis: str | None = None,
        settings: TranslationSettings | None = None,
        glossary: Glossary | None = None,
        chapters: list[Chapter] | None = None,
        provider: AsyncProvider | None = None,
    ) -> None:
        super().__init__(
            original_title,
            original_synopsis,
            title=title,
            synopsis=synopsis,
            settings=settings,
            glossary=glossary,
            chapters=chapters,
        )
        if provider is not None and not isinstance(provider, AsyncProvider):
            raise TypeError("AsyncWebNovel needs an AsyncProvider; use WebNovel for a synchronous Provider")
        self.provider = provider

    async def translate_new_chapter(self, original_title: str, original_content: str) -> Chapter:
        """Translate a new chapter in four passes within one conversation and apply it.

        Pass 1 stages glossary, notes, and running-summary updates plus the chapter's kind and
        summary; pass 2 produces the translation using that staged state; pass 3 reviews and
        fixes it (and may correct pass-1 glossary or context mistakes); pass 4 links glossary
        names in the content to their entries by id. Data requests in any pass are
        auto-fulfilled. Nothing is committed to the project until every pass succeeds — on any
        failure the project is left unchanged.

        Args:
            original_title: Source-language title.
            original_content: Raw source text.

        Returns:
            The newly added chapter.
        """
        committed = False
        restore_glossary, restore_context = self.glossary, self.context
        self.glossary = Glossary.from_dict(self.glossary.to_dict())
        self.context = Context.from_dict(self.context.to_dict())
        try:
            conversation = self._chapter_system()
            analysis_schema = analysis_output_schema(self.settings)
            conversation.append(Message(role="system", content=generate_analysis_prompt(self.settings)))
            conversation.append(_schema_message(analysis_schema))
            conversation.append(Message(role="user", content=self._analysis_user(original_title, original_content)))
            analysis = await self._ask(conversation, analysis_schema, parse_analysis)
            self._apply_analysis(analysis)

            draft_schema = draft_output_schema(self.settings)
            conversation.append(
                Message(role="system", content="Updated project state (JSON):\n" + json.dumps(self._state(), ensure_ascii=False))
            )
            conversation.append(Message(role="system", content=generate_translation_prompt(self.settings)))
            conversation.append(_schema_message(draft_schema))
            conversation.append(Message(role="user", content=self._draft_user()))
            await self._ask(conversation, draft_schema, parse_draft)  # the draft stays in the conversation for pass 3

            cleanup_schema = cleanup_output_schema(self.settings)
            conversation.append(Message(role="system", content=generate_cleanup_prompt(self.settings)))
            conversation.append(_schema_message(cleanup_schema))
            conversation.append(Message(role="user", content=self._cleanup_user()))
            cleanup = await self._ask(conversation, cleanup_schema, parse_cleanup)
            self._apply_cleanup(cleanup)

            conversation.append(
                Message(role="system", content="Updated project state (JSON):\n" + json.dumps(self._state(), ensure_ascii=False))
            )
            conversation.append(Message(role="system", content=generate_linking_prompt(self.settings)))
            conversation.append(_schema_message(draft_schema))
            conversation.append(Message(role="user", content=self._linking_user()))
            linked = await self._ask(conversation, draft_schema, parse_draft)

            chapter = self._apply_draft(linked, original_title, original_content, analysis)
            committed = True
            return chapter
        finally:
            if not committed:
                self.glossary, self.context = restore_glossary, restore_context

    async def _ask(
        self, conversation: list[Message], schema: dict[str, Any], parse: Callable[[Any], TranslationRequest | _T]
    ) -> _T:
        if self.provider is None:
            raise TranslationError("no provider configured")
        fmt = json_schema_format("wntranslate", schema)
        rounds = _MAX_REQUEST_ROUNDS
        schema_retries = _MAX_SCHEMA_RETRIES
        while True:
            raw = await self.provider.complete(conversation, response_format=fmt)
            conversation.append(Message(role="assistant", content=raw))
            try:
                result = parse(self._validate(raw, schema))
                if not isinstance(result, TranslationRequest):
                    self._check_term_refs(result)
            except _SchemaError as error:
                if schema_retries <= 0:
                    raise TranslationError("response schema invalid") from error
                schema_retries -= 1
                conversation.append(Message(role="system", content=_schema_correction(error.detail)))
                continue
            if isinstance(result, TranslationRequest):
                rounds -= 1
                if rounds <= 0:
                    raise TranslationError("model kept requesting data")
                fulfilled = [item.to_dict() for item in self.resolve_request(result)]
                conversation.append(
                    Message(role="system", content="Data you requested (JSON):\n" + json.dumps(fulfilled, ensure_ascii=False))
                )
                continue
            return result

    async def translate_metadata(self) -> TranslatedMetadata:
        """Translate the webnovel's title and synopsis and apply them.

        Returns:
            The translated title and synopsis.
        """
        metadata = await self._complete_parsed(
            self._metadata_messages(),
            json_schema_format("wntranslate_metadata", metadata_schema()),
            self._parse_metadata,
        )
        self.title = metadata.title
        self.synopsis = metadata.synopsis
        return metadata

    async def request_change(self, request: str) -> TranslationResponse:
        """Ask for a change and apply it, auto-fulfilling data requests and continuing if asked.

        If the model has more changes than fit in one response it sets ``more`` and is prompted
        to continue; each batch is applied as it arrives. The returned response is the union of
        every applied batch.

        Args:
            request: A free-form description of the change, e.g. a glossary edit.

        Returns:
            The applied change(s).
        """
        schema = output_schema(self.settings)
        conversation = self._messages(request, [])
        merged = TranslationResponse()
        for _ in range(_MAX_CONTINUATION_ROUNDS):
            response = await self._ask(conversation, schema, parse_model_output)
            self._check_policy(response)
            self.apply_response(response)
            self._merge_response(merged, response)
            if not response.more:
                return merged
            conversation.append(
                Message(role="user", content="Continue with the remaining changes; set more to false once none are left.")
            )
        raise TranslationError("too many continuation rounds")

    async def _complete_parsed(self, messages: list[Message], fmt: dict[str, Any], parse: Callable[[str], _T]) -> _T:
        if self.provider is None:
            raise TranslationError("no provider configured")
        conversation = list(messages)
        retries = _MAX_SCHEMA_RETRIES
        while True:
            raw = await self.provider.complete(conversation, response_format=fmt)
            try:
                return parse(raw)
            except _SchemaError as error:
                if retries <= 0:
                    raise TranslationError("response schema invalid") from error
                retries -= 1
                conversation = [
                    *conversation,
                    Message(role="assistant", content=raw),
                    Message(role="system", content=_schema_correction(error.detail)),
                ]
