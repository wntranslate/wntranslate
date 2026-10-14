from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import IO, Any, Self, TypeVar

import jsonschema

from ._version import __version__
from .chapter import Chapter
from .content import Content, TermNode, TextNode
from .context import Context
from .enums import GlossaryOp, RequestKind
from .glossary import Glossary
from .linker import collapse_content, link_content
from .prompts import (
    generate_analysis_prompt,
    generate_chapter_prompt,
    generate_cleanup_prompt,
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
_MAX_SCHEMA_RETRIES = 5
_MAX_CONTINUATION_ROUNDS = 100
_OVERVIEW_CHAPTERS = 10

_T = TypeVar("_T")


def _content_is_empty(content: Content) -> bool:
    """Whether a chapter's content renders to nothing (no prose text and no linked name)."""
    return not any(
        (isinstance(node, TextNode) and node.text.strip()) or isinstance(node, TermNode) for node in content.nodes
    )


def _version_below(version: str | None, target: tuple[int, int, int]) -> bool:
    """Whether a save's version string is older than target; an absent or unparseable one counts as older."""
    if not version:
        return True
    try:
        parts = tuple(int(piece) for piece in version.split(".")[:3])
    except ValueError:
        return True
    return parts + (0,) * (3 - len(parts)) < target


def _cap_runs(text: str, limit: int) -> str:
    """Shorten any run of a single repeated character longer than limit to limit copies.

    Applied to source text sent to the model so a pathological run (e.g. a 90-character scream of
    'ー') cannot prime a weak model into an endless repeat. Runs of newlines are left alone, and a
    limit of 0 or less disables it.
    """
    if limit <= 0:
        return text
    return re.sub(rf"([^\n])\1{{{limit},}}", lambda match: match.group(1) * limit, text)


def _flatten(schema: dict[str, Any]) -> dict[str, Any]:
    """Collapse a top-level request/result oneOf into one flat object keyed by response_type.

    A top-level oneOf is both rejected by Anthropic tool schemas and unreliable for models to
    fill (they omit the response_type discriminator buried in the branches). This is sent to the
    model for every provider; the library still validates the reply against the original oneOf, so
    per-branch required fields are enforced and errors stay precise. Non-oneOf schemas (e.g. the
    metadata object) are returned unchanged.
    """
    branches = schema.get("oneOf")
    if not isinstance(branches, list):
        return schema
    properties: dict[str, Any] = {}
    response_types: list[Any] = []
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        properties.update(branch.get("properties", {}))
        response_types.extend(branch.get("properties", {}).get("response_type", {}).get("enum", []))
    properties["response_type"] = {"type": "string", "enum": response_types}
    return {
        "type": "object",
        "required": ["response_type"],
        "additionalProperties": False,
        "properties": properties,
    }


def _schema_message(schema: dict[str, Any]) -> Message:
    return Message(
        role="system",
        content="Return exactly one JSON object that conforms to this JSON Schema:\n"
        + json.dumps(_flatten(schema), ensure_ascii=False),
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


def _correction(detail: str) -> str:
    return (
        f"Your previous response was rejected: {detail}\n"
        "Fix exactly that and reply again with only a JSON object that matches the schema."
    )


def _correction_messages(error: _Retryable, schema: dict[str, Any]) -> list[Message]:
    """Build the messages to send after a rejected response, re-sending the schema on a schema fail.

    The schema is re-sent first (when the failure was a schema fail) so the correction, which names
    exactly what to fix, is the final and most salient instruction before the model replies.
    """
    messages: list[Message] = []
    if isinstance(error, _SchemaError):
        messages.append(_schema_message(schema))
    messages.append(Message(role="system", content=_correction(error.detail)))
    return messages


class TranslationError(RuntimeError):
    """Raised when a model response violates the schema or configured limits."""


class _Retryable(TranslationError):
    """Internal: a rejected model response that the model may be able to fix on a retry.

    Args:
        detail: What to tell the model so it can correct its next reply.
        final: The message to raise if retries run out; defaults to detail.
    """

    def __init__(self, detail: str, final: str | None = None) -> None:
        super().__init__(final if final is not None else detail)
        self.detail = detail
        self.final = final if final is not None else detail


class _SchemaError(_Retryable):
    """A response that failed schema or term validation; retried, else 'response schema invalid'."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail, final="response schema invalid")


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

    def relink(self) -> int:
        """Re-link every chapter's text to glossary entries, deterministically.

        Useful after the glossary grows (names introduced in later chapters are back-linked into
        earlier ones) or after a bulk glossary repair. Only unlinked text is affected; existing
        term and footnote nodes are left as they are.

        Returns:
            The number of chapters whose content changed.
        """
        changed = 0
        for chapter in self.chapters:
            linked = link_content(chapter.content, self.glossary)
            if linked is not chapter.content:
                chapter.content = linked
                changed += 1
        return changed

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
        if _version_below(data.get("version"), (1, 3, 0)):
            for chapter in webnovel.chapters:
                chapter.content = collapse_content(chapter.content, webnovel.glossary)
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
        return {"glossary": self.glossary.to_dict(include_definition=False), "notes": self.context.notes}

    def _overview(self) -> dict[str, Any]:
        recent = self.chapters[-_OVERVIEW_CHAPTERS:]
        return {
            "original_title": self.original_title,
            "title": self.title,
            "original_synopsis": self.original_synopsis,
            "synopsis": self.synopsis,
            "total_chapters": len(self.chapters),
            "recent_chapters": [
                {"id": c.id, "title": c.title, "original_title": c.original_title, "kind": c.kind.value} for c in recent
            ],
        }

    def _overview_message(self) -> Message:
        return Message(role="system", content="Work overview (JSON):\n" + json.dumps(self._overview(), ensure_ascii=False))

    def _state_message(self) -> Message:
        return Message(role="system", content="Project state (JSON):\n" + json.dumps(self._state(), ensure_ascii=False))

    def _messages(self, user: str, fulfilled: list[dict[str, Any]]) -> list[Message]:
        messages = [
            Message(role="system", content=self.settings.build_prompt()),
            _schema_message(output_schema(self.settings)),
            self._overview_message(),
            self._state_message(),
        ]
        if fulfilled:
            messages.append(
                Message(role="system", content="Data you requested (JSON):\n" + json.dumps(fulfilled, ensure_ascii=False))
            )
        messages.append(Message(role="user", content=user))
        return messages

    def _metadata_messages(self) -> list[Message]:
        user = (
            "Translate this webnovel's title and synopsis.\n\n"
            f"Title:\n{self._cap(self.original_title)}\n\nSynopsis:\n{self._cap(self.original_synopsis)}"
        )
        return [
            Message(role="system", content=generate_metadata_prompt(self.settings)),
            _schema_message(metadata_schema()),
            self._overview_message(),
            self._state_message(),
            Message(role="user", content=user),
        ]

    def _chapter_system(self) -> list[Message]:
        return [
            Message(role="system", content=generate_chapter_prompt(self.settings)),
            self._overview_message(),
            self._state_message(),
        ]

    def _cap(self, text: str) -> str:
        return _cap_runs(text, self.settings.max_source_char_run)

    def _analysis_user(self, original_title: str, original_content: str) -> str:
        return f"Analyze this new chapter.\n\nTitle:\n{self._cap(original_title)}\n\nContent:\n{self._cap(original_content)}"

    def _draft_user(self, original_title: str, original_content: str) -> str:
        return (
            "Now translate THIS chapter in full from the source below — its title and every paragraph from the "
            "first line to the last, using the finalized glossary in the project state above. Translate only this "
            "source; never copy a neighbouring chapter. Never stop early, summarize, or omit any part of it.\n\n"
            f"Title:\n{self._cap(original_title)}\n\nContent:\n{self._cap(original_content)}"
        )

    def _chapter_json(self, title: str, content: Content) -> str:
        return json.dumps({"title": title, "content": content.to_dict()}, ensure_ascii=False)

    def _cleanup_user(self, draft: ChapterDraft, original_title: str, original_content: str) -> str:
        return (
            "Review this translation against the source below and return the ENTIRE corrected chapter — every node "
            "from start to finish, not only the parts you change. The title and content must be a translation of "
            "THIS source, never a copy of another chapter.\n\nSource title:\n"
            + self._cap(original_title)
            + "\n\nSource:\n"
            + self._cap(original_content)
            + "\n\nYour translation:\n"
            + self._chapter_json(draft.title, draft.content)
        )

    def _apply_glossary_and_context(self, glossary_updates: list[GlossaryUpdate], context: ContextUpdate | None) -> None:
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
        chapter = draft.to_chapter(original_title, original_content, analysis.kind, analysis.summary)
        chapter.content = link_content(chapter.content, self.glossary)
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
                raise _Retryable(f"glossary {action} not allowed")
            if counts[op] > limit:
                raise _Retryable(f"exceeds glossary {action} limit: {limit}")
        for update in updates:
            if update.op is GlossaryOp.NEW and not update.name.strip():
                raise _Retryable("a new glossary entry needs a non-blank name")
            if update.op is GlossaryOp.EDIT and update.provides("name") and not update.name.strip():
                raise _Retryable("a glossary edit must not blank the name; omit name to keep the current one")
        self._check_glossary_references(updates)

    def _check_reorder(self, position: int | None) -> None:
        if position is not None and not self.settings.allow_chapter_reorder:
            raise _Retryable("chapter reorder not allowed")

    def _validate_result(self, result: object) -> None:
        """Validate a parsed result's term refs and policy, raising _Retryable so _ask can retry."""
        self._check_content_nonempty(result)
        self._check_term_refs(result)
        if isinstance(result, ChapterAnalysis):
            self._check_glossary_policy(result.glossary_updates)
        elif isinstance(result, ChapterCleanup):
            self._check_glossary_policy(result.glossary_updates)
            self._check_reorder(result.position)
        elif isinstance(result, ChapterDraft):
            self._check_reorder(result.position)
        elif isinstance(result, TranslationResponse):
            self._check_policy(result)

    def _result_contents(self, result: object) -> list[Content]:
        """The chapter contents a result carries (skipping edits that omit content)."""
        if isinstance(result, ChapterDraft | ChapterCleanup):
            return [result.content]
        if isinstance(result, TranslationResponse):
            return [payload.content for payload in result.chapters if payload.content is not None]
        return []

    def _check_content_nonempty(self, result: object) -> None:
        """Require real content where it must exist, so a translation can never be blanked.

        A translation pass (draft/cleanup) must carry the full chapter. In a request, content that
        is PROVIDED but empty is always rejected; an edit of an existing chapter may OMIT content
        (None) to keep the current body, but a newly added chapter must include content.

        Raises:
            _SchemaError: When a pass has empty content, a chapter sends empty (not omitted)
                content, or a new chapter omits content.
        """
        if isinstance(result, ChapterDraft | ChapterCleanup):
            if _content_is_empty(result.content):
                raise _SchemaError("the chapter content is empty; return the full translated content, every paragraph")
        elif isinstance(result, TranslationResponse):
            existing = {chapter.id for chapter in self.chapters}
            for payload in result.chapters:
                if payload.content is not None and _content_is_empty(payload.content):
                    raise _SchemaError(
                        "a chapter's content is empty; to keep the existing body omit content entirely, otherwise "
                        "send the full translated content — never send empty content"
                    )
                is_edit = payload.target_id is not None and payload.target_id in existing
                if not is_edit and payload.content is None:
                    raise _SchemaError("a new chapter has no content; a new chapter must include its full translated content")

    def _check_term_refs(self, result: object) -> None:
        """Validate that every content term node points at a real entry and one of its own aliases.

        Raises:
            _SchemaError: When a term_id is not a glossary entry, or an alias_id does not belong
                to that entry — so the caller can ask the model to correct it.
        """
        for content in self._result_contents(result):
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
                raise _Retryable(f"cannot delete glossary entry in use by a chapter: {update.target_id}")
            if update.op is GlossaryOp.EDIT and entry is not None and update.provides("aliases"):
                kept = {suggestion.original for suggestion in update.aliases}
                if any(alias.id in alias_ids and alias.original not in kept for alias in entry.aliases):
                    raise _Retryable(f"cannot drop glossary alias in use by a chapter: {update.target_id}")

    def _check_policy(self, response: TranslationResponse) -> None:
        self._check_glossary_policy(response.glossary_updates)
        ids = {chapter.id for chapter in self.chapters}
        for payload in response.chapters:
            if payload.target_id is not None and payload.target_id not in ids:
                raise _Retryable(f"unknown chapter target_id: {payload.target_id}")
            self._check_reorder(payload.position)

    def _apply_glossary_update(self, update: GlossaryUpdate) -> None:
        if update.op is GlossaryOp.DELETE:
            if update.target_id is not None:
                self.glossary.remove(update.target_id)
        elif update.op is GlossaryOp.EDIT:
            if update.target_id is None:
                return
            existing = self.glossary.get(update.target_id)
            if existing is not None:
                self.glossary.replace(update.target_id, update.merged_entry(existing))
        else:
            self.glossary.add(update.to_entry())

    def _apply_chapter(self, payload: ChapterTranslation, original_title: str = "", original_content: str = "") -> None:
        existing = next((c for c in self.chapters if c.id == payload.target_id), None) if payload.target_id else None
        if existing is None:
            chapter = payload.to_chapter(original_title, original_content)
            chapter.content = link_content(chapter.content, self.glossary)
            if payload.position is None:
                self.chapters.append(chapter)
            else:
                self.chapters.insert(payload.position, chapter)
            return
        if payload.title:
            existing.title = payload.title
        if payload.content is not None and not _content_is_empty(payload.content):
            existing.content = link_content(payload.content, self.glossary)
        if payload.kind is not None:
            existing.kind = payload.kind
        if payload.summary is not None:
            existing.summary = payload.summary
        if payload.position is not None:
            self.chapters.remove(existing)
            self.chapters.insert(payload.position, existing)

    def _resolve(self, item: DataRequest) -> DataResult:
        if item.kind is RequestKind.STORY_SUMMARY:
            return DataResult(request=item, found=True, value=self.context.summary)
        if item.kind is RequestKind.GLOSSARY_ENTRY:
            entry = self.glossary.get(item.glossary_id) if item.glossary_id else None
            return DataResult(request=item, found=entry is not None, value=[entry.to_dict()] if entry else None)
        if item.kind is RequestKind.SYNOPSIS:
            return DataResult(request=item, found=True, value=self.synopsis)
        if item.kind is RequestKind.ORIGINAL_TITLE:
            return DataResult(request=item, found=True, value=self._cap(self.original_title))
        if item.kind is RequestKind.ORIGINAL_SYNOPSIS:
            return DataResult(request=item, found=True, value=self._cap(self.original_synopsis))
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
            return DataResult(request=item, found=found, value=self._cap(chapter.original_title) if chapter else None)
        if item.kind is RequestKind.CHAPTER_ORIGINAL_CONTENT:
            return DataResult(request=item, found=found, value=self._cap(chapter.original_content) if chapter else None)
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
        """Translate a new chapter in three model passes, each its own chat, and apply it.

        Pass 1 stages glossary, notes, and running-summary updates plus the chapter's kind and
        summary; pass 2 produces the translation using that staged state; pass 3 reviews and
        fixes it (and may correct pass-1 glossary or context mistakes). Each pass is a fresh
        conversation carrying only the finalized state and what that pass needs, so earlier
        passes' output does not bloat later ones. Glossary names are then linked to their entries
        deterministically against the finalized glossary (no model pass). Data requests in any
        pass are auto-fulfilled. Nothing is committed to the project until every pass succeeds —
        on any failure the project is left unchanged.

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
            analysis_schema = analysis_output_schema(self.settings)
            conversation = self._chapter_system()
            conversation.append(Message(role="system", content=generate_analysis_prompt(self.settings)))
            conversation.append(_schema_message(analysis_schema))
            conversation.append(Message(role="user", content=self._analysis_user(original_title, original_content)))
            analysis = self._ask(conversation, analysis_schema, parse_analysis)
            self._apply_analysis(analysis)

            draft_schema = draft_output_schema(self.settings)
            conversation = self._chapter_system()
            conversation.append(Message(role="system", content=generate_translation_prompt(self.settings)))
            conversation.append(_schema_message(draft_schema))
            conversation.append(Message(role="user", content=self._draft_user(original_title, original_content)))
            draft = self._ask(conversation, draft_schema, parse_draft)

            cleanup_schema = cleanup_output_schema(self.settings)
            conversation = self._chapter_system()
            conversation.append(Message(role="system", content=generate_cleanup_prompt(self.settings)))
            conversation.append(_schema_message(cleanup_schema))
            conversation.append(Message(role="user", content=self._cleanup_user(draft, original_title, original_content)))
            cleanup = self._ask(conversation, cleanup_schema, parse_cleanup)
            self._apply_cleanup(cleanup)

            final = ChapterDraft(title=cleanup.title, content=cleanup.content, position=cleanup.position)
            chapter = self._apply_draft(final, original_title, original_content, analysis)
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
        fmt = json_schema_format("wntranslate", _flatten(schema))
        rounds = _MAX_REQUEST_ROUNDS
        schema_retries = _MAX_SCHEMA_RETRIES
        while True:
            raw = self.provider.complete(conversation, response_format=fmt)
            conversation.append(Message(role="assistant", content=raw))
            try:
                result = parse(self._validate(raw, schema))
                if not isinstance(result, TranslationRequest):
                    self._validate_result(result)
            except _Retryable as error:
                if schema_retries <= 0:
                    raise TranslationError(error.final) from error
                schema_retries -= 1
                conversation.extend(_correction_messages(error, schema))
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
        schema = fmt.get("json_schema", {}).get("schema", {})
        retries = _MAX_SCHEMA_RETRIES
        while True:
            raw = self.provider.complete(conversation, response_format=fmt)
            try:
                return parse(raw)
            except _Retryable as error:
                if retries <= 0:
                    raise TranslationError(error.final) from error
                retries -= 1
                conversation = [
                    *conversation,
                    Message(role="assistant", content=raw),
                    *_correction_messages(error, schema),
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
        """Translate a new chapter in three model passes, each its own chat, and apply it.

        Pass 1 stages glossary, notes, and running-summary updates plus the chapter's kind and
        summary; pass 2 produces the translation using that staged state; pass 3 reviews and
        fixes it (and may correct pass-1 glossary or context mistakes). Each pass is a fresh
        conversation carrying only the finalized state and what that pass needs, so earlier
        passes' output does not bloat later ones. Glossary names are then linked to their entries
        deterministically against the finalized glossary (no model pass). Data requests in any
        pass are auto-fulfilled. Nothing is committed to the project until every pass succeeds —
        on any failure the project is left unchanged.

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
            analysis_schema = analysis_output_schema(self.settings)
            conversation = self._chapter_system()
            conversation.append(Message(role="system", content=generate_analysis_prompt(self.settings)))
            conversation.append(_schema_message(analysis_schema))
            conversation.append(Message(role="user", content=self._analysis_user(original_title, original_content)))
            analysis = await self._ask(conversation, analysis_schema, parse_analysis)
            self._apply_analysis(analysis)

            draft_schema = draft_output_schema(self.settings)
            conversation = self._chapter_system()
            conversation.append(Message(role="system", content=generate_translation_prompt(self.settings)))
            conversation.append(_schema_message(draft_schema))
            conversation.append(Message(role="user", content=self._draft_user(original_title, original_content)))
            draft = await self._ask(conversation, draft_schema, parse_draft)

            cleanup_schema = cleanup_output_schema(self.settings)
            conversation = self._chapter_system()
            conversation.append(Message(role="system", content=generate_cleanup_prompt(self.settings)))
            conversation.append(_schema_message(cleanup_schema))
            conversation.append(Message(role="user", content=self._cleanup_user(draft, original_title, original_content)))
            cleanup = await self._ask(conversation, cleanup_schema, parse_cleanup)
            self._apply_cleanup(cleanup)

            final = ChapterDraft(title=cleanup.title, content=cleanup.content, position=cleanup.position)
            chapter = self._apply_draft(final, original_title, original_content, analysis)
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
        fmt = json_schema_format("wntranslate", _flatten(schema))
        rounds = _MAX_REQUEST_ROUNDS
        schema_retries = _MAX_SCHEMA_RETRIES
        while True:
            raw = await self.provider.complete(conversation, response_format=fmt)
            conversation.append(Message(role="assistant", content=raw))
            try:
                result = parse(self._validate(raw, schema))
                if not isinstance(result, TranslationRequest):
                    self._validate_result(result)
            except _Retryable as error:
                if schema_retries <= 0:
                    raise TranslationError(error.final) from error
                schema_retries -= 1
                conversation.extend(_correction_messages(error, schema))
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
        schema = fmt.get("json_schema", {}).get("schema", {})
        retries = _MAX_SCHEMA_RETRIES
        while True:
            raw = await self.provider.complete(conversation, response_format=fmt)
            try:
                return parse(raw)
            except _Retryable as error:
                if retries <= 0:
                    raise TranslationError(error.final) from error
                retries -= 1
                conversation = [
                    *conversation,
                    Message(role="assistant", content=raw),
                    *_correction_messages(error, schema),
                ]
