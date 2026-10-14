from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from .chapter import Chapter
from .content import Content
from .enums import (
    ChapterKind,
    GlossaryOp,
    NameType,
    NodeType,
    ResponseType,
)
from .glossary import (
    Alias,
    CreativeWorkEntry,
    GlossaryEntry,
    ItemEntry,
    OrganizationEntry,
    OtherEntry,
    PersonEntry,
)
from .request import TranslationRequest, request_schema

if TYPE_CHECKING:
    from .settings import TranslationSettings


@dataclass
class AliasSuggestion:
    """A proposed alias in a glossary update.

    Args:
        original: Source-language form.
        translation: Output form.
        context: Optional note describing when the alias applies.
    """

    original: str
    translation: str = ""
    context: str | None = None

    def to_alias(self) -> Alias:
        """Convert to a glossary alias.

        Returns:
            The new alias.
        """
        return Alias(original=self.original, translation=self.translation, context=self.context)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {"original": self.original, "translation": self.translation, "context": self.context}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AliasSuggestion:
        """Rebuild from a dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed suggestion.
        """
        return cls(original=data["original"], translation=data.get("translation", ""), context=data.get("context"))


_MERGE_FIELDS = frozenset(
    {"name", "original_name", "type", "definition", "aliases", "pronouns", "medium", "specified_type"}
)


@dataclass
class GlossaryUpdate:
    """A glossary change proposed by the model: add, edit, or delete.

    Args:
        op: Whether to add, edit, or delete an entry.
        target_id: Existing entry id, required to edit or delete.
        name: Output name of the term.
        original_name: Source-language name.
        type: Name type.
        definition: Explanation of the term.
        aliases: Proposed aliases.
        pronouns: Pronouns, for people.
        medium: Work medium, for creative works.
        specified_type: Free-form type, for other.
    """

    op: GlossaryOp = GlossaryOp.NEW
    target_id: str | None = None
    name: str = ""
    original_name: str = ""
    type: NameType = NameType.OTHER
    definition: str = ""
    aliases: list[AliasSuggestion] = field(default_factory=list)
    pronouns: str = ""
    medium: str = ""
    specified_type: str = ""
    # Which content fields the model actually sent (set by from_dict). None means "all fields are
    # present", so a directly-constructed update behaves like a full entry.
    _provided: frozenset[str] | None = field(default=None, init=False, repr=False, compare=False)

    def provides(self, field_name: str) -> bool:
        """Whether an edit should apply this field, i.e. the model sent it (or it wasn't tracked)."""
        return self._provided is None or field_name in self._provided

    def merged_entry(self, existing: GlossaryEntry) -> GlossaryEntry:
        """Build the entry an edit should produce, keeping fields the update did not provide.

        A glossary edit is a partial update: the model sends only the fields it wants to change.
        Fields it omits are taken from the existing entry, so editing (say) a definition can never
        wipe the name. Alias ids are re-matched by Glossary.replace.

        Args:
            existing: The entry being edited.

        Returns:
            The merged entry to store in place of the existing one.
        """
        kept_aliases = [AliasSuggestion(a.original, a.translation, a.context) for a in existing.aliases]
        merged = GlossaryUpdate(
            op=GlossaryOp.EDIT,
            target_id=self.target_id,
            name=self.name if self.provides("name") else existing.name,
            original_name=self.original_name if self.provides("original_name") else existing.original_name,
            type=self.type if self.provides("type") else existing.type,
            definition=self.definition if self.provides("definition") else existing.definition,
            aliases=self.aliases if self.provides("aliases") else kept_aliases,
            pronouns=self.pronouns if self.provides("pronouns") else getattr(existing, "pronouns", ""),
            medium=self.medium if self.provides("medium") else getattr(existing, "medium", ""),
            specified_type=self.specified_type if self.provides("specified_type") else getattr(existing, "specified_type", ""),
        )
        return merged.to_entry()

    def to_entry(self) -> GlossaryEntry:
        """Build a new glossary entry of the matching subclass.

        Returns:
            The new entry.
        """
        aliases = [alias.to_alias() for alias in self.aliases]
        if self.type is NameType.PERSON:
            return PersonEntry(
                name=self.name,
                original_name=self.original_name,
                definition=self.definition,
                aliases=aliases,
                pronouns=self.pronouns,
            )
        if self.type is NameType.ITEM:
            return ItemEntry(name=self.name, original_name=self.original_name, definition=self.definition, aliases=aliases)
        if self.type is NameType.ORGANIZATION:
            return OrganizationEntry(
                name=self.name, original_name=self.original_name, definition=self.definition, aliases=aliases
            )
        if self.type is NameType.CREATIVE_WORK:
            return CreativeWorkEntry(
                name=self.name,
                original_name=self.original_name,
                definition=self.definition,
                aliases=aliases,
                medium=self.medium,
            )
        return OtherEntry(
            name=self.name,
            original_name=self.original_name,
            definition=self.definition,
            aliases=aliases,
            specified_type=self.specified_type,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {
            "op": self.op.value,
            "target_id": self.target_id,
            "name": self.name,
            "original_name": self.original_name,
            "type": self.type.value,
            "definition": self.definition,
            "aliases": [alias.to_dict() for alias in self.aliases],
            "pronouns": self.pronouns,
            "medium": self.medium,
            "specified_type": self.specified_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlossaryUpdate:
        """Rebuild from a dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed update.
        """
        update = cls(
            op=GlossaryOp(data.get("op", GlossaryOp.NEW.value)),
            target_id=data.get("target_id"),
            name=data.get("name", ""),
            original_name=data.get("original_name", ""),
            type=NameType(data.get("type", NameType.OTHER.value)),
            definition=data.get("definition", ""),
            aliases=[AliasSuggestion.from_dict(alias) for alias in data.get("aliases", [])],
            pronouns=data.get("pronouns", ""),
            medium=data.get("medium", ""),
            specified_type=data.get("specified_type", ""),
        )
        update._provided = frozenset(key for key in _MERGE_FIELDS if key in data)
        return update


@dataclass
class ChapterTranslation:
    """A translated chapter payload returned by the model.

    The source title and text are not part of the model's output; they are supplied by
    the caller of translate_new_chapter, or kept from the existing chapter on a revision.
    content, kind, and summary are optional: when editing an existing chapter (target_id set),
    an omitted field (None) means "leave it unchanged", so a title can change without resending
    the body. A newly added chapter must include content.

    Args:
        title: Translated title.
        content: Translated content nodes, or None to keep the existing chapter's content.
        kind: Kind of unit, or None to keep the existing chapter's kind.
        summary: Short summary, or None to keep the existing chapter's summary.
        target_id: Existing chapter id to replace, or None to append.
        position: Index to place the chapter at, or None to keep or append.
    """

    title: str = ""
    content: Content | None = None
    kind: ChapterKind | None = None
    summary: str | None = None
    target_id: str | None = None
    position: int | None = None

    def to_chapter(self, original_title: str = "", original_content: str = "") -> Chapter:
        """Build a new chapter from this payload and the given source title and text.

        Args:
            original_title: Source-language title to store on the chapter.
            original_content: Raw source text to store on the chapter.

        Returns:
            The new chapter.
        """
        return Chapter(
            original_title=original_title,
            original_content=original_content,
            title=self.title,
            content=self.content if self.content is not None else Content(),
            kind=self.kind if self.kind is not None else ChapterKind.CHAPTER,
            summary=self.summary,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict, omitting fields left unset for an edit.

        Returns:
            A JSON-compatible representation.
        """
        data: dict[str, Any] = {"title": self.title, "target_id": self.target_id, "position": self.position}
        if self.content is not None:
            data["content"] = self.content.to_dict()
        if self.kind is not None:
            data["kind"] = self.kind.value
        if self.summary is not None:
            data["summary"] = self.summary
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChapterTranslation:
        """Rebuild from a dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed payload.
        """
        content = data.get("content")
        kind = data.get("kind")
        return cls(
            title=data.get("title", ""),
            content=Content.from_dict(content) if content is not None else None,
            kind=ChapterKind(kind) if kind is not None else None,
            summary=data.get("summary"),
            target_id=data.get("target_id"),
            position=data.get("position"),
        )


@dataclass
class ContextUpdate:
    """An update to the running context returned by the model.

    Args:
        summary: Replacement story summary, or None to leave unchanged.
        notes: Replacement full notes list, or None to leave unchanged.
    """

    summary: str | None = None
    notes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {"summary": self.summary, "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextUpdate:
        """Rebuild from a dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed update.
        """
        notes = data.get("notes")
        return cls(summary=data.get("summary"), notes=list(notes) if notes is not None else None)


@dataclass
class TranslationResponse:
    """A translation result the model returns.

    Args:
        chapters: Translated chapters.
        glossary_updates: Glossary additions, edits, and deletions.
        context: Running context update, or None.
        more: True when the model has further changes it could not fit and wants to continue.
    """

    chapters: list[ChapterTranslation] = field(default_factory=list)
    glossary_updates: list[GlossaryUpdate] = field(default_factory=list)
    context: ContextUpdate | None = None
    more: bool = False

    response_type: ClassVar[ResponseType] = ResponseType.TRANSLATION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {
            "response_type": self.response_type.value,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "glossary_updates": [update.to_dict() for update in self.glossary_updates],
            "context": self.context.to_dict() if self.context is not None else None,
            "more": self.more,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslationResponse:
        """Rebuild from a dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed response.
        """
        context = data.get("context")
        return cls(
            chapters=[ChapterTranslation.from_dict(chapter) for chapter in data.get("chapters", [])],
            glossary_updates=[GlossaryUpdate.from_dict(update) for update in data.get("glossary_updates", [])],
            context=ContextUpdate.from_dict(context) if context is not None else None,
            more=bool(data.get("more", False)),
        )


@dataclass
class ChapterAnalysis:
    """First-pass result: glossary and context changes plus this chapter's metadata.

    Args:
        summary: Short summary of the chapter, or None.
        kind: Kind of unit (chapter, announcement, post, or other).
        glossary_updates: Glossary additions, edits, and deletions.
        context: Running context update, or None.
    """

    summary: str | None = None
    kind: ChapterKind = ChapterKind.CHAPTER
    glossary_updates: list[GlossaryUpdate] = field(default_factory=list)
    context: ContextUpdate | None = None

    response_type: ClassVar[ResponseType] = ResponseType.ANALYSIS

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {
            "response_type": self.response_type.value,
            "kind": self.kind.value,
            "summary": self.summary,
            "glossary_updates": [update.to_dict() for update in self.glossary_updates],
            "context": self.context.to_dict() if self.context is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChapterAnalysis:
        """Rebuild from a dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed analysis.
        """
        context = data.get("context")
        return cls(
            summary=data.get("summary"),
            kind=ChapterKind(data.get("kind", ChapterKind.CHAPTER.value)),
            glossary_updates=[GlossaryUpdate.from_dict(update) for update in data.get("glossary_updates", [])],
            context=ContextUpdate.from_dict(context) if context is not None else None,
        )


@dataclass
class ChapterDraft:
    """Second-pass result: the translated title and content for a new chapter.

    Args:
        title: Translated title.
        content: Translated content nodes.
        position: Index to place the chapter at, or None to append.
    """

    title: str = ""
    content: Content = field(default_factory=Content)
    position: int | None = None

    response_type: ClassVar[ResponseType] = ResponseType.TRANSLATION

    def to_chapter(
        self,
        original_title: str = "",
        original_content: str = "",
        kind: ChapterKind = ChapterKind.CHAPTER,
        summary: str | None = None,
    ) -> Chapter:
        """Build a chapter from this draft and the analysis metadata.

        Args:
            original_title: Source-language title to store on the chapter.
            original_content: Raw source text to store on the chapter.
            kind: Kind of unit, from the analysis pass.
            summary: Chapter summary, from the analysis pass.

        Returns:
            The new chapter.
        """
        return Chapter(
            original_title=original_title,
            original_content=original_content,
            title=self.title,
            content=self.content,
            kind=kind,
            summary=summary,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {
            "response_type": self.response_type.value,
            "title": self.title,
            "content": self.content.to_dict(),
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChapterDraft:
        """Rebuild from a dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed draft.
        """
        return cls(
            title=data.get("title", ""),
            content=Content.from_dict(data.get("content", {})),
            position=data.get("position"),
        )


@dataclass
class ChapterCleanup:
    """Third-pass result: the reviewed translation plus optional fixes to pass-1 glossary/context.

    Args:
        title: Corrected translated title.
        content: Corrected translated content nodes.
        position: Index to place the chapter at, or None to append.
        glossary_updates: Corrections to glossary entries the analysis pass got wrong.
        context: Corrections to the notes or running summary, or None to leave them.
    """

    title: str = ""
    content: Content = field(default_factory=Content)
    position: int | None = None
    glossary_updates: list[GlossaryUpdate] = field(default_factory=list)
    context: ContextUpdate | None = None

    response_type: ClassVar[ResponseType] = ResponseType.TRANSLATION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {
            "response_type": self.response_type.value,
            "title": self.title,
            "content": self.content.to_dict(),
            "position": self.position,
            "glossary_updates": [update.to_dict() for update in self.glossary_updates],
            "context": self.context.to_dict() if self.context is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChapterCleanup:
        """Rebuild from a dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed cleanup.
        """
        context = data.get("context")
        return cls(
            title=data.get("title", ""),
            content=Content.from_dict(data.get("content", {})),
            position=data.get("position"),
            glossary_updates=[GlossaryUpdate.from_dict(update) for update in data.get("glossary_updates", [])],
            context=ContextUpdate.from_dict(context) if context is not None else None,
        )


@dataclass
class TranslatedMetadata:
    """Translated webnovel title and synopsis.

    Args:
        title: Translated title.
        synopsis: Translated synopsis.
    """

    title: str = ""
    synopsis: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {"title": self.title, "synopsis": self.synopsis}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslatedMetadata:
        """Rebuild from a dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed metadata.
        """
        return cls(title=data.get("title", ""), synopsis=data.get("synopsis", ""))


def metadata_schema() -> dict[str, Any]:
    """Build the JSON schema for a translated title and synopsis.

    Returns:
        A JSON schema.
    """
    return {
        "type": "object",
        "required": ["title", "synopsis"],
        "additionalProperties": False,
        "properties": {"title": {"type": "string"}, "synopsis": {"type": "string"}},
    }


def parse_model_output(data: dict[str, Any]) -> TranslationRequest | TranslationResponse:
    """Parse model JSON as a data request or a translation response.

    Args:
        data: The decoded model output.

    Returns:
        A TranslationRequest when response_type is 'request', otherwise a TranslationResponse.
    """
    if ResponseType(data["response_type"]) is ResponseType.REQUEST:
        return TranslationRequest.from_dict(data)
    return TranslationResponse.from_dict(data)


def _node_schema(create_footnotes: bool) -> dict[str, Any]:
    node_types = [NodeType.TEXT.value, NodeType.TERM.value]
    properties: dict[str, Any] = {
        "text": {"type": "string"},
        "term_id": {"type": "string"},
        "alias_id": {"type": ["string", "null"]},
    }
    if create_footnotes:
        node_types.append(NodeType.FOOTNOTE.value)
        properties["content"] = {"type": "string"}
    properties["type"] = {"type": "string", "enum": node_types}
    return {"type": "object", "required": ["type"], "additionalProperties": False, "properties": properties}


def _glossary_update_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["op"],
        "additionalProperties": False,
        "properties": {
            "op": {"type": "string", "enum": [op.value for op in GlossaryOp]},
            "target_id": {"type": ["string", "null"]},
            "name": {"type": "string"},
            "original_name": {"type": "string"},
            "type": {"type": "string", "enum": [name_type.value for name_type in NameType]},
            "definition": {"type": "string"},
            "aliases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["original", "translation"],
                    "additionalProperties": False,
                    "properties": {
                        "original": {"type": "string"},
                        "translation": {"type": "string"},
                        "context": {"type": ["string", "null"]},
                    },
                },
            },
            "pronouns": {"type": "string"},
            "medium": {"type": "string"},
            "specified_type": {"type": "string"},
        },
        "allOf": [
            {
                "if": {"properties": {"op": {"const": "new"}}},
                "then": {
                    "required": ["name", "original_name", "type"],
                    "properties": {"name": {"minLength": 1}, "original_name": {"minLength": 1}},
                },
            },
            {
                "if": {"properties": {"op": {"const": "edit"}}},
                "then": {"required": ["target_id"], "properties": {"target_id": {"type": "string"}}},
            },
            {
                "if": {"properties": {"op": {"const": "delete"}}},
                "then": {"required": ["target_id"], "properties": {"target_id": {"type": "string"}}},
            },
        ],
    }


def _context_schema() -> dict[str, Any]:
    return {
        "type": ["object", "null"],
        "additionalProperties": False,
        "properties": {
            "summary": {"type": ["string", "null"]},
            "notes": {"type": ["array", "null"], "items": {"type": "string"}},
        },
    }


def _content_schema(create_footnotes: bool) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["nodes"],
        "additionalProperties": False,
        "properties": {"nodes": {"type": "array", "items": _node_schema(create_footnotes)}},
    }


def response_schema(settings: TranslationSettings) -> dict[str, Any]:
    """Build the JSON schema for a translation response from the settings.

    Args:
        settings: The settings that shape the schema.

    Returns:
        A JSON schema.
    """
    chapter_properties: dict[str, Any] = {
        "title": {"type": "string"},
        "content": _content_schema(settings.create_footnotes),
        "kind": {"type": "string", "enum": [kind.value for kind in ChapterKind]},
        "target_id": {"type": ["string", "null"]},
        "position": {"type": ["integer", "null"]},
    }
    chapter_required = ["title"]  # content/kind/summary are optional: an edit may omit them to keep the current values
    if settings.generate_chapter_summaries:
        chapter_properties["summary"] = {"type": "string"}

    return {
        "type": "object",
        "required": ["response_type", "chapters", "glossary_updates", "context"],
        "additionalProperties": False,
        "properties": {
            "response_type": {"type": "string", "enum": [ResponseType.TRANSLATION.value]},
            "chapters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": chapter_required,
                    "additionalProperties": False,
                    "properties": chapter_properties,
                },
            },
            "glossary_updates": {"type": "array", "items": _glossary_update_schema()},
            "context": _context_schema(),
            "more": {"type": "boolean"},
        },
    }


def analysis_schema(settings: TranslationSettings) -> dict[str, Any]:
    """Build the JSON schema for the first pass: glossary, context, and chapter metadata.

    Args:
        settings: The settings that shape the schema.

    Returns:
        A JSON schema.
    """
    properties: dict[str, Any] = {
        "response_type": {"type": "string", "enum": [ResponseType.ANALYSIS.value]},
        "kind": {"type": "string", "enum": [kind.value for kind in ChapterKind]},
        "glossary_updates": {"type": "array", "items": _glossary_update_schema()},
        "context": _context_schema(),
    }
    required = ["response_type", "kind", "glossary_updates", "context"]
    if settings.generate_chapter_summaries:
        properties["summary"] = {"type": "string"}
        required.append("summary")
    return {"type": "object", "required": required, "additionalProperties": False, "properties": properties}


def chapter_draft_schema(settings: TranslationSettings) -> dict[str, Any]:
    """Build the JSON schema for the second pass: the translated title and content.

    Args:
        settings: The settings that shape the schema.

    Returns:
        A JSON schema.
    """
    return {
        "type": "object",
        "required": ["response_type", "title", "content"],
        "additionalProperties": False,
        "properties": {
            "response_type": {"type": "string", "enum": [ResponseType.TRANSLATION.value]},
            "title": {"type": "string"},
            "content": _content_schema(settings.create_footnotes),
            "position": {"type": ["integer", "null"]},
        },
    }


def cleanup_schema(settings: TranslationSettings) -> dict[str, Any]:
    """Build the JSON schema for the third pass: the reviewed translation plus optional fixes.

    Args:
        settings: The settings that shape the schema.

    Returns:
        A JSON schema.
    """
    return {
        "type": "object",
        "required": ["response_type", "title", "content"],
        "additionalProperties": False,
        "properties": {
            "response_type": {"type": "string", "enum": [ResponseType.TRANSLATION.value]},
            "title": {"type": "string"},
            "content": _content_schema(settings.create_footnotes),
            "position": {"type": ["integer", "null"]},
            "glossary_updates": {"type": "array", "items": _glossary_update_schema()},
            "context": _context_schema(),
        },
    }


def output_schema(settings: TranslationSettings) -> dict[str, Any]:
    """Build the combined schema allowing exactly one of a request or a response.

    Args:
        settings: The settings that shape the response schema.

    Returns:
        A JSON schema with a top-level oneOf.
    """
    return {"oneOf": [request_schema(), response_schema(settings)]}


def analysis_output_schema(settings: TranslationSettings) -> dict[str, Any]:
    """Build the first-pass schema allowing either a data request or an analysis.

    Args:
        settings: The settings that shape the analysis schema.

    Returns:
        A JSON schema with a top-level oneOf.
    """
    return {"oneOf": [request_schema(), analysis_schema(settings)]}


def draft_output_schema(settings: TranslationSettings) -> dict[str, Any]:
    """Build the second-pass schema allowing either a data request or a chapter draft.

    Args:
        settings: The settings that shape the draft schema.

    Returns:
        A JSON schema with a top-level oneOf.
    """
    return {"oneOf": [request_schema(), chapter_draft_schema(settings)]}


def cleanup_output_schema(settings: TranslationSettings) -> dict[str, Any]:
    """Build the third-pass schema allowing either a data request or a chapter cleanup.

    Args:
        settings: The settings that shape the cleanup schema.

    Returns:
        A JSON schema with a top-level oneOf.
    """
    return {"oneOf": [request_schema(), cleanup_schema(settings)]}


def parse_analysis(data: dict[str, Any]) -> TranslationRequest | ChapterAnalysis:
    """Parse first-pass model JSON as a data request or a chapter analysis.

    Args:
        data: The decoded model output.

    Returns:
        A TranslationRequest when response_type is 'request', otherwise a ChapterAnalysis.
    """
    if ResponseType(data["response_type"]) is ResponseType.REQUEST:
        return TranslationRequest.from_dict(data)
    return ChapterAnalysis.from_dict(data)


def parse_draft(data: dict[str, Any]) -> TranslationRequest | ChapterDraft:
    """Parse second-pass model JSON as a data request or a chapter draft.

    Args:
        data: The decoded model output.

    Returns:
        A TranslationRequest when response_type is 'request', otherwise a ChapterDraft.
    """
    if ResponseType(data["response_type"]) is ResponseType.REQUEST:
        return TranslationRequest.from_dict(data)
    return ChapterDraft.from_dict(data)


def parse_cleanup(data: dict[str, Any]) -> TranslationRequest | ChapterCleanup:
    """Parse third-pass model JSON as a data request or a chapter cleanup.

    Args:
        data: The decoded model output.

    Returns:
        A TranslationRequest when response_type is 'request', otherwise a ChapterCleanup.
    """
    if ResponseType(data["response_type"]) is ResponseType.REQUEST:
        return TranslationRequest.from_dict(data)
    return ChapterCleanup.from_dict(data)
