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
        return cls(
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


@dataclass
class ChapterTranslation:
    """A translated chapter payload returned by the model.

    Args:
        original_title: Title in the source language.
        original_content: Raw source text.
        title: Translated title.
        content: Translated content nodes.
        kind: Kind of unit.
        summary: Optional short summary of the chapter.
        target_id: Existing chapter id to replace, or None to append.
        position: Index to place the chapter at, or None to keep or append.
    """

    original_title: str = ""
    original_content: str = ""
    title: str = ""
    content: Content = field(default_factory=Content)
    kind: ChapterKind = ChapterKind.CHAPTER
    summary: str | None = None
    target_id: str | None = None
    position: int | None = None

    def to_chapter(self) -> Chapter:
        """Build a chapter from this payload.

        Returns:
            The new chapter.
        """
        return Chapter(
            original_title=self.original_title,
            original_content=self.original_content,
            title=self.title,
            content=self.content,
            kind=self.kind,
            summary=self.summary,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {
            "original_title": self.original_title,
            "original_content": self.original_content,
            "title": self.title,
            "content": self.content.to_dict(),
            "kind": self.kind.value,
            "summary": self.summary,
            "target_id": self.target_id,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChapterTranslation:
        """Rebuild from a dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed payload.
        """
        return cls(
            original_title=data.get("original_title", ""),
            original_content=data.get("original_content", ""),
            title=data.get("title", ""),
            content=Content.from_dict(data.get("content", {})),
            kind=ChapterKind(data.get("kind", ChapterKind.CHAPTER.value)),
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
    """

    chapters: list[ChapterTranslation] = field(default_factory=list)
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
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "glossary_updates": [update.to_dict() for update in self.glossary_updates],
            "context": self.context.to_dict() if self.context is not None else None,
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
            {"if": {"properties": {"op": {"const": "new"}}}, "then": {"required": ["name", "original_name", "type"]}},
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


def response_schema(settings: TranslationSettings) -> dict[str, Any]:
    """Build the JSON schema for a translation response from the settings.

    Args:
        settings: The settings that shape the schema.

    Returns:
        A JSON schema.
    """
    chapter_properties: dict[str, Any] = {
        "original_title": {"type": "string"},
        "original_content": {"type": "string"},
        "title": {"type": "string"},
        "content": {
            "type": "object",
            "required": ["nodes"],
            "additionalProperties": False,
            "properties": {"nodes": {"type": "array", "items": _node_schema(settings.create_footnotes)}},
        },
        "kind": {"type": "string", "enum": [kind.value for kind in ChapterKind]},
        "target_id": {"type": ["string", "null"]},
        "position": {"type": ["integer", "null"]},
    }
    chapter_required = ["original_title", "original_content", "title", "content"]
    if settings.generate_chapter_summaries:
        chapter_properties["summary"] = {"type": "string"}
        chapter_required.append("summary")

    context_schema: dict[str, Any] = {
        "type": ["object", "null"],
        "additionalProperties": False,
        "properties": {
            "summary": {"type": ["string", "null"]},
            "notes": {"type": ["array", "null"], "items": {"type": "string"}},
        },
    }

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
            "context": context_schema,
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
