from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._ids import new_id
from .content import Content
from .enums import ChapterKind
from .glossary import Glossary


@dataclass
class Chapter:
    """A unit of a webnovel such as a chapter, announcement, or post.

    Args:
        original_title: Title in the source language.
        original_content: Raw source text.
        title: Translated title.
        content: Translated content as resolvable nodes.
        kind: What kind of unit this is.
        summary: Optional short summary of the unit.
    """

    original_title: str
    original_content: str
    title: str = ""
    content: Content = field(default_factory=Content)
    kind: ChapterKind = ChapterKind.CHAPTER
    summary: str | None = None
    _id: str = field(default_factory=new_id, init=False, repr=False)

    @property
    def id(self) -> str:
        """Stable unique identifier, generated once and never settable."""
        return self._id

    def render(self, glossary: Glossary) -> str:
        """Resolve the content's references to a markdown string.

        Args:
            glossary: Glossary used to resolve term ids.

        Returns:
            The rendered markdown.
        """
        return self.content.render(glossary)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict including its id.

        Returns:
            A JSON-compatible representation.
        """
        return {
            "id": self._id,
            "original_title": self.original_title,
            "original_content": self.original_content,
            "title": self.title,
            "kind": self.kind.value,
            "summary": self.summary,
            "content": self.content.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chapter:
        """Rebuild a chapter, preserving its id.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed chapter.
        """
        chapter = cls(
            original_title=data.get("original_title", ""),
            original_content=data.get("original_content", ""),
            title=data.get("title", ""),
            content=Content.from_dict(data.get("content", {})),
            kind=ChapterKind(data.get("kind", ChapterKind.CHAPTER.value)),
            summary=data.get("summary"),
        )
        if "id" in data:
            chapter._id = data["id"]
        return chapter
