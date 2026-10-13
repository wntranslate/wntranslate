from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .enums import ContextType, Creativity
from .prompts import generate_prompt


@dataclass
class TranslationSettings:
    """Settings controlling how a webnovel is translated.

    Args:
        source_language: Language translated from, or None if unspecified.
        target_language: Language translated into.
        create_footnotes: Whether translator footnotes are produced.
        generate_chapter_summaries: Whether a summary is generated per chapter.
        allow_glossary_add: Whether the model may add glossary entries.
        allow_glossary_edit: Whether the model may edit glossary entries.
        allow_glossary_delete: Whether the model may delete glossary entries.
        glossary_add_limit: Maximum glossary additions per response.
        glossary_edit_limit: Maximum glossary edits per response.
        glossary_delete_limit: Maximum glossary deletions per response.
        allow_chapter_reorder: Whether the model may reorder chapters.
        context_type: Strategy used to maintain the running context.
        creativity: How freely the translation may rephrase the source.
        additional_context: Any extra context for the model (the work, file format, anything), or None.
        prompt: Override for the generated prompt; editing is not recommended.
    """

    source_language: str | None = None
    target_language: str = "English"
    create_footnotes: bool = True
    generate_chapter_summaries: bool = True
    allow_glossary_add: bool = True
    allow_glossary_edit: bool = True
    allow_glossary_delete: bool = True
    glossary_add_limit: int = 100
    glossary_edit_limit: int = 100
    glossary_delete_limit: int = 5
    allow_chapter_reorder: bool = True
    context_type: ContextType = ContextType.ROLLING_SUMMARY
    creativity: Creativity = Creativity.LOW
    additional_context: str | None = None
    prompt: str | None = None

    def build_prompt(self) -> str:
        """Return the override prompt, or one generated from these settings.

        Returns:
            The effective prompt.
        """
        return self.prompt if self.prompt is not None else generate_prompt(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "create_footnotes": self.create_footnotes,
            "generate_chapter_summaries": self.generate_chapter_summaries,
            "allow_glossary_add": self.allow_glossary_add,
            "allow_glossary_edit": self.allow_glossary_edit,
            "allow_glossary_delete": self.allow_glossary_delete,
            "glossary_add_limit": self.glossary_add_limit,
            "glossary_edit_limit": self.glossary_edit_limit,
            "glossary_delete_limit": self.glossary_delete_limit,
            "allow_chapter_reorder": self.allow_chapter_reorder,
            "context_type": int(self.context_type),
            "creativity": int(self.creativity),
            "additional_context": self.additional_context,
            "prompt": self.prompt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslationSettings:
        """Rebuild settings from their dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed settings.
        """
        return cls(
            source_language=data.get("source_language"),
            target_language=data.get("target_language", "English"),
            create_footnotes=data.get("create_footnotes", True),
            generate_chapter_summaries=data.get("generate_chapter_summaries", True),
            allow_glossary_add=data.get("allow_glossary_add", True),
            allow_glossary_edit=data.get("allow_glossary_edit", True),
            allow_glossary_delete=data.get("allow_glossary_delete", True),
            glossary_add_limit=data.get("glossary_add_limit", 100),
            glossary_edit_limit=data.get("glossary_edit_limit", 100),
            glossary_delete_limit=data.get("glossary_delete_limit", 5),
            allow_chapter_reorder=data.get("allow_chapter_reorder", True),
            context_type=ContextType(data.get("context_type", ContextType.ROLLING_SUMMARY)),
            creativity=Creativity(data.get("creativity", Creativity.LOW)),
            additional_context=data.get("additional_context"),
            prompt=data.get("prompt"),
        )
