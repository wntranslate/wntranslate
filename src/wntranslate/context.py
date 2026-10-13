from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Context:
    """Running translation context carried across chapters.

    Args:
        notes: Freeform notes.
        summary: The overall/rolling context text.
    """

    notes: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {"notes": list(self.notes), "summary": self.summary}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Context:
        """Rebuild a context from its dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed context.
        """
        return cls(notes=list(data.get("notes", [])), summary=data.get("summary", ""))
