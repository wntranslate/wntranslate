from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from .enums import RequestKind, ResponseType


@dataclass
class DataRequest:
    """A single piece of data the model asks for before translating.

    Args:
        kind: What kind of data is requested.
        chapter_id: Target chapter id, when the request concerns one chapter.
        start: Start index for a partial chapter list, or None.
        end: End index (exclusive) for a partial chapter list, or None.
        detail: Optional free-form description of the request.
    """

    kind: RequestKind
    chapter_id: str | None = None
    start: int | None = None
    end: int | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {
            "kind": self.kind.value,
            "chapter_id": self.chapter_id,
            "start": self.start,
            "end": self.end,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataRequest:
        """Rebuild from a dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed request.
        """
        return cls(
            kind=RequestKind(data["kind"]),
            chapter_id=data.get("chapter_id"),
            start=data.get("start"),
            end=data.get("end"),
            detail=data.get("detail"),
        )


@dataclass
class DataResult:
    """The resolution of a single data request against a project.

    Args:
        request: The original request.
        found: Whether the requested target existed.
        value: The resolved data, if any.
    """

    request: DataRequest
    found: bool
    value: str | list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {"request": self.request.to_dict(), "found": self.found, "value": self.value}


@dataclass
class TranslationRequest:
    """Data the model requests before it can translate.

    Args:
        requests: The individual data requests.
        reason: Optional explanation of why the data is needed.
    """

    requests: list[DataRequest] = field(default_factory=list)
    reason: str | None = None

    response_type: ClassVar[ResponseType] = ResponseType.REQUEST

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {
            "response_type": self.response_type.value,
            "requests": [request.to_dict() for request in self.requests],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslationRequest:
        """Rebuild from a dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed request.
        """
        return cls(
            requests=[DataRequest.from_dict(request) for request in data.get("requests", [])],
            reason=data.get("reason"),
        )


def request_schema() -> dict[str, Any]:
    """Build the JSON schema for a data request.

    Returns:
        A JSON schema.
    """
    return {
        "type": "object",
        "required": ["response_type", "requests"],
        "additionalProperties": False,
        "properties": {
            "response_type": {"type": "string", "enum": [ResponseType.REQUEST.value]},
            "requests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["kind"],
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "enum": [kind.value for kind in RequestKind]},
                        "chapter_id": {"type": ["string", "null"]},
                        "start": {"type": ["integer", "null"]},
                        "end": {"type": ["integer", "null"]},
                        "detail": {"type": ["string", "null"]},
                    },
                },
            },
            "reason": {"type": ["string", "null"]},
        },
    }
