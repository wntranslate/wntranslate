from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from .enums import NodeType
from .glossary import Glossary


@dataclass
class TextNode:
    """Literal markdown text.

    Args:
        text: The markdown text.
    """

    text: str = ""

    type: ClassVar[NodeType] = NodeType.TEXT

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {"type": self.type.value, "text": self.text}


@dataclass
class TermNode:
    """A reference to a glossary term or alias, resolved on export.

    Args:
        term_id: The glossary entry id.
        alias_id: Optional alias id whose translation is used instead of the name.
    """

    term_id: str
    alias_id: str | None = None

    type: ClassVar[NodeType] = NodeType.TERM

    def resolve(self, glossary: Glossary) -> str:
        """Resolve the reference to its output text.

        Args:
            glossary: Glossary used to resolve the id.

        Returns:
            The resolved text.
        """
        entry = glossary.get(self.term_id)
        if entry is None:
            raise KeyError(self.term_id)
        if self.alias_id is None:
            return entry.name
        alias = entry.get_alias(self.alias_id)
        if alias is None:
            raise KeyError(self.alias_id)
        return alias.translation

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {"type": self.type.value, "term_id": self.term_id, "alias_id": self.alias_id}


@dataclass
class FootnoteNode:
    """A translator footnote whose body is markdown.

    Args:
        content: The footnote body as markdown.
    """

    content: str = ""

    type: ClassVar[NodeType] = NodeType.FOOTNOTE

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {"type": self.type.value, "content": self.content}


Node = TextNode | TermNode | FootnoteNode


def node_from_dict(data: dict[str, Any]) -> Node:
    """Rebuild a node from its dict representation.

    Args:
        data: A dict produced by a node's to_dict.

    Returns:
        The reconstructed node.
    """
    node_type = NodeType(data["type"])
    if node_type is NodeType.TEXT:
        return TextNode(text=data.get("text", ""))
    if node_type is NodeType.TERM:
        return TermNode(term_id=data["term_id"], alias_id=data.get("alias_id"))
    return FootnoteNode(content=data.get("content", ""))


@dataclass
class Content:
    """Ordered content nodes for a chapter.

    Args:
        nodes: The ordered nodes.
    """

    nodes: list[Node] = field(default_factory=list)

    def render(self, glossary: Glossary) -> str:
        """Resolve terms and footnotes to a markdown string.

        Args:
            glossary: Glossary used to resolve term ids.

        Returns:
            The rendered markdown.
        """
        parts: list[str] = []
        footnotes: list[str] = []
        for node in self.nodes:
            if isinstance(node, TextNode):
                parts.append(node.text)
            elif isinstance(node, TermNode):
                parts.append(node.resolve(glossary))
            else:
                footnotes.append(node.content)
                parts.append(f"[^{len(footnotes)}]")
        body = "".join(parts)
        if not footnotes:
            return body
        definitions = "\n".join(f"[^{index}]: {text}" for index, text in enumerate(footnotes, start=1))
        return f"{body}\n\n{definitions}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Returns:
            A JSON-compatible representation.
        """
        return {"nodes": [node.to_dict() for node in self.nodes]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Content:
        """Rebuild content from its dict representation.

        Args:
            data: A dict produced by to_dict.

        Returns:
            The reconstructed content.
        """
        return cls(nodes=[node_from_dict(node) for node in data.get("nodes", [])])
