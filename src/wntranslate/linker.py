from __future__ import annotations

import re

from .content import Content, Node, TermNode, TextNode
from .glossary import Glossary

# Only forms with an uppercase letter are linked, so all-lowercase common nouns are never wrapped.
_MIN_FORM_LENGTH = 2


def _surface_forms(glossary: Glossary) -> dict[str, tuple[str, str | None]]:
    """Map each unambiguous rendered surface form to the (term_id, alias_id) it should link to.

    A form shared by two different entries is dropped (ambiguous), so a name is never linked to the
    wrong entry. The entry name maps with alias_id None; each alias translation maps with its id.

    Args:
        glossary: The glossary to index.

    Returns:
        A form -> (term_id, alias_id) mapping with ambiguous forms removed.
    """
    forms: dict[str, tuple[str, str | None]] = {}
    ambiguous: set[str] = set()
    for entry in glossary.entries:
        candidates: list[tuple[str, tuple[str, str | None]]] = []
        if entry.name:
            candidates.append((entry.name, (entry.id, None)))
        for alias in entry.aliases:
            if alias.translation:
                candidates.append((alias.translation, (entry.id, alias.id)))
        for raw, ref in candidates:
            form = raw.strip()
            if len(form) < _MIN_FORM_LENGTH or not any(character.isupper() for character in form):
                continue
            if form in forms and forms[form] != ref:
                ambiguous.add(form)
            else:
                forms[form] = ref
    for form in ambiguous:
        forms.pop(form, None)
    return forms


def _build_pattern(forms: dict[str, tuple[str, str | None]]) -> re.Pattern[str] | None:
    """Compile a case-sensitive, word-bounded alternation of the forms, longest first.

    Longest-first ordering makes the regex prefer the longest form at a position (so "Super Chat"
    wins over "Chat"); the lookarounds keep a form from matching inside a larger word ("America" in
    "American").

    Args:
        forms: The surface-form index.

    Returns:
        The compiled pattern, or None when there is nothing to match.
    """
    if not forms:
        return None
    ordered = sorted(forms, key=len, reverse=True)
    alternation = "|".join(re.escape(form) for form in ordered)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alternation})(?![A-Za-z0-9])")


def _link_text(text: str, pattern: re.Pattern[str], forms: dict[str, tuple[str, str | None]]) -> list[Node]:
    out: list[Node] = []
    position = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        if start > position:
            out.append(TextNode(text[position:start]))
        term_id, alias_id = forms[match.group(0)]
        out.append(TermNode(term_id=term_id, alias_id=alias_id))
        position = end
    if not out:
        return [TextNode(text)]
    if position < len(text):
        out.append(TextNode(text[position:]))
    return out


def _merge_adjacent_text(nodes: list[Node]) -> list[Node]:
    merged: list[Node] = []
    for node in nodes:
        if isinstance(node, TextNode) and merged and isinstance(merged[-1], TextNode):
            merged[-1] = TextNode(merged[-1].text + node.text)
        else:
            merged.append(node)
    return merged


def link_content(content: Content, glossary: Glossary) -> Content:
    """Deterministically link glossary names in a chapter's text to term nodes.

    Scans every text node for known rendered glossary forms and splits each occurrence into a term
    node, leaving existing term and footnote nodes untouched. Matching is case-sensitive, bounded to
    whole words, longest-form-first, and skips forms shared by multiple entries, so a name is never
    linked to the wrong entry and ordinary words are left alone. Running it again is a no-op.

    Args:
        content: The content to link.
        glossary: The glossary whose names are linked.

    Returns:
        A new Content with glossary names linked, or the input unchanged when nothing matches.
    """
    forms = _surface_forms(glossary)
    pattern = _build_pattern(forms)
    if pattern is None:
        return content
    linked: list[Node] = []
    changed = False
    for node in content.nodes:
        if isinstance(node, TextNode) and node.text:
            pieces = _link_text(node.text, pattern, forms)
            if len(pieces) != 1 or not isinstance(pieces[0], TextNode):
                changed = True
            linked.extend(pieces)
        else:
            linked.append(node)
    if not changed:
        return content
    return Content(nodes=_merge_adjacent_text(linked))
