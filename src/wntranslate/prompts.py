from __future__ import annotations

from typing import TYPE_CHECKING

from .enums import ContextType, Creativity

if TYPE_CHECKING:
    from .settings import TranslationSettings

_REPHRASING = {
    Creativity.VERY_LOW: "Translate as literally as fidelity allows; do not rephrase beyond what the target language requires.",
    Creativity.LOW: "You may rephrase lightly for readability, but stay close to the source.",
    Creativity.MEDIUM: "Rephrase as needed for natural flow in the target language while still preserving relevant things (such as onomatopoeia or language nuances).",
    Creativity.HIGH: "Adapt phrasing and idiom freely for a natural, fluent read while still preserving relevant things (such as onomatopoeia or language nuances).",
    Creativity.VERY_HIGH: "Freely polish and localize the wording for the most natural result while still preserving relevant things (such as onomatopoeia or language nuances).",
}

_LOANWORDS = (
    "When a source term is one the work's target-language readership conventionally leaves untranslated "
    "(loanwords, subculture or genre vocabulary, and honorific suffixes on names such as 'Name-chan' or "
    "equivalents in other languages), keep its established form instead of translating it to a plainer equivalent."
)

_QUOTES = (
    "Use straight quotes and apostrophes (\" and '), not curly or directional ones, unless the exact character "
    "matters to the story (e.g. the cause of a bug in a coding scene)."
)


def generate_metadata_prompt(settings: TranslationSettings) -> str:
    """Build the system prompt for translating the title and synopsis.

    Args:
        settings: The settings that shape the prompt.

    Returns:
        The assembled prompt.
    """
    source = settings.source_language if settings.source_language else "the source language"
    target = settings.target_language
    parts = [
        f"You translate webnovels from {source} into {target}. Translate the title and synopsis, consistent with the glossary.",
        f"{_REPHRASING[settings.creativity]} Always preserve the original meaning and tone.",
        f"Leave glossary-specified terms and names untranslated. {_LOANWORDS}",
        _QUOTES,
    ]
    if settings.additional_context:
        parts.append(f"Additional context: {settings.additional_context}")
    parts.append("Return one JSON object with the translated title and synopsis, matching the schema.")
    return "\n\n".join(parts)


def generate_prompt(settings: TranslationSettings) -> str:
    """Build the translator system prompt from the given settings.

    Args:
        settings: The settings that shape the prompt.

    Returns:
        The assembled prompt.
    """
    source = settings.source_language if settings.source_language else "the source language"
    target = settings.target_language

    content_nodes = (
        "Put the full translation in the chapter's content as an ordered list of nodes. Write all prose in text nodes "
        '({"type": "text", "text": "..."}, Markdown allowed), including ordinary words, kept loanwords, and the names '
        "of terms not yet in the glossary. "
        'Use a term node ({"type": "term", "term_id": "<id>", "alias_id": null}) only to reference an entry that '
        "already exists in the provided glossary, by its exact id (plus an alias id when one applies); this keeps that "
        "name consistent if it is later renamed. term_id must be an id taken verbatim from the provided glossary — "
        "never a plain word, an id you made up, or an entry you are adding this response. A word is not a term node "
        "just because it is notable or foreign"
    )
    if settings.create_footnotes:
        content_nodes += '. A footnote node ({"type": "footnote", "content": "..."}) holds a short translator note'
    content_nodes += (
        ". Nodes are concatenated with nothing inserted between them, so put every line break inside the text nodes "
        "yourself (a blank line between paragraphs, as Markdown); do not rely on node boundaries for spacing. "
        "content must never be empty."
    )

    glossary_rules = []
    if settings.allow_glossary_add:
        glossary_rules.append(
            "The glossary is ONLY for names: proper nouns such as characters, places, organizations, and other "
            "uniquely named entities (including named items, skills, or titles of works that recur). You MUST add an "
            "entry for each such name the first time it appears — give its original_name, output name, type, and any "
            "aliases; the system assigns the id. Also add an alias when an existing name appears in a new form. Do NOT "
            "add common or generic words, verbs, concepts, descriptions, or punctuation; if it is not a proper name, "
            "it is not a glossary entry."
        )
    if settings.allow_glossary_edit:
        glossary_rules.append("Edit an entry by its id to correct or improve its output name, definition, or aliases.")
    if settings.allow_glossary_delete:
        glossary_rules.append("Delete an entry by its id once it is obsolete.")
    glossary_updates = " ".join(glossary_rules) if glossary_rules else "Do not change the glossary."

    sections: list[str] = [
        f"You translate webnovels from {source} into {target}, consistent with the project state you are given each request.",
        f"Do not invent facts; when unclear, translate faithfully. Treat any typo in {source} as deliberate and map it; do not correct it.",
        'Check pronouns, genders, relationships, and forms of address, and preserve meaning-bearing nuances (e.g. the masculine Japanese pronoun "ore"), including gender-bender cases.',
        content_nodes,
        "The glossary keeps names consistent across chapters and holds their output text. Reference a matched name or alias in your translation by its id, never its text (ids resolve on export); when an alias has a context, use the one that fits (e.g. the speaker). Each entry has an id, a source-language original_name, an output name, a type, and aliases; an alias is an alternate form of the same name (a nickname, an honorific or partial form such as a cut-off name, or a different reading) with its own original form, translation, and optional context.",
        glossary_updates,
        f"{_REPHRASING[settings.creativity]} Always preserve the original meaning, tone, and characterization.",
        f"{_LOANWORDS} Keep most literal onomatopoeia close to the original (e.g. 'fufufu'), but translate ones with a clear meaning or that are not literal (e.g. 'jii-' as a stare, italicized).",
        f"Do not add punctuation absent from the source (em dashes, en dashes, hyphens, colons, semicolons) unless it appears there. {_QUOTES}",
        "Each chapter you return has a translated title, a kind (chapter, announcement, post, or other), the content above, and its original_title and original_content copied verbatim. To revise an already-translated chapter instead of adding one, set its target_id to that chapter's id.",
    ]

    if settings.additional_context:
        sections.append(f"Additional context: {settings.additional_context}")

    if settings.create_footnotes:
        sections.append("Use footnotes only for necessary translator notes; keep them short.")
    else:
        sections.append("Do not produce footnotes.")

    sections.append(
        "Notes are always provided and you MUST obey them. Record a note for any project-wide fact that must stay "
        "perfectly consistent and is not a glossary entry or the story summary (e.g. a style rule or hidden plot point); "
        "when you change notes, return the full updated list, otherwise leave them unset. If you do something that you think should be saved in future new chats (every chapter is a new chat), add it to notes. For example, determined choices that can make future chats FASTER since it doesn't need to make that choice again."
    )

    if settings.context_type is ContextType.ROLLING_SUMMARY:
        sections.append(
            "A running story summary is provided and may omit details; keep continuity from it and return an updated summary."
        )
    elif settings.context_type is ContextType.NONE:
        sections.append("Do not maintain a running story summary.")

    if settings.generate_chapter_summaries:
        sections.append("Also return a short summary of each translated chapter.")
    else:
        sections.append("Do not produce chapter summaries.")

    if settings.allow_chapter_reorder:
        sections.append("You may set a chapter's position to place a new chapter or move an existing one by id.")

    sections.append(
        "Do not request the glossary or notes; they are always provided. If you need other data (the synopsis, a title, "
        "the chapter list, or another chapter's text, summary, original title, or original content), return a data "
        "request listing every item; it may not exist."
    )
    sections.append(
        "Return one JSON object matching the schema: either a data request or a translation (per response_type), never both."
    )
    return "\n\n".join(sections)
