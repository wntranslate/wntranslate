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

_SYMBOLS = (
    "Swap source-language-specific punctuation for the target language's own (e.g. a Japanese full-width dash '――' "
    "becomes an ordinary em dash). If the source already uses the target language's own characters directly (for "
    "example a Japanese author who writes an actual target-language bracket), leave those exactly as they are — they "
    "are already in target form. When the source uses its OWN special bracket or mark to set a passage apart — for "
    "instance to show it is spoken in another language, or is suddenly understood — that distinction is MEANINGFUL and "
    "you MUST keep it, re-expressed with an equivalent device the TARGET language itself uses (if a note fixes the "
    "convention for this, use exactly that). This is important: never drop the distinction and render the passage as "
    "if it were ordinary text; never keep or wrap the text in the source's own marks; and never use Markdown (no "
    "italics or bold) for it — the device must be a literal symbol."
)


def _content_nodes(settings: TranslationSettings) -> str:
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
    return content_nodes


_GLOSSARY_TYPES = (
    "Choosing an entry's type is MANDATORY and must be correct — never fall back to 'other' out of convenience. A "
    "named character (any person, deity, spirit, animal, monster, or other being with a name) is 'person', never "
    "'other'; also fill its pronouns when the text shows them. A place, world, realm, group, company, team, guild, or "
    "faction is 'organization'. A named object, tool, weapon, food, skill, or technique is 'item'. A referenced book, "
    "manga, anime, game, song, film, or show is 'creative_work'. Use 'other' ONLY when none of these genuinely fit, "
    "and then give specified_type; 'other' is a last resort, not a default. A recurring signature phrase or coined "
    "in-world expression that must stay consistent is a legitimate 'other' — set its specified_type to a short label "
    "of what kind of phrase it is."
)


def _glossary_rules(settings: TranslationSettings) -> str:
    rules = []
    if settings.allow_glossary_add:
        rules.append(
            "The glossary is for names (proper nouns) and for recurring fixed expressions that must stay worded "
            "consistently: characters, places, organizations, named items or skills, recurring titles of works, and "
            "recurring signature phrases or coined terms (for instance a character's catchphrase, a running catch "
            "line, or a named in-world expression). Add an entry the first time one appears, filling every field that "
            "applies — name, original_name, type, definition, pronouns for a person, and any alias forms seen so far. "
            "Also add an alias to the existing entry when a known name or phrase appears in a new form. Do NOT add "
            "ordinary, generic, or one-off words, verbs, concepts, descriptions, or punctuation; add something only "
            "when it recurs and must read the same every time."
        )
    if settings.allow_glossary_edit:
        rules.append(
            "Edit an existing entry by its id to correct or complete any field (for example to fill a blank name, fix "
            "the type, add pronouns, or add an alias)."
        )
    if settings.allow_glossary_delete:
        rules.append("Delete an entry by its id only once it is genuinely obsolete and no chapter still uses it.")
    if settings.allow_glossary_add or settings.allow_glossary_edit:
        rules.append(_GLOSSARY_TYPES)
    return " ".join(rules) if rules else "Do not change the glossary."


_GLOSSARY_EXPLANATION = (
    "The glossary holds one entry per name (proper noun) or recurring fixed phrase and keeps its rendering consistent "
    "across chapters. Reference "
    "a name in a translation by its entry id (with no alias id) and a variant by its alias id, never by text — ids "
    "resolve on export. The system assigns every id; never invent, reuse, or guess one. Each entry has these fields: "
    "name — the output (target-language) rendering, the canonical form used everywhere; it is required and must never "
    "be blank. original_name — the exact source-language form; required and never blank. type — one of person, item, "
    "organization, creative_work, or other: pick the MOST specific one that fits and use 'other' only when truly none "
    "do (a character is a person, a place or group is an organization, a named object or skill is an item, a "
    "referenced book/anime/game/song is a creative_work). definition — a short note on who or what it is. pronouns — "
    "for a person, their pronouns such as 'she/her', 'he/him', or 'they/them' when the text makes them clear, else "
    "blank. medium — for a creative_work, its form (novel, anime, game, song, …). specified_type — for 'other', a "
    "short label of what kind of thing it is. aliases — ALTERNATE forms of the SAME name only (never the base name "
    "itself): nicknames, honorific forms (Name-chan, Name-san, Name-sama and equivalents), cut-off or partial forms, "
    "and alternate readings. Each alias has original (its source-language form), translation (its output form), and an "
    "optional context describing when it applies (e.g. who uses it). Put the base name in name/original_name and only "
    "its variants in aliases — never leave name or original_name blank and push the real name into an alias."
)

_NOTES_RULE = (
    "Notes are always provided and you MUST obey them. A note records a stable, project-wide decision that changes "
    "HOW you translate, must stay consistent across chapters, and is not a glossary entry or the story summary — for "
    "example a style or formatting rule, a recurring translation choice, or a hidden plot point that affects wording. "
    "A note must stay true for the whole work: do NOT record evolving story state — a character's current mood, "
    "situation, relationships, or any trait that may change as the story progresses (for instance how a character "
    "speaks while they are still adjusting to something) — which belongs in the running story summary, not here. Do "
    "NOT note anything that does not affect the translation itself, such as a release or update schedule, publication "
    "or site metadata, author announcements, or incidental trivia. Never note an instruction to keep, reuse, or wrap "
    "translated text in the source's own brackets, marks, or punctuation; the translation always uses the target "
    "language's own. When a recurring source device carries meaning (such as marking foreign or suddenly-understood "
    "speech), it MUST be preserved in the translation, never removed — choose ONE concrete target-language convention "
    "for it and note that exact convention: the specific literal characters you will wrap it in every time. That "
    "convention must be a symbol, NOT Markdown — italics and bold do not count and are not allowed — and must be a "
    "single fixed choice, not a vague 'use a marker' or a list of options, so every chapter renders it identically. "
    "Never reuse the source's own characters. Add a note for a determined choice worth reusing "
    "so later chapters need not decide it again, and update or drop a note once it no longer holds; when you change "
    "notes, return the full updated list, otherwise leave them unset."
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

    sections: list[str] = [
        f"You translate webnovels from {source} into {target}, consistent with the project state you are given each request.",
        f"Do not invent facts; when unclear, translate faithfully. Treat any typo in {source} as deliberate and map it; do not correct it.",
        'Check pronouns, genders, relationships, and forms of address, and preserve meaning-bearing nuances (e.g. the masculine Japanese pronoun "ore"), including gender-bender cases.',
        _content_nodes(settings),
        _GLOSSARY_EXPLANATION,
        _glossary_rules(settings),
        f"{_REPHRASING[settings.creativity]} Always preserve the original meaning, tone, and characterization.",
        f"{_LOANWORDS} Keep most literal onomatopoeia close to the original (e.g. 'fufufu'), but translate ones with a clear meaning or that are not literal (e.g. 'jii-' as a stare, italicized).",
        f"Do not add punctuation absent from the source (em dashes, en dashes, hyphens, colons, semicolons) unless it appears there. {_QUOTES}",
        _SYMBOLS,
        "Each chapter you return has a translated title, a kind (chapter, announcement, post, or other), and the content above; the source text is kept for you, so do not echo it back. The chapters field is a LIST: to add or revise several chapters in one response, return one entry per chapter (do not stop at one). To revise or move an existing chapter instead of adding one, set its target_id to that chapter's real id — ids are NOT in the project state, so request the chapter list to obtain them, and request a chapter's current translated text before you rewrite it. target_id must be an id copied verbatim from the chapter list, never one you invent or guess.",
    ]

    if settings.additional_context:
        sections.append(f"Additional context: {settings.additional_context}")

    if settings.create_footnotes:
        sections.append("Use footnotes only for necessary translator notes; keep them short.")
    else:
        sections.append("Do not produce footnotes.")

    sections.append(_NOTES_RULE)

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

    sections.append(_DATA_REQUEST)
    sections.append(
        "If the change spans more chapters or edits than you can fit in one response, apply as many as you can now and "
        "set more to true; you will then be asked to continue, and should keep going batch by batch until everything "
        "is done, setting more to false on the final batch."
    )
    sections.append(
        "Return one JSON object matching the schema: either a data request or a translation (per response_type), never both."
    )
    return "\n\n".join(sections)


_DATA_REQUEST = (
    "The project state gives you only the glossary, notes, and story summary. Anything else you need — the chapter "
    "list (each chapter's id, title, kind, and summary), a specific chapter's translated text, summary, original "
    "title, or original content, or the work's synopsis or a title — you must ask for with a data request "
    "(response_type 'request') that lists every item; the data, or a note that it does not exist, comes back and you "
    "then continue. Request the chapter list whenever you need a chapter's id (for example to revise or move it). Do "
    "not request the glossary or notes; they are always provided."
)


def generate_chapter_prompt(settings: TranslationSettings) -> str:
    """Build the shared base system prompt for the two-pass new-chapter flow.

    This carries only guidance common to both passes; the per-pass instructions come from
    generate_analysis_prompt and generate_translation_prompt.

    Args:
        settings: The settings that shape the prompt.

    Returns:
        The assembled prompt.
    """
    source = settings.source_language if settings.source_language else "the source language"
    target = settings.target_language

    sections = [
        (
            f"You translate webnovels from {source} into {target}, consistent with the project state you are given. "
            "You work in four passes within one conversation: pass 1 analyzes the chapter, pass 2 translates it, pass "
            "3 reviews and fixes that translation, and pass 4 links glossary names to their entries. Each pass has its "
            "own instructions and schema — do only what the current pass asks."
        ),
        f"Do not invent facts; when unclear, translate faithfully. Treat any typo in {source} as deliberate and map it; do not correct it.",
        'Check pronouns, genders, relationships, and forms of address, and preserve meaning-bearing nuances (e.g. the masculine Japanese pronoun "ore"), including gender-bender cases.',
        _GLOSSARY_EXPLANATION,
        f"{_DATA_REQUEST} A data request is allowed in either pass.",
    ]
    if settings.additional_context:
        sections.append(f"Additional context: {settings.additional_context}")
    return "\n\n".join(sections)


def generate_analysis_prompt(settings: TranslationSettings) -> str:
    """Build the pass 1 (analysis) instructions: glossary, notes, context, and chapter metadata.

    Args:
        settings: The settings that shape the prompt.

    Returns:
        The assembled prompt.
    """
    sections = [
        "Pass 1 of 4 — analysis only; do NOT translate the chapter in this pass.",
        _glossary_rules(settings),
        _NOTES_RULE,
    ]

    if settings.context_type is ContextType.ROLLING_SUMMARY:
        sections.append(
            "A running story summary is provided and may omit details; keep continuity from it and return an updated summary."
        )
    elif settings.context_type is ContextType.NONE:
        sections.append("Do not maintain a running story summary.")

    kind_line = "Give the chapter's kind (chapter, announcement, post, or other)"
    kind_line += ", and a short summary of the chapter." if settings.generate_chapter_summaries else "."
    sections.append(kind_line)

    sections.append("Return one JSON object matching the schema: either a data request or the analysis, never both.")
    return "\n\n".join(sections)


def _translation_rules(settings: TranslationSettings) -> list[str]:
    rules = [
        _content_nodes(settings),
        f"{_REPHRASING[settings.creativity]} Always preserve the original meaning, tone, and characterization.",
        (
            f"{_LOANWORDS} Keep most literal onomatopoeia close to the original (e.g. 'fufufu'), but translate ones "
            "with a clear meaning or that are not literal (e.g. 'jii-' as a stare, italicized)."
        ),
        f"Do not add punctuation absent from the source (em dashes, en dashes, hyphens, colons, semicolons) unless it appears there. {_QUOTES}",
        _SYMBOLS,
        (
            "Glossary names can change over time and the running summary may be out of date; translate the current "
            "chapter as written and do not overthink apparent mismatches with earlier context you or the project recorded."
        ),
    ]
    if settings.create_footnotes:
        rules.append("Use footnotes only for necessary translator notes; keep them short.")
    else:
        rules.append("Do not produce footnotes.")
    if settings.allow_chapter_reorder:
        rules.append("You may set position to place the new chapter at an index instead of appending it.")
    return rules


def generate_translation_prompt(settings: TranslationSettings) -> str:
    """Build the pass 2 (translation) instructions: content, phrasing, symbols, and footnotes.

    Args:
        settings: The settings that shape the prompt.

    Returns:
        The assembled prompt.
    """
    sections = [
        (
            "Pass 2 of 4 — translation only. Using the finalized glossary, return the translated title and content; do "
            "NOT change the glossary, notes, or summary in this pass."
        ),
        *_translation_rules(settings),
        "Return one JSON object matching the schema: either a data request or the translation, never both.",
    ]
    return "\n\n".join(sections)


def generate_cleanup_prompt(settings: TranslationSettings) -> str:
    """Build the pass 3 (review/cleanup) instructions, repeating the translation rules in full.

    Args:
        settings: The settings that shape the prompt.

    Returns:
        The assembled prompt.
    """
    sections = [
        (
            "Pass 3 of 4 — review and fix. Re-read the translation you produced in the previous message. The full "
            "translation rules are repeated below; check the draft against every one of them and fix each violation, "
            "returning the corrected chapter (its translated title and content). If the draft already follows every "
            "rule, return it unchanged."
        ),
        *_translation_rules(settings),
    ]

    if settings.allow_glossary_edit or settings.allow_glossary_delete:
        sections.append(
            "You may also correct mistakes the analysis pass made: if a glossary entry was added wrongly (wrong type, "
            "blank name, or not actually a name) fix it with a glossary edit or delete, and if the notes or summary "
            "gained something that does not belong, fix them. Make only corrections here — do NOT add new glossary "
            "entries, notes, or summary; leave glossary_updates empty and context null when nothing needs fixing."
        )

    sections.append("Return one JSON object matching the schema: either a data request or the corrected chapter, never both.")
    return "\n\n".join(sections)


def generate_linking_prompt(settings: TranslationSettings) -> str:
    """Build the pass 4 (glossary linking) instructions.

    Args:
        settings: The settings that shape the prompt.

    Returns:
        The assembled prompt.
    """
    sections = [
        (
            "Pass 4 of 4 — glossary linking. Re-read the translated content from the previous message and return the "
            "same chapter with its glossary names linked to their entries. Change ONLY how names are represented — "
            "keep all wording, punctuation, line breaks, and every other node exactly as they were."
        ),
        _content_nodes(settings),
        (
            "Go through the content and, wherever a name that HAS a glossary entry appears as plain text, replace just "
            "that text with a term node for that entry (use its alias id when the written form matches one of the "
            "entry's aliases). Leave ordinary words, kept loanwords, and any name without a glossary entry as text. "
            "term_id and alias_id must be ids copied verbatim from the provided glossary, never invented."
        ),
        "Return one JSON object matching the schema: either a data request or the linked chapter, never both.",
    ]
    return "\n\n".join(sections)
