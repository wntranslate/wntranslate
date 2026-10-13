from __future__ import annotations

from typing import TYPE_CHECKING

from .enums import ContextType, Creativity

if TYPE_CHECKING:
    from .settings import TranslationSettings

_REPHRASING = {
    Creativity.VERY_LOW: "Translate as literally as fidelity allows; rephrase only where the target language requires.",
    Creativity.LOW: "Rephrase lightly for readability, but stay close to the source.",
    Creativity.MEDIUM: "Rephrase for natural flow in the target language, preserving nuances like onomatopoeia.",
    Creativity.HIGH: "Adapt phrasing and idiom freely for a fluent read, preserving nuances like onomatopoeia.",
    Creativity.VERY_HIGH: "Freely polish and localize the wording for the most natural result, preserving nuances like onomatopoeia.",
}

_LOANWORDS = (
    "Keep a source term the target-language readership conventionally leaves untranslated (loanwords, subculture or "
    "genre vocabulary, honorific suffixes like 'Name-chan') in its established form instead of flattening it."
)

_QUOTES = (
    "Use straight quotes and apostrophes (\" and '), not curly ones, unless the exact character matters to the plot "
    "(e.g. the cause of a bug in a coding scene)."
)

_SYMBOLS = (
    "Render source-language-specific punctuation and symbols as the target language's own (e.g. a Japanese full-width "
    "dash '――' becomes an ordinary em dash); if the author already wrote a target-language character, leave it. When "
    "the source sets a passage apart with its OWN marks (e.g. foreign or suddenly-understood speech), that distinction "
    "is meaningful and you MUST keep it, re-expressed with a device the target language uses (or the exact convention "
    "a note fixes). Never drop it, never keep or wrap text in the source's marks, and never use Markdown (italics or "
    "bold) for it — the device must be a literal symbol."
)

_PUNCTUATION = (
    "Do not add punctuation absent from the source (em dashes, en dashes, hyphens, colons, semicolons) unless it "
    f"appears there. {_QUOTES}"
)

_ONOMATOPOEIA = (
    f"{_LOANWORDS} Keep literal onomatopoeia close to the original (e.g. 'fufufu'), but translate ones with a clear "
    "meaning or that are not literal (e.g. 'jii-' as a stare, italicized). When the source stretches a sound by "
    "repeating one character (a long-vowel mark like 'ー', or a repeated letter), render it as an elongation of "
    "bounded length (a scream or stretched vowel, not thousands of letters) and then move on; never emit an extreme "
    "or runaway run of a single character."
)

_DATA_REQUEST = (
    "You are given the glossary, notes, running summary, the work's title and synopsis, and a list of the most recent "
    "chapters (each one's id, title, original title, and kind). For anything else — an older chapter's entry or the "
    "full chapter list, a chapter's translated text, its summary, or its original title or content — return a data "
    "request "
    "(response_type 'request') listing every item; the data, or a note that it does not exist, comes back and you "
    "continue. Do not request the glossary, notes, title, or synopsis — they are always provided."
)

_GLOSSARY_EXPLANATION = (
    "The glossary keeps a name's rendering consistent across chapters: reference a name by its entry id (no alias id) "
    "and a variant by its alias id, never by text (ids resolve on export). The system assigns ids — never invent or "
    "reuse one. Entry fields: name (target-language rendering, required, never blank); original_name (exact source "
    "form, required, never blank); type (person/item/organization/creative_work/other); definition (short); pronouns "
    "(for a person, when the text shows them); medium (for a creative_work); specified_type (for other). aliases are "
    "ALTERNATE forms of the same name only (nicknames, honorifics like Name-chan/-san/-sama, cut-off forms, alternate "
    "readings), each with original (source form), translation (output form), and optional context. Put the base name "
    "in name/original_name and only its variants in aliases — never blank the name and hide it in an alias."
)

_GLOSSARY_TYPES = (
    "Set each entry's type correctly; never default to 'other'. A named character or being is person (fill pronouns "
    "when the text shows them). A place, world, group, company, team, or faction is organization. A named object, "
    "tool, weapon, food, skill, or technique is item. A referenced book, manga, anime, game, song, or film is "
    "creative_work. Use 'other' only when none fit, with a short specified_type; a recurring coined phrase or "
    "catchphrase is a valid 'other'."
)

_NOTES_RULE = (
    "Notes are always provided and you MUST obey them. A note is a stable, work-wide decision that changes HOW you "
    "translate and is not a glossary entry or the story summary — a style or formatting rule, a recurring translation "
    "choice, or a wording-affecting hidden plot point. Keep notes true for the whole work: do NOT note evolving story "
    "state (moods, situations, relationships, or a trait that changes as the story goes — e.g. how someone speaks "
    "while still adjusting); that belongs in the running summary. Do NOT note anything that does not affect the "
    "translation (release schedules, site metadata, author announcements, trivia). Never note keeping, reusing, or "
    "wrapping text in the source's own brackets or marks; the translation uses the target's own. When a recurring "
    "source device carries meaning (e.g. marking foreign or suddenly-understood speech) you MUST preserve it, never "
    "remove it: pick ONE concrete target convention and note the exact literal characters you will use every time (a "
    "symbol, not Markdown; not vague, not a list of options). Add a note for a reusable decision, update or drop it "
    "once it no longer holds, and return the full updated list whenever you change notes (otherwise leave unset)."
)

_COMPLETENESS = (
    "Fill every required field. ALWAYS translate the chapter's title into the target language and return it in the "
    "title field — never leave the title in the source language, blank, missing, or just copied from the source. The "
    "content must hold the whole chapter from first line to last."
)


def _content_nodes(settings: TranslationSettings) -> str:
    text = (
        "Put the full translation in content as an ordered list of nodes. Write prose in text nodes "
        '({"type": "text", "text": "..."}, Markdown allowed), including ordinary words, kept loanwords, and names not '
        "yet in the glossary. "
        'A term node ({"type": "term", "term_id": "<id>", "alias_id": null}) references an existing glossary entry by '
        "its exact id so the name stays consistent if later renamed. term_id must be a real id from the glossary "
        "(never a plain word, an id you made up, or an entry you are adding now). alias_id is null unless you use one "
        "of THAT entry's own aliases (an alias listed under that term_id) — never an alias id from a different entry. "
        "A word is not a term node just because it is notable or foreign"
    )
    if settings.create_footnotes:
        text += '. A footnote node ({"type": "footnote", "content": "..."}) holds a short translator note'
    text += (
        ". Nodes are concatenated with nothing between them, so put line breaks inside the text nodes (a blank line "
        "between paragraphs); do not rely on node boundaries for spacing. content must never be empty."
    )
    return text


def _glossary_rules(settings: TranslationSettings) -> str:
    rules = []
    if settings.allow_glossary_add:
        rules.append(
            "The glossary holds names (proper nouns) and recurring fixed expressions that must read the same every "
            "time: characters, places, organizations, named items or skills, recurring work titles, and "
            "catchphrases or coined terms. Add an entry the first time one appears, filling name, original_name, "
            "type, definition, pronouns (for a person), and any alias forms seen so far. Add an alias to the existing "
            "entry when a known name or phrase appears in a new form. Do NOT add ordinary, generic, or one-off words, "
            "verbs, concepts, or descriptions."
        )
    if settings.allow_glossary_edit:
        rules.append("Edit an entry by its id to fix or complete any field (a blank name, wrong type, pronouns, an alias).")
    if settings.allow_glossary_delete:
        rules.append("Delete an entry by its id only once it is obsolete and no chapter still uses it.")
    if settings.allow_glossary_add or settings.allow_glossary_edit:
        rules.append(_GLOSSARY_TYPES)
    return " ".join(rules) if rules else "Do not change the glossary."


def _summary_rule(settings: TranslationSettings) -> str | None:
    if settings.context_type is ContextType.ROLLING_SUMMARY:
        return (
            "A running story summary is provided and may omit details. Keep continuity from it and return an updated "
            "summary, but keep it COMPRESSED: rewrite and condense rather than appending — merge and generalize older "
            "events, and drop stale, resolved, or fine-grained details no longer needed to translate later chapters. "
            "It may grow as the story does, but prune so it stays a concise digest, never an ever-growing log."
        )
    if settings.context_type is ContextType.NONE:
        return "Do not maintain a running story summary."
    return None


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
        f"You translate webnovels from {source} into {target}, consistent with the project state given each request.",
        f"Do not invent facts; when unclear, translate faithfully. Treat any typo in {source} as deliberate and map it; do not correct it.",
        'Check pronouns, genders, relationships, and forms of address, and keep meaning-bearing nuances (e.g. the masculine Japanese pronoun "ore"), including gender-bender cases.',
        _content_nodes(settings),
        _GLOSSARY_EXPLANATION,
        _glossary_rules(settings),
        f"{_REPHRASING[settings.creativity]} Always preserve the original meaning, tone, and characterization.",
        _ONOMATOPOEIA,
        _PUNCTUATION,
        _SYMBOLS,
        (
            "Each returned chapter has a translated title, a kind (chapter, announcement, post, or other), and the "
            "content above; the source text is kept for you, so do not echo it back. chapters is a LIST: to add or "
            "revise several chapters at once, return one entry each (do not stop at one). To revise or move an "
            "existing chapter, set target_id to its id (from the recent-chapters list, or request the chapter list "
            "for an older one) and request its current translated text first; never invent an id."
        ),
    ]

    if settings.additional_context:
        sections.append(f"Additional context: {settings.additional_context}")

    sections.append(
        "Use footnotes only for necessary translator notes; keep them short."
        if settings.create_footnotes
        else "Do not produce footnotes."
    )
    sections.append(_NOTES_RULE)
    summary = _summary_rule(settings)
    if summary is not None:
        sections.append(summary)
    sections.append(
        "Also return a short summary of each translated chapter."
        if settings.generate_chapter_summaries
        else "Do not produce chapter summaries."
    )
    if settings.allow_chapter_reorder:
        sections.append("You may set a chapter's position to place a new chapter or move an existing one by id.")

    sections.append(_DATA_REQUEST)
    sections.append(
        "If the change spans more chapters or edits than fit in one response, apply as many as you can and set more to "
        "true; you will be asked to continue, batch by batch, until done — set more to false on the final batch."
    )
    sections.append("Return one JSON object matching the schema: a data request or a translation (per response_type), never both.")
    return "\n\n".join(sections)


def generate_chapter_prompt(settings: TranslationSettings) -> str:
    """Build the shared base system prompt for the four-pass new-chapter flow.

    This carries only guidance common to every pass; per-pass instructions come from
    generate_analysis_prompt, generate_translation_prompt, generate_cleanup_prompt, and
    generate_linking_prompt.

    Args:
        settings: The settings that shape the prompt.

    Returns:
        The assembled prompt.
    """
    source = settings.source_language if settings.source_language else "the source language"
    target = settings.target_language

    sections = [
        (
            f"You translate webnovels from {source} into {target}, consistent with the project state given you. You "
            "work in four passes in one conversation: pass 1 analyzes the chapter, pass 2 translates it, pass 3 "
            "reviews and fixes that translation, pass 4 links glossary names to their entries. Each pass has its own "
            "instructions and schema — do only what the current pass asks."
        ),
        f"Do not invent facts; when unclear, translate faithfully. Treat any typo in {source} as deliberate and map it; do not correct it.",
        'Check pronouns, genders, relationships, and forms of address, and keep meaning-bearing nuances (e.g. the masculine Japanese pronoun "ore"), including gender-bender cases.',
        _GLOSSARY_EXPLANATION,
        f"{_DATA_REQUEST} A data request is allowed in any pass.",
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
    summary = _summary_rule(settings)
    if summary is not None:
        sections.append(summary)
    kind_line = "Give the chapter's kind (chapter, announcement, post, or other)"
    kind_line += ", and a short summary of the chapter." if settings.generate_chapter_summaries else "."
    sections.append(kind_line)
    sections.append("Return one JSON object matching the schema: a data request or the analysis, never both.")
    return "\n\n".join(sections)


def _translation_rules(settings: TranslationSettings) -> list[str]:
    rules = [
        _COMPLETENESS,
        _content_nodes(settings),
        f"{_REPHRASING[settings.creativity]} Always preserve the original meaning, tone, and characterization.",
        _ONOMATOPOEIA,
        _PUNCTUATION,
        _SYMBOLS,
        (
            "Glossary names and the running summary may be slightly out of date; translate the current chapter as "
            "written and do not overthink apparent mismatches with earlier context."
        ),
    ]
    rules.append(
        "Use footnotes only for necessary translator notes; keep them short."
        if settings.create_footnotes
        else "Do not produce footnotes."
    )
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
            "Pass 2 of 4 — translation only. Using the finalized glossary, return the translated title and content; "
            "do NOT change the glossary, notes, or summary in this pass."
        ),
        *_translation_rules(settings),
        "Return one JSON object matching the schema: a data request or the translation, never both.",
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
            "Pass 3 of 4 — review and fix. Re-read the translation you produced. The full translation rules are "
            "repeated below; check the draft against every one and fix each violation, returning the corrected title "
            "and content. If it already follows every rule, return it unchanged."
        ),
        *_translation_rules(settings),
    ]
    if settings.allow_glossary_edit or settings.allow_glossary_delete:
        sections.append(
            "You MUST also correct mistakes the analysis pass made (required, not optional). Check every entry and "
            "alias it just added against the rest of the glossary: if one duplicates an entry that already existed "
            "(same name, person, or thing), delete the newly-added duplicate and move anything unique it had (an "
            "alias, a better definition) onto the surviving (established) entry with an edit. Also fix an entry with "
            "the wrong type, a blank name, or that is not a name, and remove from notes or summary anything that does "
            "not belong. Make only corrections — do NOT add brand-new glossary entries, notes, or summary; leave "
            "glossary_updates empty and context null when nothing needs fixing. The linking pass references whichever "
            "entry survives."
        )
    sections.append("Return one JSON object matching the schema: a data request or the corrected chapter, never both.")
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
            "Pass 4 of 4 — glossary linking. Re-read the translated content and return the same chapter with its "
            "glossary names linked. Change ONLY how names are represented — keep all wording, punctuation, line "
            "breaks, the title, and every other node exactly as they were."
        ),
        _COMPLETENESS,
        _content_nodes(settings),
        (
            "Wherever a name that HAS a glossary entry appears as plain text, replace just that text with a term node "
            "for that entry. Set alias_id only when the written form matches one of THAT entry's own aliases, and it "
            "must be an alias id listed under that term_id — never one from another entry (matching the wording is "
            "not enough). Leave ordinary words, kept loanwords, and names with no glossary entry as text. term_id and "
            "alias_id must be ids copied verbatim from the glossary, never invented."
        ),
        "Return one JSON object matching the schema: a data request or the linked chapter, never both.",
    ]
    return "\n\n".join(sections)
