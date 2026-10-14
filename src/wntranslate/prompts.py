from __future__ import annotations

from typing import TYPE_CHECKING

from .enums import ContextType, Creativity

if TYPE_CHECKING:
    from .settings import TranslationSettings

_REPHRASING = {
    Creativity.VERY_LOW: "Translate as literally as fidelity allows; rephrase only where the target language requires.",
    Creativity.LOW: "Rephrase lightly for readability, staying close to the source.",
    Creativity.MEDIUM: "Rephrase for natural flow, preserving nuances like onomatopoeia.",
    Creativity.HIGH: "Adapt phrasing and idiom freely for a fluent read, preserving nuances like onomatopoeia.",
    Creativity.VERY_HIGH: "Freely polish and localize for the most natural result, preserving nuances like onomatopoeia.",
}

_LOANWORDS = (
    "Keep source terms the target readership leaves untranslated (loanwords, genre vocabulary, honorific suffixes "
    "like 'Name-chan') in their established form; do not flatten them."
)

_QUOTES = "Use straight quotes and apostrophes (\" and '), not curly ones, unless the exact character matters to the plot."

_SYMBOLS = (
    "Render source-specific punctuation and symbols as the target language's own (e.g. a Japanese dash '――' becomes "
    "an em dash); leave target-language characters the author already used. When the source sets a passage apart with "
    "its OWN marks (e.g. foreign or suddenly-understood speech), that distinction is meaningful: keep it, re-expressed "
    "with a target-language device (or the exact convention a note sets). Never drop it, never keep or wrap text in "
    "the source's marks, and never use Markdown for it — use a literal symbol."
)

_PUNCTUATION = (
    "Do not add punctuation absent from the source (em/en dashes, hyphens, colons, semicolons) unless it appears "
    f"there. {_QUOTES}"
)

_ONOMATOPOEIA = (
    f"{_LOANWORDS} Keep literal onomatopoeia close to the original (e.g. 'fufufu'), but translate ones with a clear "
    "meaning or that aren't literal (e.g. 'jii-' as a stare, italicized). When the source stretches a sound by "
    "repeating one character (a long-vowel mark 'ー' or a repeated letter), render a bounded elongation (a scream, "
    "not thousands of letters), then move on; never emit a runaway run of a single character."
)

_DATA_REQUEST = (
    "You are given the glossary, notes, running summary, the work's title and synopsis, and the most recent chapters "
    "(id, title, original title, kind). For anything else — an older chapter's entry, the full chapter list, or a "
    "chapter's translated text, summary, or original title or content — return a data request (response_type "
    "'request') listing every item; it comes back and you continue. Never request the glossary, notes, title, or "
    "synopsis — they are always provided."
)

_GLOSSARY_EXPLANATION = (
    "The glossary keeps a name's rendering consistent across chapters: reference a name by its entry id (no alias id) "
    "and a variant by its alias id, never by text. The system assigns ids — never invent or reuse one. Entry fields: "
    "name (target rendering, required, never blank); original_name (exact source form, required, never blank); type "
    "(person/item/organization/creative_work/other); definition (a few words identifying who or what it is — not a "
    "description, backstory, or full sentence); pronouns (for a person, when shown); medium (for a creative_work); "
    "specified_type (for other). aliases are ALTERNATE forms of the SAME name only (nicknames, honorifics like "
    "Name-chan/-san/-sama, cut-off forms, alternate readings), each with original (source form), translation (output "
    "form), and optional context. Put the base name in name/original_name and only its variants in aliases."
)

_GLOSSARY_TYPES = (
    "Type each entry correctly, never defaulting to 'other': a named character or being is person (fill pronouns when "
    "shown); a place, world, group, company, or faction is organization; a uniquely proper-named object or artifact "
    "is item (a generic item, skill, or technique is NOT an entry); a referenced book, manga, anime, game, song, or "
    "film is creative_work; use 'other' only when none fit, with a short specified_type — a recurring catchphrase is a "
    "valid 'other'."
)

_NOTES_RULE = (
    "Notes are always provided and you MUST obey them. A note is a stable, work-wide decision about HOW to translate "
    "(a style or formatting rule, a recurring choice, a wording-affecting hidden plot point) — not a glossary entry "
    "or the story summary. Do NOT note evolving story state (moods, situations, relationships, a changing trait); "
    "that belongs in the running summary. Do NOT note anything that does not affect the translation (schedules, site "
    "metadata, trivia). Never note keeping or wrapping text in the source's own marks. When a recurring source device "
    "carries meaning (e.g. marking foreign speech) you MUST preserve it: pick ONE concrete target convention and note "
    "the exact literal characters you will use every time (a symbol, not Markdown; not vague). Return the full "
    "updated notes list whenever you change it; otherwise leave it unset."
)

_COMPLETENESS = (
    "Fill every required field. ALWAYS translate the chapter's title into the target language in the title field — "
    "never leave it in the source language, blank, or copied from the source — and put the whole chapter, first line "
    "to last, in content."
)

_NO_REPLY = (
    "A request only EDITS the project; you are not chatting. Never reply to, answer, or acknowledge the user, and "
    "never put a reply, answer, question, or commentary in a chapter's title or content. Change a chapter's content "
    "only when explicitly told to retranslate or replace it; to change only the title or other metadata, edit that "
    "field and OMIT content — the existing body is kept."
)


def _content_nodes(settings: TranslationSettings) -> str:
    text = (
        "Put the full translation in content as an ordered list of nodes. Prose goes in text nodes "
        '({"type": "text", "text": "..."}, Markdown allowed), including ordinary words, kept loanwords, and names not '
        'yet in the glossary. A term node ({"type": "term", "term_id": "<id>", "alias_id": null}) references an '
        "existing glossary entry by its exact id so the name stays consistent if later renamed. term_id must be a "
        "real glossary id (never a plain word, an invented id, or an entry you are adding now); alias_id is null "
        "unless the written form matches one of THAT entry's own aliases — never an alias id from another entry"
    )
    if settings.create_footnotes:
        text += '. A footnote node ({"type": "footnote", "content": "..."}) holds a short translator note'
    text += (
        ". Nodes join with nothing between them, so put line breaks inside text nodes (a blank line between "
        "paragraphs). content must never be empty."
    )
    return text


def _glossary_rules(settings: TranslationSettings) -> str:
    rules = []
    if settings.allow_glossary_add:
        rules.append(
            "The glossary holds ONLY proper NAMES that must read the same every time (specific characters, places, "
            "organizations, titles of works) and recurring CATCHPHRASES or coined signature expressions. Before "
            "adding, scan the glossary already provided and reuse any existing entry or alias — never add a duplicate. "
            "Otherwise add an entry the first time a qualifying name or catchphrase appears (name, original_name, "
            "type, short definition, pronouns for a person, and aliases seen so far); add an alias to the existing "
            "entry when a known name appears in a new form. Do NOT add generic, common, or one-off names ('the man', "
            "'the guild'), ordinary words, verbs, concepts, descriptions, or generic items, skills, or techniques — "
            "only proper names and catchphrases belong."
        )
    if settings.allow_glossary_edit:
        rules.append(
            "Edit an entry by its id, sending ONLY the fields you are changing — every field you omit keeps its "
            "current value, so revising a definition or adding an alias must NOT resend or blank the name. Use it to "
            "fix or complete a field (a blank name, wrong type, pronouns, a new alias)."
        )
    if settings.allow_glossary_delete:
        rules.append("Delete an entry by its id only once it is obsolete and no chapter still uses it.")
    if settings.allow_glossary_add or settings.allow_glossary_edit:
        rules.append(_GLOSSARY_TYPES)
    return " ".join(rules) if rules else "Do not change the glossary."


def _summary_rule(settings: TranslationSettings) -> str | None:
    if settings.context_type is ContextType.ROLLING_SUMMARY:
        return (
            "A running summary of the WHOLE story so far is provided. BUILD ON IT: fold in the current chapter and "
            "return the updated whole-story summary — never replace it with only recent events, and never drop the "
            "earlier story. Keep every still-relevant fact (who the characters are, relationships, the setting, "
            "unresolved plot threads); condense by merging older, fully-resolved details rather than deleting what "
            "still matters, so it stays a cumulative digest of the entire story, not a log of recent chapters."
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
    parts = [
        (
            f"You translate webnovels from {source} into {settings.target_language}. Translate the title and "
            "synopsis consistent with the glossary."
        ),
        f"{_REPHRASING[settings.creativity]} Always preserve the original meaning and tone.",
        f"Leave glossary-specified names untranslated. {_LOANWORDS}",
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

    sections: list[str] = [
        (
            f"You translate webnovels from {source} into {settings.target_language}, consistent with the project "
            "state given each request."
        ),
        _NO_REPLY,
        (
            f"Do not invent facts; when unclear, translate faithfully. Treat any typo in {source} as deliberate and "
            "map it; do not correct it."
        ),
        (
            'Keep pronouns, genders, relationships, forms of address, and meaning-bearing nuances (e.g. the '
            'masculine Japanese pronoun "ore"), including gender-bender cases.'
        ),
        _content_nodes(settings),
        _GLOSSARY_EXPLANATION,
        _glossary_rules(settings),
        f"{_REPHRASING[settings.creativity]} Always preserve the original meaning, tone, and characterization.",
        _ONOMATOPOEIA,
        _PUNCTUATION,
        _SYMBOLS,
        (
            "Each returned chapter has a translated title, a kind (chapter, announcement, post, or other), and "
            "content; the source text is kept for you, so never echo it. chapters is a LIST — to change several at "
            "once, return one entry each. To revise or move an existing chapter, set target_id to its id (from the "
            "recent-chapters list, or request the chapter list for an older one), never an invented id. An edit "
            "updates ONLY the fields you send: to change just the title, send target_id and title and omit content "
            "(the body is kept); include content only when you are rewriting the body, and then it must be the full "
            "chapter. A NEW chapter must include full content."
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
        "If the change spans more chapters or edits than fit in one response, apply as many as you can and set more "
        "to true; you will be asked to continue batch by batch — set more to false on the final batch."
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

    sections = [
        (
            f"You translate webnovels from {source} into {settings.target_language}, consistent with the project "
            "state given you. You work in four passes in one conversation: pass 1 analyzes, pass 2 translates, pass "
            "3 reviews and fixes, pass 4 links glossary names. Each pass has its own instructions and schema — do "
            "only what the current pass asks."
        ),
        (
            f"Do not invent facts; when unclear, translate faithfully. Treat any typo in {source} as deliberate and "
            "map it; do not correct it."
        ),
        (
            'Keep pronouns, genders, relationships, forms of address, and meaning-bearing nuances (e.g. the '
            'masculine Japanese pronoun "ore"), including gender-bender cases.'
        ),
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
    ]
    if settings.allow_glossary_add:
        sections.append(
            "Be conservative: add ONLY genuine proper names and catchphrases, never a generic name, and never a "
            "duplicate of an entry already in the provided glossary — scan it first and reuse the match."
        )
    sections.append(_NOTES_RULE)
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
            "The glossary and running summary may be slightly out of date; translate the chapter as written and do "
            "not overthink mismatches with earlier context."
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
            "do NOT change the glossary, notes, or summary here."
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
            "Pass 3 of 4 — review and fix. Re-read your translation and check it against every rule below, fixing "
            "each violation and returning the corrected title and content in full; if it already follows every "
            "rule, return it unchanged."
        ),
        *_translation_rules(settings),
    ]
    if settings.allow_glossary_edit or settings.allow_glossary_delete:
        sections.append(
            "You MUST also fix mistakes the analysis pass made. Check each entry and alias it added against the rest "
            "of the glossary: if one duplicates an entry that already existed, delete the new duplicate and move "
            "anything unique (an alias, a better definition) onto the surviving entry with an edit. Also fix a wrong "
            "type, a blank name, or an entry that is not a name, and drop anything that does not belong from notes or "
            "summary. Make only corrections — do NOT add brand-new entries, notes, or summary; leave glossary_updates "
            "empty and context null when nothing needs fixing."
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
            "Pass 4 of 4 — glossary linking. Return the same chapter with its glossary names linked. Change ONLY how "
            "names are represented — keep all wording, punctuation, line breaks, the title, and every other node "
            "exactly as they were."
        ),
        _COMPLETENESS,
        _content_nodes(settings),
        (
            "Wherever a name that HAS a glossary entry appears as plain text, replace just that text with a term node "
            "for that entry. Set alias_id only when the written form matches one of THAT entry's own aliases (an "
            "alias id listed under that term_id) — never one from another entry. Leave ordinary words, kept "
            "loanwords, and names with no entry as text. term_id and alias_id are copied verbatim from the glossary, "
            "never invented."
        ),
        "Return one JSON object matching the schema: a data request or the linked chapter, never both.",
    ]
    return "\n\n".join(sections)
