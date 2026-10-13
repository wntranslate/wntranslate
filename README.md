# WNTranslate
Simple yet comprehensive AI-assisted webnovel translation tool. Keeps track of the plot properly to ensure smooth translation.

Tested with local llama.cpp self-hosted Qwen3.6 35B A3B model, and Deepseek V4.1 Flash. I recommend using this library with a provider that supports `json_schema` (which Deepseek does not support) although Deepseek worked decently well.

A high parameter model is recommended to understand the prompt and properly apply translations. This library uses a four-pass system (analyze, translate, review, apply glossary) every single translation, and thinking mode is recommended to be OFF (`ReasoningEffort.NONE`) or LOW (`ReasoningEffort.LOW`), otherwise each chapter will take a lot of time.

It takes under 60s to translate a single chapter (standard length) using Deepseek V4.1 Flash with no reasoning, costing under $0.02 per chapter.

### Features
- **Glossary** for consistent naming across chapters:
  - typed entries (person, item, organization, creative work, or a custom type)
  - aliases with an optional per-alias context, so one source term resolves differently by situation (e.g. a name cut off mid-speech, or depending on who speaks)
  - terms are referenced by ID instead of copied as text, so editing a translation updates every chapter at once
- **Context tracking** to keep long stories coherent:
  - rolling summary plus freeform notes
  - optional per-chapter summaries that the AI can reference
- **Settings-driven prompt and response schema** to regenerate from your options: source/target language, creativity, footnotes, context mode, chapter summaries, and glossary/reorder permissions and limits.
- **Structured model protocol** - glossary add/edit/delete, chapter reordering, and context changes apply straight back into the project, and the model can request data it still needs (the chapter list, another chapter's text, …), which is fulfilled automatically:
  - new chapters translate in two passes within one conversation: first the glossary, notes, and running summary are settled, then the translation is produced against the finalized names
- **HTML output** for a full reader (table of contents, prev/next arrows, keyboard navigation) or a single standalone chapter, with Markdown and jump-to/jump-back footnotes rendered.
- **Versioned project files** - the whole project can save and load as JSON.

### Install
```sh
pip install wntranslate # [async] for async usage
```

### Usage

Set up a project with a model provider. `base_url` can point at any OpenAI-compatible endpoint:
```python
from wntranslate import WebNovel, TranslationSettings, Creativity, OpenAIProvider

wn = WebNovel(
    "小説のタイトル",       # original (source-language) title, required
    "元のあらすじ",         # original synopsis, required ("" if none)
    settings=TranslationSettings(source_language="Japanese", target_language="English", creativity=Creativity.LOW),
    provider=OpenAIProvider(model="gpt-4o-mini", api_key="sk-...", base_url="https://api.openai.com/v1"),
)

# Fill the translated title/synopsis (optional; left as None otherwise):
wn.translate_metadata()
```

The model is always given the glossary and your project notes, and it can request the synopsis, titles, or other chapters' text when it needs them. Notes hold project-wide facts that must stay perfectly consistent across every chapter (and that aren't glossary terms or story summary); the model keeps them updated, or you can steer them with `request_change`.

Translate a new chapter. This runs in two passes within one conversation: first the model updates the glossary, notes, and running summary and classifies the chapter; then, with the finalized glossary, it produces the translation. Every response is validated against the generated schema, and any data the model requests (another chapter's text, the synopsis, …) is auto-fulfilled. The newly added chapter is returned:
```python
chapter = wn.translate_new_chapter(original_title="第1話", original_content="<source text>")
print(chapter.title)
```

Ask for a specific change in plain language; it is applied and returned:
```python
wn.request_change("Rename the glossary entry 'Ayami' to 'Ayame'.")
wn.request_change("In chapter 3 only, render Ayami as 'Aya'.") # Creates a new alias in the glossary and applies it to chapter 3 only.
```

Read it. Export Markdown, a full HTML reader, or one standalone chapter:
```python
print(wn.export())                       # markdown
with open("book.html", "w", encoding="utf-8") as fp:
    fp.write(wn.to_html())               # full reader: TOC + prev/next + footnotes
one = wn.chapter_to_html(wn.chapters[0]) # standalone chapter document
```

Save and load the whole project (versioned JSON):
```python
data = wn.dumps()  # compact; pass indent=2 for pretty output
restored = WebNovel.loads(data)
```

#### Building by hand

You can also populate the glossary and chapters directly instead of via the model. Every entry and alias gets a stable, auto-generated `id`:
```python
from wntranslate import PersonEntry, Alias

ayami = PersonEntry(
    name="Ayami",
    original_name="彩美",
    pronouns="she/her",
    aliases=[Alias(original="彩-", translation="Aya-", context="name cut off in speech")],
)
wn.glossary.add(ayami)

wn.glossary.get(ayami.id)      # by id
wn.glossary.lookup("彩美")      # by name, original name, or alias
```

Chapter content references glossary IDs instead of raw text, so editing a term updates every chapter; IDs resolve when rendering:
```python
from wntranslate import Chapter, Content, TextNode, TermNode, FootnoteNode

wn.chapters.append(Chapter(
    original_title="第1話",
    original_content="<source text>",
    title="Chapter 1",
    content=Content(nodes=[
        TextNode(text="Hello, "),
        TermNode(term_id=ayami.id),                               # -> "Ayami"
        TextNode(text="! Or should I say "),
        TermNode(term_id=ayami.id, alias_id=ayami.aliases[0].id), # -> "Aya-"
        FootnoteNode(content="A play on her name."),
    ]),
    summary="They meet for the first time.",
))
```
