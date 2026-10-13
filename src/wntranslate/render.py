from __future__ import annotations

from html import escape

from markdown_it import MarkdownIt

from .content import Content, TermNode, TextNode
from .glossary import Glossary

_MD = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])

_STYLE = """
:root { color-scheme: light dark; --fg: #1a1a1a; --bg: #fdfdfc; --muted: #666; --accent: #3b5bdb; --rule: #e3e3e0; }
@media (prefers-color-scheme: dark) { :root { --fg: #e6e6e6; --bg: #1a1a1a; --muted: #9a9a9a; --accent: #8ab4f8; --rule: #333; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: Georgia, "Times New Roman", serif; line-height: 1.7; }
.top { position: sticky; top: 0; display: flex; align-items: center; gap: 12px; padding: 12px 16px; background: var(--bg); border-bottom: 1px solid var(--rule); }
.top h1 { flex: 1; margin: 0; font-size: 1.1rem; text-align: center; font-family: system-ui, sans-serif; }
.top button { font-size: 1.2rem; line-height: 1; padding: 6px 12px; cursor: pointer; background: none; border: 1px solid var(--rule); border-radius: 6px; color: var(--fg); }
.top button:disabled { opacity: 0.35; cursor: default; }
.wrap { max-width: 42rem; margin: 0 auto; padding: 0 16px 6rem; }
.toc { font-family: system-ui, sans-serif; }
.toc h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.toc ol { padding-left: 1.2rem; }
.toc a { color: var(--accent); text-decoration: none; }
.toc a.current { font-weight: 700; }
.chapter-title { font-family: system-ui, sans-serif; }
a { color: var(--accent); }
.footnote-ref a { text-decoration: none; }
.footnotes { margin-top: 2.5rem; border-top: 1px solid var(--rule); padding-top: 1rem; font-size: 0.9rem; color: var(--muted); }
.footnotes a.footnote-back { color: inherit; text-decoration: none; }
.footnotes a.footnote-back:hover { color: var(--accent); }
"""

_SCRIPT = """
const chapters = Array.from(document.querySelectorAll('.chapter'));
const links = Array.from(document.querySelectorAll('.toc a'));
const prev = document.getElementById('prev');
const next = document.getElementById('next');
let current = 0;
function show(i) {
  if (i < 0 || i >= chapters.length) return;
  chapters.forEach((c, idx) => c.hidden = idx !== i);
  links.forEach((a, idx) => a.classList.toggle('current', idx === i));
  current = i;
  prev.disabled = i === 0;
  next.disabled = i === chapters.length - 1;
  window.scrollTo(0, 0);
  history.replaceState(null, '', '#ch-' + i);
}
links.forEach((a, idx) => a.addEventListener('click', e => { e.preventDefault(); show(idx); }));
prev.addEventListener('click', () => show(current - 1));
next.addEventListener('click', () => show(current + 1));
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') show(current - 1);
  else if (e.key === 'ArrowRight') show(current + 1);
});
const match = location.hash.match(/^#ch-(\\d+)$/);
show(match ? parseInt(match[1], 10) : 0);
"""


def render_content_html(content: Content, glossary: Glossary, *, id_prefix: str = "") -> str:
    """Render content to HTML, resolving terms and linking footnotes.

    Args:
        content: The content to render.
        glossary: Glossary used to resolve term ids.
        id_prefix: Prefix for footnote ids, to keep them unique across chapters.

    Returns:
        The HTML fragment.
    """
    parts: list[str] = []
    footnotes: list[str] = []
    for node in content.nodes:
        if isinstance(node, TextNode):
            parts.append(node.text)
        elif isinstance(node, TermNode):
            parts.append(node.resolve(glossary))
        else:
            footnotes.append(node.content)
            parts.append(f"@@WNFN{len(footnotes)}@@")

    body: str = _MD.render("".join(parts))
    for index in range(1, len(footnotes) + 1):
        marker = (
            f'<sup class="footnote-ref" id="{id_prefix}fnref-{index}">'
            f'<a href="#{id_prefix}fn-{index}">{index}</a></sup>'
        )
        body = body.replace(f"@@WNFN{index}@@", marker)
    if not footnotes:
        return body

    items = [
        f'<li id="{id_prefix}fn-{index}">'
        f'<a class="footnote-back" href="#{id_prefix}fnref-{index}">{_MD.renderInline(text)}</a></li>'
        for index, text in enumerate(footnotes, start=1)
    ]
    return f'{body}\n<section class="footnotes"><ol>{"".join(items)}</ol></section>'


def chapter_document(title: str, body_html: str) -> str:
    """Wrap a single chapter's HTML in a standalone document.

    Args:
        title: Document title.
        body_html: The chapter's rendered HTML.

    Returns:
        A complete HTML document.
    """
    safe = escape(title)
    return (
        f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{safe}</title>\n<style>{_STYLE}</style>\n</head>\n<body>\n"
        f'<main class="wrap">\n<h1 class="chapter-title">{safe}</h1>\n{body_html}\n</main>\n</body>\n</html>\n'
    )


def reader_document(title: str, chapters: list[tuple[str, str]]) -> str:
    """Build a full reader document with a table of contents and navigation.

    Args:
        title: Book title.
        chapters: Ordered (title, body_html) pairs.

    Returns:
        A complete HTML document.
    """
    safe_title = escape(title)
    toc = "".join(
        f'<li><a href="#ch-{index}">{escape(name)}</a></li>' for index, (name, _) in enumerate(chapters)
    )
    articles = "".join(
        f'<article class="chapter" id="ch-{index}" hidden>\n'
        f'<h2 class="chapter-title">{escape(name)}</h2>\n{body}\n</article>'
        for index, (name, body) in enumerate(chapters)
    )
    return (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{safe_title}</title>\n<style>{_STYLE}</style>\n</head>\n<body>\n"
        f'<header class="top"><button id="prev" title="Previous">&larr;</button>'
        f'<h1>{safe_title}</h1><button id="next" title="Next">&rarr;</button></header>\n'
        f'<main class="wrap">\n<nav class="toc"><h2>Contents</h2><ol>{toc}</ol></nav>\n{articles}\n</main>\n'
        f"<script>{_SCRIPT}</script>\n</body>\n</html>\n"
    )
