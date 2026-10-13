from wntranslate import (
    Chapter,
    Content,
    FootnoteNode,
    PersonEntry,
    TermNode,
    TextNode,
    WebNovel,
)


def _project() -> tuple[WebNovel, PersonEntry, Chapter]:
    wn = WebNovel("src", "", title="Book")
    person = PersonEntry(name="Ayami")
    wn.glossary.add(person)
    chapter = Chapter(
        original_title="原題",
        original_content="原文",
        title="Chapter 1",
        content=Content(
            nodes=[
                TextNode(text="# Heading\n\nHello "),
                TermNode(term_id=person.id),
                TextNode(text=" and **bye**"),
                FootnoteNode(content="a *note*"),
            ]
        ),
    )
    wn.chapters.append(chapter)
    return wn, person, chapter


def test_chapter_to_html() -> None:
    wn, _, chapter = _project()
    html = wn.chapter_to_html(chapter)
    assert "<!DOCTYPE html>" in html
    assert "<h1>Heading</h1>" in html
    assert "<strong>bye</strong>" in html
    assert "Ayami" in html
    assert 'id="fnref-1"' in html and 'href="#fn-1"' in html
    assert 'id="fn-1"' in html and 'href="#fnref-1"' in html
    assert "<em>note</em>" in html


def test_to_html_reader() -> None:
    wn, _, _ = _project()
    wn.chapters.append(
        Chapter(original_title="o2", original_content="c2", title="Chapter 2", content=Content(nodes=[TextNode(text="Second")]))
    )
    html = wn.to_html()
    assert "Chapter 1" in html and "Chapter 2" in html
    assert 'class="toc"' in html
    assert 'id="ch-0"' in html and 'id="ch-1"' in html
    assert 'id="c0-fn-1"' in html
    assert "ArrowLeft" in html and "ArrowRight" in html


def test_html_escapes_raw_markup() -> None:
    wn = WebNovel("src", "", title="Book")
    wn.chapters.append(
        Chapter(
            original_title="o",
            original_content="c",
            title="T",
            content=Content(nodes=[TextNode(text="<script>alert(1)</script>")]),
        )
    )
    html = wn.to_html()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
