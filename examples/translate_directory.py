"""Lazily translate a directory of numbered chapter text files and serve them over HTTP.

Files are numbered (e.g. 0001.txt, 2.txt, 000003.txt) and sorted by that number.
Each file has its title on the first line and its content on the rest. Chapters are
translated on demand by an async worker: visiting one starts translating it (and up to
--pretranslate chapters ahead). Pages update live over a WebSocket — badges, streamed
tokens, and request status change in place without reloading; already-translated chapters
stay servable while another chapter is translating. The project is saved after each
chapter finishes.

    python examples/translate_directory.py ./chapters \
        --model gpt-4o-mini --base-url https://api.openai.com/v1 --api-key sk-... --pretranslate 5 --stream

    python examples/translate_directory.py ./chapters \
        --provider deepseek --model deepseek-chat --api-key sk-... --stream
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

from aiohttp import web

from wntranslate import (
    AsyncProvider,
    AsyncWebNovel,
    Chapter,
    Creativity,
    Message,
    ReasoningEffort,
    TranslationResponse,
    TranslationSettings,
)
from wntranslate.providers.async_openai import (
    AsyncDeepSeekProvider,
    AsyncOpenAIProvider,
)
from wntranslate.render import render_content_html

_PROVIDERS = {"openai": AsyncOpenAIProvider, "deepseek": AsyncDeepSeekProvider}

_CSS = """
body { margin: 0; background: #fdfdfc; color: #1a1a1a; font-family: Georgia, serif; line-height: 1.7; }
@media (prefers-color-scheme: dark) { body { background: #1a1a1a; color: #e6e6e6; } }
.wrap { max-width: 42rem; margin: 0 auto; padding: 0 16px 4rem; }
.nav { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 12px 16px;
       border-bottom: 1px solid #8884; font-family: system-ui, sans-serif; position: sticky; top: 0; background: inherit; }
.nav a { color: #3b5bdb; text-decoration: none; }
.nav .disabled { opacity: 0.35; }
.synopsis { color: #555; font-size: 0.95rem; }
@media (prefers-color-scheme: dark) { .synopsis { color: #aaa; } }
.toc-list { font-family: system-ui, sans-serif; line-height: 2; padding-left: 1.4rem; }
.toc-list a { color: #3b5bdb; text-decoration: none; }
.badge { font-size: 0.8rem; color: #888; }
.status { text-align: center; margin-top: 4rem; font-family: system-ui, sans-serif; color: #666; }
.stream { white-space: pre-wrap; word-break: break-word; font-size: 0.8rem; color: #888; background: #8881; padding: 12px; border-radius: 6px; }
.error { color: #c92a2a; white-space: pre-wrap; }
.request { margin-top: 2.5rem; border-top: 1px solid #8884; padding-top: 1.5rem; display: flex; flex-direction: column;
           gap: 8px; font-family: system-ui, sans-serif; }
.request label { font-size: 0.9rem; color: #666; }
.request textarea { width: 100%; box-sizing: border-box; font: inherit; padding: 8px; resize: vertical; }
.request button { align-self: flex-start; font: inherit; padding: 6px 14px; }
.request-status { font-size: 0.9rem; color: #666; margin: 4px 0 0; }
.footnotes { margin-top: 2.5rem; border-top: 1px solid #8884; padding-top: 1rem; font-size: 0.9rem; }
.footnotes a.footnote-back { color: inherit; text-decoration: none; }
"""

_BADGES = {"done": "done", "translating": "translating…", "error": "failed", "pending": ""}


def load_chapter(path: Path) -> tuple[str, str]:
    """Split a file into its title (first line) and content (the rest).

    Args:
        path: The chapter file.

    Returns:
        A (title, content) pair.
    """
    title, _, content = path.read_text(encoding="utf-8").partition("\n")
    return title.strip(), content.strip()


def _nav(index: int, total: int) -> str:
    prev = f'<a href="/chapter/{index - 1}">&larr; Prev</a>' if index > 0 else '<span class="disabled">&larr; Prev</span>'
    nxt = (
        f'<a href="/chapter/{index + 1}">Next &rarr;</a>'
        if index < total - 1
        else '<span class="disabled">Next &rarr;</span>'
    )
    return f'<nav class="nav">{prev}<a href="/">Chapter {index + 1} / {total}</a>{nxt}</nav>'


_WS_SCRIPT = """
function esc(s){var d=document.createElement('div');d.textContent=(s==null?'':s);return d.innerHTML;}
(function(){
  function connect(){
    var proto=(location.protocol==='https:'?'wss://':'ws://');
    var ws=new WebSocket(proto+location.host+'/ws');
    ws.onmessage=function(e){try{if(window.applyUpdate){window.applyUpdate(JSON.parse(e.data));}}catch(_){}};
    ws.onclose=function(){setTimeout(connect,1000);};
  }
  connect();
})();
"""


_TOC_SCRIPT = r"""
window.applyUpdate = function (d) {
  var t = document.getElementById('book-title'); if (t) { t.textContent = d.title; }
  var syn = document.getElementById('synopsis');
  if (syn) { syn.innerHTML = (d.synopsis || '').split('\n\n').filter(function (p) { return p.trim(); }).map(function (p) { return '<p>' + esc(p) + '</p>'; }).join(''); }
  var ol = document.getElementById('toc');
  if (ol) { ol.innerHTML = d.chapters.map(function (c) { return '<li><a href="/chapter/' + c.index + '">' + esc(c.title) + '</a> <span class="badge">' + esc(c.badge) + '</span></li>'; }).join(''); }
  var rs = document.getElementById('request-status'); if (rs) { rs.textContent = d.request; }
};
"""

_CHAPTER_SCRIPT = r"""
var INDEX = __INDEX__, STATE = __STATE__;
window.applyUpdate = function (d) {
  var chap = d.chapters[INDEX];
  if (chap && chap.state !== STATE) { location.reload(); return; }
  var pre = document.getElementById('stream');
  var text = d.streams[String(INDEX)];
  if (pre && text != null) { pre.textContent = text; }
};
"""


def _page(title: str, body: str, *, script: str = "") -> bytes:
    page_script = f"<script>{script}</script>" if script else ""
    document = (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_CSS}</style></head><body>{body}"
        f"{page_script}<script>{_WS_SCRIPT}</script></body></html>"
    )
    return document.encode("utf-8")


def _summarize_response(response: TranslationResponse) -> str:
    parts = []
    if response.glossary_updates:
        parts.append(f"{len(response.glossary_updates)} glossary change(s)")
    if response.chapters:
        parts.append(f"{len(response.chapters)} chapter change(s)")
    if response.context is not None:
        parts.append("context updated")
    return ", ".join(parts) or "no changes"


class LoggingProvider(AsyncProvider):
    """Wraps an async provider and prints each request's messages and response."""

    def __init__(self, inner: AsyncProvider) -> None:
        self.inner = inner

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        for message in messages:
            print(f"\n--- {message.role} ---\n{message.content}", file=sys.stderr)
        reply = await self.inner.complete(messages, response_format=response_format, extra_body=extra_body)
        print(f"\n--- response ---\n{reply}\n", file=sys.stderr)
        return reply


class Translator:
    """Owns the project, an async translation worker, per-chapter state, and autosave."""

    def __init__(self, project: AsyncWebNovel, sources: list[tuple[str, str]], pretranslate: int, save_path: Path) -> None:
        self.project = project
        self.sources = sources
        self.pretranslate = pretranslate
        self.save_path = save_path
        self.status: dict[int, str] = {}
        self.errors: dict[int, str] = {}
        self.translated: dict[int, Chapter] = {}
        self.pending: set[int] = set()
        self.requested: int | None = None
        self.metadata_state = "done" if project.title is not None else "pending"
        self.stream_text: dict[int, str] = {}
        self.token_counts: dict[int, int] = {}
        self._streaming: int | None = None
        self.requests: list[str] = []
        self.request_status: str | None = None
        self.request_error: str | None = None
        self.request_result: str | None = None
        self.cond = asyncio.Condition()
        self.listeners: set[asyncio.Event] = set()
        self._seed_from_project()

    def _notify(self) -> None:
        """Wake every connected WebSocket so it pushes a fresh snapshot."""
        for event in self.listeners:
            event.set()

    def feed_token(self, piece: str) -> None:
        """Append a streamed token to the chapter currently being translated."""
        if self._streaming is not None:
            self.stream_text[self._streaming] = self.stream_text.get(self._streaming, "") + piece
            self.token_counts[self._streaming] = self.token_counts.get(self._streaming, 0) + 1
            self._notify()

    def _request_status_text(self) -> str:
        return {
            "queued": "Request queued…",
            "running": "Applying request…",
            "done": f"Applied: {self.request_result or 'no changes'}",
            "error": f"Request failed: {self.request_error or 'error'}",
        }.get(self.request_status or "", "")

    def snapshot(self) -> dict[str, Any]:
        """Build the JSON state the WebSocket pushes to every page."""
        chapters = []
        for index in range(len(self.sources)):
            state = self.status.get(index, "pending")
            chapter = self.translated.get(index)
            title = chapter.title if chapter is not None and chapter.title else self.sources[index][0]
            chapters.append(
                {"index": index, "title": title or f"Chapter {index + 1}", "state": state, "badge": _BADGES.get(state, "")}
            )
        streams = {str(index): text for index, text in self.stream_text.items()}
        return {
            "title": self.project.title or self.project.original_title,
            "synopsis": self.project.synopsis or self.project.original_synopsis or "",
            "chapters": chapters,
            "streams": streams,
            "request": self._request_status_text(),
        }

    def start(self) -> None:
        asyncio.create_task(self._run())
        if self.metadata_state == "pending":
            asyncio.create_task(self._translate_metadata())

    async def _translate_metadata(self) -> None:
        async with self.cond:
            self.metadata_state = "translating"
            self.cond.notify_all()
            self._notify()
        started = time.monotonic()
        try:
            await self.project.translate_metadata()
            self._save()
        except Exception as exc:  # noqa: BLE001 - surface any failure and keep serving
            state = "error"
            print(f"Failed metadata translation after {time.monotonic() - started:.1f}s: {exc}", file=sys.stderr)
        else:
            state = "done"
            print(f"Translated metadata in {time.monotonic() - started:.1f}s", file=sys.stderr)
        async with self.cond:
            self.metadata_state = state
            self.cond.notify_all()
            self._notify()

    def _seed_from_project(self) -> None:
        claimed: set[int] = set()
        for chapter in self.project.chapters:
            key = (chapter.original_title, chapter.original_content)
            for index, source in enumerate(self.sources):
                if index not in claimed and source == key:
                    self.translated[index] = chapter
                    self.status[index] = "done"
                    claimed.add(index)
                    break

    async def _request(self, index: int) -> None:
        async with self.cond:
            self.requested = index
            last = len(self.sources) - 1
            for candidate in range(index, min(index + self.pretranslate, last) + 1):
                if self.status.get(candidate) not in ("translating", "done") and candidate not in self.pending:
                    self.pending.add(candidate)
            self.cond.notify_all()
            self._notify()

    def _next_chapter(self) -> int | None:
        if not self.pending:
            return None
        index = self.requested if self.requested in self.pending else min(self.pending)
        self.pending.discard(index)
        return index

    async def submit_request(self, text: str) -> None:
        """Queue a free-form change request; it runs after the current translation, before the next.

        Args:
            text: The written request to apply via request_change.
        """
        async with self.cond:
            self.requests.append(text)
            if self.request_status != "running":
                self.request_status = "queued"
            self.cond.notify_all()
            self._notify()

    async def _run(self) -> None:
        while True:
            text: str | None = None
            index: int | None = None
            async with self.cond:
                while not self.requests and not self.pending:
                    await self.cond.wait()
                if self.requests:
                    text = self.requests.pop(0)
                    self.request_status = "running"
                    self.request_error = None
                    self.request_result = None
                    self.cond.notify_all()
                else:
                    index = self._next_chapter()
                    assert index is not None
                    self.status[index] = "translating"
            self._notify()
            if text is not None:
                await self._run_request(text)
            elif index is not None:
                await self._translate(index)

    async def _run_request(self, text: str) -> None:
        started = time.monotonic()
        error: str | None = None
        result: str | None = None
        try:
            result = _summarize_response(await self.project.request_change(text))
            self._save()
        except Exception as exc:  # noqa: BLE001 - surface any failure and keep serving
            error = str(exc)
        elapsed = time.monotonic() - started
        async with self.cond:
            if error is not None:
                self.request_status = "error"
                self.request_error = error
                print(f"Request failed after {elapsed:.1f}s: {error}", file=sys.stderr)
            else:
                self.request_status = "done"
                self.request_result = result
                print(f"Applied request in {elapsed:.1f}s ({result})", file=sys.stderr)
            self.cond.notify_all()
            self._notify()

    async def _translate(self, index: int) -> None:
        title, content = self.sources[index]
        started = time.monotonic()
        error: str | None = None
        chapter: Chapter | None = None
        self.stream_text[index] = ""
        self.token_counts[index] = 0
        self._streaming = index
        try:
            chapter = await self.project.translate_new_chapter(title, content)
            self._save()
        except Exception as exc:  # noqa: BLE001 - surface any failure on the chapter page
            error = str(exc)
        finally:
            self._streaming = None
        elapsed = time.monotonic() - started
        async with self.cond:
            if error is not None:
                self.status[index] = "error"
                self.errors[index] = error
                print(f"Failed chapter {index + 1} ({title}) after {elapsed:.1f}s: {error}", file=sys.stderr)
            else:
                assert chapter is not None
                self.translated[index] = chapter
                self.status[index] = "done"
                self.stream_text.pop(index, None)
                tokens = self.token_counts.pop(index, 0)
                suffix = f" and {tokens} tokens" if tokens else ""
                print(f"Translated chapter {index + 1} ({chapter.title or title}) in {elapsed:.1f}s{suffix}", file=sys.stderr)
            self.cond.notify_all()
            self._notify()

    def _save(self) -> None:
        self.save_path.write_text(self.project.dumps(indent=4, separators=(", ", ": ")), encoding="utf-8")

    async def retranslate(self, index: int) -> None:
        """Drop an existing translation and queue the chapter to translate again."""
        async with self.cond:
            chapter = self.translated.pop(index, None)
            if chapter is not None and chapter in self.project.chapters:
                self.project.chapters.remove(chapter)
            self.status.pop(index, None)
            self.errors.pop(index, None)
        self._save()
        await self._request(index)

    def render_toc(self) -> bytes:
        items = []
        for index in range(len(self.sources)):
            state = self.status.get(index, "pending")
            chapter = self.translated.get(index)
            title = chapter.title if chapter is not None and chapter.title else self.sources[index][0]
            items.append(
                f'<li><a href="/chapter/{index}">{escape(title or f"Chapter {index + 1}")}</a> '
                f'<span class="badge">{_BADGES.get(state, "")}</span></li>'
            )
        heading = escape(self.project.title or self.project.original_title)
        synopsis = self.project.synopsis or self.project.original_synopsis or ""
        paragraphs = "".join(f"<p>{escape(part)}</p>" for part in synopsis.split("\n\n") if part.strip())
        body = (
            f'<main class="wrap"><h1 id="book-title">{heading}</h1>'
            f'<div id="synopsis" class="synopsis">{paragraphs}</div>'
            f'<ol id="toc" class="toc-list">{"".join(items)}</ol>{self._render_request_box()}</main>'
        )
        return _page("Contents", body, script=_TOC_SCRIPT)

    def _render_request_box(self) -> str:
        return (
            '<form method="post" action="/request" class="request">'
            '<label for="request">Request a change (glossary edit, retitle, add a note, reorder, …)</label>'
            '<textarea id="request" name="request" rows="3" '
            'placeholder="e.g. Rename the protagonist to Iroha everywhere and note that the tone is comedic."></textarea>'
            "<button>Submit request</button>"
            f'<p id="request-status" class="request-status">{escape(self._request_status_text())}</p></form>'
        )

    async def render_chapter(self, index: int) -> bytes:
        await self._request(index)
        total = len(self.sources)
        async with self.cond:
            state = self.status.get(index, "pending")
            chapter = self.translated.get(index)
            error = self.errors.get(index)
        script = _CHAPTER_SCRIPT.replace("__INDEX__", str(index)).replace("__STATE__", json.dumps(state))
        if state == "done" and chapter is not None:
            body = render_content_html(chapter.content, self.project.glossary)
            heading = escape(chapter.title or f"Chapter {index + 1}")
            retranslate = (
                f'<form method="post" action="/chapter/{index}/retranslate" class="retranslate">'
                f"<button>Retranslate</button></form>"
            )
            inner = f'{_nav(index, total)}<main class="wrap"><h1>{heading}</h1>{body}{retranslate}</main>'
            return _page(heading, inner, script=script)
        stream = f'<pre id="stream" class="stream">{escape(self.stream_text.get(index, ""))}</pre>'
        if state == "error":
            message = f'<p class="status error">{escape(error or "error")}</p>'
            return _page(f"Chapter {index + 1}", f'{_nav(index, total)}<main class="wrap">{message}{stream}</main>', script=script)
        message = f'<p class="status">Translating chapter {index + 1}…</p>'
        return _page(f"Chapter {index + 1}", f'{_nav(index, total)}<main class="wrap">{message}{stream}</main>', script=script)


async def _toc(request: web.Request) -> web.Response:
    translator: Translator = request.app["translator"]
    return web.Response(body=translator.render_toc(), content_type="text/html", charset="utf-8")


async def _ws(request: web.Request) -> web.WebSocketResponse:
    translator: Translator = request.app["translator"]
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    event = asyncio.Event()
    translator.listeners.add(event)
    try:
        await ws.send_json(translator.snapshot())
        while not ws.closed:
            await event.wait()
            event.clear()
            await ws.send_json(translator.snapshot())
    except (ConnectionError, asyncio.CancelledError):
        pass  # the client went away; stop pushing to it
    finally:
        translator.listeners.discard(event)
    return ws


async def _submit_request(request: web.Request) -> web.Response:
    translator: Translator = request.app["translator"]
    data = await request.post()
    text = str(data.get("request", "")).strip()
    if text:
        await translator.submit_request(text)
    raise web.HTTPFound("/")


async def _chapter(request: web.Request) -> web.Response:
    translator: Translator = request.app["translator"]
    raw = request.match_info["index"]
    if not raw.isdigit() or not (0 <= int(raw) < len(translator.sources)):
        raise web.HTTPNotFound
    return web.Response(body=await translator.render_chapter(int(raw)), content_type="text/html", charset="utf-8")


async def _retranslate(request: web.Request) -> web.Response:
    translator: Translator = request.app["translator"]
    raw = request.match_info["index"]
    if not raw.isdigit() or not (0 <= int(raw) < len(translator.sources)):
        raise web.HTTPNotFound
    await translator.retranslate(int(raw))
    raise web.HTTPFound(f"/chapter/{raw}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Lazily translate and serve a directory of numbered chapter files.")
    parser.add_argument("directory", type=Path, help="Directory of numbered .txt chapter files")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pretranslate", type=int, default=0, help="Chapters to translate ahead in the background")
    parser.add_argument("--save", type=Path, default=None, help="Project JSON path (default: <directory>.wnt.json)")
    parser.add_argument("--provider", choices=_PROVIDERS, default="openai", help="Provider preset to use")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--base-url", default=None, help="API base URL (required unless the provider supplies one)")
    parser.add_argument("--api-key", default=None, help="API key (omit for endpoints that need none)")
    parser.add_argument("--timeout", type=float, default=600.0, help="Request timeout in seconds (0 for no limit)")
    parser.add_argument(
        "--reasoning-effort",
        choices=["off", *(effort.value for effort in ReasoningEffort)],
        default=ReasoningEffort.NONE.value,
        help="Reasoning effort to request; 'off' omits the field for endpoints that reject it",
    )
    parser.add_argument("--source-language", default="Japanese")
    parser.add_argument("--target-language", default="English")
    parser.add_argument("--additional-context", default=None)
    parser.add_argument("--stream", action="store_true", help="Stream and print generated tokens as they arrive")
    parser.add_argument("--show-logs", action="store_true", help="Print each request's messages and response")
    args = parser.parse_args()
    if args.base_url is None and args.provider == "openai":
        parser.error("--base-url is required for --provider openai")

    files = sorted((path for path in args.directory.iterdir() if path.suffix == ".txt"), key=lambda path: int(path.stem))
    if not files:
        print(f"No .txt files found in {args.directory}", file=sys.stderr)
        return 1

    provider_kwargs: dict[str, Any] = {
        "model": args.model,
        "api_key": args.api_key,
        "timeout": args.timeout if args.timeout > 0 else None,
        "stream": args.stream,
        "reasoning_effort": None if args.reasoning_effort == "off" else ReasoningEffort(args.reasoning_effort),
    }
    if args.base_url is not None:
        provider_kwargs["base_url"] = args.base_url
    base_provider = _PROVIDERS[args.provider](**provider_kwargs)
    provider: AsyncProvider = LoggingProvider(base_provider) if args.show_logs else base_provider

    save_path = args.save if args.save is not None else Path(f"{args.directory.name}.wnt.json")
    if save_path.exists():
        project = AsyncWebNovel.loads(save_path.read_text(encoding="utf-8"))
        project.provider = provider
        print(f"Resumed project from {save_path}", file=sys.stderr)
    else:
        project = AsyncWebNovel(
            args.directory.name,
            "",
            settings=TranslationSettings(
                source_language=args.source_language,
                target_language=args.target_language,
                additional_context=args.additional_context,
                creativity=Creativity.MEDIUM
            ),
            provider=provider,
        )

    translator = Translator(project, [load_chapter(path) for path in files], max(args.pretranslate, 0), save_path)

    if args.stream:
        echo_to_console = args.show_logs

        def on_token(piece: str) -> None:
            translator.feed_token(piece)
            if echo_to_console:
                sys.stderr.buffer.write(piece.encode("utf-8", "replace"))
                sys.stderr.buffer.flush()

        base_provider.on_token = on_token

    app = web.Application()
    app["translator"] = translator
    app.add_routes(
        [
            web.get("/", _toc),
            web.get("/ws", _ws),
            web.post("/request", _submit_request),
            web.get("/chapter/{index}", _chapter),
            web.post("/chapter/{index}/retranslate", _retranslate),
        ]
    )
    app.on_startup.append(lambda _: _start(translator))

    done = sum(1 for state in translator.status.values() if state == "done")
    print(f"Serving {len(files)} chapters ({done} already translated) at http://127.0.0.1:{args.port}/", file=sys.stderr)
    print(f"Saving project to {save_path}", file=sys.stderr)
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)
    return 0


async def _start(translator: Translator) -> None:
    translator.start()


if __name__ == "__main__":
    raise SystemExit(main())
