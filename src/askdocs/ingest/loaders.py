"""Turn PDF / Markdown / HTML files into Documents made of heading-scoped Sections."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

from askdocs.models import Document, Section

SUPPORTED = {".md", ".markdown", ".html", ".htm", ".pdf"}
IGNORED_NAMES = {"attribution.md", "readme.md", "license.md"}  # repo housekeeping, not corpus
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _flush(stack: list[str], buf: list[str], out: list[Section], page: int | None = None) -> None:
    text = "\n".join(buf).strip()
    if text:
        out.append(Section(heading_path=[h for h in stack if h], text=text, page=page))
    buf.clear()


def load_markdown(text: str, fallback_title: str) -> tuple[str, list[Section]]:
    stack: list[str] = []
    buf: list[str] = []
    sections: list[Section] = []
    in_fence = False
    title = ""
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _MD_HEADING.match(line)
        if m:
            _flush(stack, buf, sections)
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and not title:
                title = heading
            stack = stack[: level - 1] + [""] * max(0, level - 1 - len(stack)) + [heading]
        else:
            buf.append(line)
    _flush(stack, buf, sections)
    return title or fallback_title, sections


def load_html(html: str, fallback_title: str) -> tuple[str, list[Section]]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
    page_title = re.split(r"\s+\|\s+", page_title)[0]  # "Pod Security | Kubernetes" -> "Pod Security"

    stack: list[str] = []
    buf: list[str] = []
    sections: list[Section] = []
    title = ""
    block_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "td", "th", "dt", "dd", "blockquote"]
    for el in root.find_all(block_tags):
        assert isinstance(el, Tag)
        # skip blocks nested inside another captured block (e.g. <p> inside <li>)
        if el.find_parent(["li", "pre", "td", "th", "dd", "blockquote"]) and el.name not in {"pre"}:
            continue
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if re.fullmatch(r"h[1-6]", el.name):
            _flush(stack, buf, sections)
            level = int(el.name[1])
            if level == 1 and not title:
                title = text
            stack = stack[: level - 1] + [""] * max(0, level - 1 - len(stack)) + [text]
        else:
            buf.append(("- " + text) if el.name == "li" else text)
    _flush(stack, buf, sections)
    return title or page_title or fallback_title, sections


def load_pdf(path: Path) -> tuple[str, list[Section]]:
    reader = PdfReader(str(path))
    meta_title = (reader.metadata.title if reader.metadata else None) or path.stem
    sections: list[Section] = []
    current_heading = meta_title
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        buf: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            # Heuristic heading detection: short, no trailing period, Title Case or numbered.
            if (
                3 <= len(stripped) <= 80
                and not stripped.endswith((".", ",", ";", ":"))
                and (re.match(r"^\d+(\.\d+)*\s+[A-Z]", stripped) or stripped.istitle())
                and len(stripped.split()) <= 10
            ):
                _flush([meta_title, current_heading], buf, sections, page=page_no)
                current_heading = stripped
            else:
                buf.append(line)
        _flush([meta_title, current_heading], buf, sections, page=page_no)
    return meta_title, sections


def load_file(path: Path, root: Path) -> Document:
    suffix = path.suffix.lower()
    rel = str(path.relative_to(root))
    fallback = path.stem.replace("-", " ").replace("_", " ").title()
    if suffix in {".md", ".markdown"}:
        title, sections = load_markdown(path.read_text(encoding="utf-8"), fallback)
    elif suffix in {".html", ".htm"}:
        title, sections = load_html(path.read_text(encoding="utf-8"), fallback)
    elif suffix == ".pdf":
        title, sections = load_pdf(path)
    else:
        raise ValueError(f"unsupported file type: {path}")
    return Document(source=rel, title=title, sections=sections)


def iter_documents(root: Path) -> Iterator[Document]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED and path.name.lower() not in IGNORED_NAMES:
            doc = load_file(path, root)
            if doc.sections:
                yield doc
