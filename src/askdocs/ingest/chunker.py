"""Structure-aware chunking.

Strategy: never let a chunk span two sections (headings are natural topic boundaries).
Inside a long section, slide a word window of `chunk_size` with `overlap` words of context.
Tiny sections are merged with the following sibling so we don't index 5-word fragments.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from askdocs.models import Chunk, Document, Section


def _slug(text: str, max_len: int = 24) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].strip("-") or "doc"


def make_chunk_id(source: str, section: str, index: int) -> str:
    """Short, stable and human-readable, e.g. `configmap-3fa2c1`. Stable across re-indexing."""
    digest = hashlib.sha1(f"{source}|{section}|{index}".encode()).hexdigest()[:6]
    stem = source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return f"{_slug(stem)}-{digest}"


def _merge_small(sections: list[Section], min_words: int) -> list[Section]:
    merged: list[Section] = []
    carry: Section | None = None
    for sec in sections:
        if carry is not None:
            sec = Section(
                heading_path=carry.heading_path if len(carry.heading_path) <= len(sec.heading_path) else sec.heading_path,
                text=carry.text + "\n\n" + sec.text,
                page=carry.page or sec.page,
            )
            carry = None
        if len(sec.text.split()) < min_words:
            carry = sec
        else:
            merged.append(sec)
    if carry is not None:
        if merged:
            last = merged[-1]
            merged[-1] = Section(heading_path=last.heading_path, text=last.text + "\n\n" + carry.text, page=last.page)
        else:
            merged.append(carry)
    return merged


def _windows(words: list[str], size: int, overlap: int) -> Iterable[list[str]]:
    if len(words) <= size:
        yield words
        return
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + size]
        # avoid a tiny trailing window that's fully contained in the previous one
        if start > 0 and len(window) <= overlap:
            break
        yield window
        if start + size >= len(words):
            break


def chunk_document(doc: Document, chunk_size: int = 220, overlap: int = 40, min_words: int = 25) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for sec in _merge_small(doc.sections, min_words):
        # drop the doc title from the breadcrumb; it's stored separately
        path = sec.heading_path[1:] if sec.heading_path and sec.heading_path[0] == doc.title else sec.heading_path
        section_name = " > ".join(path)
        words = sec.text.split()
        for i, window in enumerate(_windows(words, chunk_size, overlap)):
            cid = make_chunk_id(doc.source, section_name, i)
            if cid in seen:  # repeated heading names inside one doc (e.g. two "Example" sections)
                cid = make_chunk_id(doc.source, f"{section_name}#{len(chunks)}", i)
            seen.add(cid)
            chunks.append(
                Chunk(
                    id=cid,
                    source=doc.source,
                    title=doc.title,
                    section=section_name,
                    page=sec.page,
                    text=" ".join(window),
                )
            )
    return chunks
