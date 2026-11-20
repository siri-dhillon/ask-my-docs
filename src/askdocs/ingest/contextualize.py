"""Contextual retrieval (Anthropic, Sept 2024): before indexing, ask an LLM to write 1-2 sentences
that situate each chunk within its whole document, and index `context + chunk`.

The full document goes in a cached system prompt, so every chunk of the same document after the
first one reads the document from the prompt cache at a fraction of the input price.
Results are cached on disk (keyed by chunk id) so re-indexing is free.
"""

from __future__ import annotations

import json
from pathlib import Path

from askdocs.generate.providers import LLMProvider, Usage
from askdocs.models import Chunk, Document

PROMPT = (
    "Here is the chunk we want to situate within the whole document:\n<chunk>\n{chunk}\n</chunk>\n"
    "Give a short succinct context (1-2 sentences) to situate this chunk within the overall document "
    "for the purposes of improving search retrieval of the chunk. Answer only with the succinct context."
)


def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out[rec["id"]] = rec["context"]
    return out


def contextualize(docs_chunks: list[tuple[Document, list[Chunk]]], llm: LLMProvider, cache_path: Path) -> Usage:
    cache = _load_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    usage = Usage()
    with cache_path.open("a", encoding="utf-8") as fh:
        for doc, chunks in docs_chunks:
            doc_text = "\n\n".join(s.text for s in doc.sections)
            system = f"<document title={doc.title!r}>\n{doc_text}\n</document>"
            for c in chunks:
                if c.id in cache:
                    c.context = cache[c.id]
                    continue
                ctx, u = llm.text(system, PROMPT.format(chunk=c.text), max_tokens=150, cache_system=True)
                usage.add(u)
                c.context = ctx
                fh.write(json.dumps({"id": c.id, "context": ctx}) + "\n")
    return usage
