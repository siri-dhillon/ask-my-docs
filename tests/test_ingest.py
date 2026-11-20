from pathlib import Path

import pytest

from askdocs.ingest.chunker import chunk_document, make_chunk_id
from askdocs.ingest.loaders import iter_documents, load_html, load_markdown
from askdocs.models import Document, Section

MD = """# Pods

Intro paragraph about pods.

## Lifecycle

Pods move through phases.

```bash
# not a heading inside a code fence
kubectl get pods
```

### Restart policy

Always, OnFailure, Never.
"""


def test_markdown_sections_follow_heading_tree():
    title, sections = load_markdown(MD, "fallback")
    assert title == "Pods"
    paths = [s.heading_path for s in sections]
    assert paths == [["Pods"], ["Pods", "Lifecycle"], ["Pods", "Lifecycle", "Restart policy"]]
    assert "kubectl get pods" in sections[1].text  # fenced '#' line did not start a section


def test_html_sections_and_title():
    html = """<html><head><title>Secrets | Kubernetes</title></head><body>
    <nav>menu junk</nav><main><h1>Secrets</h1><p>Secrets hold sensitive data.</p>
    <h2>Types</h2><ul><li>Opaque</li><li>kubernetes.io/tls</li></ul></main></body></html>"""
    title, sections = load_html(html, "fallback")
    assert title == "Secrets"
    assert sections[1].heading_path == ["Secrets", "Types"]
    assert "- Opaque" in sections[1].text
    assert all("menu junk" not in s.text for s in sections)


def test_pdf_loader_records_pages(tmp_path: Path):
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    pdf = tmp_path / "guide.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 750, "1 Introduction")
    c.drawString(72, 730, "This guide explains how probes work in detail.")
    c.showPage()
    c.drawString(72, 750, "2 Liveness Probes")
    c.drawString(72, 730, "A liveness probe restarts a container that is stuck.")
    c.save()
    docs = list(iter_documents(tmp_path))
    assert len(docs) == 1
    pages = {s.page for s in docs[0].sections}
    assert pages == {1, 2}
    assert any(s.heading_path[-1] == "2 Liveness Probes" for s in docs[0].sections)


def _doc(words: int) -> Document:
    text = " ".join(f"w{i}" for i in range(words))
    return Document(source="a/b.md", title="T", sections=[Section(heading_path=["T", "S"], text=text)])


def test_chunk_overlap_and_metadata():
    chunks = chunk_document(_doc(500), chunk_size=200, overlap=50)
    assert len(chunks) == 3
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-50:] == second[:50]  # configured overlap is honoured
    assert all(c.section == "S" and c.source == "a/b.md" for c in chunks)


def test_chunks_never_cross_sections():
    doc = Document(
        source="x.md",
        title="T",
        sections=[
            Section(heading_path=["T", "A"], text="alpha " * 60),
            Section(heading_path=["T", "B"], text="beta " * 60),
        ],
    )
    chunks = chunk_document(doc, chunk_size=200, overlap=20)
    assert [c.section for c in chunks] == ["A", "B"]
    assert "beta" not in chunks[0].text


def test_small_sections_are_merged():
    doc = Document(
        source="x.md",
        title="T",
        sections=[Section(heading_path=["T", "Tiny"], text="few words"), Section(heading_path=["T", "Big"], text="x " * 80)],
    )
    chunks = chunk_document(doc, min_words=25)
    assert len(chunks) == 1 and chunks[0].text.startswith("few words")


def test_ids_are_stable_and_unique():
    assert make_chunk_id("a.md", "S", 0) == make_chunk_id("a.md", "S", 0)
    chunks = chunk_document(_doc(1000), chunk_size=100, overlap=10)
    assert len({c.id for c in chunks}) == len(chunks)


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_document(_doc(10), chunk_size=10, overlap=10)
