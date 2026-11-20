"""Build the demo corpus: a slice of the Kubernetes "Concepts" docs.

Source: https://github.com/kubernetes/website (content licensed CC BY 4.0).
- Markdown sections are pulled from the repo at a pinned commit and their Hugo
  shortcodes are rewritten into plain Markdown.
- The "security" section is downloaded as rendered HTML from kubernetes.io so the
  corpus exercises the HTML loader too.

Usage:  uv run python scripts/fetch_corpus.py [--out data/k8s]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

REPO = "https://github.com/kubernetes/website.git"
COMMIT = "be58364903ad78d60314019b81759f05c33b4af5"  # pinned for reproducibility
MD_SECTIONS = ["workloads", "services-networking", "configuration", "storage", "containers", "overview"]
HTML_SECTION = "security"
HEADING_MAP = {
    "whatsnext": "What's next",
    "prerequisites": "Before you begin",
    "objectives": "Objectives",
    "cleanup": "Clean up",
    "synopsis": "Synopsis",
    "options": "Options",
    "seealso": "See also",
}


def _attr(blob: str, name: str) -> str | None:
    m = re.search(rf'{name}\s*=\s*"([^"]*)"', blob)
    return m.group(1) if m else None


def load_glossary(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in (root / "content/en/docs/reference/glossary").glob("*.md"):
        text = f.read_text(encoding="utf-8")
        body = text.split("---", 2)[-1]
        body = re.sub(r"<!--.*?-->", "", body, flags=re.S).strip()
        first_para = body.split("\n\n")[0].replace("\n", " ").strip()
        out[f.stem] = first_para
    return out


def clean_markdown(text: str, glossary: dict[str, str]) -> str:
    # Front matter -> keep only the title as an H1.
    title = ""
    if text.startswith("---"):
        _, fm, text = text.split("---", 2)
        m = re.search(r"^title:\s*(.+)$", fm, flags=re.M)
        title = m.group(1).strip().strip('"') if m else ""

    def shortcode(m: re.Match[str]) -> str:
        if m.group(0).lstrip("{<% ").startswith("/"):
            return ""  # closing tag
        name, args = m.group(1), m.group(2) or ""
        if name == "glossary_tooltip":
            return _attr(args, "text") or _attr(args, "term_id") or ""
        if name == "glossary_definition":
            term = _attr(args, "term_id") or ""
            prepend = _attr(args, "prepend") or ""
            definition = clean_markdown(glossary.get(term, ""), {}) if term in glossary else ""
            definition = re.sub(r"^#.*\n", "", definition).strip()
            return f"{prepend} {definition}".strip()
        if name == "feature-state":
            ver = _attr(args, "for_k8s_version") or ""
            state = _attr(args, "state") or ""
            return f"FEATURE STATE: Kubernetes {ver} [{state}]".strip()
        if name == "heading":
            key = (re.search(r'"([^"]+)"', args) or [None, ""])[1]
            return HEADING_MAP.get(key, key.title())
        if name in {"note", "caution", "warning"}:
            return f"**{name.title()}:** "
        if name in {"code_sample", "codenew"}:
            return f"(See code sample `{_attr(args, 'file') or ''}`.)"
        if name == "param":
            return (re.search(r'"([^"]+)"', args) or [None, ""])[1]
        if name == "skew":
            return "the current version"
        return ""  # tabs, figure, mermaid, table wrappers... drop the tag, keep inner text

    text = re.sub(r"\{\{[<%]\s*/?\s*([a-zA-Z_-]+)(.*?)\s*[>%]\}\}", shortcode, text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return f"# {title}\n\n{text}\n" if title else text + "\n"


def fetch_markdown(out: Path) -> tuple[int, list[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "website"
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", REPO, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set",
                        "content/en/docs/concepts", "content/en/docs/reference/glossary"], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", COMMIT], check=True)
        glossary = load_glossary(repo)
        n = 0
        for section in MD_SECTIONS:
            for f in sorted((repo / "content/en/docs/concepts" / section).rglob("*.md")):
                rel = f.relative_to(repo / "content/en/docs/concepts")
                name = "index" if f.stem == "_index" else f.stem
                dest = out / "md" / rel.parent / f"{name}.md"
                cleaned = clean_markdown(f.read_text(encoding="utf-8"), glossary)
                if len(cleaned) < 300:  # skip near-empty section index pages
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(cleaned, encoding="utf-8")
                n += 1
        # Remember which security pages exist so we can fetch their rendered HTML.
        sec = repo / "content/en/docs/concepts" / HTML_SECTION
        pages = sorted(p.stem for p in sec.glob("*.md") if p.stem != "_index")
    return n, pages


def fetch_html(out: Path, pages: list[str]) -> int:
    n = 0
    dest_dir = out / "html" / HTML_SECTION
    dest_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=30, headers={"User-Agent": "askdocs-corpus-builder"}) as client:
        for page in pages:
            url = f"https://kubernetes.io/docs/concepts/{HTML_SECTION}/{page}/"
            r = client.get(url)
            if r.status_code != 200:
                print(f"skip {url}: {r.status_code}")
                continue
            # Keep only the article body (drop site nav/footer) to keep the repo small.
            soup = BeautifulSoup(r.text, "lxml")
            main = soup.select_one("div.td-content") or soup.find("main") or soup.body
            for tag in main.select("script, style, nav, .td-page-meta, .feedback--title, .pageinfo"):
                tag.decompose()
            title = soup.title.get_text(strip=True) if soup.title else page
            html = f"<html><head><title>{title}</title><meta name='source' content='{url}'></head><body><main>{main}</main></body></html>"
            (dest_dir / f"{page}.html").write_text(html, encoding="utf-8")
            n += 1
            time.sleep(0.3)  # be polite
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/k8s")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    n_md, pages = fetch_markdown(out)
    n_html = fetch_html(out, pages)
    print(f"wrote {n_md} markdown + {n_html} html docs to {out}")


if __name__ == "__main__":
    main()
