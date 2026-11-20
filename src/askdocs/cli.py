"""`askdocs ingest` builds the index; `askdocs ask "question"` queries it."""

from __future__ import annotations

import argparse
import json
import time

from dotenv import load_dotenv

from askdocs.config import get_settings


def cmd_ingest(args: argparse.Namespace) -> None:
    from askdocs.generate.providers import build_provider
    from askdocs.index import DocIndex
    from askdocs.ingest.chunker import chunk_document
    from askdocs.ingest.contextualize import contextualize
    from askdocs.ingest.loaders import iter_documents

    s = get_settings()
    t0 = time.perf_counter()
    docs = list(iter_documents(s.data_dir))
    pairs = [(d, chunk_document(d, s.chunk_size, s.chunk_overlap, s.min_chunk_words)) for d in docs]
    chunks = [c for _, cs in pairs for c in cs]
    print(f"loaded {len(docs)} documents -> {len(chunks)} chunks")
    if s.contextualize or args.contextualize:
        usage = contextualize(pairs, build_provider(s), s.context_cache)
        print(
            f"contextualized chunks (new tokens in={usage.input_tokens} cached={usage.cache_read_tokens} out={usage.output_tokens})"
        )
    idx = DocIndex(s)
    idx.recreate()
    idx.upsert(chunks)
    print(f"indexed {idx.count()} chunks into '{s.collection}' in {time.perf_counter() - t0:.1f}s")


def cmd_ask(args: argparse.Namespace) -> None:
    from askdocs.service import AskService

    resp = AskService().ask(args.question, mode=args.mode)
    if args.json:
        print(resp.model_dump_json(indent=2))
        return
    print(resp.answer)
    for c in resp.citations:
        print(f"  [{c.chunk_id}] {c.source} — {c.section}")
    print(f"\nconfidence={resp.confidence:.2f} attempts={resp.attempts} timings={json.dumps(resp.timings_ms)}")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="askdocs")
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("ingest", help="load, chunk and index the corpus")
    p.add_argument("--contextualize", action="store_true", help="add LLM-written context to each chunk")
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p.add_argument("--mode", choices=["vector", "hybrid", "hybrid_rerank"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
