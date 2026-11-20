"""Evaluate retrieval (all three modes) and, optionally, end-to-end answers with an LLM judge.

  uv run python eval/run_eval.py --retrieval-only          # free, no API key
  uv run python eval/run_eval.py                            # + answers (needs an LLM provider)
  uv run python eval/run_eval.py --check eval/eval_thresholds.yaml   # exit 1 if below thresholds

Writes eval/reports/latest.json (+ a timestamped copy) and eval/reports/results.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import first_relevant_passage_rank, first_relevant_rank, judge_answer, mrr, recall_at_k  # noqa: E402

from askdocs.config import Settings  # noqa: E402
from askdocs.generate.providers import Usage, build_provider  # noqa: E402
from askdocs.index import DocIndex  # noqa: E402
from askdocs.retrieve.pipeline import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODES = ["vector", "hybrid", "hybrid_rerank"]
KS = [1, 3, 5, 10]
# $ per million tokens (input, output); used for the cost line in the report
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def eval_retrieval(retriever: Retriever, golden: list[dict]) -> tuple[dict, list[dict]]:
    answerable = [g for g in golden if g["answerable"]]
    out: dict[str, dict] = {}
    per_q: list[dict] = []
    for mode in MODES:
        ranks, pranks, lat = [], [], []
        for g in answerable:
            t0 = time.perf_counter()
            res = retriever.retrieve(g["question"], mode=mode, top_k=max(KS))  # type: ignore[arg-type]
            lat.append((time.perf_counter() - t0) * 1000)
            r = first_relevant_rank([c.chunk.source for c in res.chunks], set(g["sources"]))
            pr = first_relevant_passage_rank(
                [(c.chunk.source, c.chunk.text) for c in res.chunks], set(g["sources"]), g["evidence"]
            )
            ranks.append(r)
            pranks.append(pr)
            per_q.append(
                {
                    "id": g["id"],
                    "mode": mode,
                    "doc_rank": r,
                    "passage_rank": pr,
                    "top_source": res.chunks[0].chunk.source if res.chunks else None,
                }
            )
        # document-level: any chunk from a gold doc counts. passage-level: must contain the evidence text.
        out[mode] = {f"doc_recall@{k}": round(recall_at_k(ranks, k), 4) for k in KS}
        out[mode]["doc_mrr@10"] = round(mrr(ranks), 4)
        out[mode].update({f"recall@{k}": round(recall_at_k(pranks, k), 4) for k in KS})
        out[mode]["mrr@10"] = round(mrr(pranks), 4)
        out[mode]["latency_p50_ms"] = round(p(lat, 0.5), 1)
        out[mode]["latency_p95_ms"] = round(p(lat, 0.95), 1)
        out[mode]["n"] = len(answerable)

    # Retrieval-level refusal check on the out-of-scope questions (reranker confidence gate)
    unans = [g for g in golden if not g["answerable"]]
    if unans:
        refused = [not retriever.is_confident(retriever.retrieve(g["question"], mode="hybrid_rerank")) for g in unans]
        answ_conf = [retriever.retrieve(g["question"], mode="hybrid_rerank") for g in answerable]
        false_refusals = sum(not retriever.is_confident(r) for r in answ_conf)
        out["confidence_gate"] = {
            "unanswerable_refused_rate": round(sum(refused) / len(unans), 4),
            "answerable_false_refusal_rate": round(false_refusals / len(answerable), 4),
        }
    return out, per_q


def eval_answers(settings: Settings, index: DocIndex, golden: list[dict], judge_model: str | None) -> tuple[dict, list[dict]]:
    from askdocs.service import AskService

    svc = AskService(settings, index=index)
    judge = build_provider(settings, judge_model) if judge_model else svc.llm
    judge_usage = Usage()
    rows: list[dict] = []
    for g in golden:
        t0 = time.perf_counter()
        r = svc.ask(g["question"])
        latency = (time.perf_counter() - t0) * 1000
        row = {
            "id": g["id"],
            "answerable": g["answerable"],
            "refused": r.refused,
            "refusal_reason": r.refusal_reason,
            "attempts": r.attempts,
            "latency_ms": latency,
            "usage": r.usage,
            "answer": r.answer,
            "n_claims": len(r.claims),
            "pipeline_citation_pass_rate": (sum(bool(c.supported) for c in r.claims) / len(r.claims)) if r.claims else None,
        }
        if g["answerable"] and not r.refused:
            context = {rc.chunk.id: f"{rc.chunk.title} > {rc.chunk.section}\n{rc.chunk.text}" for rc in r.retrieved}
            claims = [{"text": c.text, "citations": c.citations} for c in r.claims]
            j, u = judge_answer(judge, g["question"], g["answer"], claims, context)
            judge_usage.add(u)
            n = max(len(j.claims), 1)
            row["faithfulness"] = sum(c.supported_by_context for c in j.claims) / n
            row["citation_accuracy"] = sum(c.supported_by_citations for c in j.claims) / n
            row["relevance"] = (j.relevance - 1) / 4
            row["correctness"] = (j.correctness - 1) / 4
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    judged = [r for r in ans if "faithfulness" in r]
    unans = [r for r in rows if not r["answerable"]]

    def mean(key: str) -> float:
        vals = [r[key] for r in judged]
        return round(statistics.fmean(vals), 4) if vals else 0.0

    in_tok = sum(r["usage"].get("input_tokens", 0) for r in rows)
    out_tok = sum(r["usage"].get("output_tokens", 0) for r in rows)
    pin, pout = PRICES.get(settings.llm_model, (0.0, 0.0))
    lat = [r["latency_ms"] for r in rows]
    summary = {
        "faithfulness": mean("faithfulness"),
        "citation_accuracy": mean("citation_accuracy"),
        "answer_relevance": mean("relevance"),
        "answer_correctness": mean("correctness"),
        "answer_rate": round(sum(not r["refused"] for r in ans) / len(ans), 4) if ans else 0.0,
        "unanswerable_refusal_rate": round(sum(r["refused"] for r in unans) / len(unans), 4) if unans else 0.0,
        "retry_rate": round(sum(r["attempts"] > 1 for r in rows) / len(rows), 4),
        "latency_p50_ms": round(p(lat, 0.5), 1),
        "latency_p95_ms": round(p(lat, 0.95), 1),
        "cost_per_query_usd": round((in_tok * pin + out_tok * pout) / 1e6 / len(rows), 6),
        "judge_tokens": {"input": judge_usage.input_tokens, "output": judge_usage.output_tokens},
        "provider": f"{svc.llm.name}/{svc.llm.model}",
        "n": len(rows),
    }
    return summary, rows


def check(report: dict, thresholds_path: Path) -> list[str]:
    """Compare report against thresholds. Keys are dotted paths, e.g. retrieval.hybrid_rerank.recall@5."""
    th = yaml.safe_load(thresholds_path.read_text())
    failures = []
    for section, metrics in th.items():
        if section == "max":
            continue
        for dotted, minimum in _flatten(metrics).items():
            node = report.get(section)
            if node is None:
                print(f"  (skip) {section}.{dotted}: section not evaluated")
                continue
            for part in dotted.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                failures.append(f"{section}.{dotted}: missing from report")
            elif node < minimum:
                failures.append(f"{section}.{dotted}: {node:.3f} < {minimum}")
            else:
                print(f"  ok     {section}.{dotted}: {node:.3f} >= {minimum}")
    for dotted, maximum in _flatten(th.get("max", {})).items():
        node = report
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            print(f"  (skip) max.{dotted}: not evaluated")
        elif node > maximum:
            failures.append(f"{dotted}: {node:.3f} > max {maximum}")
        else:
            print(f"  ok     {dotted}: {node:.3f} <= {maximum}")
    return failures


def _flatten(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = float(v)
    return out


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "Passage-level (the retrieved chunk must contain the answer evidence):",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Doc Recall@5 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    names = {"vector": "Vector only", "hybrid": "Hybrid (BM25 + dense, RRF)", "hybrid_rerank": "Hybrid + rerank"}
    for m in MODES:
        r = report["retrieval"][m]
        lines.append(
            f"| {names[m]} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['recall@5']:.3f} | {r['recall@10']:.3f} | "
            f"{r['mrr@10']:.3f} | {r['doc_recall@5']:.3f} | {r['latency_p50_ms']:.0f} | {r['latency_p95_ms']:.0f} |"
        )
    if "answer" in report:
        a = report["answer"]
        lines += [
            "",
            f"End-to-end ({a['provider']}, hybrid + rerank, n={a['n']}):",
            "",
            "| Faithfulness | Citation accuracy | Relevance | Correctness | Answer rate | Out-of-scope refusal | p50 ms | p95 ms | $/query |",
            "|---|---|---|---|---|---|---|---|---|",
            f"| {a['faithfulness']:.3f} | {a['citation_accuracy']:.3f} | {a['answer_relevance']:.3f} | {a['answer_correctness']:.3f} | "
            f"{a['answer_rate']:.3f} | {a['unanswerable_refusal_rate']:.3f} | {a['latency_p50_ms']:.0f} | {a['latency_p95_ms']:.0f} | "
            f"{a['cost_per_query_usd']:.5f} |",
        ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "eval/golden.jsonl"))
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--judge-model", default=None, help="defaults to ASKDOCS_JUDGE_MODEL or the answer model")
    ap.add_argument("--check", default=None, help="thresholds yaml; exit 1 on regression")
    ap.add_argument("--out", default=str(ROOT / "eval/reports"))
    args = ap.parse_args()

    settings = Settings()
    golden = load_golden(Path(args.golden))
    index = DocIndex(settings)
    if index.count() == 0:
        sys.exit("index is empty: run `uv run askdocs ingest` first")
    retriever = Retriever(settings, index)

    report: dict = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            k: str(v)
            for k, v in settings.model_dump().items()
            if k
            in {
                "chunk_size",
                "chunk_overlap",
                "dense_model",
                "sparse_model",
                "rerank_model",
                "rerank_pool",
                "candidates_per_retriever",
                "top_k",
                "min_confidence",
                "llm_provider",
                "llm_model",
                "citation_check",
                "contextualize",
            }
        },
        "n_questions": len(golden),
    }
    print(f"retrieval eval on {sum(g['answerable'] for g in golden)} answerable questions...")
    report["retrieval"], report["retrieval_per_question"] = eval_retrieval(retriever, golden)
    for m in MODES:
        print(f"  {m:14s} {report['retrieval'][m]}")
    if "confidence_gate" in report["retrieval"]:
        print(f"  gate           {report['retrieval']['confidence_gate']}")

    if not args.retrieval_only:
        print("answer eval (LLM + judge)...")
        report["answer"], report["answer_per_question"] = eval_answers(
            settings, index, golden, args.judge_model or settings.judge_model
        )
        print(f"  {json.dumps(report['answer'])}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp"].replace(":", "").replace("-", "")
    for name in ("latest.json", f"report-{stamp}.json"):
        (out / name).write_text(json.dumps(report, indent=2))
    write_markdown(report, out / "results.md")
    print(f"wrote {out / 'latest.json'} and results.md")

    if args.check:
        print(f"checking thresholds in {args.check}")
        failures = check(report, Path(args.check))
        if failures:
            print("EVAL GATE FAILED:\n  " + "\n  ".join(failures))
            sys.exit(1)
        print("eval gate passed")


if __name__ == "__main__":
    main()
