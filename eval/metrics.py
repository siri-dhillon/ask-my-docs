"""Retrieval and answer-quality metrics."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from askdocs.generate.providers import LLMProvider, Usage


# ---------- retrieval ----------
def first_relevant_rank(retrieved_sources: Sequence[str], gold: set[str]) -> int | None:
    for i, src in enumerate(retrieved_sources, start=1):
        if src in gold:
            return i
    return None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def passage_is_relevant(chunk_source: str, chunk_text: str, gold: set[str], evidence: str) -> bool:
    """Strict relevance: the chunk comes from a gold document AND contains the evidence phrase."""
    return chunk_source in gold and _norm(evidence) in _norm(chunk_text)


def first_relevant_passage_rank(chunks: Sequence[tuple[str, str]], gold: set[str], evidence: str) -> int | None:
    for i, (src, text) in enumerate(chunks, start=1):
        if passage_is_relevant(src, text, gold, evidence):
            return i
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Share of questions with at least one gold document in the top k (a.k.a. hit rate@k).
    Each question here has one target document, so this equals recall@k."""
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: Sequence[int | None]) -> float:
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks) if ranks else 0.0


# ---------- answers (LLM judge) ----------
class ClaimJudgement(BaseModel):
    claim_index: int
    supported_by_context: bool = Field(description="Is the claim supported by ANY of the retrieved context?")
    supported_by_citations: bool = Field(description="Is the claim supported by the chunks it specifically cites?")


class AnswerJudgement(BaseModel):
    claims: list[ClaimJudgement]
    relevance: int = Field(ge=1, le=5, description="How directly the answer addresses the question (5 = fully).")
    correctness: int = Field(ge=1, le=5, description="Agreement with the reference answer (5 = same facts).")


JUDGE_SYSTEM = """You are a strict evaluator of a retrieval-augmented QA system.
For each numbered claim, judge (a) whether ANY retrieved context chunk supports it, and (b) whether the chunks
it cites support it. Then rate relevance of the whole answer to the question (1-5) and its correctness against
the reference answer (1-5; a correct but less detailed answer can still score 4-5)."""


def judge_answer(
    judge: LLMProvider,
    question: str,
    reference: str,
    claims: list[dict],
    context: dict[str, str],
) -> tuple[AnswerJudgement, Usage]:
    ctx = "\n\n".join(f'<chunk id="{cid}">\n{text}\n</chunk>' for cid, text in context.items())
    cl = "\n".join(f"{i}. {c['text']}  (cites: {', '.join(c['citations'])})" for i, c in enumerate(claims))
    user = f"<question>{question}</question>\n<reference>{reference}</reference>\n<context>\n{ctx}\n</context>\n<claims>\n{cl}\n</claims>"
    return judge.structured(JUDGE_SYSTEM, user, AnswerJudgement, max_tokens=2000)
