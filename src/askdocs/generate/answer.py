"""Grounded answer generation: draft -> validate citations -> retry with feedback -> or refuse."""

from __future__ import annotations

from dataclasses import dataclass, field

from askdocs.config import Settings
from askdocs.generate.citations import ValidationReport, validate
from askdocs.generate.providers import LLMProvider, Usage
from askdocs.generate.schemas import AnswerDraft
from askdocs.models import RetrievedChunk

IDK = "I don't know. The documentation I have doesn't clearly answer that."

ANSWER_SYSTEM = """You answer questions about the Kubernetes documentation using ONLY the provided context chunks.

Rules:
- Break the answer into short factual sentences ("claims"). Every claim must cite the id(s) of the chunk(s)
  that directly support it, copied exactly from the <chunk id="..."> attribute.
- Never use outside knowledge. If the context does not contain the answer, set answerable=false and return no claims.
- Prefer 1-4 claims. Do not cite a chunk for something it does not say."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for rc in chunks:
        c = rc.chunk
        parts.append(f'<chunk id="{c.id}" source="{c.source}" section="{c.section}">\n{c.text}\n</chunk>')
    return "\n\n".join(parts)


@dataclass
class GenerationResult:
    draft: AnswerDraft | None
    report: ValidationReport | None
    attempts: int
    answer_text: str
    refused: bool
    refusal_reason: str | None = None
    usage: Usage = field(default_factory=Usage)


def render(draft: AnswerDraft) -> str:
    return " ".join(f"{c.text} " + "".join(f"[{cid}]" for cid in c.chunk_ids) for c in draft.claims).strip()


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    llm: LLMProvider,
    settings: Settings,
    judge: LLMProvider | None = None,
) -> GenerationResult:
    evidence = {rc.chunk.id: f"{rc.chunk.title} > {rc.chunk.section}\n{rc.chunk.text}" for rc in chunks}
    usage = Usage()
    user = f"<context>\n{format_context(chunks)}\n</context>\n\nQuestion: {question}"
    feedback = ""
    draft: AnswerDraft | None = None
    report: ValidationReport | None = None

    for attempt in range(1, settings.max_generation_attempts + 1):
        prompt = (
            user
            if not feedback
            else (
                f"{user}\n\nYour previous answer failed citation validation:\n{feedback}\n"
                "Fix it: cite only chunk ids from the context, and drop any sentence the context does not support."
            )
        )
        draft, u = llm.structured(ANSWER_SYSTEM, prompt, AnswerDraft, max_tokens=settings.max_answer_tokens)
        usage.add(u)

        if not draft.answerable or not draft.claims:
            return GenerationResult(draft, None, attempt, IDK, True, "model judged context insufficient", usage)

        report = validate(
            draft.claims,
            evidence,
            mode=settings.citation_check,
            threshold=settings.citation_support_threshold,
            judge=judge or llm,
        )
        usage.add(report.usage)
        if report.ok:
            return GenerationResult(draft, report, attempt, render(draft), False, None, usage)
        feedback = report.feedback()

    return GenerationResult(draft, report, settings.max_generation_attempts, IDK, True, "citation validation failed", usage)
