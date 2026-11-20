"""Wires the stages together. The API, CLI, UI, MCP server and eval all go through `AskService.ask`."""

from __future__ import annotations

import time

from askdocs.config import Settings, get_settings
from askdocs.generate.answer import IDK, generate_answer
from askdocs.generate.providers import LLMProvider, Usage, build_provider
from askdocs.generate.schemas import RewrittenQuery
from askdocs.index import DocIndex
from askdocs.models import AskResponse, Citation, Claim
from askdocs.retrieve.pipeline import Mode, Retriever

REWRITE_SYSTEM = (
    "Rewrite the user's question into a standalone search query for the Kubernetes documentation. "
    "Expand abbreviations and add the precise Kubernetes terms a doc page would use. Keep it under 25 words."
)


class AskService:
    def __init__(
        self,
        settings: Settings | None = None,
        index: DocIndex | None = None,
        retriever: Retriever | None = None,
        llm: LLMProvider | None = None,
        judge: LLMProvider | None = None,
    ):
        self.s = settings or get_settings()
        self.index = index or DocIndex(self.s)
        self.retriever = retriever or Retriever(self.s, self.index)
        self.llm = llm or build_provider(self.s)
        self.judge = judge or (build_provider(self.s, self.s.judge_model) if self.s.judge_model else self.llm)

    def ask(self, question: str, mode: Mode | None = None, top_k: int | None = None) -> AskResponse:
        timings: dict[str, float] = {}
        usage = Usage()
        t_start = time.perf_counter()

        query = question
        if self.s.rewrite_queries:
            t0 = time.perf_counter()
            rq, u = self.llm.structured(REWRITE_SYSTEM, question, RewrittenQuery, max_tokens=200)
            usage.add(u)
            query = rq.query
            timings["rewrite_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        result = self.retriever.retrieve(query, mode=mode, top_k=top_k)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        timings.update(result.timings_ms)

        def finish(**kw) -> AskResponse:
            timings["total_ms"] = (time.perf_counter() - t_start) * 1000
            return AskResponse(
                question=question,
                retrieved=result.chunks,
                confidence=result.confidence,
                timings_ms={k: round(v, 1) for k, v in timings.items()},
                usage={
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read_tokens": usage.cache_read_tokens,
                },
                **kw,
            )

        if not result.chunks or not self.retriever.is_confident(result):
            return finish(
                answer=IDK,
                claims=[],
                citations=[],
                refused=True,
                refusal_reason=f"retrieval confidence {result.confidence:.2f} below threshold",
            )

        t0 = time.perf_counter()
        gen = generate_answer(question, result.chunks, self.llm, self.s, judge=self.judge)
        timings["generation_ms"] = (time.perf_counter() - t0) * 1000
        usage.add(gen.usage)

        by_id = {rc.chunk.id: rc.chunk for rc in result.chunks}
        claims: list[Claim] = []
        cited: dict[str, Citation] = {}
        if gen.draft and gen.report:
            for chk in gen.report.checks:
                claims.append(Claim(text=chk.claim.text, citations=chk.claim.chunk_ids, supported=chk.ok))
                for cid in chk.claim.chunk_ids:
                    if cid in by_id and cid not in cited:
                        c = by_id[cid]
                        cited[cid] = Citation(chunk_id=cid, source=c.source, section=c.section, quote=c.text[:240])
        return finish(
            answer=gen.answer_text,
            claims=claims,
            citations=[] if gen.refused else list(cited.values()),
            refused=gen.refused,
            refusal_reason=gen.refusal_reason,
            attempts=gen.attempts,
        )
