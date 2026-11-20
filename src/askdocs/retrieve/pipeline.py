"""Retrieval pipeline with three modes, so the eval can compare them head-to-head.

vector         dense search only                                    (baseline)
hybrid         dense + BM25, merged with RRF
hybrid_rerank  dense + BM25 -> RRF -> cross-encoder rerank of the fused candidates
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from askdocs.config import Settings
from askdocs.index import DocIndex
from askdocs.models import Chunk, RetrievedChunk
from askdocs.retrieve.fusion import reciprocal_rank_fusion
from askdocs.retrieve.rerank import CrossEncoderReranker, Reranker

Mode = Literal["vector", "hybrid", "hybrid_rerank"]


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    confidence: float  # comparable within a mode; used for the refusal threshold
    mode: Mode
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(self, settings: Settings, index: DocIndex, reranker: Reranker | None = None):
        self.s = settings
        self.index = index
        self._reranker = reranker

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.s.rerank_model)
        return self._reranker

    def retrieve(self, query: str, mode: Mode | None = None, top_k: int | None = None) -> RetrievalResult:
        mode = mode or self.s.retrieval_mode
        top_k = top_k or self.s.top_k
        depth = self.s.candidates_per_retriever
        t: dict[str, float] = {}

        t0 = time.perf_counter()
        dense = self.index.dense_search(query, depth)
        t["dense_ms"] = (time.perf_counter() - t0) * 1000
        dense_rank = {c.id: i for i, (c, _) in enumerate(dense, start=1)}

        if mode == "vector":
            out = [RetrievedChunk(chunk=c, score=s, dense_rank=dense_rank[c.id]) for c, s in dense[:top_k]]
            conf = out[0].score if out else 0.0
            return RetrievalResult(out, conf, mode, t)

        t0 = time.perf_counter()
        sparse = self.index.sparse_search(query, depth)
        t["sparse_ms"] = (time.perf_counter() - t0) * 1000
        sparse_rank = {c.id: i for i, (c, _) in enumerate(sparse, start=1)}

        by_id: dict[str, Chunk] = {c.id: c for c, _ in dense} | {c.id: c for c, _ in sparse}
        fused = reciprocal_rank_fusion([[c.id for c, _ in dense], [c.id for c, _ in sparse]], k=self.s.rrf_k)

        def mk(cid: str, score: float, rr: float | None = None) -> RetrievedChunk:
            return RetrievedChunk(
                chunk=by_id[cid], score=score, dense_rank=dense_rank.get(cid), sparse_rank=sparse_rank.get(cid), rerank_score=rr
            )

        if mode == "hybrid":
            out = [mk(str(cid), s) for cid, s in fused[:top_k]]
            # RRF scores are tiny; normalise by the max possible (rank 1 in both lists) for a 0-1 confidence
            conf = (out[0].score / (2 / (self.s.rrf_k + 1))) if out else 0.0
            return RetrievalResult(out, conf, mode, t)

        # hybrid_rerank: rerank a deeper fused pool than top_k so the reranker can promote buried hits
        pool = [str(cid) for cid, _ in fused[: self.s.rerank_pool]]
        t0 = time.perf_counter()
        scores = self.reranker.score(query, [by_id[cid].embedding_text() for cid in pool])
        t["rerank_ms"] = (time.perf_counter() - t0) * 1000
        ranked = sorted(zip(pool, scores, strict=True), key=lambda x: x[1], reverse=True)[:top_k]
        out = [mk(cid, s, s) for cid, s in ranked]
        conf = out[0].score if out else 0.0
        return RetrievalResult(out, conf, mode, t)

    def is_confident(self, result: RetrievalResult) -> bool:
        if result.mode == "hybrid_rerank":
            return result.confidence >= self.s.min_confidence
        if result.mode == "vector":
            return result.confidence >= self.s.min_vector_score
        return True  # RRF scores carry no absolute relevance signal; never refuse on them alone
