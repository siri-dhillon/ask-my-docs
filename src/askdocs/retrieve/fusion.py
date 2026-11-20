"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).

score(d) = Σ_r 1 / (k + rank_r(d))

Why RRF instead of adding raw scores: BM25 scores are unbounded and cosine scores live in
[-1, 1], so they aren't comparable. RRF only uses ranks, needs no tuning beyond k (60 is
the paper's default), and rewards documents that several retrievers agree on.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hashable]], k: int = 60) -> list[tuple[Hashable, float]]:
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
