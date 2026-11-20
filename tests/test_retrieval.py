from askdocs.retrieve.fusion import reciprocal_rank_fusion
from askdocs.retrieve.pipeline import Retriever


class StubReranker:
    """Scores by keyword overlap so tests are fast and deterministic."""

    def score(self, query, texts):
        q = set(query.lower().split())
        return [len(q & set(t.lower().split())) / max(len(q), 1) for t in texts]


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert fused[0][0] == "b"  # ranked 2nd and 1st beats 1st and 3rd
    assert len(fused) == 3


def test_rrf_handles_disjoint_lists():
    fused = dict(reciprocal_rank_fusion([["a"], ["z"]], k=60))
    assert fused["a"] == fused["z"]


def test_index_roundtrip(index):
    assert index.count() == 5
    got = index.get(["secret-000002", "missing-id"])
    assert list(got) == ["secret-000002"]


def test_sparse_search_finds_exact_terms(index):
    hits = index.sparse_search("MiB", 3)
    assert hits[0][0].id == "configmap-000001"


def test_all_modes_find_the_right_chunk(settings, index):
    r = Retriever(settings, index, reranker=StubReranker())
    for mode in ("vector", "hybrid", "hybrid_rerank"):
        res = r.retrieve("when does the kubelet restart a container liveness probe", mode=mode, top_k=3)
        assert res.chunks[0].chunk.id == "probes-000003", mode
        assert len(res.chunks) == 3


def test_hybrid_records_ranks_from_both_retrievers(settings, index):
    res = Retriever(settings, index, reranker=StubReranker()).retrieve("ConfigMap size limit MiB", mode="hybrid")
    top = res.chunks[0]
    assert top.chunk.id == "configmap-000001"
    assert top.dense_rank is not None and top.sparse_rank is not None


def test_low_confidence_is_flagged(settings, index):
    r = Retriever(settings, index, reranker=StubReranker())
    res = r.retrieve("zzz qqq", mode="hybrid_rerank")
    assert not r.is_confident(res)
