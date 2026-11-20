"""Qdrant-backed index holding a dense vector AND a BM25 sparse vector for every chunk.

One collection, two named vectors:
  - "dense": bge-small-en-v1.5 embeddings (cosine)      -> semantic matches
  - "bm25":  Qdrant/bm25 term weights with IDF modifier  -> exact keyword matches
Qdrant computes IDF server-side, so BM25 stays correct as documents are added.

Runs embedded (on-disk, no server) by default; set ASKDOCS_QDRANT_URL to use docker-compose.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from functools import cached_property

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from askdocs.config import Settings
from askdocs.models import Chunk

DENSE = "dense"
SPARSE = "bm25"


class DocIndex:
    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self.s = settings
        if client is not None:
            self.client = client
        elif settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            self.client = QdrantClient(path=str(settings.qdrant_path))

    # Models load lazily: the API process pays the cost once, tests that don't search pay nothing.
    @cached_property
    def dense_model(self) -> TextEmbedding:
        return TextEmbedding(self.s.dense_model)

    @cached_property
    def sparse_model(self) -> SparseTextEmbedding:
        return SparseTextEmbedding(self.s.sparse_model)

    def recreate(self) -> None:
        dim = self.dense_model.embedding_size
        if self.client.collection_exists(self.s.collection):
            self.client.delete_collection(self.s.collection)
        self.client.create_collection(
            self.s.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )

    def upsert(self, chunks: Sequence[Chunk], batch_size: int = 64) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.embedding_text() for c in batch]
            dense = list(self.dense_model.embed(texts))
            sparse = list(self.sparse_model.embed(texts))
            points = [
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)),
                    vector={
                        DENSE: d.tolist(),
                        SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                    },
                    payload=c.model_dump(),
                )
                for c, d, s in zip(batch, dense, sparse, strict=True)
            ]
            self.client.upsert(self.s.collection, points=points)

    def count(self) -> int:
        if not self.client.collection_exists(self.s.collection):
            return 0
        return self.client.count(self.s.collection).count

    def dense_search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        vec = next(iter(self.dense_model.query_embed(query))).tolist()
        res = self.client.query_points(self.s.collection, query=vec, using=DENSE, limit=limit, with_payload=True)
        return [(Chunk(**p.payload), p.score) for p in res.points]  # type: ignore[arg-type]

    def sparse_search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        sv = next(iter(self.sparse_model.query_embed(query)))
        res = self.client.query_points(
            self.s.collection,
            query=models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
            using=SPARSE,
            limit=limit,
            with_payload=True,
        )
        return [(Chunk(**p.payload), p.score) for p in res.points]  # type: ignore[arg-type]

    def get(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, cid)) for cid in chunk_ids]
        recs = self.client.retrieve(self.s.collection, ids=ids, with_payload=True)
        return {r.payload["id"]: Chunk(**r.payload) for r in recs}  # type: ignore[index, arg-type]
