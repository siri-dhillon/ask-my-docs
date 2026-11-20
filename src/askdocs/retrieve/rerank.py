"""Cross-encoder reranking.

Bi-encoders (the dense retriever) embed query and chunk separately, which is fast but lossy.
A cross-encoder reads (query, chunk) together and scores relevance directly: far more
accurate, far too slow for the whole corpus, perfect for re-ordering ~30-60 candidates.
"""

from __future__ import annotations

import math
from functools import cached_property
from typing import Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 8):
        self.model_name = model_name
        self.batch_size = batch_size

    @cached_property
    def model(self) -> TextCrossEncoder:
        return TextCrossEncoder(self.model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return relevance probabilities in [0, 1] (sigmoid of the raw logits)."""
        if not texts:
            return []
        return [sigmoid(float(s)) for s in self.model.rerank(query, texts, batch_size=self.batch_size)]
