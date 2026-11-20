"""Shared data types passed between pipeline stages."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    """A contiguous piece of a document under one heading path."""

    heading_path: list[str]  # e.g. ["ConfigMaps", "Using ConfigMaps", "Immutable ConfigMaps"]
    text: str
    page: int | None = None  # PDFs only


class Document(BaseModel):
    source: str  # path relative to the data dir
    title: str
    sections: list[Section]


class Chunk(BaseModel):
    id: str
    source: str
    title: str
    section: str  # heading path joined with " > "
    page: int | None = None
    text: str
    context: str = ""  # optional LLM-written situating context (contextual retrieval)

    def embedding_text(self) -> str:
        """What we actually embed/index: breadcrumb + optional context + body.

        Prepending the breadcrumb is a cheap form of contextual retrieval: a chunk that just
        says "set it to true" becomes findable by the section it lives in.
        """
        head = f"{self.title} > {self.section}" if self.section else self.title
        parts = [head, self.context, self.text]
        return "\n".join(p for p in parts if p)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # the score of the final stage (rerank prob, RRF score or cosine)
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    chunk_id: str
    source: str
    section: str
    quote: str = ""  # short excerpt of the cited chunk


class Claim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    supported: bool | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    claims: list[Claim]
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    refused: bool = False
    refusal_reason: str | None = None
    confidence: float
    attempts: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
