"""Structured-output schemas the LLM must fill. Pydantic -> JSON Schema -> constrained decoding."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DraftClaim(BaseModel):
    text: str = Field(description="One factual sentence that answers part of the question.")
    chunk_ids: list[str] = Field(description="IDs of the context chunks that directly support this sentence.")


class AnswerDraft(BaseModel):
    answerable: bool = Field(description="False if the context does not contain the answer.")
    claims: list[DraftClaim] = Field(description="The answer as a list of cited sentences. Empty if not answerable.")


class SupportVerdict(BaseModel):
    claim_index: int
    supported: bool
    reason: str


class SupportVerdicts(BaseModel):
    verdicts: list[SupportVerdict]


class RewrittenQuery(BaseModel):
    query: str = Field(description="A standalone, keyword-rich search query.")
