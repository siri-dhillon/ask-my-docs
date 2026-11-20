"""Central configuration. Every knob is overridable via env vars (prefix ASKDOCS_) or .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASKDOCS_", env_file=".env", extra="ignore")

    # --- data / ingestion ---
    data_dir: Path = Path("data")
    chunk_size: int = 220  # words per chunk (≈ 300 tokens)
    chunk_overlap: int = 40  # words shared between consecutive chunks of one section
    min_chunk_words: int = 25  # tiny tail sections are merged into their neighbour

    # --- vector store ---
    qdrant_url: str | None = None  # e.g. http://localhost:6333 (docker-compose). None => embedded on-disk mode
    qdrant_path: Path = Path(".qdrant")
    collection: str = "askdocs"
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"

    # --- retrieval ---
    retrieval_mode: Literal["vector", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    candidates_per_retriever: int = 30  # how deep each retriever searches before fusion
    rrf_k: int = 60  # standard RRF constant from Cormack et al. (2009)
    rerank_pool: int = 30  # fused candidates sent to the cross-encoder (latency ∝ pool size)
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # or BAAI/bge-reranker-base (bigger, slower)
    top_k: int = 5  # chunks handed to the LLM
    min_confidence: float = 0.15  # refuse when reranker probability of best chunk is below this
    min_vector_score: float = 0.55  # same idea for modes without a reranker (cosine similarity)

    # --- contextual retrieval (optional, costs LLM calls at index time) ---
    contextualize: bool = False
    context_cache: Path = Path(".cache/contexts.jsonl")

    # --- generation ---
    llm_provider: Literal["anthropic", "openai", "ollama", "fake"] = "anthropic"
    llm_model: str = "claude-haiku-5-5"
    judge_model: str | None = None  # defaults to llm_model
    ollama_base_url: str = "http://localhost:11434/v1"
    max_answer_tokens: int = 2000
    citation_check: Literal["lexical", "llm"] = "lexical"
    citation_support_threshold: float = 0.5  # lexical: share of claim keywords found in the cited chunk
    max_generation_attempts: int = 2  # first try + one retry with validator feedback
    rewrite_queries: bool = False  # LLM query rewriting for vague questions


@lru_cache
def get_settings() -> Settings:
    return Settings()
