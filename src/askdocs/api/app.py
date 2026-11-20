"""FastAPI service. `uvicorn askdocs.api.app:app`"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from askdocs import __version__
from askdocs.models import AskResponse
from askdocs.service import AskService


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["vector", "hybrid", "hybrid_rerank"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


def create_app(service: AskService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_dotenv()
        app.state.service = service or AskService()
        if app.state.service.index.count() == 0:
            raise RuntimeError("index is empty: run `askdocs ingest` first")
        yield

    app = FastAPI(title="Ask My Docs", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        svc: AskService = request.app.state.service
        return {
            "status": "ok",
            "chunks_indexed": svc.index.count(),
            "provider": svc.llm.name,
            "model": svc.llm.model,
            "retrieval_mode": svc.s.retrieval_mode,
        }

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest, request: Request) -> AskResponse:
        svc: AskService = request.app.state.service
        try:
            # model inference + LLM call are blocking; keep the event loop free
            return await run_in_threadpool(svc.ask, req.question, req.mode, req.top_k)
        except Exception as e:  # surface provider errors as 502 rather than a bare 500
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e

    return app


app = create_app()
