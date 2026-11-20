FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy FASTEMBED_CACHE_PATH=/models
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra ui
COPY src ./src
COPY ui ./ui
COPY data ./data
RUN uv sync --frozen --no-dev --extra ui
# Pre-download embedding / reranker models into the image so containers start fast and offline
RUN uv run python -c "from fastembed import TextEmbedding, SparseTextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('Qdrant/bm25'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "askdocs.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
