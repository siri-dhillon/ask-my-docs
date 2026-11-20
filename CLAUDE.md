# CLAUDE.md — ask-my-docs

Production-style RAG over a slice of the Kubernetes docs. Portfolio project #1 of 5
(inspired by Aishwarya Srinivasan's "5 AI Engineer Projects to Build in 2026").

## Layout
- `src/askdocs/ingest/` loaders (md/html/pdf) → heading-aware chunker
- `src/askdocs/index.py` Qdrant store: named dense vector + BM25 sparse vector per chunk
- `src/askdocs/retrieve/` dense + sparse search → RRF fusion → cross-encoder rerank
- `src/askdocs/generate/` pluggable LLM providers, prompts, citation validation
- `src/askdocs/api/` FastAPI `/ask`; `ui/` Streamlit; `src/askdocs/mcp_server.py` MCP tool
- `eval/` golden set, metrics, `run_eval.py`, `eval_thresholds.yaml` (CI gate)

## Commands
- `uv sync --all-extras` · `uv run pytest -q` · `uv run ruff check .`
- `uv run askdocs ingest` (build index) · `uv run askdocs ask "..."`
- `uv run uvicorn askdocs.api.app:app --reload` · `uv run streamlit run ui/streamlit_app.py`
- `uv run python eval/run_eval.py --retrieval-only` (no API key needed)

## Conventions
- Config only through `askdocs.config.Settings` (env prefix `ASKDOCS_`); no hardcoded keys.
- Tests must not call real LLMs: use the `fake` provider.
- Chunk IDs are stable hashes (`<doc-slug>-<6 hex>`); the golden set references sources, not IDs.
