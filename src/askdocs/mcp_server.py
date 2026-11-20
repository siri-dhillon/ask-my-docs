"""Expose the RAG pipeline as an MCP tool so agents (Claude Code, Claude Desktop, IDEs) can
query the docs with citations.

Register with Claude Code:
    claude mcp add ask-my-docs -- uv --directory /path/to/ask-my-docs run askdocs-mcp
"""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from mcp.server import MCPServer
from pydantic import BaseModel

from askdocs.service import AskService

mcp = MCPServer(
    name="ask-my-docs",
    instructions="Answers questions about the Kubernetes documentation with verified citations.",
)


class CitedAnswer(BaseModel):
    answer: str
    refused: bool
    citations: list[dict[str, str]]
    confidence: float


@lru_cache
def _service() -> AskService:
    load_dotenv()
    return AskService()


@mcp.tool()
def ask_docs(question: str) -> CitedAnswer:
    """Answer a question using the Kubernetes docs. Every sentence is backed by cited doc chunks;
    returns refused=true with "I don't know" when the docs don't cover it."""
    r = _service().ask(question)
    return CitedAnswer(
        answer=r.answer,
        refused=r.refused,
        citations=[{"chunk_id": c.chunk_id, "source": c.source, "section": c.section, "quote": c.quote} for c in r.citations],
        confidence=round(r.confidence, 3),
    )


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
