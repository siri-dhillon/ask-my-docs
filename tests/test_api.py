import pytest
from fastapi.testclient import TestClient

from askdocs.api.app import create_app
from askdocs.generate.fake_handlers import default_fake
from askdocs.retrieve.pipeline import Retriever
from askdocs.service import AskService
from tests.test_retrieval import StubReranker


@pytest.fixture
def client(settings, index):
    svc = AskService(settings, index=index, retriever=Retriever(settings, index, StubReranker()), llm=default_fake())
    with TestClient(create_app(svc)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks_indexed"] == 5 and body["provider"] == "fake"


def test_ask_returns_answer_citations_and_chunks(client):
    r = client.post("/ask", json={"question": "What does the Retain reclaim policy do for persistent volumes?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert body["citations"][0]["source"] == "persistent-volumes.md"
    assert len(body["retrieved"]) > 0 and "total_ms" in body["timings_ms"]


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "mode": "bogus"}).status_code == 422


def test_mcp_tool_registered():
    import asyncio

    from askdocs.mcp_server import mcp

    tools = asyncio.run(mcp.list_tools())
    assert [t.name for t in tools] == ["ask_docs"]
