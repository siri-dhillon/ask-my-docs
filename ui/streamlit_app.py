"""Minimal UI over the /ask API.  `streamlit run ui/streamlit_app.py`"""

from __future__ import annotations

import os
import re

import httpx
import streamlit as st

API = os.environ.get("ASKDOCS_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Ask My Docs", page_icon="📚", layout="wide")
st.title("📚 Ask My Docs — Kubernetes")
st.caption("Hybrid retrieval (BM25 + dense) → RRF → cross-encoder rerank → cited answer with validated citations.")

with st.sidebar:
    mode = st.radio("Retrieval mode", ["hybrid_rerank", "hybrid", "vector"], index=0)
    top_k = st.slider("Chunks given to the LLM (top-k)", 1, 10, 5)
    try:
        h = httpx.get(f"{API}/health", timeout=5).json()
        st.success(f"API up · {h['chunks_indexed']} chunks · {h['provider']}/{h['model']}")
    except Exception as e:  # noqa: BLE001
        st.error(f"API not reachable at {API}: {e}")

examples = [
    "How large can a ConfigMap be?",
    "What happens when a liveness probe fails?",
    "What is the difference between a Deployment and a StatefulSet?",
    "Who won the 2022 World Cup?",
]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples, strict=True):
    if col.button(ex, use_container_width=True):
        st.session_state["q"] = ex

q = st.text_input("Question", key="q", placeholder="Ask anything about Kubernetes concepts…")
if q:
    with st.spinner("Retrieving and generating…"):
        r = httpx.post(f"{API}/ask", json={"question": q, "mode": mode, "top_k": top_k}, timeout=120)
    if r.status_code != 200:
        st.error(r.text)
        st.stop()
    data = r.json()

    answer = re.sub(r"\[([a-z0-9-]+-[0-9a-f]{6})\]", r"`\1`", data["answer"])
    (st.warning if data["refused"] else st.success)(answer)
    if data["refused"] and data.get("refusal_reason"):
        st.caption(f"Refusal reason: {data['refusal_reason']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{data['confidence']:.2f}")
    m2.metric("Attempts", data["attempts"])
    m3.metric("Total latency", f"{data['timings_ms'].get('total_ms', 0):.0f} ms")
    m4.metric("Tokens in/out", f"{data['usage'].get('input_tokens', 0)}/{data['usage'].get('output_tokens', 0)}")

    if data["citations"]:
        st.subheader("Citations")
        for c in data["citations"]:
            with st.expander(f"`{c['chunk_id']}` — {c['source']} › {c['section']}"):
                st.write(c["quote"] + "…")

    st.subheader("Retrieved chunks")
    rows = [
        {
            "chunk": rc["chunk"]["id"],
            "source": rc["chunk"]["source"],
            "section": rc["chunk"]["section"],
            "score": round(rc["score"], 3),
            "dense rank": rc["dense_rank"],
            "bm25 rank": rc["sparse_rank"],
        }
        for rc in data["retrieved"]
    ]
    st.dataframe(rows, use_container_width=True)
    with st.expander("Stage timings (ms)"):
        st.json(data["timings_ms"])
