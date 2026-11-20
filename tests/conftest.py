"""Test fixtures: an in-memory Qdrant index over a tiny synthetic corpus.

Real embedding models are small and cached by fastembed, so retrieval tests use them for
realism; LLM calls always go through the deterministic FakeProvider.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from askdocs.config import Settings
from askdocs.index import DocIndex
from askdocs.models import Chunk

CORPUS = [
    Chunk(
        id="configmap-000001",
        source="configmap.md",
        title="ConfigMaps",
        section="Overview",
        text="The data stored in a ConfigMap cannot exceed 1 MiB. Use a volume or a database for larger settings.",
    ),
    Chunk(
        id="secret-000002",
        source="secret.md",
        title="Secrets",
        section="Overview",
        text="A Secret is an object that contains a small amount of sensitive data such as a password, a token, or a key.",
    ),
    Chunk(
        id="probes-000003",
        source="probes.md",
        title="Liveness, Readiness and Startup Probes",
        section="Liveness probe",
        text="Liveness probes determine when to restart a container. The kubelet restarts a container whose liveness probe fails.",
    ),
    Chunk(
        id="hpa-000004",
        source="hpa.md",
        title="Horizontal Pod Autoscaling",
        section="How it works",
        text="The HorizontalPodAutoscaler automatically updates a workload resource such as a Deployment to scale it to match demand.",
    ),
    Chunk(
        id="pv-000005",
        source="persistent-volumes.md",
        title="Persistent Volumes",
        section="Reclaim policy",
        text="The Retain reclaim policy allows for manual reclamation of the resource. Delete removes the PersistentVolume object.",
    ),
]


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(llm_provider="fake", collection="test", qdrant_url=None, _env_file=None)  # type: ignore[call-arg]


@pytest.fixture(scope="session")
def index(settings: Settings) -> DocIndex:
    idx = DocIndex(settings, client=QdrantClient(":memory:"))
    idx.recreate()
    idx.upsert(CORPUS)
    return idx
