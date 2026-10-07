import os
import sys

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import Settings  # noqa: E402
from app.embeddings import HashEmbedder  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture()
def settings() -> Settings:
    return Settings(embedding_provider="hash", collection_prefix="test", chunk_size=300,
                    chunk_overlap=50, score_threshold=0.0, qdrant_connect_retries=1,
                    log_level="WARNING", max_upload_mb=1, embedding_model="test/model-a",
                    embedding_models="test/model-a,test/model-b")


@pytest.fixture()
def client(settings):
    """A full app wired to an in-memory Qdrant and the hash embedder.
    Using TestClient as a context manager runs the lifespan (startup/shutdown)."""
    app = create_app(settings, embedder=HashEmbedder(384),
                     qdrant=AsyncQdrantClient(location=":memory:"))
    with TestClient(app) as c:
        yield c
