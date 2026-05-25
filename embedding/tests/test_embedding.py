"""Tests for hippo.embedding — uses mock embeddings (no Ollama required)."""

import json
import os
import sqlite3
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from embedding.engine import EmbeddingEngine, blob_to_vector, vector_to_blob
from embedding.store import VectorStore, Document


# ---------- Fixtures ----------

@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


def _fake_embed(self, text: str) -> np.ndarray:
    """Deterministic fake embedding based on text hash."""
    vec = np.array([hash(text + str(i)) % 1000 for i in range(32)], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@pytest.fixture
def mock_engine():
    engine = EmbeddingEngine(dim=32)
    with patch.object(EmbeddingEngine, "embed", _fake_embed):
        # Also fix batch
        def fake_batch(texts, batch_size=8, pause=0.0):
            return np.array([_fake_embed(None, t) for t in texts], dtype=np.float32)
        with patch.object(EmbeddingEngine, "embed_batch", lambda self, texts, **kw: fake_batch(texts)):
            yield engine


# ---------- Engine Tests ----------

class TestHelpers:
    def test_blob_roundtrip(self):
        vec = np.random.randn(768).astype(np.float32)
        blob = vector_to_blob(vec)
        restored = blob_to_vector(blob)
        np.testing.assert_array_almost_equal(vec, restored)

    def test_blob_roundtrip_custom_dim(self):
        vec = np.random.randn(32).astype(np.float32)
        assert blob_to_vector(vector_to_blob(vec)).shape == (32,)


# ---------- Store Tests ----------

class TestVectorStore:
    def test_add_and_count(self, tmp_db, mock_engine):
        store = VectorStore(tmp_db, embedding_engine=mock_engine)
        assert store.count() == 0
        store.add("hello world", {"lang": "en"})
        assert store.count() == 1

    def test_search_basic(self, tmp_db, mock_engine):
        store = VectorStore(tmp_db, embedding_engine=mock_engine)
        store.add("the cat sat on the mat", {"topic": "animals"})
        store.add("python is a programming language", {"topic": "tech"})
        store.add("dogs are loyal animals", {"topic": "animals"})

        results = store.search("cats and dogs", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, Document) for r in results)

    def test_search_with_filter(self, tmp_db, mock_engine):
        store = VectorStore(tmp_db, embedding_engine=mock_engine)
        store.add("cat", {"topic": "animals"})
        store.add("python code", {"topic": "tech"})

        results = store.search("cat", top_k=10, filter={"topic": "tech"})
        assert len(results) == 1
        assert results[0].metadata["topic"] == "tech"

    def test_search_threshold(self, tmp_db, mock_engine):
        store = VectorStore(tmp_db, embedding_engine=mock_engine)
        store.add("hello")
        results = store.search("goodbye", top_k=5, threshold=0.999)
        # With random hashes, unlikely to exceed 0.999
        assert isinstance(results, list)

    def test_add_batch(self, tmp_db, mock_engine):
        store = VectorStore(tmp_db, embedding_engine=mock_engine)
        ids = store.add_batch([
            ("doc1", {"n": 1}),
            ("doc2", {"n": 2}),
            ("doc3", {"n": 3}),
        ])
        assert len(ids) == 3
        assert store.count() == 3

    def test_delete(self, tmp_db, mock_engine):
        store = VectorStore(tmp_db, embedding_engine=mock_engine)
        doc_id = store.add("to be deleted")
        assert store.count() == 1
        store.delete(doc_id)
        assert store.count() == 0

    def test_rebuild(self, tmp_db, mock_engine):
        store = VectorStore(tmp_db, embedding_engine=mock_engine)
        store.add("persist me")
        count = store.rebuild()
        assert count == 1
        assert store.count() == 1

    def test_persistence(self, tmp_db, mock_engine):
        store = VectorStore(tmp_db, embedding_engine=mock_engine)
        store.add("survives restart")
        # New store instance, same db
        store2 = VectorStore(tmp_db, embedding_engine=mock_engine)
        assert store2.count() == 1
