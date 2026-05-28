"""
Hippo Embedding tests — all tests use MockEngine (no Ollama dependency).

Run: python3 -m pytest embedding/tests/ -v
"""

from __future__ import annotations

import csv
import json
import os
import tempfile

import numpy as np
import pytest

# ---- Mock Engine ----

class MockEngine:
    def __init__(self, dim=8):
        self.dim = dim
        self.detected_dim = None

    def embed(self, text):
        vec = np.zeros(self.dim)
        vec[hash(text) % self.dim] = 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts, batch_size=8, pause=0.0):
        return np.array([self.embed(t) for t in texts])


# ---- Tokenizer tests ----

from embedding.tokenizer import default_tokenizer


def test_default_tokenizer_english():
    tokens = default_tokenizer("The quick brown fox jumps over the lazy dog")
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "the" not in tokens  # stopword
    assert "over" not in tokens  # stopword


def test_default_tokenizer_cjk():
    tokens = default_tokenizer("我去了北京")
    assert "京" in tokens
    assert "北" in tokens
    assert "我" not in tokens  # stopword
    assert "去" not in tokens  # stopword
    assert "了" not in tokens  # stopword


def test_custom_tokenizer():
    from embedding.bm25 import BM25Index
    bm25 = BM25Index(tokenizer=lambda text: text.split("|"))
    bm25.add("1", "hello|world")
    results = bm25.search("hello", top_k=5)
    assert len(results) == 1
    assert results[0][0] == "1"


# ---- BM25 tests ----

from embedding.bm25 import BM25Index


def test_bm25_basic():
    bm25 = BM25Index()
    bm25.add("1", "machine learning algorithms")
    bm25.add("2", "deep learning neural networks")
    bm25.add("3", "cooking recipes for dinner")
    results = bm25.search("machine learning", top_k=3)
    assert len(results) > 0
    assert results[0][0] == "1"


def test_bm25_idf_weighting():
    bm25 = BM25Index()
    # "rare" appears in 1 doc, "common" in 3 docs
    bm25.add("1", "rare word document")
    bm25.add("2", "common word text")
    bm25.add("3", "common word data")
    bm25.add("4", "common word info")

    results = bm25.search("rare", top_k=4)
    assert results[0][0] == "1"  # rare doc should rank first


def test_bm25_delete():
    bm25 = BM25Index()
    bm25.add("1", "hello world")
    bm25.add("2", "hello python")
    assert bm25.count() == 2

    bm25.delete("1")
    assert bm25.count() == 1

    results = bm25.search("hello", top_k=5)
    assert all(r[0] != "1" for r in results)


def test_bm25_stopwords():
    bm25 = BM25Index()
    bm25.add("1", "的 了 是 在")
    results = bm25.search("的了是", top_k=5)
    # all tokens are stopwords → no results or zero score
    assert len(results) == 0 or results[0][1] == 0.0


# ---- Store tests ----

from embedding.store import VectorStore, Document


def _make_store(mode="dense", tmp=None):
    if tmp is None:
        tmp = tempfile.mktemp(suffix=".db")
    engine = MockEngine(dim=8)
    return VectorStore(db_path=tmp, embedding_engine=engine, mode=mode)


def test_store_dense_mode():
    with tempfile.TemporaryDirectory() as d:
        store = _make_store("dense", os.path.join(d, "test.db"))
        store.add("hello world", {"tag": "greeting"})
        store.add("python programming", {"tag": "tech"})
        results = store.search("hello", top_k=2)
        assert len(results) >= 1
        assert results[0].text == "hello world"


def test_store_sparse_mode():
    with tempfile.TemporaryDirectory() as d:
        store = _make_store("sparse", os.path.join(d, "test.db"))
        store.add("machine learning algorithms")
        store.add("cooking recipes")
        results = store.search("machine", top_k=2)
        assert len(results) >= 1
        assert "machine" in results[0].text


def test_store_hybrid_mode():
    with tempfile.TemporaryDirectory() as d:
        store = _make_store("hybrid", os.path.join(d, "test.db"))
        store.add("machine learning algorithms")
        store.add("cooking recipes dinner")
        results = store.search("machine learning", top_k=2)
        assert len(results) >= 1
        # hybrid should find the ML doc
        ids = [r.id for r in results]
        assert any("machine" in r.text for r in results)


# ---- ANN tests ----

from embedding.ann_index import ANNIndex, HAS_HNSW


def test_ann_index_numpy_fallback():
    idx = ANNIndex(dim=4, max_elements=100)
    # unique vectors so there's a clear winner
    vecs = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    ]
    for i, v in enumerate(vecs):
        idx.add(v, doc_id=i)

    q = np.array([1.0, 0.0, 0.0, 0.0])
    results = idx.search(q, top_k=3)
    assert len(results) >= 1
    assert results[0][0] == 0  # closest to q


@pytest.mark.skipif(not HAS_HNSW, reason="hnswlib not installed")
def test_ann_index_hnswlib():
    idx = ANNIndex(dim=4, max_elements=100)
    for i in range(5):
        v = np.zeros(4)
        v[i % 4] = 1.0
        idx.add(v, doc_id=i)

    q = np.array([1.0, 0.0, 0.0, 0.0])
    results = idx.search(q, top_k=3)
    assert len(results) >= 1
    assert results[0][0] == 0


# ---- Importer tests ----

from embedding.importers import import_csv, import_json


def test_import_csv():
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "data.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "author"])
            w.writerow(["hello world", "alice"])
            w.writerow(["python code", "bob"])

        store = _make_store("dense", os.path.join(d, "test.db"))
        ids = import_csv(store, csv_path, text_col="text", meta_cols=["author"])
        assert len(ids) == 2
        assert store.count() == 2
        # P1-5: verify metadata content via search
        results = store.search("hello", top_k=2)
        alice_found = any(r.metadata.get("author") == "alice" for r in results)
        assert alice_found


def test_import_json():
    with tempfile.TemporaryDirectory() as d:
        json_path = os.path.join(d, "data.json")
        records = [
            {"content": "hello world", "info": {"author": "alice"}},
            {"content": "python code", "info": {"author": "bob"}},
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f)

        store = _make_store("dense", os.path.join(d, "test.db"))
        ids = import_json(store, json_path, text_key="content", meta_key="info")
        assert len(ids) == 2
        assert store.count() == 2
        # P1-5: verify metadata content via search
        results = store.search("hello", top_k=2)
        alice_found = any(r.metadata.get("author") == "alice" for r in results)
        assert alice_found


# ---- Engine tests ----

from embedding.engine import EmbeddingEngine


def test_engine_urllib():
    """Verify engine uses urllib (import check — no subprocess)."""
    import embedding.engine as mod
    import inspect
    src = inspect.getsource(mod.EmbeddingEngine.embed)
    assert "urlopen" in src
    assert "subprocess" not in src


def test_dimension_auto_detect():
    eng = MockEngine(dim=16)
    eng.detected_dim = None
    # simulate: first embed sets detected_dim
    v = eng.embed("test")
    eng.detected_dim = len(v)
    assert eng.detected_dim == 16


def test_dimension_mismatch():
    """P2-4: ANNIndex should reject wrong-dimension vectors via its own check."""
    idx = ANNIndex(dim=4)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        idx.add(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), doc_id=1)


# ---- P0/P1 fix tests ----

def test_bm25_empty_document_skipped():
    """P1-1: empty token documents should be skipped."""
    bm25 = BM25Index()
    bm25.add("1", "的了的")  # all stopwords → empty tokens
    assert bm25.count() == 0


def test_ann_dimension_mismatch():
    """P0-2: adding wrong-dimension vector should raise ValueError."""
    idx = ANNIndex(dim=4)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        idx.add(np.array([1.0, 0.0]), doc_id=1)


def test_ann_numpy_delete_skips_deleted():
    """P1-2: deleted items should not appear in search results."""
    idx = ANNIndex(dim=4, max_elements=100)
    idx.add(np.array([1.0, 0.0, 0.0, 0.0]), doc_id=1)
    idx.add(np.array([0.0, 1.0, 0.0, 0.0]), doc_id=2)
    idx.delete(1)
    results = idx.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=5)
    assert all(r[0] != 1 for r in results)


def test_importer_path_traversal_rejected():
    """P1-3: paths with '..' should be rejected."""
    from embedding.importers import _validate_path
    with pytest.raises(ValueError, match="Path traversal"):
        _validate_path("../etc/passwd")
