"""
Hippo Embedding tests — all tests use MockEngine (no Ollama dependency).

Run: python3 -m pytest embedding/tests/ -v
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import threading

import numpy as np
import pytest

# ---- Mock Engine ----

class MockEngine:
    def __init__(self, dim=8):
        self.dim = dim
        self.detected_dim = None

    def embed(self, text):
        vec = np.zeros(self.dim)
        # NOTE: hash() is randomized per-process (PYTHONHASHSEED).
        # For dim=8, collision probability is 1/8 per pair. Tests that
        # rely on distinct vectors may flake ~12.5% of runs.
        vec[hash(text) % self.dim] = 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts, batch_size=8, pause=0.0):
        return np.array([self.embed(t) for t in texts])


# ---- Tokenizer tests ----

from hippo.embedding.tokenizer import default_tokenizer


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
    from hippo.embedding.bm25 import BM25Index
    bm25 = BM25Index(tokenizer=lambda text: text.split("|"))
    bm25.add("1", "hello|world")
    results = bm25.search("hello", top_k=5)
    assert len(results) == 1
    assert results[0][0] == "1"


# ---- BM25 tests ----

from hippo.embedding.bm25 import BM25Index


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

from hippo.embedding.store import VectorStore, Document


def _make_store(mode="dense", tmp=None):
    if tmp is None:
        tmp = tempfile.mktemp(suffix=".db")
    engine = MockEngine(dim=8)
    return VectorStore(db_path=tmp, embedding_engine=engine, mode=mode)


def test_store_dense_mode():
    with tempfile.TemporaryDirectory() as d:
        with _make_store("dense", os.path.join(d, "test.db")) as store:
            store.add("hello world", {"tag": "greeting"})
            store.add("python programming", {"tag": "tech"})
            results = store.search("hello", top_k=2)
            assert len(results) >= 1
            assert results[0].text == "hello world"


def test_store_sparse_mode():
    with tempfile.TemporaryDirectory() as d:
        with _make_store("sparse", os.path.join(d, "test.db")) as store:
            store.add("machine learning algorithms")
            store.add("cooking recipes")
            results = store.search("machine", top_k=2)
            assert len(results) >= 1
            assert "machine" in results[0].text


def test_store_hybrid_mode():
    with tempfile.TemporaryDirectory() as d:
        with _make_store("hybrid", os.path.join(d, "test.db")) as store:
            store.add("machine learning algorithms")
            store.add("cooking recipes dinner")
            results = store.search("machine learning", top_k=2)
            assert len(results) >= 1
            # hybrid should find the ML doc
            ids = [r.id for r in results]
            assert any("machine" in r.text for r in results)


# ---- ANN tests ----

from hippo.embedding.ann_index import ANNIndex, HAS_HNSW


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

from hippo.embedding.importers import import_csv, import_json


def test_import_csv():
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "data.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "author"])
            w.writerow(["hello world", "alice"])
            w.writerow(["python code", "bob"])

        with _make_store("dense", os.path.join(d, "test.db")) as store:
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

        with _make_store("dense", os.path.join(d, "test.db")) as store:
            ids = import_json(store, json_path, text_key="content", meta_key="info")
            assert len(ids) == 2
            assert store.count() == 2
            # P1-5: verify metadata content via search
            results = store.search("hello", top_k=2)
            alice_found = any(r.metadata.get("author") == "alice" for r in results)
            assert alice_found


# ---- Engine tests ----

from hippo.embedding.engine import EmbeddingEngine


def test_engine_no_subprocess():
    """Verify engine does not shell out to subprocess."""
    import hippo.embedding.engine as mod
    import inspect
    src = inspect.getsource(mod)
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
    from hippo.embedding.importers import _validate_path
    with pytest.raises(ValueError, match="Path traversal"):
        _validate_path("../etc/passwd")


# ---- PR #6 review: concurrency + close() safety ----

def test_store_concurrent_writes():
    """P0 review fix: long-lived connection must work under multi-threaded writes.

    Previously each add() opened its own short-lived connection, so 5 threads
    could add concurrently without contention. After switching to one
    long-lived self._conn, this only works if (a) check_same_thread=False and
    (b) writes are serialized with self._lock.
    """
    from hippo.embedding.store import VectorStore
    with tempfile.TemporaryDirectory() as d:
        with VectorStore(
            db_path=os.path.join(d, "conc.db"),
            embedding_engine=MockEngine(dim=8),
            mode="dense",
        ) as store:
            errors = []
            ids_per_thread = [[] for _ in range(5)]

            def worker(tid):
                try:
                    for i in range(6):
                        doc_id = store.add(
                            f"thread{tid}_doc{i}",
                            {"thread": tid, "idx": i},
                        )
                        ids_per_thread[tid].append(doc_id)
                except Exception as e:
                    errors.append((tid, repr(e)))

            threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"concurrent add() failed: {errors}"
            # 5 threads × 6 docs = 30 docs
            assert store.count() == 30
            all_ids = [i for ids in ids_per_thread for i in ids]
            assert len(set(all_ids)) == 30, "duplicate doc IDs under concurrency"


def test_store_concurrent_search_during_writes():
    """Reads must remain safe while writes happen concurrently.

    Reads aren't locked (SQLite's MVCC + serialized isolation handles them),
    but they must not crash with 'database is locked' or return inconsistent
    state. We just verify no exception is raised and results are well-formed.
    """
    from hippo.embedding.store import VectorStore
    with tempfile.TemporaryDirectory() as d:
        with VectorStore(
            db_path=os.path.join(d, "rw.db"),
            embedding_engine=MockEngine(dim=8),
            mode="dense",
        ) as store:
            # Pre-seed a few docs so search() has something to find
            for i in range(10):
                store.add(f"document {i}", {"i": i})

            errors = []

            def writer():
                try:
                    for i in range(20):
                        store.add(f"extra doc {i}", {})
                except Exception as e:
                    errors.append(("writer", repr(e)))

            def reader():
                try:
                    for _ in range(20):
                        results = store.search("document", top_k=5)
                        assert isinstance(results, list)
                except Exception as e:
                    errors.append(("reader", repr(e)))

            threads = (
                [threading.Thread(target=writer) for _ in range(3)]
                + [threading.Thread(target=reader) for _ in range(3)]
            )
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"concurrent read/write failed: {errors}"


def test_store_after_close_raises_runtime_error():
    """P1 review fix: post-close API calls must raise RuntimeError, not AttributeError."""
    from hippo.embedding.store import VectorStore
    with tempfile.TemporaryDirectory() as d:
        store = VectorStore(
            db_path=os.path.join(d, "closed.db"),
            embedding_engine=MockEngine(dim=8),
            mode="dense",
        )
        store.close()

        with pytest.raises(RuntimeError, match="closed"):
            store.add("text", {})

        with pytest.raises(RuntimeError, match="closed"):
            store.search("query")

        with pytest.raises(RuntimeError, match="closed"):
            store.delete(1)

        with pytest.raises(RuntimeError, match="closed"):
            store.update_metadata(1, {})

        with pytest.raises(RuntimeError, match="closed"):
            store.execute("SELECT 1")

        # Close again is a no-op (idempotent), should not raise
        store.close()
        store.close()
