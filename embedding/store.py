"""
Hippo VectorStore — lightweight vector search backed by SQLite.

Changed: added mode parameter (dense/hybrid/sparse), BM25 integration, RRF fusion.
Dependencies: numpy, sqlite3 (stdlib)
"""

from __future__ import annotations

import json as _json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .bm25 import BM25Index
from .engine import EmbeddingEngine, blob_to_vector, vector_to_blob
from .tokenizer import default_tokenizer

__all__ = ["VectorStore", "Document"]


@dataclass
class Document:
    id: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    score: float = 0.0


class VectorStore:
    """
    SQLite-backed vector store with in-memory index.

    mode: "dense" (default), "hybrid" (BM25+dense RRF), "sparse" (pure BM25)
    """

    def __init__(
        self,
        db_path: str = "vectorstore.db",
        embedding_engine: Optional[EmbeddingEngine] = None,
        mode: str = "dense",
        tokenizer: Callable = None,
    ):
        self.db_path = db_path
        self.engine = embedding_engine or EmbeddingEngine()
        self.mode = mode
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._entries: List[tuple] = []  # (id, text, metadata_json, vec)
        self._entry_map: Dict[int, tuple] = {}  # doc_id → entry tuple
        self._init_db()
        self._load_all()

        # BM25 for hybrid/sparse modes
        self._bm25: Optional[BM25Index] = None
        if mode != "dense":
            tok = tokenizer or default_tokenizer
            self._bm25 = BM25Index(tokenizer=tok)
            # reindex existing docs into BM25
            for doc_id, text, _, _ in self._entries:
                self._bm25.add(str(doc_id), text)

    # ---- internal ----

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _load_all(self):
        self._entries.clear()
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute("SELECT id, text, metadata, embedding FROM documents"):
                mid, text, meta_json, blob = row
                vec = blob_to_vector(blob)
                self._entries.append((mid, text, meta_json, vec))
                self._entry_map[mid] = self._entries[-1]

    def _rrf_fuse(self, dense_results: List[Document], sparse_results: List[tuple],
                  k: int = 60, dense_weight: float = 1.0, sparse_weight: float = 1.0) -> List[Document]:
        """Reciprocal Rank Fusion with configurable dense/sparse weights.

        RRF score = dense_weight / (k + rank_dense + 1) + sparse_weight / (k + rank_sparse + 1)
        """
        scores: Dict[int, float] = {}
        doc_map: Dict[int, Document] = {}

        for rank, doc in enumerate(dense_results):
            scores[doc.id] = scores.get(doc.id, 0) + dense_weight / (k + rank + 1)
            doc_map[doc.id] = doc

        # sparse_results = [(doc_id_str, score), ...]
        for rank, (doc_id_str, _) in enumerate(sparse_results):
            did = int(doc_id_str)
            scores[did] = scores.get(did, 0) + sparse_weight / (k + rank + 1)
            if did not in doc_map:
                # find in entries
                for mid, text, meta_json, vec in self._entries:
                    if mid == did:
                        doc_map[did] = Document(id=mid, text=text, metadata=_json_loads(meta_json))
                        break

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results: List[Document] = []
        for doc_id, score in ranked:
            doc = doc_map.get(doc_id)
            if doc:
                doc.score = round(score, 6)
                results.append(doc)
        return results

    # ---- public API ----

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a document and return its ID."""
        metadata = metadata or {}
        vec = self.engine.embed(text)
        meta_json = _json_dumps(metadata)
        blob = vector_to_blob(vec)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO documents (text, metadata, embedding) VALUES (?, ?, ?)",
                (text, meta_json, blob),
            )
            doc_id = cur.lastrowid
        self._entries.append((doc_id, text, meta_json, vec))
        self._entry_map[doc_id] = self._entries[-1]

        if self._bm25 is not None:
            self._bm25.add(str(doc_id), text)

        return doc_id

    def add_batch(self, items: List[tuple]) -> List[int]:
        """Add multiple documents. *items* = [(text, metadata), ...]."""
        texts = [t for t, _ in items]
        vecs = self.engine.embed_batch(texts)
        ids: List[int] = []
        with sqlite3.connect(self.db_path) as conn:
            for i, (text, meta) in enumerate(items):
                meta_json = _json_dumps(meta or {})
                blob = vector_to_blob(vecs[i])
                cur = conn.execute(
                    "INSERT INTO documents (text, metadata, embedding) VALUES (?, ?, ?)",
                    (text, meta_json, blob),
                )
                doc_id = cur.lastrowid
                ids.append(doc_id)
                self._entries.append((doc_id, text, meta_json, vecs[i]))
                self._entry_map[doc_id] = self._entries[-1]
                if self._bm25 is not None:
                    self._bm25.add(str(doc_id), text)
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Search by mode:
          - dense: cosine similarity
          - sparse: BM25
          - hybrid: RRF fusion of dense + sparse
        """
        if self.mode == "sparse":
            return self._search_sparse(query, top_k)
        elif self.mode == "hybrid":
            return self._search_hybrid(query, top_k, threshold, filter)
        else:
            return self._search_dense(query, top_k, threshold, filter)

    def _search_dense(self, query, top_k, threshold, filter):
        qvec = self.engine.embed(query)
        results: List[Document] = []
        for mid, text, meta_json, vec in self._entries:
            if filter:
                meta = _json_loads(meta_json)
                if not all(meta.get(k) == v for k, v in filter.items()):
                    continue
            score = float(np.dot(qvec, vec))
            if score >= threshold:
                results.append(Document(
                    id=mid, text=text,
                    metadata=_json_loads(meta_json),
                    score=round(score, 4),
                ))
        results.sort(key=lambda d: d.score, reverse=True)
        return results[:top_k]

    def _search_sparse(self, query, top_k):
        raw = self._bm25.search(query, top_k)
        results: List[Document] = []
        for doc_id_str, score in raw:
            did = int(doc_id_str)
            entry = self._entry_map.get(did)
            if entry:
                results.append(Document(
                    id=did, text=entry[1],
                    metadata=_json_loads(entry[2]),
                    score=round(score, 4),
                ))
        return results

    def _search_hybrid(self, query, top_k, threshold, filter):
        dense = self._search_dense(query, top_k * 2, threshold, filter)
        sparse = self._bm25.search(query, top_k * 2)
        fused = self._rrf_fuse(dense, sparse)
        return fused[:top_k]

    def delete(self, doc_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._entries = [e for e in self._entries if e[0] != doc_id]
        self._entry_map.pop(doc_id, None)
        if self._bm25 is not None:
            self._bm25.delete(str(doc_id))
        return True

    def count(self) -> int:
        return len(self._entries)

    def rebuild(self) -> int:
        """Reload index from disk."""
        self._load_all()
        if self._bm25 is not None:
            self._bm25 = BM25Index(tokenizer=self._bm25._tokenizer)
            for doc_id, text, _, _ in self._entries:
                self._bm25.add(str(doc_id), text)
        return len(self._entries)


# ---- JSON helpers ----

def _json_dumps(obj):
    return _json.dumps(obj, ensure_ascii=False)

def _json_loads(s):
    try:
        return _json.loads(s)
    except Exception:
        return {}
