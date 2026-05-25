"""
Hippo VectorStore — lightweight vector search backed by SQLite.

Features:
  - Store documents with metadata + embedding vectors
  - Cosine similarity search (dot product on L2-normalized vectors)
  - Filter by arbitrary metadata key/value pairs
  - Full in-memory index for fast queries (<1 ms on thousands of entries)
  - Persistent SQLite storage
"""

from __future__ import annotations

import json as _json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .engine import EmbeddingEngine, blob_to_vector, vector_to_blob

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

    Usage::

        store = VectorStore("/tmp/my_store.db")
        store.add("Hello world", {"source": "greeting"})
        results = store.search("hi", top_k=3)
    """

    def __init__(
        self,
        db_path: str = "vectorstore.db",
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        self.db_path = db_path
        self.engine = embedding_engine or EmbeddingEngine()
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._entries: List[tuple] = []  # (id, text, metadata_json, vec)
        self._init_db()
        self._load_all()

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
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Search by cosine similarity.

        Args:
            query: Query text.
            top_k: Max results.
            threshold: Minimum similarity score.
            filter: Metadata key/value pairs to filter on.

        Returns:
            List of Document sorted by score descending.
        """
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

    def delete(self, doc_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._entries = [e for e in self._entries if e[0] != doc_id]
        return True

    def count(self) -> int:
        return len(self._entries)

    def rebuild(self) -> int:
        """Reload index from disk."""
        self._load_all()
        return len(self._entries)


# ---- JSON helpers ----

def _json_dumps(obj):
    return _json.dumps(obj, ensure_ascii=False)

def _json_loads(s):
    try:
        return _json.loads(s)
    except Exception:
        return {}
