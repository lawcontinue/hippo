"""
Hippo VectorStore — lightweight vector search backed by SQLite.

Modes:
  - sparse (default): BM25 only, zero external dependencies beyond numpy
  - dense: cosine similarity via sentence-transformers
  - hybrid: RRF fusion of BM25 + dense

Dependencies: numpy, sqlite3 (stdlib)
Optional: sentence-transformers (for dense/hybrid modes)
"""

from __future__ import annotations

import json as _json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .bm25 import BM25Index
from .tokenizer import default_tokenizer

__all__ = ["VectorStore", "Document"]


@dataclass
class Document:
    id: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    score: float = 0.0


def _get_engine(embedding_engine=None):
    """Lazy import EmbeddingEngine to avoid requiring sentence-transformers for sparse mode."""
    if embedding_engine is not None:
        return embedding_engine
    from .engine import EmbeddingEngine
    return EmbeddingEngine()


def _get_blob_helpers():
    """Lazy import blob helpers."""
    from .engine import blob_to_vector, vector_to_blob
    return blob_to_vector, vector_to_blob


class VectorStore:
    """
    SQLite-backed vector store with in-memory index.

    mode: "sparse" (default, BM25 only, zero deps), "dense", "hybrid" (BM25+dense RRF)

    sparse mode needs only numpy. dense/hybrid need sentence-transformers
    (install with ``pip install hippo-llm[embedding]``).
    """

    def __init__(
        self,
        db_path: str = "vectorstore.db",
        embedding_engine: Optional[Any] = None,  # EmbeddingEngine, lazy imported
        mode: str = "sparse",
        tokenizer: Callable = None,
    ):
        self.db_path = db_path
        self.mode = mode
        self._engine = None  # lazy init
        self._engine_arg = embedding_engine  # stored for lazy init

        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._entries: List[tuple] = []  # (id, text, metadata_json, vec_or_none)
        self._entry_map: Dict[int, tuple] = {}  # doc_id → entry tuple

        # For sparse mode, no embedding column needed
        self._use_embedding = mode != "sparse"

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

        # Warn if hybrid mode but most docs lack embeddings (e.g. sparse→hybrid upgrade)
        if self._use_embedding and self._entries:
            null_count = sum(1 for _, _, _, vec in self._entries if vec is None)
            if null_count > len(self._entries) * 0.5:
                import warnings
                warnings.warn(
                    f"{null_count}/{len(self._entries)} documents have no embedding vectors. "
                    f"Hybrid search degrades to sparse-only for those docs. "
                    f"Run store.rebuild_embeddings(engine) to generate embeddings.",
                    UserWarning,
                    stacklevel=2,
                )

    @property
    def engine(self):
        """Lazy-load embedding engine only when needed (dense/hybrid modes)."""
        if self._engine is None and self._use_embedding:
            self._engine = _get_engine(self._engine_arg)
        return self._engine

    # ---- internal ----

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            if self._use_embedding:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        embedding BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Migration: sparse→dense/hybrid upgrade — add embedding column if missing
            if self._use_embedding:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()]
                if 'embedding' not in cols:
                    conn.execute("ALTER TABLE documents ADD COLUMN embedding BLOB")

    def _load_all(self):
        self._entries.clear()
        blob_to_vector, _ = _get_blob_helpers() if self._use_embedding else (None, None)
        with sqlite3.connect(self.db_path) as conn:
            if self._use_embedding:
                for row in conn.execute("SELECT id, text, metadata, embedding FROM documents"):
                    mid, text, meta_json, blob = row
                    vec = blob_to_vector(blob) if blob else None
                    self._entries.append((mid, text, meta_json, vec))
                    self._entry_map[mid] = self._entries[-1]
            else:
                for row in conn.execute("SELECT id, text, metadata FROM documents"):
                    mid, text, meta_json = row
                    self._entries.append((mid, text, meta_json, None))
                    self._entry_map[mid] = self._entries[-1]

    def _rrf_fuse(self, dense_results: List[Document], sparse_results: List[tuple],
                  k: int = 60, dense_weight: float = 1.0, sparse_weight: float = 1.0) -> List[Document]:
        """Reciprocal Rank Fusion with configurable dense/sparse weights."""
        scores: Dict[int, float] = {}
        doc_map: Dict[int, Document] = {}

        for rank, doc in enumerate(dense_results):
            scores[doc.id] = scores.get(doc.id, 0) + dense_weight / (k + rank + 1)
            doc_map[doc.id] = doc

        for rank, (doc_id_str, _) in enumerate(sparse_results):
            did = int(doc_id_str)
            scores[did] = scores.get(did, 0) + sparse_weight / (k + rank + 1)
            if did not in doc_map:
                entry = self._entry_map.get(did)
                if entry:
                    doc_map[did] = Document(id=did, text=entry[1], metadata=_json_loads(entry[2]))

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
        meta_json = _json_dumps(metadata)

        if self._use_embedding:
            _, vector_to_blob = _get_blob_helpers()
            vec = self.engine.embed(text)
            blob = vector_to_blob(vec)
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "INSERT INTO documents (text, metadata, embedding) VALUES (?, ?, ?)",
                    (text, meta_json, blob),
                )
                doc_id = cur.lastrowid
            self._entries.append((doc_id, text, meta_json, vec))
        else:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "INSERT INTO documents (text, metadata) VALUES (?, ?)",
                    (text, meta_json),
                )
                doc_id = cur.lastrowid
            self._entries.append((doc_id, text, meta_json, None))

        self._entry_map[doc_id] = self._entries[-1]
        if self._bm25 is not None:
            self._bm25.add(str(doc_id), text)
        return doc_id

    def add_batch(self, items: List[Any], engine: Any = None) -> List[int]:
        """Add multiple documents.

        *items* format:
          - list of dicts: [{"text": "...", "metadata": {...}}, ...]
          - list of tuples: [("text", metadata_dict), ...]
        """
        # Normalize to (text, metadata) tuples
        normalized = []
        for item in items:
            if isinstance(item, dict):
                normalized.append((item.get("text", ""), item.get("metadata", {})))
            elif isinstance(item, (list, tuple)):
                text = item[0]
                meta = item[1] if len(item) > 1 else {}
                normalized.append((text, meta))
            else:
                normalized.append((str(item), {}))

        ids: List[int] = []

        if self._use_embedding:
            _, vector_to_blob = _get_blob_helpers()
            # Use provided engine or self.engine
            eng = engine or self.engine
            texts = [t for t, _ in normalized]
            vecs = eng.embed_batch(texts)
            with sqlite3.connect(self.db_path) as conn:
                for i, (text, meta) in enumerate(normalized):
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
        else:
            with sqlite3.connect(self.db_path) as conn:
                for text, meta in normalized:
                    meta_json = _json_dumps(meta or {})
                    cur = conn.execute(
                        "INSERT INTO documents (text, metadata) VALUES (?, ?)",
                        (text, meta_json),
                    )
                    doc_id = cur.lastrowid
                    ids.append(doc_id)
                    self._entries.append((doc_id, text, meta_json, None))
                    self._entry_map[doc_id] = self._entries[-1]

        # BM25 index
        if self._bm25 is not None:
            for i, (text, _) in enumerate(normalized):
                self._bm25.add(str(ids[i]), text)

        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
        engine: Any = None,
    ) -> List[Document]:
        """
        Search by mode:
          - sparse (default): BM25
          - dense: cosine similarity
          - hybrid: RRF fusion of dense + sparse
        """
        if self.mode == "sparse":
            return self._search_sparse(query, top_k)
        elif self.mode == "hybrid":
            return self._search_hybrid(query, top_k, threshold, filter, engine)
        else:
            return self._search_dense(query, top_k, threshold, filter, engine)

    def _search_dense(self, query, top_k, threshold, filter, engine=None):
        eng = engine or self.engine
        qvec = eng.embed(query)
        results: List[Document] = []
        for mid, text, meta_json, vec in self._entries:
            if vec is None:
                continue
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

    def _search_hybrid(self, query, top_k, threshold, filter, engine=None):
        dense = self._search_dense(query, top_k * 2, threshold, filter, engine)
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

    def rebuild_embeddings(self, engine=None) -> int:
        """Generate embeddings for documents that lack them (e.g. after sparse→hybrid upgrade).

        Returns the number of documents re-embedded.
        """
        if not self._use_embedding:
            raise ValueError("rebuild_embeddings() requires dense or hybrid mode")

        eng = engine or self.engine
        _, vector_to_blob = _get_blob_helpers()

        # Find docs with NULL embeddings
        to_embed = [(mid, text) for mid, text, _, vec in self._entries if vec is None]
        if not to_embed:
            return 0

        texts = [t for _, t in to_embed]
        vecs = eng.embed_batch(texts)

        with sqlite3.connect(self.db_path) as conn:
            for i, (mid, _) in enumerate(to_embed):
                blob = vector_to_blob(vecs[i])
                conn.execute("UPDATE documents SET embedding = ? WHERE id = ?", (blob, mid))

        # Reload
        self._load_all()
        return len(to_embed)


# ---- JSON helpers ----

def _json_dumps(obj):
    return _json.dumps(obj, ensure_ascii=False)

def _json_loads(s):
    try:
        return _json.loads(s)
    except Exception:
        return {}
