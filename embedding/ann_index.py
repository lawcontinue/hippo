"""
Hippo ANN Index — approximate nearest neighbor with hnswlib or numpy fallback.

Dependencies: numpy (required), hnswlib (optional)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    import hnswlib
    HAS_HNSW = True
except ImportError:
    HAS_HNSW = False

__all__ = ["ANNIndex", "HAS_HNSW"]


class ANNIndex:
    """ANN index using hnswlib if available, else numpy linear scan."""

    def __init__(self, dim: int, max_elements: int = 10000, metric: str = "cosine"):
        self._dim = dim
        self._max_elements = max_elements
        self._metric = metric
        self._id_map: dict = {}  # internal_idx → doc_id
        self._reverse_map: dict = {}  # doc_id → internal_idx
        self._next_idx = 0

        if HAS_HNSW:
            space = "cosine" if metric == "cosine" else "l2"
            self._index = hnswlib.Index(space=space, dim=dim)
            self._index.init_index(max_elements=max_elements, ef_construction=200, M=16)
            self._index.set_ef(50)
        else:
            self._vectors: List[np.ndarray] = []
            self._deleted: set = set()  # indices marked as deleted

    def add(self, vec: np.ndarray, doc_id: int) -> None:
        """Add a vector with associated doc_id."""
        if len(vec) != self._dim:
            raise ValueError(f"Dimension mismatch: expected {self._dim}, got {len(vec)}")
        if HAS_HNSW:
            if self._next_idx >= self._max_elements:
                self._index.resize_index(self._max_elements * 2)
                self._max_elements *= 2
            self._index.add_items(vec.reshape(1, -1), [self._next_idx])
            self._id_map[self._next_idx] = doc_id
            self._reverse_map[doc_id] = self._next_idx
            self._next_idx += 1
        else:
            idx = len(self._vectors)
            self._vectors.append(vec.copy())
            self._id_map[idx] = doc_id
            self._reverse_map[doc_id] = idx

    def search(self, vec: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return list of (doc_id, distance) sorted by similarity."""
        if self._next_idx == 0 and not self._vectors:
            return []

        if HAS_HNSW:
            labels, distances = self._index.knn_query(vec.reshape(1, -1), k=min(top_k, self._next_idx))
            return [(self._id_map[int(l)], float(d)) for l, d in zip(labels[0], distances[0])]
        else:
            if not self._vectors:
                return []
            mat = np.array(self._vectors)
            if self._metric == "cosine":
                scores = mat @ vec
            else:
                scores = -np.linalg.norm(mat - vec, axis=1)
            # skip deleted entries
            valid = [(self._id_map[int(i)], float(scores[i])) for i in range(len(scores)) if i not in self._deleted and i in self._id_map]
            valid.sort(key=lambda x: x[1], reverse=True)
            return valid[:top_k]

    def delete(self, doc_id: int) -> bool:
        """Mark a document as deleted."""
        if doc_id not in self._reverse_map:
            return False
        if HAS_HNSW:
            internal = self._reverse_map.pop(doc_id)
            self._index.mark_deleted(internal)
            del self._id_map[internal]
        else:
            internal = self._reverse_map.pop(doc_id)
            self._deleted.add(internal)
            del self._id_map[internal]
            # compact if >50% deleted
            if self._deleted and len(self._deleted) > len(self._vectors) // 2:
                self._compact()
        return True

    def count(self) -> int:
        """Return number of active (non-deleted) vectors."""
        if HAS_HNSW:
            return self._next_idx
        return len(self._vectors) - len(self._deleted)

    def _compact(self) -> None:
        """Rebuild vectors/id_map to remove deleted entries (numpy fallback only)."""
        new_vectors = []
        new_id_map = {}
        new_reverse_map = {}
        for i, vec in enumerate(self._vectors):
            if i not in self._deleted and i in self._id_map:
                new_idx = len(new_vectors)
                doc_id = self._id_map[i]
                new_vectors.append(vec)
                new_id_map[new_idx] = doc_id
                new_reverse_map[doc_id] = new_idx
        self._vectors = new_vectors
        self._id_map = new_id_map
        self._reverse_map = new_reverse_map
        self._deleted.clear()
