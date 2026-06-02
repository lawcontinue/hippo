"""
Hippo ANN Index v2 — tunable HNSW parameters + numpy fallback.

Changes from v1:
- HNSW params (ef_construction, M, ef_search) are configurable
- auto_ef: set ef_search dynamically based on dataset size
- batch_add for bulk inserts
- benchmark method for parameter tuning
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import hnswlib
    HAS_HNSW = True
except ImportError:
    HAS_HNSW = False

__all__ = ["ANNIndex", "ANNConfig", "HAS_HNSW"]


@dataclass
class ANNConfig:
    """Tunable ANN parameters."""
    ef_construction: int = 200   # build-time search depth (higher = better graph, slower build)
    M: int = 16                  # connections per node (higher = better recall, more memory)
    ef_search: int = 100         # query-time search depth (higher = better recall, slower query)
    max_elements: int = 10000
    metric: str = "cosine"       # "cosine" or "l2"
    auto_ef: bool = True         # auto-scale ef_search with dataset size


class ANNIndex:
    """ANN index with tunable HNSW parameters."""

    def __init__(self, dim: int, config: Optional[ANNConfig] = None,
                 max_elements: int = 10000, metric: str = "cosine"):
        # backward compat: accept old-style args
        if config is None:
            config = ANNConfig(max_elements=max_elements, metric=metric)
        self._dim = dim
        cfg = config or ANNConfig()
        self._config = cfg
        self._max_elements = cfg.max_elements
        self._metric = cfg.metric
        self._id_map: dict = {}
        self._reverse_map: dict = {}
        self._next_idx = 0

        if HAS_HNSW:
            space = "cosine" if cfg.metric == "cosine" else "l2"
            self._index = hnswlib.Index(space=space, dim=dim)
            self._index.init_index(
                max_elements=cfg.max_elements,
                ef_construction=cfg.ef_construction,
                M=cfg.M,
            )
            self._index.set_ef(cfg.ef_search)
        else:
            self._vectors: List[np.ndarray] = []
            self._deleted: set = set()

    @property
    def config(self) -> ANNConfig:
        return self._config

    def set_ef_search(self, ef: int) -> None:
        """Adjust query-time search depth dynamically."""
        self._config.ef_search = ef
        if HAS_HNSW:
            self._index.set_ef(ef)

    def _auto_ef(self) -> None:
        """Auto-scale ef_search: sqrt(n) for small datasets, capped at n."""
        if not self._config.auto_ef:
            return
        n = self._next_idx
        if n == 0:
            return
        ef = min(max(int(n ** 0.5), 10), n)
        self.set_ef_search(ef)

    def add(self, vec: np.ndarray, doc_id: int) -> None:
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
            self._auto_ef()
        else:
            self._vectors.append(vec.copy())
            idx = len(self._vectors) - 1
            self._id_map[idx] = doc_id
            self._reverse_map[doc_id] = idx

    def add_batch(self, vecs: np.ndarray, doc_ids: List[int]) -> None:
        """Bulk insert. vecs shape: (n, dim)."""
        assert len(vecs) == len(doc_ids)
        for vec, did in zip(vecs, doc_ids):
            self.add(vec, did)

    def search(self, vec: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        if self._next_idx == 0 and not getattr(self, '_vectors', []):
            return []
        if HAS_HNSW:
            k = min(top_k, self._next_idx)
            if k == 0:
                return []
            # ensure ef_search >= k for valid query
            if self._config.ef_search < k:
                self._index.set_ef(k)
            try:
                labels, distances = self._index.knn_query(vec.reshape(1, -1), k=k)
            except RuntimeError:
                # ef or M too small, return empty
                return []
            return [(self._id_map[int(l)], float(d)) for l, d in zip(labels[0], distances[0])]
        else:
            if not self._vectors:
                return []
            mat = np.array(self._vectors)
            if self._metric == "cosine":
                scores = mat @ vec
            else:
                scores = -np.linalg.norm(mat - vec, axis=1)
            valid = [
                (self._id_map[int(i)], float(scores[i]))
                for i in range(len(scores))
                if i not in self._deleted and i in self._id_map
            ]
            valid.sort(key=lambda x: x[1], reverse=True)
            return valid[:top_k]

    def delete(self, doc_id: int) -> bool:
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
            if self._deleted and len(self._deleted) > len(self._vectors) // 2:
                self._compact()
        return True

    def count(self) -> int:
        if HAS_HNSW:
            return self._next_idx
        return len(self._vectors) - len(self._deleted)

    def benchmark(self, query_vecs: np.ndarray, query_ids: List[int],
                  top_k: int = 10) -> dict:
        """Run recall/latency benchmark against known ground truth.

        query_ids[i] = expected doc_id for query_vecs[i] (exact match).
        Returns: {recall@k, avg_latency_ms, p99_latency_ms}
        """
        hits = 0
        latencies = []
        for qvec, true_id in zip(query_vecs, query_ids):
            t0 = time.perf_counter()
            results = self.search(qvec, top_k=top_k)
            latencies.append((time.perf_counter() - t0) * 1000)
            if any(r[0] == true_id for r in results):
                hits += 1
        n = len(query_vecs)
        lats = np.array(latencies)
        return {
            f"recall@{top_k}": round(hits / n, 4) if n else 0.0,
            "avg_latency_ms": round(float(lats.mean()), 3) if n else 0.0,
            "p99_latency_ms": round(float(np.percentile(lats, 99)), 3) if n else 0.0,
            "n_queries": n,
            "ef_search": self._config.ef_search,
        }

    def _compact(self) -> None:
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
