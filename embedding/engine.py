"""
Hippo Embedding Engine — generate embeddings via sentence-transformers.

Dependencies (sparse mode): numpy only
Dependencies (dense/hybrid): numpy, sentence-transformers

Changed: Ollama API → sentence-transformers local inference (2026-05-31)
         Lazy import with friendly error (2026-06-13)
"""

from __future__ import annotations

import os
import struct
import threading
from typing import Dict, List, Optional

import numpy as np

__all__ = ["EmbeddingEngine", "blob_to_vector", "vector_to_blob"]

# ---------- helpers ----------

DEFAULT_MODEL = os.environ.get("HIPPO_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
# Local path override (e.g. E:/models/modelscope_cache/Xorbits/bge-small-zh on Windows)
LOCAL_MODEL_PATH = os.environ.get("HIPPO_EMBED_MODEL_PATH", "")
DEFAULT_DIM = int(os.environ.get("HIPPO_EMBED_DIM", "512"))

# Quality-first alternative (larger, slower, higher accuracy)
# Set HIPPO_EMBED_MODEL=BAAI/bge-m3 and HIPPO_EMBED_DIM=1024 to use
QUALITY_MODEL = "BAAI/bge-m3"


def blob_to_vector(blob: bytes) -> np.ndarray:
    """Deserialize SQLite BLOB → numpy float32 vector (auto-detect dim)."""
    dim = len(blob) // 4
    return np.array(struct.unpack(f"<{dim}f", blob), dtype=np.float32)


def vector_to_blob(vec: np.ndarray) -> bytes:
    """Serialize numpy float32 vector → SQLite BLOB."""
    return struct.pack(f"<{len(vec)}f", *vec)


# ---------- EmbeddingEngine ----------

class EmbeddingEngine:
    """Generate embeddings through sentence-transformers (local, no Ollama).

    Requires ``sentence-transformers`` — install with ``pip install hippo-llm[embedding]``.
    """

    _global_model = None
    _global_lock = threading.Lock()

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        base_url: Optional[str] = None,  # ignored, kept for API compat
        cache_size: int = 512,
    ):
        self.model = model
        self.dim = dim
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self.detected_dim: Optional[int] = None

    @classmethod
    def _get_model(cls, model_name: str):
        """Lazy-load sentence-transformers model (singleton)."""
        if cls._global_model is None:
            with cls._global_lock:
                if cls._global_model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError:
                        raise ImportError(
                            "Dense embedding requires sentence-transformers. "
                            "Install with: pip install hippo-llm[embedding]"
                        )
                    path = LOCAL_MODEL_PATH or model_name
                    cls._global_model = SentenceTransformer(path, local_files_only=True)
        return cls._global_model

    def __len__(self) -> int:
        return len(self._cache)

    # ---- public API ----

    def embed(self, text: str) -> np.ndarray:
        """Return L2-normalized embedding for *text*."""
        key = text[:200]
        if key in self._cache:
            return self._cache[key]

        st_model = self._get_model(self.model)
        vec = st_model.encode(text, normalize_embeddings=True).astype(np.float32)

        # dimension auto-detect + assert
        if self.detected_dim is None:
            self.detected_dim = len(vec)
        else:
            assert len(vec) == self.detected_dim, (
                f"Dimension mismatch: expected {self.detected_dim}, got {len(vec)}"
            )

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # FIFO eviction
        if len(self._cache) >= self._cache_size:
            del self._cache[next(iter(self._cache))]
        self._cache[key] = vec
        return vec

    def embed_batch(self, texts: List[str], batch_size: int = 32, pause: float = 0.0) -> np.ndarray:
        """Batch-embed texts using sentence-transformers batch encode."""
        st_model = self._get_model(self.model)
        vecs = st_model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        arr = np.array(vecs, dtype=np.float32)
        if self.detected_dim is None and len(arr) > 0:
            self.detected_dim = arr.shape[1]
        return arr

    def clear_cache(self) -> None:
        self._cache.clear()
