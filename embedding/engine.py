"""
Hippo Embedding Engine — generate embeddings via Ollama-compatible API.

Dependencies: numpy, urllib (stdlib)
Changed: curl → urllib.request, added dimension auto-detect, __len__
"""

from __future__ import annotations

import json
import os
import struct
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

import numpy as np

__all__ = ["EmbeddingEngine", "blob_to_vector", "vector_to_blob"]

# ---------- helpers ----------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_DIM = 768


def blob_to_vector(blob: bytes) -> np.ndarray:
    """Deserialize SQLite BLOB → numpy float32 vector (auto-detect dim)."""
    dim = len(blob) // 4
    return np.array(struct.unpack(f"<{dim}f", blob), dtype=np.float32)


def vector_to_blob(vec: np.ndarray) -> bytes:
    """Serialize numpy float32 vector → SQLite BLOB."""
    return struct.pack(f"<{len(vec)}f", *vec)


# ---------- EmbeddingEngine ----------

class EmbeddingEngine:
    """Generate embeddings through an Ollama-compatible /api/embeddings endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        base_url: Optional[str] = None,
        cache_size: int = 512,
    ):
        self.model = model
        self.dim = dim
        self.base_url = (base_url or OLLAMA_URL).rstrip("/")
        self._endpoint = f"{self.base_url}/api/embeddings"
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self.detected_dim: Optional[int] = None

    def __len__(self) -> int:
        return len(self._cache)

    # ---- public API ----

    def embed(self, text: str) -> np.ndarray:
        """Return L2-normalized embedding for *text*."""
        key = text[:200]
        if key in self._cache:
            return self._cache[key]

        payload = json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"Embedding request failed: {e}") from e

        vec = np.array(body["embedding"], dtype=np.float32)

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

    def embed_batch(self, texts: List[str], batch_size: int = 8, pause: float = 0.05) -> np.ndarray:
        """Batch-embed texts with optional pause between mini-batches.

        The first call triggers dimension auto-detection via ``embed()``,
        which sets ``self.detected_dim``.  Subsequent embeddings are
        validated against the detected dimension.
        """
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            for t in texts[i : i + batch_size]:
                out.append(self.embed(t))
            if i + batch_size < len(texts):
                time.sleep(pause)
        return np.array(out, dtype=np.float32)

    def clear_cache(self) -> None:
        self._cache.clear()
