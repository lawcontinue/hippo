"""
Hippo Embedding Engine — generate embeddings via Ollama-compatible API.

Supports:
  - Any Ollama embedding model (default: nomic-embed-text)
  - L2-normalized output (cosine similarity = dot product)
  - In-process LRU cache
  - Batch embedding with rate limiting
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import time
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

    # ---- public API ----

    def embed(self, text: str) -> np.ndarray:
        """Return L2-normalized embedding for *text*."""
        key = text[:200]
        if key in self._cache:
            return self._cache[key]

        payload = json.dumps({"model": self.model, "prompt": text})
        result = subprocess.run(
            [
                "curl", "-s", "-X", "POST", self._endpoint,
                "-H", "Content-Type: application/json",
                "-d", payload,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"Embedding request failed: {result.stderr}")

        vec = np.array(json.loads(result.stdout)["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # FIFO eviction
        if len(self._cache) >= self._cache_size:
            del self._cache[next(iter(self._cache))]
        self._cache[key] = vec
        return vec

    def embed_batch(self, texts: List[str], batch_size: int = 8, pause: float = 0.05) -> np.ndarray:
        """Batch-embed texts with optional pause between mini-batches."""
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            for t in texts[i : i + batch_size]:
                out.append(self.embed(t))
            if i + batch_size < len(texts):
                time.sleep(pause)
        return np.array(out, dtype=np.float32)

    def clear_cache(self) -> None:
        self._cache.clear()
