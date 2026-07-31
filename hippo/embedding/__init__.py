"""Hippo Embedding — lightweight embedding + vector search over SQLite."""

from .engine import EmbeddingEngine
from .store import VectorStore

__all__ = ["EmbeddingEngine", "VectorStore"]
