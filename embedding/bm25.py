"""
Hippo BM25 Index — Okapi BM25 ranking with pluggable tokenizer.

Dependencies: math (stdlib)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

from .tokenizer import default_tokenizer

__all__ = ["BM25Index"]


class BM25Index:
    """Okapi BM25 index (k1=1.5, b=0.75)."""

    def __init__(self, tokenizer: Callable[[str], List[str]] = default_tokenizer):
        self._tokenizer = tokenizer
        self._k1 = 1.5
        self._b = 0.75
        # doc_id → token list
        self._docs: Dict[str, List[str]] = {}
        # doc_id → {token: count}
        self._tf: Dict[str, Dict[str, int]] = {}
        # token → set of doc_ids
        self._inverted: Dict[str, set] = defaultdict(set)
        # token → document frequency
        self._df: Dict[str, int] = defaultdict(int)
        # doc_id → doc length (token count)
        self._dl: Dict[str, int] = {}
        self._avgdl: float = 0.0

    def _update_avgdl(self):
        n = len(self._docs)
        self._avgdl = sum(self._dl.values()) / n if n > 0 else 0.0

    def add(self, doc_id: str, text: str) -> None:
        """Add or replace a document."""
        # remove old if exists
        if doc_id in self._docs:
            self.delete(doc_id)

        tokens = self._tokenizer(text)
        if not tokens:
            return  # skip empty documents
        self._docs[doc_id] = tokens
        self._dl[doc_id] = len(tokens)
        # precompute term frequencies
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        self._tf[doc_id] = tf

        seen = set()
        for t in tokens:
            self._inverted[t].add(doc_id)
            if t not in seen:
                self._df[t] += 1
                seen.add(t)

        self._update_avgdl()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return list of (doc_id, score) sorted by BM25 score descending."""
        if not self._docs:
            return []

        query_tokens = self._tokenizer(query)
        N = len(self._docs)
        scores: Dict[str, float] = defaultdict(float)

        for t in query_tokens:
            if t not in self._df:
                continue
            df = self._df[t]
            idf = math.log((N - df + 0.5) / (df + 0.5))
            for doc_id in self._inverted[t]:
                tf = self._tf[doc_id].get(t, 0)
                dl = self._dl[doc_id]
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                scores[doc_id] += idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def delete(self, doc_id: str) -> bool:
        """Delete a document. Returns False if not found."""
        if doc_id not in self._docs:
            return False

        old_tokens = self._docs.pop(doc_id)
        del self._dl[doc_id]
        del self._tf[doc_id]

        # decrement df and clean inverted index
        seen = set()
        for t in old_tokens:
            self._inverted[t].discard(doc_id)
            if not self._inverted[t]:
                del self._inverted[t]
            if t not in seen:
                self._df[t] -= 1
                if self._df[t] <= 0:
                    del self._df[t]
                seen.add(t)

        self._update_avgdl()
        return True

    def count(self) -> int:
        return len(self._docs)
