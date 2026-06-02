"""
Hippo Embedding Benchmark — ANN params + RRF weights + mode comparison.

Usage:
    python3 -m embedding.benchmark            # full benchmark
    python3 -m embedding.benchmark --quick     # fast version (500 docs)
    python3 -m embedding.benchmark --ann-only  # ANN parameter sweep only

Output: Markdown table of results.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .ann_index import ANNConfig, ANNIndex, HAS_HNSW
from .bm25 import BM25Index
from .engine import EmbeddingEngine
from .store import VectorStore


# ---- Synthetic data generator ----

def _generate_corpus(n: int, dim: int, seed: int = 42) -> Tuple[np.ndarray, List[str]]:
    """Generate synthetic documents with clustered embeddings."""
    rng = np.random.RandomState(seed)
    n_clusters = max(n // 10, 5)
    centers = rng.randn(n_clusters, dim)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    vecs = np.zeros((n, dim))
    texts = []
    for i in range(n):
        c = centers[i % n_clusters]
        noise = rng.randn(dim) * 0.1
        v = c + noise
        v /= np.linalg.norm(v)
        vecs[i] = v
        texts.append(f"document cluster {i % n_clusters} item {i} topic {'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu'[i % 12:].split()[0]}")

    return vecs, texts


# ---- ANN parameter sweep ----

def bench_ann_params(vecs: np.ndarray, dim: int, n_queries: int = 50) -> List[Dict]:
    """Sweep HNSW parameters and measure recall/latency."""
    if not HAS_HNSW:
        print("⚠️  hnswlib not installed, skipping ANN sweep")
        return []

    n = len(vecs)
    results = []
    rng = np.random.RandomState(99)
    query_idx = rng.choice(n, size=min(n_queries, n), replace=False)

    configs = [
        ANNConfig(ef_construction=100, M=8,  ef_search=20,  max_elements=n * 2),
        ANNConfig(ef_construction=100, M=16, ef_search=50,  max_elements=n * 2),
        ANNConfig(ef_construction=200, M=16, ef_search=50,  max_elements=n * 2),
        ANNConfig(ef_construction=200, M=16, ef_search=100, max_elements=n * 2),
        ANNConfig(ef_construction=200, M=32, ef_search=100, max_elements=n * 2),
        ANNConfig(ef_construction=400, M=32, ef_search=200, max_elements=n * 2),
    ]

    for cfg in configs:
        idx = ANNIndex(dim=dim, config=cfg)
        # build
        t0 = time.perf_counter()
        for i, v in enumerate(vecs):
            idx.add(v, doc_id=i)
        build_ms = (time.perf_counter() - t0) * 1000

        # benchmark
        query_vecs = vecs[query_idx]
        stats = idx.benchmark(query_vecs, list(query_idx), top_k=10)
        stats["ef_construction"] = cfg.ef_construction
        stats["M"] = cfg.M
        stats["build_ms"] = round(build_ms, 1)
        stats["n_docs"] = n
        results.append(stats)

    return results


# ---- RRF weight sweep ----

def bench_rrf_weights(vecs: np.ndarray, texts: List[str], dim: int,
                      n_queries: int = 50) -> List[Dict]:
    """Sweep dense_weight / sparse_weight ratios for hybrid search."""
    rng = np.random.RandomState(99)
    n = len(vecs)
    query_idx = rng.choice(n, size=min(n_queries, n), replace=False)
    query_texts = [texts[i] for i in query_idx]

    class _MockEngine:
        def __init__(self, dim):
            self.dim = dim
        def embed(self, text):
            idx = int(text.split("item")[1].split()[0]) if "item" in text else 0
            if idx < len(vecs):
                return vecs[idx].copy()
            return np.zeros(dim)

    weight_combos = [
        (1.0, 1.0),   # equal
        (1.0, 0.5),   # dense-heavy
        (0.5, 1.0),   # sparse-heavy
        (1.0, 0.0),   # pure dense (via RRF)
        (0.0, 1.0),   # pure sparse (via RRF)
        (2.0, 1.0),   # 2:1 dense
        (1.0, 2.0),   # 2:1 sparse
    ]

    results = []
    for dw, sw in weight_combos:
        with tempfile.TemporaryDirectory() as d:
            import os
            store = VectorStore(
                db_path=os.path.join(d, "bench.db"),
                embedding_engine=_MockEngine(dim),
                mode="hybrid",
            )
            for i, (v, t) in enumerate(zip(vecs, texts)):
                store._entries.append((i, t, "{}", v))
                store._entry_map[i] = (i, t, "{}", v)
            # init bm25
            from .tokenizer import default_tokenizer
            store._bm25 = BM25Index(tokenizer=default_tokenizer)
            for i, t in enumerate(texts):
                store._bm25.add(str(i), t)

            hits = 0
            latencies = []
            for qi, qt in zip(query_idx, query_texts):
                t0 = time.perf_counter()
                dense = store._search_dense(qt, 20, 0.0, None)
                sparse = store._bm25.search(qt, 20)
                fused = store._rrf_fuse(dense, sparse, dense_weight=dw, sparse_weight=sw)
                latencies.append((time.perf_counter() - t0) * 1000)
                if any(r.id == qi for r in fused[:10]):
                    hits += 1

            recall = round(hits / len(query_idx), 4)
            avg_lat = round(np.mean(latencies), 3)
            results.append({
                "dense_w": dw, "sparse_w": sw,
                "recall@10": recall,
                "avg_ms": avg_lat,
            })

    return results


# ---- Mode comparison ----

def bench_modes(vecs: np.ndarray, texts: List[str], dim: int,
                n_queries: int = 50) -> List[Dict]:
    """Compare dense / sparse / hybrid modes."""
    rng = np.random.RandomState(99)
    n = len(vecs)
    query_idx = rng.choice(n, size=min(n_queries, n), replace=False)
    query_texts = [texts[i] for i in query_idx]

    # hard queries: add noise so exact match isn't trivial
    query_vecs_hard = vecs[query_idx].copy()
    noise = rng.randn(*query_vecs_hard.shape) * 0.3
    query_vecs_hard += noise
    query_vecs_hard /= np.linalg.norm(query_vecs_hard, axis=1, keepdims=True)

    class _MockEngine:
        def __init__(self, dim):
            self.dim = dim
        def embed(self, text):
            idx = int(text.split("item")[1].split()[0]) if "item" in text else 0
            if idx < len(vecs):
                return vecs[idx].copy()
            return np.zeros(dim)

    results = []
    for mode in ["dense", "sparse", "hybrid"]:
        with tempfile.TemporaryDirectory() as d:
            import os
            store = VectorStore(
                db_path=os.path.join(d, "bench.db"),
                embedding_engine=_MockEngine(dim),
                mode=mode,
            )
            for i, (v, t) in enumerate(zip(vecs, texts)):
                store._entries.append((i, t, "{}", v))
                store._entry_map[i] = (i, t, "{}", v)
            if mode in ("sparse", "hybrid"):
                from .tokenizer import default_tokenizer
                store._bm25 = BM25Index(tokenizer=default_tokenizer)
                for i, t in enumerate(texts):
                    store._bm25.add(str(i), t)

            hits = 0
            latencies = []
            for qi, qvec_hard, qt in zip(query_idx, query_vecs_hard, query_texts):
                # use perturbed vector for dense search
                qvec_orig = store.engine.embed(qt)
                store.engine.embed = lambda t, _q=qvec_hard: _q  # temp override
                t0 = time.perf_counter()
                res = store.search(qt, top_k=10)
                latencies.append((time.perf_counter() - t0) * 1000)
                if any(r.id == qi for r in res):
                    hits += 1
                store.engine.embed = lambda t, _v=qvec_orig: _v  # restore

            recall = round(hits / len(query_idx), 4)
            avg_lat = round(np.mean(latencies), 3)
            results.append({
                "mode": mode,
                "recall@10": recall,
                "avg_ms": avg_lat,
            })

    return results


# ---- Markdown output ----

def _print_table(title: str, rows: List[Dict], columns: List[str]):
    print(f"\n### {title}\n")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    print(header)
    print(sep)
    for row in rows:
        vals = [str(row.get(c, "")) for c in columns]
        print("| " + " | ".join(vals) + " |")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Hippo Embedding Benchmark")
    parser.add_argument("--quick", action="store_true", help="500 docs instead of 2000")
    parser.add_argument("--ann-only", action="store_true", help="ANN sweep only")
    parser.add_argument("--n-docs", type=int, default=None)
    args = parser.parse_args()

    n_docs = args.n_docs or (500 if args.quick else 2000)
    dim = 64

    print(f"🔬 Hippo Embedding Benchmark — {n_docs} docs, dim={dim}")
    print(f"   hnswlib: {'✅' if HAS_HNSW else '❌ (numpy fallback)'}")

    vecs, texts = _generate_corpus(n_docs, dim)

    # 1. ANN parameter sweep
    if HAS_HNSW:
        print("\n⏳ Running ANN parameter sweep...")
        ann_results = bench_ann_params(vecs, dim)
        if ann_results:
            _print_table("ANN Parameter Sweep (HNSW)", ann_results,
                        ["ef_construction", "M", "ef_search", "recall@10", "avg_latency_ms", "build_ms"])

    if args.ann_only:
        return

    # 2. Mode comparison
    print("\n⏳ Running mode comparison...")
    mode_results = bench_modes(vecs, texts, dim)
    _print_table("Search Mode Comparison", mode_results,
                ["mode", "recall@10", "avg_ms"])

    # 3. RRF weight sweep
    print("\n⏳ Running RRF weight sweep...")
    rrf_results = bench_rrf_weights(vecs, texts, dim)
    _print_table("RRF Weight Sweep (hybrid mode)", rrf_results,
                ["dense_w", "sparse_w", "recall@10", "avg_ms"])

    print("\n✅ Benchmark complete.")


if __name__ == "__main__":
    main()
