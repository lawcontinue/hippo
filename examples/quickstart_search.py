"""
Quick start: Hippo embedding + hybrid search in 30 seconds.

Usage:
    # Start Ollama first, then:
    python3 examples/quickstart_search.py
"""

from hippo.embedding import EmbeddingEngine, VectorStore

# 1. Setup (uses local Ollama for embeddings)
engine = EmbeddingEngine(model="nomic-embed-text")
store = VectorStore("quickstart.db", mode="hybrid")  # BM25 + dense RRF fusion

# 2. Add documents
docs = [
    {"text": "Pipeline parallelism splits model layers across devices", "metadata": {"source": "readme"}},
    {"text": "BM25 handles exact keyword matches without embeddings", "metadata": {"source": "docs"}},
    {"text": "Hippo uses SQLite for persistence — no external vector DB needed", "metadata": {"source": "readme"}},
    {"text": "Chinese tokenizer uses character bigrams, no jieba dependency", "metadata": {"source": "docs"}},
    {"text": "ANN index handles >10K documents with sub-ms queries", "metadata": {"source": "benchmarks"}},
]
store.add_batch(docs, engine=engine)
print(f"Indexed {len(docs)} documents\n")

# 3. Search
queries = [
    "how to run big models",
    "中文分词",
    "vector database",
]
for q in queries:
    results = store.search(q, engine=engine, top_k=2)
    print(f"Q: {q}")
    for doc in results:
        print(f"  [{doc.score:.3f}] {doc.text[:60]}...")
    print()
