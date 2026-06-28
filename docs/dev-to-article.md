# Local document search in 30 seconds with Hippo (no vector DB, no cloud, no jieba)

Most RAG tutorials start with "install ChromaDB, download an embedding model, configure a vector store." By the time you've done all that, you haven't even searched anything yet.

Hippo takes a different approach: start with BM25, upgrade to hybrid when you need it.

## The 30-second version

```bash
pip install hippo-llm
```

```python
from hippo.embedding import VectorStore
store = VectorStore("docs.db")
store.add_batch([{"text": "Your documents here"}])
results = store.search("your query")
```

That's it. No embedding model download. No vector DB server. No API keys. SQLite for storage, numpy for similarity, stdlib tokenizer for Chinese text.

## Why start with BM25?

Because it works. For most document collections under 10K, BM25 with a decent tokenizer gets you 80% of the way there. No GPU needed. No model download. Sub-millisecond queries.

You can validate your search use case in minutes instead of hours. Then upgrade when BM25 isn't enough.

## Upgrading to hybrid

```bash
pip install hippo-llm[embedding]
```

```python
from hippo.embedding import EmbeddingEngine, VectorStore

engine = EmbeddingEngine(model="nomic-embed-text")  # local sentence-transformers
store = VectorStore("docs.db", mode="hybrid", embedding_engine=engine)
```

Same API. Now you get BM25 + dense vectors with RRF fusion. Uses [sentence-transformers](https://github.com/UKPLab/sentence-transformers) under the hood — no Ollama, no cloud API, no external services.

Your existing database migrates automatically — no re-indexing needed.

## The numbers

From production use (not synthetic benchmarks):

| Method | Latency | OOD Accuracy (110 queries) |
|--------|---------|---------------------------|
| BM25 | <1ms | keyword baseline |
| bge-small-zh (512d) | 5ms | 85.5% top-1 |
| Hybrid RRF | <1ms overhead | 91.8% top-1 |

The key insight: keyword search and embedding search have completely different error patterns. Each gets 42 queries right that the other misses. Fusion isn't averaging — it's complementarity.

## Full RAG pipeline

Combine Hippo's hybrid search with any LLM for a complete RAG system:

```python
from hippo.embedding import EmbeddingEngine, VectorStore
import openai

# 1. Index your documents
engine = EmbeddingEngine(model="nomic-embed-text")
store = VectorStore("knowledge.db", mode="hybrid", embedding_engine=engine)

store.add_batch([
    {"text": "Hippo splits model layers across multiple devices using TCP."},
    {"text": "Each device only loads its shard of layers, reducing memory per device."},
    {"text": "The loop detector catches semantic repetition using Jaccard similarity."},
    {"text": "BM25 hybrid search combines keyword matching with semantic similarity."},
], engine=engine)

# 2. Search
query = "how does hippo handle memory?"
results = store.search(query, engine=engine, top_k=2)
context = "\n".join(doc.text for doc in results)

# 3. Generate answer with any LLM (local or cloud)
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="qwen3-30b-a3b-q3",
    messages=[
        {"role": "system", "content": f"Answer based on this context:\n{context}"},
        {"role": "user", "content": query}
    ]
)
print(response.choices[0].message.content)
```

If you started with sparse mode and want to upgrade an existing database:

```python
store = VectorStore("knowledge.db", mode="hybrid", embedding_engine=engine)
store.rebuild_embeddings(engine)  # backfill embeddings for existing docs
```

No re-indexing. No data loss. Same API.

## Chinese support

Built-in CJK tokenizer: single-character segmentation for Chinese/Japanese/Korean, whitespace splitting for English, with stop word filtering for both. No jieba, no external dependencies.

```python
store.add_batch([{"text": "中文文档直接搜索，无需配置分词器"}])
store.search("搜索")  # works immediately
```

The tokenizer handles mixed Chinese/English text naturally — CJK characters get single-char tokens, English words get whitespace-split tokens. This is simpler than jieba but sufficient for BM25 ranking, where exact match precision matters more than linguistic accuracy.

## When to use what

**Sparse (default)** — prototyping, small collections, no GPU needed:

```python
store = VectorStore("docs.db")  # mode="sparse" by default
```

**Hybrid** — production RAG, semantic routing, agent memory:

```bash
pip install hippo-llm[embedding]
```

```python
store = VectorStore("docs.db", mode="hybrid", embedding_engine=engine)
```

**Dense** — pure similarity search, no keyword matching needed:

```python
store = VectorStore("docs.db", mode="dense", embedding_engine=engine)
```

## Install

```bash
pip install hippo-llm                    # BM25 search, zero external services
pip install hippo-llm[embedding]         # add semantic search + hybrid RRF
```

GitHub: https://github.com/lawcontinue/hippo

---

**Tags:** #python #rag #localllm #search #bm25 #embeddings #opensource #chinese-nlp
