# Hippo 首发推广文案（v3，clone即用叙事）

## HackerNews (Show HN)

```
Show HN: Hippo – Search your documents locally, in 30 seconds, zero dependencies

pip install hippo-llm. That's it. No ChromaDB, no jieba, no cloud API, no embedding model to download.

I built Hippo because I got tired of the RAG stack tax. Embedding model here. Vector DB there. Chinese? Install jieba separately. By the time you're done, you've maintained 3 services to do a db.search(query).

Hippo starts with BM25 only — works immediately after pip install:

from hippo.embedding import VectorStore
store = VectorStore("docs.db")  # sparse mode, BM25 only
store.add_batch([{"text": "你的中文文档"}, {"text": "English docs too"}])
store.search("搜索")  # done

Need semantic search? One command upgrades to hybrid:

pip install hippo-llm[embedding]

store = VectorStore("docs.db", mode="hybrid", embedding_engine=engine)
# Same API. Now BM25 + dense vectors, RRF fused.

Your sparse-mode database migrates automatically — no re-indexing needed.

Real numbers from production use:
- BM25 query: <1ms (pure Python + SQLite)
- Dense embedding (bge-small-zh, 512d): 5ms/query, 85.5% top-1 standalone
- Hybrid RRF fusion: 91.8% top-1 on OOD 110 queries (keyword and embedding are complementary — 42 correct unique to each)
- Chinese tokenizer built-in, zero config

Bonus: pipeline parallelism for local LLM inference. Split any GGUF model across machines with plain TCP. Two Mac Minis run Qwen3-30B-A3B at 78 tok/s. Mac + PC mixed.

MIT licensed, 32 tests. pip install hippo-llm

GitHub: https://github.com/lawcontinue/hippo
```

## Reddit r/LocalLLaMA

```
Title: Hippo – pip install, 30 seconds, you have local search. Upgrade to hybrid when you need it.

I built Hippo because I was running multi-agent experiments on a Mac Mini + RTX 5060 Ti, and every agent needed document search. The existing options all required setting up separate services:

- ChromaDB: run a server, manage collections
- LangChain embeddings: call OpenAI/Cohere per query
- Chinese search: install jieba, configure tokenizer

All I wanted was store.search(query).

What Hippo does differently:

Start with BM25 only. pip install hippo-llm gives you working search immediately — no embedding model download, no external DB, no API keys. SQLite for storage, numpy for similarity, stdlib Chinese tokenizer.

from hippo.embedding import VectorStore
store = VectorStore("docs.db")
store.add_batch([{"text": "管道并行将模型层拆分到多台设备上"}])
store.search("怎么拆模型")  # works now

Upgrade to hybrid when you're ready. pip install hippo-llm[embedding] adds sentence-transformers. Same API, now with BM25 + dense vectors + RRF fusion. Your existing database migrates automatically — no re-indexing.

Benchmarks from actual use (not synthetic):

| Component | Latency | Accuracy (OOD 110 queries) |
|-----------|---------|---------------------------|
| BM25 | <1ms | keyword-only baseline |
| bge-small-zh (512d) | 5ms | 85.5% top-1 |
| Hybrid RRF | <1ms overhead | 91.8% top-1 |
| ANN (10K docs) | ~2s build | — |

The fusion works because keyword and embedding have completely different error modes — each gets 42 queries right that the other misses. That's the real value of hybrid: not one-or-the-other, but both.

Also includes pipeline parallelism for local inference: split GGUF models across machines with plain TCP. Mac + PC mixed.

| Setup | Model | Speed |
|-------|-------|-------|
| RTX 5060 Ti 16GB | Qwen3-14B-Q4 | 41 tok/s |
| 2× Mac Mini M2 16GB | Qwen3-30B-A3B-Q3 | 78 tok/s |

MIT, 32 tests, PyPI: pip install hippo-llm

GitHub: https://github.com/lawcontinue/hippo

Happy to answer questions about the sparse→hybrid upgrade path or the fusion benchmarks.
```

## 掘金

```
Title: Hippo：pip install 30 秒搞定本地搜索，零依赖启动，按需升级 Hybrid

## 30 秒上手

pip install hippo-llm 之后直接写代码：

```python
from hippo.embedding import VectorStore

store = VectorStore("docs.db")  # 默认 sparse，BM25 即用
store.add_batch([
    {"text": "管道并行将模型层拆分到多台设备上"},
    {"text": "混合搜索结合了关键词匹配和语义相似度"},
    {"text": "BM25 handles exact keyword matches"},
])
results = store.search("怎么拆模型")
for doc in results:
    print(f"[{doc.score:.3f}] {doc.text}")
```

没有 ChromaDB。没有 jieba。没有网络请求。没有 embedding 模型下载。pip install 直接跑。

中文分词内置（CJK 单字切分 + 停用词过滤），零配置。

## 为什么要造这个轮子

之前做 multi-agent 实验（Mac Mini + RTX 5060Ti），每个 agent 需要语义搜索和 RAG。现有方案要么要单独起服务（ChromaDB），要么走云端 API（OpenAI embedding），要么中文要装 jieba 配半天。

我想要的就是 store.search(query)，不应该这么难。

所以 Hippo 的设计原则：**先跑起来，再按需升级**。

## 按需升级到 Hybrid

BM25 验证完效果后，一行升级到语义搜索：

```bash
pip install hippo-llm[embedding]  # 加 sentence-transformers
```

```python
from hippo.embedding import EmbeddingEngine, VectorStore

engine = EmbeddingEngine(model="nomic-embed-text")
store = VectorStore("docs.db", mode="hybrid", embedding_engine=engine)
# 同一个 API，加了 dense vector + BM25 RRF 融合
```

重点：之前 sparse 模式建的数据库直接能用，不需要重新索引。Hippo 自动做 ALTER TABLE 迁移。

## 实测数据

| 指标 | 数值 |
|------|------|
| BM25 查询延迟 | <1ms |
| Dense embedding (bge-small-zh, 512d) | 5ms/query |
| Hybrid RRF 融合 OOD top-1 | 91.8%（110 题测试集） |
| bge-small-zh 独立 OOD | 85.5% |
| ANN 10K 文档索引构建 | ~2s |

关键词和 embedding 互补性很强：各独有 42 条正确，融合后比单独任一都准。这就是 hybrid 的价值——不是二选一，是两个都要。

## 还能跑本地推理

附带了 pipeline parallelism，可以把 GGUF 模型拆到多台机器上跑：

- RTX 5060 Ti 16GB：Qwen3-14B-Q4，41 tok/s
- 2× Mac Mini M2 16GB：Qwen3-30B-A3B-Q3，78 tok/s

Mac + PC 混搭，普通 TCP，不需要 MPI。

## 安装

```bash
pip install hippo-llm                    # BM25 搜索，零依赖
pip install hippo-llm[embedding]         # 加语义搜索 + hybrid RRF
```

MIT 协议，32 测试全绿。

GitHub: https://github.com/lawcontinue/hippo
PyPI: pip install hippo-llm
```

## dev.to

```
Title: Local document search in 30 seconds with Hippo (no vector DB, no cloud, no jieba)

Most RAG tutorials start with "install ChromaDB, download an embedding model, configure a vector store." By the time you've done all that, you haven't even searched anything yet.

Hippo takes a different approach: start with BM25, upgrade to hybrid when you need it.

## The 30-second version

pip install hippo-llm

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

pip install hippo-llm[embedding]

```python
from hippo.embedding import EmbeddingEngine, VectorStore
engine = EmbeddingEngine(model="nomic-embed-text")
store = VectorStore("docs.db", mode="hybrid", embedding_engine=engine)
```

Same API. Now you get BM25 + dense vectors with RRF fusion. Your existing database migrates automatically — no re-indexing needed.

## The numbers

From production use (not synthetic benchmarks):

| Method | Latency | OOD Accuracy (110 queries) |
|--------|---------|---------------------------|
| BM25 | <1ms | keyword baseline |
| bge-small-zh (512d) | 5ms | 85.5% top-1 |
| Hybrid RRF | <1ms overhead | 91.8% top-1 |

The key insight: keyword search and embedding search have completely different error patterns. Each gets 42 queries right that the other misses. Fusion isn't averaging — it's complementarity.

## Full RAG pipeline

Combine Hippo's hybrid search with a local LLM for a complete RAG system:

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

# 3. Generate answer with local LLM (OpenAI-compatible API)
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
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

store.add_batch([{"text": "中文文档直接搜索，无需配置分词器"}])
store.search("搜索")  # works immediately

The tokenizer handles mixed Chinese/English text naturally — CJK characters get single-char tokens, English words get whitespace-split tokens. This is simpler than jieba but sufficient for BM25 ranking, where exact match precision matters more than linguistic accuracy.

## When to use what

- `mode="sparse"` (default): prototyping, small collections, no GPU
- `mode="hybrid"`: production RAG, semantic routing, agent memory
- `mode="dense"`: pure similarity search, no keyword matching needed

pip install hippo-llm
GitHub: https://github.com/lawcontinue/hippo
```

## 首发时序

| D+ | 平台 | 文案 | 状态 |
|----|------|------|------|
| D0 | HN | Show HN（痛点共鸣式） | ✅ 就绪 |
| D1 | Reddit r/LocalLLaMA | 功能+数据+故事 | ✅ 就绪 |
| D2 | 掘金 | 中文实战体验文 | ✅ 就绪 |
| D3+ | dev.to | Tutorial walk-through | ✅ 就绪 |

## 首发前检查清单

- [x] README embedding 优先 + clone 即用叙事
- [x] 实测数据嵌入（5ms, 85.5%, 91.8%）
- [x] 中文场景突出
- [x] HN/Reddit/掘金/dev.to 四篇文案就绪
- [x] sparse→hybrid migration 自动化
- [x] GitHub push（57dc897）
- [ ] demo_search.png 更新（突出 sparse 输出）
- [ ] HN 发帖
