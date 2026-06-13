# Hippo 首发推广文案（v2，embedding 优先叙事）

## HackerNews (Show HN)

```
Show HN: Hippo – pip install, 30 seconds, you have hybrid search. No vector DB, no jieba, no cloud.

I built Hippo because I got tired of the RAG stack tax. Embedding model here. Vector DB there. Chinese? Install jieba separately. By the time you're done, you've maintained 3 services to do a `db.search(query)`.

Hippo does all of it in one `pip install`:

- Hybrid search: BM25 + dense vectors with RRF fusion, backed by SQLite. No external vector DB.
- Chinese-optimized: built-in tokenizer with stop words. No jieba dependency.
- ANN index for >10K documents, sub-ms queries.
- Local embedding via Ollama (bge-small-zh: 5ms/query, 85.5% OOD top-1 on 110 queries).
- Pipeline parallelism bonus: split any GGUF model across machines. Two Mac Minis run Qwen3-30B-A3B at 78 tok/s.

Real numbers from production use:
- BM25 query: <1ms
- Dense search (bge-small-zh, 512d): 5ms
- Keyword + embedding fusion: 91.8% top-1 on OOD 110 queries (the two classifiers have completely different error modes — 42 correct unique each, TOOLS #94)
- ANN build (10K docs): ~2s

The use case: you're building an AI agent that needs semantic search, RAG, or memory. You don't want cloud APIs (latency + cost + privacy), and you don't want to maintain 3 services. Hippo is one dependency.

Also handles inference with OpenAI-compatible API, loop detection, and auto memory budget for sharding.

MIT licensed. GitHub: https://github.com/lawcontinue/hippo
PyPI: pip install hippo-llm
```

## Reddit r/LocalLLaMA

```
Title: Hippo – Local embedding + hybrid search (BM25 + dense, Chinese-optimized) in one pip install

Body:

Tired of the "install ChromaDB + embedding model + connector + jieba for Chinese" dance? I built Hippo to be the single dependency for local semantic search.

What it does:
- Hybrid search: BM25 + dense vectors with RRF fusion, SQLite-backed, sub-ms queries
- Built-in Chinese tokenizer (stop words included), no jieba
- ANN index for >10K documents
- Local embedding via Ollama — bge-small-zh runs at 5ms/query with 85.5% OOD top-1 accuracy
- Keyword + embedding fusion hits 91.8% top-1 (they're complementary: 42 correct unique to each)

Why I built it:
Running multi-agent experiments on a Mac Mini + RTX 5060 Ti. Needed semantic routing, RAG, and agent memory. Every option either required cloud (privacy concern), separate services (overhead for one person), or didn't handle Chinese well. Hippo packages the whole stack.

Bonus: pipeline parallelism for inference. Split any GGUF model across machines with plain TCP.

| Setup | Model | Speed |
|-------|-------|-------|
| RTX 5060 Ti 16GB | Qwen3-14B-Q4 | 41 tok/s |
| 2× Mac Mini M2 16GB | Qwen3-30B-A3B-Q3 | 78 tok/s |
| bge-small-zh embedding | 512d, OOD 110q | 5ms/query |

MIT, 143 tests, PyPI: `pip install hippo-llm`

GitHub: https://github.com/lawcontinue/hippo

Happy to answer questions about the hybrid search design or the fusion benchmarks.
```

## 掘金

```
Title: Hippo：一个 pip install 搞定本地搜索，BM25 零依赖启动，按需升级 Hybrid

## 30 秒上手

```python
from hippo.embedding import VectorStore
store = VectorStore("docs.db")  # 默认 sparse，BM25 即用
store.add_batch([
    {"text": "管道并行将模型层拆分到多台设备上"},
    {"text": "混合搜索结合了关键词匹配和语义相似度"},
])
print(store.search("怎么拆模型"))
```

没有 ChromaDB。没有 jieba。没有网络请求。`pip install hippo-llm` 直接跑。

## 为什么要造这个轮子？

之前做 multi-agent 实验（Mac Mini + RTX 5060Ti），每个 agent 需要语义搜索和 RAG。现有方案：
- ChromaDB：要单独起服务
- LangChain embedding：走 OpenAI API，有延迟+成本+隐私问题
- 中文搜索：装 jieba，配 tokenizer，搞半天

我想要的就是 `store.search(query)`，不应该这么难。

## 按需升级

BM25 验证完效果后，一行升级到 hybrid：

```bash
pip install hippo-llm[embedding]  # 加 sentence-transformers
```

```python
from hippo.embedding import EmbeddingEngine, VectorStore
engine = EmbeddingEngine(model="nomic-embed-text")
store = VectorStore("docs.db", mode="hybrid", embedding_engine=engine)
# 同一个 API，加了 dense vector + RRF 融合
```

## 实测数据

| 指标 | 数值 |
|------|------|
| BM25 查询 | <1ms |
| Dense (bge-small-zh) | 5ms/query |
| Hybrid 融合 OOD top-1 | 91.8% |
| ANN 10K 文档索引 | ~2s |

关键词 + embedding 互补性很强：各独有 42 条正确，融合比单独任一都准。

GitHub: https://github.com/lawcontinue/hippo
PyPI: pip install hippo-llm
```

## dev.to

```
Title: Build a local RAG pipeline in 15 lines with Hippo (no ChromaDB needed)

摘要：Walk through building a RAG pipeline using Hippo's embedding + hybrid search. Show the 15-line example, explain BM25 + dense fusion, benchmark against cloud APIs. Chinese-optimized out of the box.

重点：
- 15 行代码完整 RAG
- 不需要 ChromaDB/jieba/云端 API
- 实测：5ms/query 本地 embedding
- 中文零配置
```

## 首发前检查清单

- [x] README embedding 优先叙事
- [x] 实测数据嵌入（5ms, 85.5%, 91.8%）
- [x] 中文场景突出
- [x] HN/Reddit/掘金文案更新
- [ ] demo_search.png 更新（突出 embedding 输出）
- [ ] GitHub README push
- [ ] HN 发帖（时序：D1 HN → D2 Reddit → D3 掘金）
