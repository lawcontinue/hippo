# Hippo 首发推广文案

## HackerNews (Show HN)

```
Show HN: Hippo – Run 30B models on consumer hardware, search without ChromaDB

I built Hippo because I was tired of the RAG stack tax. Every project needs inference + vector search, but you end up installing Ollama, ChromaDB, dealing with embeddings separately, and maintaining two services.

Hippo does both in one `pip install`:

- Pipeline parallelism: split any GGUF model across machines (Mac + PC mixed, plain TCP, no MPI). Two Mac Minis run Qwen3-30B-A3B at 78 tok/s.
- Built-in embedding + search: BM25 + dense vectors with RRF fusion, backed by SQLite. Chinese-optimized tokenizer included, no jieba needed.
- ANN index for >10K documents, sub-ms queries.
- OpenAI-compatible API so it drops into existing code.

The use case: you have a consumer GPU (RTX 5060 Ti 16GB = ¥3800) or a Mac Mini, and you want local LLM + RAG without the cloud dependency or the service sprawl.

MIT licensed, 7.7K lines, 143 tests.

GitHub: https://github.com/lawcontinue/hippo
PyPI: pip install hippo-llm
```

## Reddit r/LocalLLaMA

```
Title: Hippo – One pip install for local LLM inference + hybrid search (BM25 + dense, Chinese-optimized)

Body:

Tired of the "install Ollama + ChromaDB + embedding model + connector" dance? I built Hippo to be the single dependency for local AI.

What it does:
- Run GGUF models on consumer hardware (30B-A3B at 78 tok/s on two Mac Minis via pipeline parallelism)
- Hybrid search (BM25 + dense vectors, RRF fused) with built-in Chinese tokenizer
- SQLite-backed, works offline, sub-ms queries for <10K docs, ANN for larger collections
- OpenAI-compatible API (/v1/chat/completions) — works with LangChain, LlamaIndex, etc.

Why I built it:
I was running multi-agent experiments on a Mac Mini + RTX 5060 Ti and needed both inference and document search. Existing solutions either required cloud (too slow for my use case), separate services (too much overhead for one person), or didn't handle Chinese well.

Benchmarks:
| Setup | Model | Speed |
|-------|-------|-------|
| RTX 5060 Ti 16GB | Qwen3-14B-Q4 | 41 tok/s |
| 2× Mac Mini M2 16GB | Qwen3-30B-A3B-Q3 | 78 tok/s |
| Single Mac Mini M2 | Qwen3-30B-A3B-Q3 | 24 tok/s |

The pipeline parallelism is the interesting part — it splits model layers across machines using plain TCP, no MPI or special networking. You can mix Mac and PC.

MIT, 143 tests, PyPI: `pip install hippo-llm`

GitHub: https://github.com/lawcontinue/hippo

Happy to answer questions about the pipeline parallelism implementation or the hybrid search design.
```

## Reddit r/MachineLearning

```
Title: [R] Hippo: Pipeline-parallel local LLM inference with built-in hybrid search

Body:

We open-sourced Hippo, a local inference framework that combines pipeline parallelism with integrated embedding/search.

Key technical contributions:

1. Pipeline parallelism over plain TCP — splits GGUF model layers across N devices with automatic memory budget calculation. No MPI dependency. Tested with mixed Mac + PC setups.

2. Hybrid BM25 + dense search with RRF fusion — eliminates the need for a separate vector DB. SQLite-backed persistence, numpy similarity computation. Chinese-optimized BM25 tokenizer built-in (character bigrams + stop words, no jieba dependency).

3. ANN index for scalable retrieval — approximate nearest neighbor using IVF-style partitioning, handles >10K document collections.

4. Loop detection via Jaccard similarity — catches semantic repetition in generated text that standard repeat_penalty misses.

The framework is 7.7K lines, 143 tests, MIT licensed. Designed for researchers and developers running experiments on consumer hardware without cloud dependencies.

Paper/benchmarks: in repo
Code: https://github.com/lawcontinue/hippo
```

## V2EX / 即刻 / 掘金（中文推广）

```
标题：花 3800 块的显卡跑 30B 模型，还自带搜索——Hippo 开源了

我做 Hippo 是因为受够了 RAG 的"全家桶"：装 Ollama 推理 + ChromaDB 向量库 + embedding 模型 + 连接器，一套下来半天才跑通。

Hippo 一个 pip install 搞定：

- 流水线并行：把模型按层拆到多台机器上，Mac 和 PC 可以混搭，普通 TCP 不需要 MPI。两台 Mac Mini 跑 Qwen3-30B-A3B 能到 78 tok/s。
- 内置混合搜索：BM25 + 稠密向量 + RRF 融合，SQLite 存储，中文分词器自带（不需要 jieba）。
- 万级文档 ANN 索引，亚毫秒查询。
- OpenAI 兼容 API，LangChain/LlamaIndex 直接用。

场景：有一张消费级显卡或 Mac Mini，想本地跑 LLM + RAG，不想上云也不想装一堆服务。

MIT 协议，7700 行代码，143 个测试。

GitHub: https://github.com/lawcontinue/hippo
安装: pip install hippo-llm
```

## 发布节奏建议

| 平台 | 时机 | 注意事项 |
|------|------|---------|
| HackerNews | 周二-周四 美东上午（北京晚上） | Show HN 格式，配 demo GIF |
| Reddit r/LocalLLaMA | HN 发后 2 小时 | 回复每个评论 |
| Reddit r/MachineLearning | 同上 | 技术向，侧重方法论 |
| V2EX / 即刻 / 掘金 | 同一天或次日 | 中文受众，配中文 README |
| Twitter/X | HN 发的同时 | 简短 + 截图 + 链接 |

## 首发前检查清单

- [ ] README 微调（补 demo GIF/截图）
- [ ] GitHub topics/tags 添加（llm, rag, embedding, pipeline-parallelism, local-ai）
- [ ] LICENSE 文件确认存在
- [ ] PyPI 版本确认最新
- [ ] 至少 1 个 example 脚本可直接运行
- [ ] GitHub release 创建（v0.3.1 tag）
