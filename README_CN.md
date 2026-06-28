# Hippo 🦛

`pip install hippo-llm` | Python 3.10+ | MIT

搜索你的文档。BM25 开箱即用，按需升级 Hybrid。不需要 ChromaDB。不需要云 API。不需要 jieba。一个 `pip install` 全搞定。

[English](./README.md)

## 30 秒上手

```python
from hippo.embedding import VectorStore

store = VectorStore("docs.db")  # 默认 sparse，BM25 即用

store.add_batch([
    {"text": "管道并行将模型层拆分到多台设备上"},
    {"text": "BM25 处理精确关键词匹配"},
    {"text": "混合搜索结合了关键词匹配和语义相似度"},
])

results = store.search("怎么跑大模型", top_k=5)
for doc in results:
    print(f"[{doc.score:.3f}] {doc.text}")
```

不需要向量数据库。不需要 embedding 模型下载。SQLite 持久化，开箱即用。

> **→ 需要语义搜索？** `pip install hippo-llm[embedding]` 切换 `mode="hybrid"`，同一个 API，加了 dense 向量 + RRF 融合。

**中文搜索友好**——内置分词器和停用词表，不需要 jieba，零配置。

<details>
<summary>双机部署</summary>

```bash
# 机器 1
hippo-pipeline serve --model gemma-3-12b --mode pipeline --rank 0

# 机器 2
hippo-pipeline serve --model gemma-3-12b --mode pipeline --rank 1 \
  --coordinator http://192.168.1.10:9000
```

模型拆成两半，每台只加载自己的分片。跑单机装不下的模型。

</details>

## 一个包搞定推理 + 搜索

搭 RAG 通常要装两个服务：sentence-transformers 做向量 + ChromaDB 做向量库。Hippo 一个 `pip install` 全给你。

```python
from hippo.embedding import EmbeddingEngine, VectorStore

engine = EmbeddingEngine(model="nomic-embed-text")  # 本地 sentence-transformers，无需外部服务
store = VectorStore("docs.db", mode="hybrid")  # BM25 + 语义混合检索

# 添加文档
store.add_batch([
    {"text": "流水线并行把模型层拆分到多台设备", "metadata": {"source": "readme"}},
    {"text": "BM25 负责精确关键词匹配", "metadata": {"source": "docs"}},
    {"text": "投机解码可以把延迟降低 2-3 倍", "metadata": {"source": "benchmarks"}},
], engine=engine)

# 混合搜索（BM25 + 语义，RRF 融合）
results = store.search("怎么在小显卡上跑大模型", engine=engine, top_k=5)
for doc in results:
    print(f"[{doc.score:.3f}] {doc.text}")
```

不需要额外的向量数据库。SQLite 持久化，numpy 算相似度，离线也能用。

**中文搜索友好**——内置中文分词器和停用词表，不需要装 jieba，开箱即用。

<details>
<summary>完整 RAG 示例（推理 + 搜索联动）</summary>

```python
from hippo.embedding import EmbeddingEngine, VectorStore
import openai

# 1. 建索引（一次性）
engine = EmbeddingEngine(model="nomic-embed-text")
store = VectorStore("knowledge.db", mode="hybrid")

documents = [
    "Hippo 通过 TCP 把模型层拆分到多台设备上。",
    "每台设备只加载自己那部分层，降低单机内存需求。",
    "循环检测器用 Jaccard 相似度捕捉语义重复。",
    "BM25 混合搜索结合了关键词匹配和语义相似度。",
]
store.add_batch([{"text": d} for d in documents], engine=engine)

# 2. 检索
query = "hippo 怎么处理内存问题？"
results = store.search(query, engine=engine, top_k=2)
context = "\n".join(doc.text for doc in results)

# 3. 生成回答
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="qwen3-30b-a3b-q3",
    messages=[
        {"role": "system", "content": f"根据以下内容回答问题：\n{context}"},
        {"role": "user", "content": query}
    ]
)
print(response.choices[0].message.content)
```

</details>

## 功能一览

| 功能 | 说明 |
|------|------|
| **流水线并行** | 把 HF 模型拆到 N 台机器上。Mac 和 PC 可以混用。纯 TCP 通信，不需要 MPI。 |
| **循环检测** | Jaccard 相似度检测语义重复，`repeat_penalty` 抓不到的它抓得到。 |
| **Embedding & 搜索** | 语义向量 + BM25 + 混合 RRF 融合。SQLite 存储，亚毫秒查询。 |
| **中文 BM25** | 内置中文分词器和停用词，不需要 jieba。 |
| **ANN 索引** | 大规模文档（>1万条）的近似最近邻检索。 |
| **OpenAI 兼容 API** | 直接对接 `/v1/chat/completions`。兼容 LangChain、LlamaIndex 等工具。 |
| **自动显存预算** | 根据可用显存自动计算模型分片方案。 |

## 适用场景

| 你想... | 用这个 |
|---------|--------|
| 单机本地推理 | `--mode standalone` + 任意 GGUF 模型 |
| 跑单机装不下的大模型 | `--mode pipeline` 跨 2+ 台设备 |
| 搭 RAG 但不想装 ChromaDB | `VectorStore(mode="hybrid")` |
| 搜中文文档 | 内置中文分词的 BM25 |

## 安装

```bash
pip install hippo-llm
```

需要：Python 3.10+。语义搜索需要 `pip install hippo-llm[embedding]`（自动装 sentence-transformers）。无需 Ollama 或任何外部服务。

## 性能基准

| 配置 | 模型 | 速度 |
|------|------|------|
| Mac Mini M2 (16GB) | Qwen3-4B-Q4 | 41 tok/s |
| RTX 5060 Ti (16GB) | Qwen3-14B-Q4 | 41 tok/s |
| 2× Mac Mini (各16GB) | Qwen3-30B-A3B-Q3 | 78 tok/s |
| Mac Mini M2 (16GB) | Qwen3-30B-A3B-Q3 | 24 tok/s |

## 路线图

- **v0.3**: ANN 索引（>1万文档） ✅
- **v0.4**: 多设备分片（>2台）、自动层均衡
- **v0.5**: 跨设备投机解码
- **v0.6**: 内置模型下载 + GGUF 自动转换

## 许可证

MIT

## 作者

lawcontinue — [GitHub](https://github.com/lawcontinue)
