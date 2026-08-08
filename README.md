# Hippo 🦛

[![CI](https://github.com/lawcontinue/hippo/actions/workflows/ci.yml/badge.svg)](https://github.com/lawcontinue/hippo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hippo-llm.svg)](https://pypi.org/project/hippo-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`pip install hippo-llm` | [中文文档](./README_CN.md) | [Examples](./examples/)

Search your documents locally. BM25 works in 30 seconds, upgrade to hybrid when you need it.

No ChromaDB. No cloud API. No jieba. One `pip install`.

<p align="center">
  <img src="docs/demo_ui.png" width="80%" alt="Hippo search demo">
</p>

## 30-second search

```python
from hippo.embedding import VectorStore

# sparse mode: BM25 only, zero extra dependencies
store = VectorStore("docs.db")  # default mode="sparse"

store.add_batch([
    {"text": "Pipeline parallelism splits layers across devices"},
    {"text": "BM25 handles exact keyword matches"},
    {"text": "Speculative decoding improves latency by 2-3x"},
])

results = store.search("how to run big models on small GPUs", top_k=5)
for doc in results:
    print(f"[{doc.score:.3f}] {doc.text}")
```

No external vector DB. No embedding model download. SQLite for persistence. Works offline immediately.

> **→ Need semantic search?** `pip install hippo-llm[embedding]` and switch to `mode="hybrid"` — same API, adds dense vectors + RRF fusion. [See hybrid example ↓](#full-rag-example-with-local-llm-hybrid-mode)

**Chinese-optimized**: Built-in tokenizer with stop words. No jieba dependency.

```python
store.add_batch([
    {"text": "管道并行将模型层拆分到多台设备上"},
    {"text": "混合搜索结合了关键词匹配和语义相似度"},
])
results = store.search("怎么在低端显卡上跑大模型", top_k=3)
```

<details>
<summary>Hybrid mode: BM25 + dense embedding with RRF fusion</summary>

```bash
pip install hippo-llm[embedding]
```

```python
from hippo.embedding import EmbeddingEngine, VectorStore

engine = EmbeddingEngine(model="nomic-embed-text")  # local, no API key
store = VectorStore("docs.db", mode="hybrid", embedding_engine=engine)

store.add_batch([
    {"text": "Pipeline parallelism splits layers across devices"},
    {"text": "BM25 handles exact keyword matches"},
], engine=engine)

# RRF fusion: BM25 exact match + semantic similarity
results = store.search("how to run big models on small GPUs", engine=engine, top_k=5)
for doc in results:
    print(f"[{doc.score:.3f}] {doc.text}")
```

</details>

<details>
<summary>Full RAG example with local LLM (hybrid mode)</summary>

```python
from hippo.embedding import EmbeddingEngine, VectorStore
import openai

# 1. Index documents (one-time)
engine = EmbeddingEngine(model="nomic-embed-text")
store = VectorStore("knowledge.db", mode="hybrid", embedding_engine=engine)

documents = [
    "Hippo splits model layers across multiple devices using TCP.",
    "Each device only loads its shard of layers, reducing memory per device.",
    "The loop detector catches semantic repetition using Jaccard similarity.",
    "BM25 hybrid search combines keyword matching with semantic similarity.",
]
store.add_batch([{"text": d} for d in documents], engine=engine)

# 2. RAG query
query = "how does hippo handle memory?"
results = store.search(query, engine=engine, top_k=2)
context = "\n".join(doc.text for doc in results)

# 3. Generate answer with local LLM
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

</details>

## Why Hippo for embedding?

Every RAG pipeline, semantic router, and agent memory layer needs embeddings. Most people call cloud APIs (OpenAI, Cohere) and pay per token. Hippo runs everything locally.

| Problem | Hippo's answer |
|---------|---------------|
| Cloud embedding APIs: latency + cost + privacy | Local embedding, zero network calls |
| Installing ChromaDB + connector + embedding model separately | `VectorStore(mode="hybrid")` — one class, SQLite-backed |
| Chinese search needs jieba + extra config | Built-in tokenizer, zero config |
| BM25 vs dense — which to pick? | RRF fusion combines both, no choosing needed |
| >10K documents? | ANN index, sub-ms queries |
| Pipeline parallelism for big models | Split any GGUF across machines (Mac + PC mixed) |

**Real numbers from production use**:

| Metric | Value |
|--------|-------|
| BM25 query latency | <1ms |
| Dense search (bge-small-zh, 512d) | 5ms |
| Hybrid RRF fusion | <1ms overhead |
| ANN index build (10K docs) | ~2s |
| OOD accuracy (110 queries) | 85.5% top-1 (bge-small-zh), 92.7% (bge-m3) |
| Keyword + embedding fusion | 91.8% top-1 (互补, TOOLS #94) |

## Inference: run big models on cheap hardware

```bash
hippo-pipeline serve --model qwen3-30b-a3b-q3 --mode standalone
# → OpenAI-compatible API at localhost:8000/v1/chat/completions
```

<details>
<summary>Two-machine pipeline parallelism</summary>

```bash
# Machine 1
hippo-pipeline serve --model gemma-3-12b --mode pipeline --rank 0

# Machine 2
hippo-pipeline serve --model gemma-3-12b --mode pipeline --rank 1 \
  --coordinator http://192.168.1.10:9000
```

Split the model across machines. Plain TCP, no MPI. Mac + PC mixed.

</details>

## Acceleration Strategies

Hippo includes a unified acceleration abstraction (`acceleration.py`) that auto-selects the best inference acceleration strategy for your hardware:

| Strategy | Hardware | Description |
|----------|----------|-------------|
| **DFlash** | Apple Silicon (M-series) | KV injection via MLX, optimized for Mac Mini / MacBook |
| **MTP** | NVIDIA GPU | vLLM-style speculative decoding with multi-token prediction |
| **Pipeline** | Dual-machine (mixed) | Split model layers across 2+ machines via TCP (Mac + PC) |
| **None** | Fallback | No acceleration, baseline inference |

The `AccelerationOrchestrator` auto-detects your hardware and selects the best strategy — no manual configuration needed.

```python
from acceleration import AccelerationOrchestrator

orchestrator = AccelerationOrchestrator()
strategy = orchestrator.select(hardware="apple_silicon", language="chinese", task_type="chat")
# → DFlashStrategy
```

## What's inside

| Feature | Details |
|---------|---------|
| **Embedding + Hybrid Search** | Dense + BM25 + RRF fusion. SQLite-backed, sub-ms queries. |
| **Chinese-optimized BM25** | Built-in tokenizer with stop words. No jieba needed. |
| **ANN Index** | Approximate nearest neighbor for large collections (>10K docs). |
| **Pipeline Parallelism** | Split any GGUF model across N machines. → [hippo-pipeline](https://github.com/lawcontinue/hippo-pipeline) |
| **Loop Detection** | Jaccard-similarity detector catches semantic repetition. |
| **OpenAI-Compatible API** | Drop-in `/v1/chat/completions`. Works with LangChain, LlamaIndex. |
| **SafetyGuard** | 3-layer prompt injection defense: L1 regex → L2 TF-IDF → L3 embedding. |
| **Eval Toolkit** | Fusion evaluation, drift detection, reward assessment, chaos engineering. |
| **Auto Memory Budget** | Calculates shard splits from available VRAM automatically. |

## When to use Hippo

| You want... | Use this |
|-------------|----------|
| Search documents in 30 seconds | `VectorStore("docs.db")` — BM25, zero config |
| Search Chinese documents | Built-in tokenizer, zero config |
| Agent memory / semantic routing | `pip install hippo-llm[embedding]` → hybrid RRF |
| Local inference on one machine | `--mode standalone` with any GGUF model |
| Run a model too big for one device | `--mode pipeline` across 2+ machines |

## Install

```bash
pip install hippo-llm
```

Zero dependencies beyond numpy. BM25 search works immediately.

```bash
pip install hippo-llm[embedding]  # add dense vectors + hybrid RRF fusion
```

Requirements: Python 3.10+. Dense embedding uses [sentence-transformers](https://github.com/UKPLab/sentence-transformers) (auto-installed with `pip install hippo-llm[embedding]`). No external services needed.

### Configuration

Hippo supports environment variables for embedding model customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `HIPPO_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | Model name (HuggingFace ID or local path) |
| `HIPPO_EMBED_MODEL_PATH` | _(empty)_ | Override path to a local model directory (e.g. ModelScope cache). Takes priority over `HIPPO_EMBED_MODEL`. |
| `HIPPO_EMBED_DIM` | `512` | Embedding dimension. Set to `1024` if using bge-m3. |

**Speed vs Quality**:

| Profile | Model | Size | Latency | Accuracy (OOD 110q) |
|---------|-------|------|---------|---------------------|
| **Default (speed)** | `BAAI/bge-small-zh-v1.5` | 183MB | ~5ms | 85.5% top-1 |
| **Quality** | `BAAI/bge-m3` | 2.2GB | ~630ms | 92.7% top-1 |

To use the quality-first model:
```bash
export HIPPO_EMBED_MODEL=BAAI/bge-m3
export HIPPO_EMBED_DIM=1024
```

For offline / no-HuggingFace access, download the model locally (e.g. from [ModelScope](https://modelscope.cn)) and point to the directory:
```bash
export HIPPO_EMBED_MODEL_PATH=/path/to/bge-small-zh-v1.5
```

### Import paths: wheel vs source

The `hippo-llm` wheel flattens packages to top-level. Use these imports:

```python
# Wheel install (pip install hippo-llm)
from embedding.store import VectorStore
from embedding.engine import EmbeddingEngine
from embedding.memory_safety import add_with_source, search_with_confidence
from hippo.safety_guard import SafetyGuard
```

If running from a source checkout (not wheel):
```python
# Source tree
from hippo.embedding import VectorStore
```

## Roadmap

- **v0.3**: ANN index + Chinese tokenizer + hybrid RRF + sparse default ✅
- **v0.3.2**: SafetyGuard CN L1 fix + path unification + eval toolkit ✅
- **v0.4**: Built-in embedding models (bge-small-zh 5ms sweet spot), reranker, real-time routing (<10ms)
- **v0.5**: Agent memory layer (embedding-backed episodic memory) — [Design Doc](docs/V05_AGENT_MEMORY_DESIGN.md)
- **v0.6**: Multi-shard support (>2 devices), speculative decoding

## Evaluation Toolkit

`hippo.eval` provides deterministic evaluation for AI agent quality — no LLM-as-judge circularity.

```python
from hippo.eval import evaluate, DriftDetector, RewardEvaluator, FaultInjector

# 1. Rule-based quality gate (batch filter, borderline → LLM)
result = evaluate("your text here")
print(result.verdict)  # pass / needs_review / reject

# 2. Distribution drift detection (KL/JS divergence)
detector = DriftDetector()
detector.fit_reference(eval_distribution)
drift = detector.detect(production_distribution)
print(drift.is_drifted, drift.severity)

# 3. Reward design assessment (sparsity, conflict, Goodhart)
evaluator = RewardEvaluator()
report = evaluator.sparsity_report(signals)
print(report['overall_sparsity'], report['level'])

# 4. Chaos engineering (fault injection)
injector = FaultInjector.create_enabled()
injected = injector.inject(step, FaultType.NETWORK_TIMEOUT)
```

Install with: `pip install hippo-llm[eval]`

See [`examples/quickstart_eval.py`](examples/quickstart_eval.py) for a full runnable demo.

## Benchmarks

| Setup | Model | Speed |
|-------|-------|-------|
| Mac Mini M2 (16GB) | Qwen3-4B-Q4 | 41 tok/s |
| RTX 5060 Ti (16GB) | Qwen3-14B-Q4 | 41 tok/s |
| 2× Mac Mini (16GB each) | Qwen3-30B-A3B-Q3 | 78 tok/s |
| Mac Mini M2 (16GB) | Qwen3-30B-A3B-Q3 | 24 tok/s |

## License

MIT

## Author

lawcontinue — [GitHub](https://github.com/lawcontinue)
