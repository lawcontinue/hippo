# Hippo 🦛

`pip install hippo-llm` | Python 3.10+ | MIT

Run 30B models on a ¥3800 GPU at 78 tok/s. Then search through your documents without installing ChromaDB.

## 30-second setup

```bash
hippo-pipeline serve --model qwen3-30b-a3b-q3 --mode standalone
# → OpenAI-compatible API at localhost:8000/v1/chat/completions
```

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="qwen3-30b-a3b-q3",
    messages=[{"role": "user", "content": "Explain pipeline parallelism"}],
    max_tokens=500
)
print(r.choices[0].message.content)
```

<details>
<summary>Two-machine setup</summary>

```bash
# Machine 1
hippo-pipeline serve --model gemma-3-12b --mode pipeline --rank 0

# Machine 2
hippo-pipeline serve --model gemma-3-12b --mode pipeline --rank 1 \
  --coordinator http://192.168.1.10:9000
```

Split the model across machines. Run what doesn't fit on one GPU.

</details>

## One install for inference + search

Most RAG setups need two services: Ollama for inference + ChromaDB for vectors. Hippo gives you both in one `pip install`.

```python
from hippo.embedding import EmbeddingEngine, VectorStore

engine = EmbeddingEngine(model="nomic-embed-text")  # uses local Ollama
store = VectorStore("docs.db", mode="hybrid")  # BM25 + dense RRF fusion

# Add documents
store.add_batch([
    {"text": "Pipeline parallelism splits layers across devices", "metadata": {"source": "readme"}},
    {"text": "BM25 handles exact keyword matches", "metadata": {"source": "docs"}},
    {"text": "Speculative decoding improves latency by 2-3x", "metadata": {"source": "benchmarks"}},
], engine=engine)

# Hybrid search (BM25 + semantic, RRF fused)
results = store.search("how to run big models on small GPUs", engine=engine, top_k=5)
for doc in results:
    print(f"[{doc.score:.3f}] {doc.text}")
```

No external vector DB. SQLite for persistence, numpy for similarity. Works offline.

<details>
<summary>Full RAG example with local LLM</summary>

```python
from hippo.embedding import EmbeddingEngine, VectorStore
import openai

# 1. Index your documents (one-time)
engine = EmbeddingEngine(model="nomic-embed-text")
store = VectorStore("knowledge.db", mode="hybrid")

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

## What's inside

| Feature | Details |
|---------|---------|
| **Pipeline Parallelism** | Split any HF model across N machines. Mac + PC mixed. Plain TCP, no MPI. |
| **Loop Detection** | Jaccard-similarity detector catches semantic repetition that `repeat_penalty` misses. |
| **Embedding & Search** | Dense + BM25 + hybrid RRF fusion. SQLite-backed, sub-ms queries. |
| **Chinese-optimized BM25** | Built-in Chinese tokenizer with stop words. No jieba needed. |
| **ANN Index** | Approximate nearest neighbor for large collections (>10K docs). |
| **OpenAI-Compatible API** | Drop-in `/v1/chat/completions`. Works with LangChain, LlamaIndex, anything. |
| **Auto Memory Budget** | Calculates shard splits from available VRAM automatically. |

## When to use Hippo

| You want... | Use this |
|-------------|----------|
| Local inference on one machine | `--mode standalone` with any GGUF model |
| Run a model too big for one device | `--mode pipeline` across 2+ machines |
| RAG without installing ChromaDB | `VectorStore(mode="hybrid")` |
| Search Chinese documents | BM25 with built-in tokenizer |

## Install

```bash
pip install hippo-llm
```

Requirements: Python 3.10+, [Ollama](https://ollama.ai) running locally for model weights and embeddings.

## Roadmap

- **v0.3**: ANN index for >10K document collections ✅
- **v0.4**: Multi-shard support (>2 devices), automatic layer balancing
- **v0.5**: Speculative decoding across shards
- **v0.6**: Built-in model download + GGUF auto-conversion

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
