"""
Quick start: RAG with local LLM + Hippo search.

Prerequisites:
    1. Start a local LLM server (e.g. hippo-pipeline serve)
    2. Start Hippo server: hippo-pipeline serve --model qwen3-4b-q4 --mode standalone
    3. Run this script: python3 examples/quickstart_rag.py
"""

import openai
from hippo.embedding import EmbeddingEngine, VectorStore

# 1. Index your documents (one-time)
engine = EmbeddingEngine(model="nomic-embed-text")
store = VectorStore("knowledge.db", mode="hybrid")

documents = [
    "Hippo splits model layers across multiple devices using plain TCP.",
    "Each device only loads its shard of layers, reducing memory per device.",
    "The loop detector catches semantic repetition using Jaccard similarity.",
    "BM25 hybrid search combines keyword matching with semantic similarity via RRF fusion.",
    "Chinese tokenizer uses character bigrams with stop words, no jieba needed.",
    "ANN index supports >10K documents with approximate nearest neighbor search.",
]
store.add_batch([{"text": d} for d in documents], engine=engine)
print("Indexed documents.\n")

# 2. RAG query
query = "how does hippo handle memory and search?"
results = store.search(query, engine=engine, top_k=2)
context = "\n".join(doc.text for doc in results)
print(f"Retrieved context:\n{context}\n")

# 3. Generate answer with local LLM
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="qwen3-4b",
    messages=[
        {"role": "system", "content": f"Answer based on this context:\n{context}"},
        {"role": "user", "content": query},
    ],
    max_tokens=300,
)
print(f"Answer:\n{response.choices[0].message.content}")
