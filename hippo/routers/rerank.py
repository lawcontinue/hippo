"""Rerank routes: /api/rerank, /v1/rerank

Supports reranker models (e.g., bge-reranker, ms-marco-MiniLM)
loaded via llama-cpp. Uses cross-encoder scoring: concatenates
query + document and scores the pair.

API format follows Jina/Cohere rerank convention:
    POST /api/rerank
    {
        "model": "bge-reranker-v2-m3",
        "query": "What is deep learning?",
        "documents": ["Deep learning is...", "Machine learning is..."],
        "top_n": 3  // optional
    }

Reference: ollama/ollama#3368 (reranking model support)
"""

import time
import logging
import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from hippo.dependencies import _get_manager, _check_auth
from hippo import metrics

logger = logging.getLogger("hippo")

router = APIRouter()

# Keywords for auto-detecting reranker models from name
# Maximum number of documents per rerank request (DoS protection)
MAX_DOCUMENTS = 1000

# Default prompt template for cross-encoder scoring.
# Users can override via prompt_template in the request body.
# Placeholders: {query} and {document}
DEFAULT_PROMPT_TEMPLATE = "query: {query}\ndocument: {document}"


def _score_single(llama, query: str, document: str, prompt_template: str) -> float:
    """Score a single query-document pair using cross-encoder.

    Returns a relevance score (higher = more relevant).
    Uses the logit of the [CLS] token or the first token's probability
    as the relevance signal, depending on model output.

    Args:
        llama: Loaded llama-cpp model instance.
        query: Search query string.
        document: Document text to score against query.
        prompt_template: Format string with {query} and {document} placeholders.
    """
    prompt = prompt_template.format(query=query, document=document)

    # Try with logprobs first; falls back to no-logprobs if model doesn't support it
    try:
        result = llama.create_completion(
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            echo=False,
            logprobs=1,
        )
    except ValueError:
        # Model loaded without logits_all — score without logprobs
        result = llama.create_completion(
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            echo=False,
        )

    # Extract score from model output
    # Strategy 1: Most reranker models output a numeric score as text
    text = result["choices"][0].get("text", "").strip()

    try:
        return float(text)
    except (ValueError, TypeError):
        pass

    # Strategy 2: Use logprob of first token as relevance signal
    # Note: requires model loaded with logits_all=True
    logprobs_data = result["choices"][0].get("logprobs")
    if logprobs_data and logprobs_data.get("tokens"):
        token_logprobs = logprobs_data.get("token_logprobs", [])
        if token_logprobs and token_logprobs[0] is not None:
            return token_logprobs[0]

    # Strategy 3: No usable signal — return negative infinity
    # Using -inf instead of 0.0 so unscorable docs sort to the bottom
    return float("-inf")


@router.post("/api/rerank")
async def rerank(request: Request):
    """Rerank documents by relevance to a query.

    Supports burst-aware on-demand loading: reranker models are loaded
    on first request and auto-unloaded after idle timeout (like all models).
    """
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    manager = _get_manager(request)
    body = await request.json()

    model_name = body.get("model", "")
    query = body.get("query", "")
    documents = body.get("documents", [])
    top_n = body.get("top_n", len(documents))
    prompt_template = body.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)

    if not model_name:
        return JSONResponse({"error": "model is required"}, status_code=400)
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)
    if not documents or not isinstance(documents, list):
        return JSONResponse({"error": "documents must be a non-empty list"}, status_code=400)
    if len(documents) > MAX_DOCUMENTS:
        return JSONResponse(
            {"error": f"too many documents ({len(documents)}), max is {MAX_DOCUMENTS}"},
            status_code=400,
        )

    # Validate prompt_template has required placeholders
    try:
        prompt_template.format(query="test", document="test")
    except (KeyError, IndexError):
        return JSONResponse(
            {"error": "prompt_template must contain {query} and {document} placeholders"},
            status_code=400,
        )

    # Clamp top_n
    top_n = min(max(1, top_n), len(documents))

    start_time = time.time()
    queue_start = time.time()

    try:
        llama = manager.get(model_name)
    except FileNotFoundError as e:
        metrics.inference_requests_total.labels(
            model=model_name, endpoint="/api/rerank", status="error"
        ).inc()
        return JSONResponse({"error": str(e)}, status_code=404)

    queue_time_ms = (time.time() - queue_start) * 1000

    # Score all documents in executor to avoid blocking event loop
    try:
        scores = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: [_score_single(llama, query, doc, prompt_template) for doc in documents],
        )
    except Exception as e:
        duration = time.time() - start_time
        metrics.inference_requests_total.labels(
            model=model_name, endpoint="/api/rerank", status="error"
        ).inc()
        metrics.inference_duration_seconds.labels(
            model=model_name, endpoint="/api/rerank"
        ).observe(duration)
        logger.exception("Rerank scoring error")
        return JSONResponse({"error": str(e)}, status_code=500)

    # Build results sorted by score descending
    scored = [
        {"index": i, "relevance_score": round(scores[i], 4), "document": {"text": documents[i]}}
        for i in range(len(documents))
    ]
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    results = scored[:top_n]

    duration = time.time() - start_time
    scoring_time_ms = duration * 1000 - queue_time_ms

    metrics.inference_requests_total.labels(
        model=model_name, endpoint="/api/rerank", status="success"
    ).inc()
    metrics.inference_duration_seconds.labels(
        model=model_name, endpoint="/api/rerank"
    ).observe(duration)

    return {
        "model": model_name,
        "results": results,
        "total_documents": len(documents),
        "timings": {
            "queue_time_ms": round(queue_time_ms, 2),
            "scoring_time_ms": round(scoring_time_ms, 2),
            "total_time_ms": round(duration * 1000, 2),
        },
    }
