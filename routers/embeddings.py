"""Embedding routes: /api/embeddings, /api/embed, /v1/embeddings"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from hippo.dependencies import _check_auth, _get_manager

logger = logging.getLogger("hippo")

router = APIRouter()


@router.post("/api/embeddings")
async def embeddings(request: Request):
    """Generate embeddings (Ollama compatible).

    Supports both legacy format (prompt) and new format (input).
    Returns a single embedding vector for the given text.
    """
    # P0-2 fix: require auth for inference endpoints
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    manager = _get_manager(request)
    body = await request.json()
    model_name = body.get("model", "")
    # Ollama accepts both "prompt" (legacy) and "input" (newer)
    prompt = body.get("prompt", "") or body.get("input", "")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""

    try:
        llama = manager.get(model_name)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)

    try:
        result = llama.create_embedding(prompt)
        embedding = result["data"][0]["embedding"]
        return {"model": model_name, "embedding": embedding}
    except Exception as e:
        logger.exception("Embedding error")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/embed")
async def embed(request: Request):
    """Generate embeddings (Ollama /api/embed format, batch support).

    Accepts {"model": "...", "input": "text" | ["text1", "text2"]}
    Returns {"model": "...", "embeddings": [[...], [...]]}
    """
    # P0-2 fix: require auth for inference endpoints
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    manager = _get_manager(request)
    body = await request.json()
    model_name = body.get("model", "")
    raw_input = body.get("input", body.get("prompt", ""))

    if isinstance(raw_input, str):
        inputs = [raw_input]
    elif isinstance(raw_input, list):
        inputs = raw_input
    else:
        return JSONResponse({"error": "input must be a string or list of strings"}, status_code=400)

    if not inputs:
        return JSONResponse({"error": "input cannot be empty"}, status_code=400)

    try:
        llama = manager.get(model_name)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)

    try:
        all_embeddings = []
        for text in inputs:
            result = llama.create_embedding(text)
            all_embeddings.append(result["data"][0]["embedding"])

        return {
            "model": model_name,
            "embeddings": all_embeddings,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
        }
    except Exception as e:
        logger.exception("Embed error")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/v1/embeddings")
async def embeddings_openai(request: Request):
    """Generate embeddings (OpenAI-compatible format).

    Accepts {"model": "...", "input": "text" | ["text1"] | [token_ids]}
    Returns OpenAI-style response with usage stats.
    """
    # P0-2 fix: require auth for inference endpoints
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    manager = _get_manager(request)
    body = await request.json()
    model_name = body.get("model", "")
    raw_input = body.get("input", "")

    if isinstance(raw_input, str):
        inputs = [raw_input]
    elif isinstance(raw_input, list):
        # Could be list of strings or list of token ids (ints)
        if raw_input and isinstance(raw_input[0], int):
            # Token IDs not supported, treat as error
            return JSONResponse(
                {"error": "Token ID input not supported, use text input"},
                status_code=400,
            )
        inputs = raw_input
    else:
        return JSONResponse({"error": "input must be a string or list"}, status_code=400)

    try:
        llama = manager.get(model_name)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)

    try:
        data = []
        total_tokens = 0
        for i, text in enumerate(inputs):
            result = llama.create_embedding(text)
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": result["data"][0]["embedding"],
            })
            total_tokens += result.get("usage", {}).get("prompt_tokens", 0)

        return {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }
    except Exception as e:
        logger.exception("OpenAI embedding error")
        return JSONResponse({"error": str(e)}, status_code=500)
