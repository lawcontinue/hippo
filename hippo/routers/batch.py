"""Batch inference routes: /api/batch/generate, /api/batch/chat"""

import json
import time
import logging
import asyncio
from typing import Any, Dict, Callable, Awaitable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from hippo.dependencies import _get_config, _get_manager, _check_auth
from hippo import metrics

logger = logging.getLogger("hippo")

router = APIRouter()


async def _do_single_generate(
    manager,
    model_name: str,
    prompt: str,
    options: Dict[str, Any],
    config,
) -> Dict[str, Any]:
    """Execute a single generate request."""
    try:
        llama = manager.get(model_name)
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e)}

    repeat_penalty = options.get("repeat_penalty", config.defaults.repeat_penalty)
    params = {
        "max_tokens": options.get("num_predict", config.defaults.max_tokens),
        "temperature": options.get("temperature", config.defaults.temperature),
        "top_p": options.get("top_p", 0.9),
        "repeat_penalty": repeat_penalty,
    }

    timeout_seconds = options.get("timeout", config.batch.timeout_seconds)

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: llama.create_completion(prompt=prompt, stream=False, **params)
            ),
            timeout=timeout_seconds,
        )
        return {"status": "success", "response": result["choices"][0]["text"]}
    except asyncio.TimeoutError:
        return {"status": "error", "error": f"Timed out after {timeout_seconds}s"}
    except Exception as e:
        logger.exception("Batch generate error")
        return {"status": "error", "error": str(e)}


async def _do_single_chat(
    manager,
    model_name: str,
    messages: list,
    options: Dict[str, Any],
    config,
) -> Dict[str, Any]:
    """Execute a single chat request."""
    try:
        llama = manager.get(model_name)
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e)}

    repeat_penalty = options.get("repeat_penalty", config.defaults.repeat_penalty)
    params = {
        "max_tokens": options.get("num_predict", config.defaults.max_tokens),
        "temperature": options.get("temperature", config.defaults.temperature),
        "repeat_penalty": repeat_penalty,
    }

    timeout_seconds = options.get("timeout", config.batch.timeout_seconds)

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: llama.create_chat_completion(messages=messages, stream=False, **params)
            ),
            timeout=timeout_seconds,
        )
        msg = result["choices"][0]["message"]
        return {"status": "success", "response": msg}
    except asyncio.TimeoutError:
        return {"status": "error", "error": f"Timed out after {timeout_seconds}s"}
    except Exception as e:
        logger.exception("Batch chat error")
        return {"status": "error", "error": str(e)}


async def _batch_execute(
    request: Request,
    endpoint: str,
    process_fn: Callable[[Any], Awaitable[Dict[str, Any]]],
) -> JSONResponse:
    """P1-2: shared batch execution logic (eliminates code duplication).

    Args:
        request: FastAPI request object.
        endpoint: Endpoint path for metrics labels.
        process_fn: Async callable that takes a single request dict and returns a result dict.

    Returns:
        JSONResponse with batch results.
    """
    # Authentication required
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    config = _get_config(request)

    # P0-2 fix: limit request body size to prevent OOM (default 10MB)
    raw_body = await request.body()
    max_body_bytes = getattr(config, "batch_max_body_bytes", 10 * 1024 * 1024)
    if len(raw_body) > max_body_bytes:
        return JSONResponse(
            {"error": f"Request body {len(raw_body)} bytes exceeds limit {max_body_bytes}"},
            status_code=413,
        )

    body = json.loads(raw_body)
    requests_list = body.get("requests", [])
    max_concurrent = body.get("max_concurrent", config.batch.max_concurrent)

    # Validation: batch size limit
    batch_max_size = config.batch_max_size
    if len(requests_list) > batch_max_size:
        return JSONResponse(
            {"error": f"Batch size {len(requests_list)} exceeds maximum {batch_max_size}"},
            status_code=400,
        )

    # Validation: max_concurrent limit
    if max_concurrent > 16:
        return JSONResponse(
            {"error": f"max_concurrent {max_concurrent} exceeds maximum 16"},
            status_code=400,
        )

    start_time = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(req: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            result = await process_fn(req)
            # Metrics: record result
            metrics.inference_requests_total.labels(
                model=req.get("model", "") or "unknown",
                endpoint=endpoint,
                status=result["status"],
            ).inc()
            return {
                "response": result.get("response", ""),
                "status": result["status"],
                "error": result.get("error"),
            }

    results = await asyncio.gather(
        *[run_with_semaphore(req) for req in requests_list],
        return_exceptions=True,
    )

    # Handle unexpected exceptions
    processed_results = []
    for r in results:
        if isinstance(r, Exception):
            processed_results.append({
                "response": "",
                "status": "error",
                "error": f"Unexpected error: {str(r)}",
            })
        else:
            processed_results.append(r)

    success_count = sum(1 for r in processed_results if r["status"] == "success")
    error_count = sum(1 for r in processed_results if r["status"] == "error")
    total_duration_ns = time.time_ns() - int(start_time * 1e9)

    # Metrics: record batch duration
    metrics.inference_duration_seconds.labels(
        model="batch", endpoint=endpoint,
    ).observe(time.time() - start_time)

    return JSONResponse({
        "results": processed_results,
        "total_duration_ns": total_duration_ns,
        "success_count": success_count,
        "error_count": error_count,
    })


@router.post("/api/batch/generate")
async def batch_generate(request: Request):
    """Batch generate completion (multiple requests in parallel)."""
    config = _get_config(request)
    manager = _get_manager(request)

    async def process_fn(req: Dict[str, Any]) -> Dict[str, Any]:
        return await _do_single_generate(
            manager, req.get("model", ""), req.get("prompt", ""),
            req.get("options", {}), config,
        )

    return await _batch_execute(request, "/api/batch/generate", process_fn)


@router.post("/api/batch/chat")
async def batch_chat(request: Request):
    """Batch chat completion (multiple requests in parallel)."""
    config = _get_config(request)
    manager = _get_manager(request)

    async def process_fn(req: Dict[str, Any]) -> Dict[str, Any]:
        return await _do_single_chat(
            manager, req.get("model", ""), req.get("messages", []),
            req.get("options", {}), config,
        )

    return await _batch_execute(request, "/api/batch/chat", process_fn)
