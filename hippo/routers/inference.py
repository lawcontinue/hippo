"""Inference routes: /api/generate, /api/chat"""

import time
import json
import logging
import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from hippo.dependencies import _get_config, _get_manager, _check_auth
from hippo import metrics

logger = logging.getLogger("hippo")

router = APIRouter()


@router.post("/api/generate")
async def generate(request: Request):
    """Generate completion (Ollama compatible)."""
    # P0-2 fix: require auth for inference endpoints
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    config = _get_config(request)
    manager = _get_manager(request)
    body = await request.json()
    model_name = body.get("model", "")
    prompt = body.get("prompt", "")
    stream = body.get("stream", True)
    options = body.get("options", {})

    # Metrics: record start time
    start_time = time.time()

    try:
        llama = manager.get(model_name)
    except FileNotFoundError as e:
        # Metrics: record failed request
        metrics.inference_requests_total.labels(
            model=model_name or "unknown",
            endpoint="/api/generate",
            status="error"
        ).inc()
        return JSONResponse({"error": str(e)}, status_code=404)

    repeat_penalty = options.get("repeat_penalty", config.defaults.repeat_penalty)
    params = {
        "max_tokens": options.get("num_predict", config.defaults.max_tokens),
        "temperature": options.get("temperature", config.defaults.temperature),
        "top_p": options.get("top_p", 0.9),
        "repeat_penalty": repeat_penalty,
    }

    # P2-4 fix: inference timeout protection (default 120s)
    timeout_seconds = options.get("timeout", 120)

    if not stream:
        t0 = time.time_ns()
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: llama.create_completion(prompt=prompt, stream=False, **params)
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            # Metrics: record timeout
            duration = time.time() - start_time
            metrics.inference_requests_total.labels(
                model=model_name,
                endpoint="/api/generate",
                status="timeout"
            ).inc()
            metrics.inference_duration_seconds.labels(
                model=model_name,
                endpoint="/api/generate"
            ).observe(duration)
            return JSONResponse(
                {"error": f"Inference timed out after {timeout_seconds}s"},
                status_code=504,
            )
        except Exception as e:
            # Metrics: record error
            duration = time.time() - start_time
            metrics.inference_requests_total.labels(
                model=model_name,
                endpoint="/api/generate",
                status="error"
            ).inc()
            metrics.inference_duration_seconds.labels(
                model=model_name,
                endpoint="/api/generate"
            ).observe(duration)
            logger.exception("Inference error")
            return JSONResponse({"error": str(e)}, status_code=500)

        elapsed = time.time_ns() - t0
        duration = time.time() - start_time

        # Metrics: record successful request
        metrics.inference_requests_total.labels(
            model=model_name,
            endpoint="/api/generate",
            status="success"
        ).inc()
        metrics.inference_duration_seconds.labels(
            model=model_name,
            endpoint="/api/generate"
        ).observe(duration)

        return {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
            "response": result["choices"][0]["text"],
            "done": True,
            "done_reason": "stop",
            "context": [],
            "total_duration": elapsed,
        }

    def stream_generate():
        t0 = time.time_ns()
        try:
            full_response = ""
            for chunk in llama.create_completion(prompt=prompt, stream=True, **params):
                text = chunk["choices"][0].get("text", "")
                full_response += text
                yield json.dumps({
                    "model": model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                    "response": text,
                    "done": False,
                }) + "\n"

            elapsed = time.time_ns() - t0
            duration = time.time() - start_time

            # Metrics: record successful streaming request
            metrics.inference_requests_total.labels(
                model=model_name,
                endpoint="/api/generate",
                status="success"
            ).inc()
            metrics.inference_duration_seconds.labels(
                model=model_name,
                endpoint="/api/generate"
            ).observe(duration)

            yield json.dumps({
                "model": model_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                "response": "",
                "done": True,
                "done_reason": "stop",
                "context": [],
                "total_duration": elapsed,
            }) + "\n"
        except Exception as e:
            # Metrics: record streaming error
            duration = time.time() - start_time
            metrics.inference_requests_total.labels(
                model=model_name,
                endpoint="/api/generate",
                status="error"
            ).inc()
            metrics.inference_duration_seconds.labels(
                model=model_name,
                endpoint="/api/generate"
            ).observe(duration)
            logger.exception("Stream generate error")
            yield json.dumps({
                "model": model_name,
                "error": str(e),
                "done": True,
                "done_reason": "error",
            }) + "\n"

    return StreamingResponse(stream_generate(), media_type="application/x-ndjson")


@router.post("/api/chat")
async def chat(request: Request):
    """Chat completion (Ollama compatible)."""
    # P0-2 fix: require auth for inference endpoints
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    config = _get_config(request)
    manager = _get_manager(request)
    body = await request.json()
    model_name = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", True)
    options = body.get("options", {})

    # Metrics: record start time
    start_time = time.time()

    try:
        llama = manager.get(model_name)
    except FileNotFoundError as e:
        # Metrics: record failed request
        metrics.inference_requests_total.labels(
            model=model_name or "unknown",
            endpoint="/api/chat",
            status="error"
        ).inc()
        return JSONResponse({"error": str(e)}, status_code=404)

    repeat_penalty = options.get("repeat_penalty", config.defaults.repeat_penalty)
    params = {
        "max_tokens": options.get("num_predict", config.defaults.max_tokens),
        "temperature": options.get("temperature", config.defaults.temperature),
        "repeat_penalty": repeat_penalty,
    }

    # P2-4 fix: inference timeout protection (default 120s)
    timeout_seconds = options.get("timeout", 120)

    if not stream:
        t0 = time.time_ns()
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: llama.create_chat_completion(messages=messages, stream=False, **params)
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            # Metrics: record timeout
            duration = time.time() - start_time
            metrics.inference_requests_total.labels(
                model=model_name,
                endpoint="/api/chat",
                status="timeout"
            ).inc()
            metrics.inference_duration_seconds.labels(
                model=model_name,
                endpoint="/api/chat"
            ).observe(duration)
            return JSONResponse(
                {"error": f"Inference timed out after {timeout_seconds}s"},
                status_code=504,
            )
        except Exception as e:
            # Metrics: record error
            duration = time.time() - start_time
            metrics.inference_requests_total.labels(
                model=model_name,
                endpoint="/api/chat",
                status="error"
            ).inc()
            metrics.inference_duration_seconds.labels(
                model=model_name,
                endpoint="/api/chat"
            ).observe(duration)
            logger.exception("Inference error")
            return JSONResponse({"error": str(e)}, status_code=500)

        msg = result["choices"][0]["message"]
        elapsed = time.time_ns() - t0
        duration = time.time() - start_time

        # Metrics: record successful request
        metrics.inference_requests_total.labels(
            model=model_name,
            endpoint="/api/chat",
            status="success"
        ).inc()
        metrics.inference_duration_seconds.labels(
            model=model_name,
            endpoint="/api/chat"
        ).observe(duration)

        return {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
            "message": msg,
            "done": True,
            "done_reason": "stop",
            "total_duration": elapsed,
        }

    def stream_chat():
        try:
            for chunk in llama.create_chat_completion(messages=messages, stream=True, **params):
                delta = chunk["choices"][0].get("delta", {})
                yield json.dumps({
                    "model": model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                    "message": delta,
                    "done": False,
                }) + "\n"

            duration = time.time() - start_time

            # Metrics: record successful streaming request
            metrics.inference_requests_total.labels(
                model=model_name,
                endpoint="/api/chat",
                status="success"
            ).inc()
            metrics.inference_duration_seconds.labels(
                model=model_name,
                endpoint="/api/chat"
            ).observe(duration)

            yield json.dumps({
                "model": model_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            }) + "\n"
        except Exception as e:
            # Metrics: record streaming error
            duration = time.time() - start_time
            metrics.inference_requests_total.labels(
                model=model_name,
                endpoint="/api/chat",
                status="error"
            ).inc()
            metrics.inference_duration_seconds.labels(
                model=model_name,
                endpoint="/api/chat"
            ).observe(duration)
            logger.exception("Stream chat error")
            yield json.dumps({
                "model": model_name,
                "error": str(e),
                "done": True,
                "done_reason": "error",
            }) + "\n"

    return StreamingResponse(stream_chat(), media_type="application/x-ndjson")
