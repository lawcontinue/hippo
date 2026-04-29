"""Inference routes: /api/generate, /api/chat"""

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from hippo import metrics
from hippo.dependencies import _check_auth, _get_config, _get_manager

logger = logging.getLogger("hippo")

router = APIRouter()


def _get_predictor(request: Request):
    """Get the QueryPredictor from app state, if available."""
    return getattr(request.app.state, "predictor", None)


def _record_query(request: Request, endpoint: str, body: dict):
    """Record a query to the predictor (non-blocking, best-effort)."""
    try:
        predictor = _get_predictor(request)
        if predictor:
            predictor.record(endpoint, body)
    except Exception:
        pass  # predictor failures must not affect inference


def _check_predictor_cache(request: Request, endpoint: str, body: dict):
    """Check predictor cache for a pre-computed result. Returns dict or None."""
    try:
        predictor = _get_predictor(request)
        if predictor:
            return predictor.lookup(endpoint, body)
    except Exception:
        pass
    return None


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

    # Sleep-time Compute: record query for prediction
    _record_query(request, "/api/generate", body)

    # Sleep-time Compute: check predictor cache (non-streaming only)
    if not stream:
        cached = _check_predictor_cache(request, "/api/generate", body)
        if cached:
            cached["model"] = model_name
            cached["cached"] = True
            return cached

    # Metrics: record start time
    start_time = time.time()
    queue_start = time.time()

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

    queue_time_ms = (time.time() - queue_start) * 1000  # model load wait time

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
        generation_time_ms = elapsed / 1e6

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
            "timings": {
                "queue_time_ms": round(queue_time_ms, 2),
                "ttft_ms": round(generation_time_ms, 2),  # non-streaming: TTFT = total
                "generation_time_ms": round(generation_time_ms, 2),
                "mean_itl_ms": 0,  # non-streaming: no inter-token latency
            },
        }

    def stream_generate():
        t0 = time.time_ns()
        ttft_ns = None
        token_times = []
        try:
            full_response = ""
            for chunk in llama.create_completion(prompt=prompt, stream=True, **params):
                text = chunk["choices"][0].get("text", "")
                if ttft_ns is None:
                    ttft_ns = time.time_ns() - t0
                token_times.append(time.time_ns())
                full_response += text
                yield json.dumps({
                    "model": model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                    "response": text,
                    "done": False,
                }) + "\n"

            elapsed = time.time_ns() - t0
            duration = time.time() - start_time
            generation_time_ms = (elapsed - (ttft_ns or elapsed)) / 1e6
            mean_itl_ms = 0.0
            if len(token_times) > 1:
                itls = [(token_times[i+1] - token_times[i]) / 1e6 for i in range(len(token_times)-1)]
                mean_itl_ms = sum(itls) / len(itls)

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
                "timings": {
                    "queue_time_ms": round(queue_time_ms, 2),
                    "ttft_ms": round((ttft_ns or elapsed) / 1e6, 2),
                    "generation_time_ms": round(generation_time_ms, 2),
                    "mean_itl_ms": round(mean_itl_ms, 2),
                },
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

    # Sleep-time Compute: record query for prediction
    _record_query(request, "/api/chat", body)

    # Sleep-time Compute: check predictor cache (non-streaming only)
    if not stream:
        cached = _check_predictor_cache(request, "/api/chat", body)
        if cached:
            cached["model"] = model_name
            cached["cached"] = True
            return cached

    # Metrics: record start time
    start_time = time.time()
    queue_start = time.time()

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

    queue_time_ms = (time.time() - queue_start) * 1000

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
        generation_time_ms = elapsed / 1e6

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
            "timings": {
                "queue_time_ms": round(queue_time_ms, 2),
                "ttft_ms": round(generation_time_ms, 2),
                "generation_time_ms": round(generation_time_ms, 2),
                "mean_itl_ms": 0,  # non-streaming: no inter-token latency
            },
        }

    def stream_chat():
        t0 = time.time_ns()
        ttft_ns = None
        token_times = []
        try:
            for chunk in llama.create_chat_completion(messages=messages, stream=True, **params):
                if ttft_ns is None:
                    ttft_ns = time.time_ns() - t0
                token_times.append(time.time_ns())
                delta = chunk["choices"][0].get("delta", {})
                yield json.dumps({
                    "model": model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
                    "message": delta,
                    "done": False,
                }) + "\n"

            duration = time.time() - start_time
            elapsed = time.time_ns() - t0
            generation_time_ms = (elapsed - (ttft_ns or elapsed)) / 1e6
            mean_itl_ms = 0.0
            if len(token_times) > 1:
                itls = [(token_times[i+1] - token_times[i]) / 1e6 for i in range(len(token_times)-1)]
                mean_itl_ms = sum(itls) / len(itls)

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
                "timings": {
                    "queue_time_ms": round(queue_time_ms, 2),
                    "ttft_ms": round((ttft_ns or elapsed) / 1e6, 2),
                    "generation_time_ms": round(generation_time_ms, 2),
                    "mean_itl_ms": round(mean_itl_ms, 2),
                },
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
