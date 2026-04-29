"""Batch inference routes: /api/batch/* — Continuous batching support.

作者：忒弥斯 (T-Mind) 🔮
创建日期：2026-04-22
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from hippo.continuous_batch import ContinuousBatchEngine as CBE
from hippo.dependencies import _get_config

logger = logging.getLogger("hippo")

router = APIRouter()


def _get_batch_engine(request: Request, model_name: str) -> Optional[ContinuousBatchEngine]:
    """Get or create batch engine for a model."""
    # Lazy-load batch engines
    if not hasattr(request.app.state, "batch_engines"):
        request.app.state.batch_engines = {}

    batch_engines = request.app.state.batch_engines

    if model_name not in batch_engines:
        config = _get_config(request)
        # Find model path
        model_path = config.routes.get(model_name, model_name)

        try:
            engine = CBE(
                model_path=model_path,
                max_slots=7,  # Family members
                max_memory_gb=14.0  # Safety margin
            )
            batch_engines[model_name] = engine
            logger.info(f"Created batch engine for {model_name}")
        except Exception as e:
            logger.error(f"Failed to create batch engine for {model_name}: {e}")
            return None

    return batch_engines.get(model_name)


@router.post("/api/batch/submit")
async def batch_submit(request: Request):
    """Submit a request to the batch queue.

    Request body:
    {
        "model": "gemma-3-12b-it-qat-4bit",
        "prompt": "Hello world",
        "max_tokens": 100
    }

    Response:
    {
        "request_id": "uuid",
        "queue_position": 3,
        "status": "queued"
    }
    """
    body = await request.json()
    model_name = body.get("model", "")
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 100)

    if not model_name:
        raise HTTPException(status_code=400, detail="model is required")

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # Get batch engine
    engine = _get_batch_engine(request, model_name)
    if not engine:
        raise HTTPException(status_code=500, detail=f"Failed to create batch engine for {model_name}")

    # Create request
    req = Request(
        id=str(uuid.uuid4()),
        prompt=prompt,
        max_tokens=max_tokens
    )

    # Submit to queue
    if not engine.submit(req):
        raise HTTPException(status_code=503, detail="Queue full or memory insufficient")

    return {
        "request_id": req.id,
        "queue_position": len(engine.queue),
        "status": "queued"
    }


@router.post("/api/batch/run")
async def batch_run(request: Request):
    """Run a batch (process all queued requests).

    Response:
    {
        "results": [
            {
                "request_id": "uuid",
                "text": "generated text",
                "tokens": [123, 456, ...],
                "elapsed_sec": 10.5,
                "tok_s": 9.5
            }
        ]
    }
    """
    body = await request.json()
    model_name = body.get("model", "")

    if not model_name:
        raise HTTPException(status_code=400, detail="model is required")

    # Get batch engine
    engine = _get_batch_engine(request, model_name)
    if not engine:
        raise HTTPException(status_code=500, detail=f"Batch engine not found for {model_name}")

    # Run batch
    results = engine.run_batch()

    # Convert to dict
    return {
        "results": [
            {
                "request_id": r.request_id,
                "text": r.text,
                "tokens": r.tokens,
                "elapsed_sec": r.elapsed_sec,
                "tok_s": r.tok_s
            }
            for r in results
        ]
    }


@router.get("/api/batch/stats")
async def batch_stats(request: Request):
    """Get batch engine statistics.

    Response:
    {
        "queue_size": 3,
        "active_requests": 5,
        "memory_gb": 7.5,
        "memory_peak_gb": 8.2
    }
    """
    model_name = request.query_params.get("model", "")

    if not model_name:
        raise HTTPException(status_code=400, detail="model is required")

    # Get batch engine
    engine = _get_batch_engine(request, model_name)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Batch engine not found for {model_name}")

    stats = engine.stats()

    return {
        "queue_size": stats["queue_size"],
        "active_requests": stats["active_requests"],
        "memory_gb": stats["memory_gb"],
        "memory_peak_gb": stats["memory_peak_gb"]
    }
