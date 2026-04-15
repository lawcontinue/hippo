"""Stats router: /api/stats endpoint for TUI dashboard.

Returns real-time statistics: QPS, memory usage, error rate, etc.
"""

import time

from fastapi import APIRouter, Request
from prometheus_client import REGISTRY

from hippo.dependencies import _get_manager, _get_config

router = APIRouter()

# Track request timing
_request_start_times = {}


def calculate_qps() -> float:
    """Calculate queries per second from Prometheus metrics."""
    try:
        # Get inference_requests_total counter
        for metric in REGISTRY.collect():
            if metric.name == "hippo_inference_requests_total":
                # Get total requests
                total = 0.0
                for sample in metric.samples:
                    if sample.name.endswith("_total"):
                        total += float(sample.value)
                # Simple QPS: requests / uptime (rough estimate)
                # For better accuracy, we'd need a sliding window
                return round(total, 2)
    except Exception:
        pass
    return 0.0


def calculate_memory_mb(manager) -> float:
    """Calculate total memory usage in MB."""
    try:
        total_mb = 0.0
        for entry in manager.list_loaded():
            # Get model size from manager
            model_path = manager._resolve_model_path(entry["name"])
            if model_path:
                size_bytes = model_path.stat().st_size
                total_mb += size_bytes / (1024 ** 2)
        return round(total_mb, 2)
    except Exception:
        return 0.0


def calculate_error_rate() -> float:
    """Calculate error rate percentage."""
    try:
        success_count = 0
        error_count = 0

        for metric in REGISTRY.collect():
            if metric.name == "hippo_inference_requests_total":
                for sample in metric.samples:
                    if "status" in sample.labels:
                        status = sample.labels["status"]
                        value = float(sample.value)
                        if status == "success":
                            success_count += value
                        else:
                            error_count += value

        total = success_count + error_count
        if total > 0:
            return round((error_count / total) * 100, 2)
    except Exception:
        pass
    return 0.0


def calculate_total_requests() -> int:
    """Calculate total number of requests."""
    try:
        total = 0
        for metric in REGISTRY.collect():
            if metric.name == "hippo_inference_requests_total":
                for sample in metric.samples:
                    total += int(float(sample.value))
        return total
    except Exception:
        return 0


@router.get("/api/stats")
async def get_stats(request: Request):
    """Get real-time statistics for TUI dashboard.

    Returns:
        - qps: Queries per second
        - memory_mb: Total memory usage in MB
        - error_rate: Error rate percentage
        - total_requests: Total number of requests
        - loaded_models: Number of loaded models
    """
    manager = _get_manager(request)
    config = _get_config(request)

    # Calculate statistics
    qps = calculate_qps()
    memory_mb = calculate_memory_mb(manager)
    error_rate = calculate_error_rate()
    total_requests = calculate_total_requests()
    loaded_models = len(manager.list_loaded())

    return {
        "qps": qps,
        "memory_mb": memory_mb,
        "error_rate": error_rate,
        "total_requests": total_requests,
        "loaded_models": loaded_models,
        "uptime_seconds": int(time.time() - getattr(manager, "_start_time", time.time())),
        "metrics_enabled": getattr(config, "metrics_enabled", True),
    }
