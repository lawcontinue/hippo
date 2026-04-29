"""Prometheus metrics for Hippo."""

from prometheus_client import Counter, Gauge, Histogram

# Inference request counter
inference_requests_total = Counter(
    "hippo_inference_requests_total",
    "Total number of inference requests",
    ["model", "endpoint", "status"],
)

# Inference duration histogram
inference_duration_seconds = Histogram(
    "hippo_inference_duration_seconds",
    "Inference request duration in seconds",
    ["model", "endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

# Models loaded gauge — updated by ModelManager._load() and unload()
models_loaded = Gauge(
    "hippo_models_loaded",
    "Whether a model is currently loaded (1=loaded, 0=unloaded)",
    ["model"],
)

# Memory usage gauge — updated by ModelManager._load() and unload()
memory_usage_bytes = Gauge(
    "hippo_memory_usage_bytes",
    "Memory usage in bytes by model (estimated from file size)",
    ["model"],
)
