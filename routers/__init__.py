"""FastAPI routers for Hippo API."""

from hippo.routers import (
    batch,
    embeddings,
    inference,
    management,
    metrics,
    models,
    system,
)

__all__ = ["models", "inference", "embeddings", "management", "system", "metrics", "batch"]
