"""FastAPI routers for Hippo API."""

from hippo.routers import models, inference, embeddings, management, system, metrics, batch

__all__ = ["models", "inference", "embeddings", "management", "system", "metrics", "batch"]
