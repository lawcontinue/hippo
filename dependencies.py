"""Shared dependency functions for FastAPI routes."""

from fastapi import Request
from fastapi.responses import JSONResponse

from hippo.audit import AuditLogger
from hippo.config import HippoConfig
from hippo.model_manager import ModelManager


def _get_audit(request: Request) -> AuditLogger:
    """Get audit logger from app state."""
    return getattr(request.app.state, "audit", AuditLogger(None))


def _get_config(request: Request) -> HippoConfig:
    """Get config from app state."""
    return request.app.state.config


def _get_manager(request: Request) -> ModelManager:
    """Get model manager from app state."""
    return request.app.state.manager


async def _check_auth(request: Request):
    """Verify API key for write operations if HIPPO_API_KEY is set.

    Returns None if auth succeeds or is not required.
    Returns JSONResponse if auth fails.
    """
    import secrets

    cfg = _get_config(request)
    api_key = cfg.api_key
    if not api_key:
        return None  # No key configured, skip auth

    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        # P0-1 fix: use secrets.compare_digest() to prevent timing attacks
        if secrets.compare_digest(token, api_key):
            return None

    return JSONResponse(
        {"error": "Unauthorized: valid API key required"},
        status_code=401,
    )


async def _check_concurrency(request: Request, active_requests: int, active_lock):
    """Check if concurrency limit is reached.

    NOTE: This is only used by middleware.py's concurrency_middleware.
    Route handlers should NOT call this directly — the middleware handles it.
    """
    cfg = _get_config(request)
    limit = cfg.max_concurrent_requests
    if limit <= 0:
        return None
    with active_lock:
        if active_requests >= limit:
            return JSONResponse({"error": "Server busy, try again later"}, status_code=503)
    return None
