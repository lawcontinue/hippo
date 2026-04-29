"""FastAPI middleware for security, audit, and concurrency control."""

import json
import logging
import threading
import time

from fastapi import Request

from hippo.dependencies import _get_audit, _get_config

logger = logging.getLogger("hippo")

# Concurrency tracking
_active_requests = 0
_active_lock = threading.Lock()


async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    # HSTS only if using HTTPS
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


async def audit_middleware(request: Request, call_next):
    """Audit middleware — logs every API call to JSONL audit trail."""
    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000

    audit = _get_audit(request)
    if audit.enabled:
        # P1-2 fix: extract model from query params or URL path, avoid reading body
        model = request.query_params.get("model")
        if not model:
            # For POST, try lightweight peek (FastAPI caches body internally)
            try:
                body = await request.body()
                if body and len(body) < 65536:  # skip large/streaming bodies
                    data = json.loads(body)
                    model = data.get("model")
            except Exception:
                pass

        client_ip = request.client.host if request.client else None

        # P0-4 fix: hash API key before passing to audit log
        import hashlib
        auth_header = request.headers.get("Authorization", "")
        api_key = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            api_key = hashlib.sha256(token.encode()).hexdigest()[:32]

        audit.log(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=latency_ms,
            model=model,
            client_ip=client_ip,
            api_key=api_key,
        )

    return response


async def concurrency_middleware(request: Request, call_next):
    """Enforce concurrency limit on active requests."""
    global _active_requests

    cfg = _get_config(request)
    limit = cfg.max_concurrent_requests
    if limit > 0:
        with _active_lock:
            if _active_requests >= limit:
                from fastapi.responses import JSONResponse
                return JSONResponse({"error": "Server busy, try again later"}, status_code=503)
            _active_requests += 1

    try:
        response = await call_next(request)
    finally:
        if limit > 0:
            with _active_lock:
                _active_requests -= 1

    return response
