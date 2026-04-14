"""Metrics router: /metrics endpoint for Prometheus."""

from fastapi import APIRouter, Request, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from hippo.dependencies import _get_config, _check_auth

router = APIRouter()


@router.get("/metrics")
async def metrics(request: Request):
    """Prometheus metrics endpoint.

    P0-3 fix: requires auth when HIPPO_API_KEY is configured.
    If no API key is set (local dev), metrics are accessible without auth.
    """
    config = _get_config(request)

    # Security check: metrics must be enabled
    if not getattr(config, "metrics_enabled", True):
        return Response(
            content="Metrics are disabled. Set metrics_enabled: true in config.",
            status_code=403,
            media_type="text/plain",
        )

    # P0-3 fix: require auth if API key is configured
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    # Generate Prometheus text format metrics
    metrics_data = generate_latest()
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST,
    )
