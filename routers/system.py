"""System routes: /api/version, /, /api/predict
"""

from fastapi import APIRouter, Request

from hippo import __version__
from hippo.dependencies import _check_auth

router = APIRouter()


@router.get("/api/version")
async def version():
    """Return server version (Ollama compatible)."""
    return {"version": __version__}


@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hippo is running", "version": __version__}


@router.get("/api/predict")
async def get_predictions(request: Request):
    """Get current query predictions and predictor statistics.

    Sleep-time Compute introspection endpoint.
    P1-3: requires auth (same as inference endpoints).
    """
    auth_result = await _check_auth(request)
    if auth_result:
        return auth_result

    predictor = getattr(request.app.state, "predictor", None)
    if not predictor:
        return {"enabled": False, "predictions": [], "stats": {}}

    return {
        "enabled": True,
        "predictions": predictor.get_predictions(),
        "stats": predictor.get_stats(),
    }
