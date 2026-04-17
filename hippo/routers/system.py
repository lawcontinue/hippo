"""System routes: /api/version, /, /api/predict
"""

from fastapi import APIRouter, Request

from hippo import __version__

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
    """
    predictor = getattr(request.app.state, "predictor", None)
    if not predictor:
        return {"enabled": False, "predictions": [], "stats": {}}

    return {
        "enabled": True,
        "predictions": predictor.get_predictions(),
        "stats": predictor.get_stats(),
    }
