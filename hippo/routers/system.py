"""System routes: /api/version, /"""

from fastapi import APIRouter

from hippo import __version__

router = APIRouter()


@router.get("/api/version")
async def version():
    """Return server version (Ollama compatible)."""
    return {"version": __version__}


@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hippo 🦛 is running", "version": __version__}
