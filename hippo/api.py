"""FastAPI server — Ollama-compatible API."""

import os
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from hippo.config import HippoConfig
from hippo.model_manager import ModelManager
from hippo.audit import AuditLogger
from hippo.middleware import (
    security_headers_middleware,
    audit_middleware,
    concurrency_middleware,
)
from hippo.routers import models, inference, embeddings, management, system, metrics, batch, stats, rerank

from hippo import __version__

logger = logging.getLogger("hippo")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: startup logging + pre-cache, shutdown cleanup."""
    # Startup
    log_dir = os.path.expanduser("~/.hippo")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(log_dir, "hippo.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logging.getLogger("hippo").addHandler(file_handler)

    # Audit logger
    audit_path = getattr(app.state, "_audit_log_path", None)
    app.state.audit = AuditLogger(audit_path)
    if app.state.audit.enabled:
        logger.info(f"Audit logging enabled: {audit_path}")

    logger.info("Hippo server started")

    # Pre-cache model families in background asyncio task
    manager = getattr(app.state, "manager", None)
    if manager:
        async def _precache():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _precache_sync, manager)

        def _precache_sync(mgr):
            for p in mgr.config.models_dir.rglob("*.gguf"):
                try:
                    mgr._detect_family(p)
                except Exception:
                    pass
            logger.info("Model family pre-caching complete")

        asyncio.create_task(_precache())

    yield

    # Shutdown
    if manager:
        manager.stop_cleanup_thread()
        manager.unload_all()
        logger.info("Hippo server shutdown complete")


def create_app(config: HippoConfig, manager: ModelManager, audit_log_path: str = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Hippo 🦛",
        version=__version__,
        lifespan=lifespan,
    )

    # Store state for dependencies
    app.state.config = config
    app.state.manager = manager
    if audit_log_path:
        app.state._audit_log_path = audit_log_path

    # Register middleware (order matters: first registered = outermost)
    app.middleware("http")(security_headers_middleware)
    app.middleware("http")(audit_middleware)
    app.middleware("http")(concurrency_middleware)

    # Register routers
    app.include_router(models.router)
    app.include_router(inference.router)
    app.include_router(embeddings.router)
    app.include_router(management.router)
    app.include_router(system.router)
    app.include_router(metrics.router)
    app.include_router(batch.router)
    app.include_router(stats.router)
    app.include_router(rerank.router)

    return app


# P1-6 fix: lazy default app to avoid side effects on import
# cli.py sets up app.state before using it, so the default instance
# is only a fallback for direct imports.
_default_app = None


def _get_default_app():
    """Get or create the default app instance (lazy, for CLI compatibility)."""
    global _default_app
    if _default_app is None:
        _default_app = create_app(
            config=HippoConfig(),
            manager=ModelManager(HippoConfig()),
        )
    return _default_app


# Use module-level __getattr__ for lazy access (Python 3.7+)
def __getattr__(name):
    if name == "app":
        return _get_default_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
