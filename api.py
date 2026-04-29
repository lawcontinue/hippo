"""FastAPI server — Ollama-compatible API."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from hippo import __version__
from hippo.audit import AuditLogger
from hippo.config import HippoConfig
from hippo.middleware import (
    audit_middleware,
    concurrency_middleware,
    security_headers_middleware,
)
from hippo.model_manager import ModelManager
from hippo.query_predictor import QueryPredictor
from hippo.routers import (
    batch,
    embeddings,
    inference,
    management,
    metrics,
    models,
    rerank,
    stats,
    system,
)

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

    # Query Predictor (Sleep-time Compute)
    from pathlib import Path as _Path
    predictor = QueryPredictor(
        manager=None,  # set after manager is available
        persist_path=_Path(os.path.join(log_dir, "predictor_stats.json")),
    )
    app.state.predictor = predictor

    # Pre-cache model families in background asyncio task
    manager = getattr(app.state, "manager", None)
    if manager:
        # Wire predictor to manager
        app.state.predictor.set_manager(manager)
        app.state.predictor.start_background(interval_seconds=60)
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

    # Start cluster worker/gateway if configured
    cluster_worker = getattr(app.state, "_cluster_worker", None)
    cluster_gateway = getattr(app.state, "_cluster_gateway", None)
    if cluster_worker:
        logger.info("Starting cluster worker...")
        await cluster_worker.start()
    if cluster_gateway:
        logger.info("Starting cluster gateway...")
        await cluster_gateway.start()

    yield

    # Shutdown cluster
    if cluster_worker:
        logger.info("Stopping cluster worker...")
        await cluster_worker.stop()
    if cluster_gateway:
        logger.info("Stopping cluster gateway...")
        await cluster_gateway.stop()

    # Shutdown
    predictor = getattr(app.state, "predictor", None)
    if predictor:
        predictor.stop_background()
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
