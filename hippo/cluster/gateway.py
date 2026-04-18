"""Gateway — cluster coordinator and unified API entry point.

The Gateway:
1. Accepts registration from workers
2. Maintains a Scheduler for model placement
3. Routes inference requests to the right worker
4. Provides a unified OpenAI-compatible API
5. Handles failover if a worker goes offline
"""

import time
import logging
import asyncio
from typing import Optional

import aiohttp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hippo.cluster.discovery import DiscoveryService, NodeInfo
from hippo.cluster.scheduler import Scheduler, WorkerState

logger = logging.getLogger("hippo.cluster.gateway")

HEARTBEAT_TIMEOUT = 45  # seconds before marking worker unhealthy
CLEANUP_INTERVAL = 30  # seconds between worker cleanup checks


# --- API Models ---

class RegisterRequest(BaseModel):
    worker_id: str
    host: str
    port: int
    gpu_memory_gb: float
    mlx: bool = False
    models: list[str] = []
    platform: str = "Darwin"
    cpu_cores: int = 4


class HeartbeatRequest(BaseModel):
    worker_id: str
    status: str = "healthy"
    loaded_models: list[str] = []
    gpu_memory_used_gb: float = 0.0


class DeregisterRequest(BaseModel):
    worker_id: str


class InferenceRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


# --- Gateway Service ---

class GatewayService:
    """Cluster gateway — coordinates workers and routes requests.

    Usage:
        gw = GatewayService()
        await gw.start()  # starts mDNS broadcast + cleanup loop

        # Expose REST API via FastAPI router
        app.include_router(gw.router)

        await gw.stop()
    """

    def __init__(self, port: int = 11434):
        self.port = port
        self._scheduler = Scheduler()
        self._discovery = DiscoveryService(role="gateway", port=port)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        self._local_manager = None  # fallback to local ModelManager

        # FastAPI router for cluster endpoints
        self.router = APIRouter(prefix="/cluster", tags=["cluster"])
        self._setup_routes()

    async def start(self):
        """Start the gateway."""
        logger.info(f"Starting gateway on port {self.port}")
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        )

        # Broadcast via mDNS
        self._discovery.start_broadcast()

        # Start worker cleanup loop
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Also browse for workers that might already be running
        self._discovery.start_browse(self._on_node_discovered)

        logger.info("Gateway started")

    async def stop(self):
        """Stop the gateway."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._discovery.shutdown()

        if self._session:
            await self._session.close()

        logger.info("Gateway stopped")

    def set_local_manager(self, manager):
        """Set the local ModelManager for fallback inference."""
        self._local_manager = manager

    async def route_inference(self, request: InferenceRequest) -> dict:
        """Route an inference request to the appropriate worker.

        Falls back to local inference if no worker has the model.
        """
        assignment = self._scheduler.get_assignment(request.model)

        if assignment:
            worker = self._scheduler._workers.get(assignment.worker_id)
            if worker and worker.status == "healthy":
                # Forward to worker
                url = f"http://{worker.host}:{worker.port}/v1/completions"
                payload = {
                    "model": request.model,
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "stream": request.stream,
                }
                try:
                    async with self._session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        else:
                            logger.warning(f"Worker {worker.worker_id} returned {resp.status}")
                except Exception as e:
                    logger.error(f"Worker {worker.worker_id} error: {e}")
                    # Mark unhealthy and try fallback
                    worker.status = "unhealthy"

        # Fallback: local inference
        if self._local_manager:
            logger.info(f"Using local fallback for model {request.model}")
            # TODO: call local ModelManager
            return {"error": "local fallback not yet implemented"}

        raise HTTPException(status_code=503, detail=f"No worker available for model {request.model}")

    def get_cluster_status(self) -> dict:
        """Return cluster status."""
        return self._scheduler.get_cluster_status()

    def _on_node_discovered(self, event: str, node: NodeInfo):
        """Handle discovered nodes via mDNS."""
        if event == "added" and node.role == "worker":
            self._scheduler.register_worker(
                worker_id=node.name,
                host=node.host,
                port=node.port,
                gpu_memory_gb=node.gpu_memory_gb,
                models=node.models,
                mlx="mlx" in node.properties.get("platform", "").lower(),
            )

    async def _cleanup_loop(self):
        """Periodically check for unhealthy workers."""
        while self._running:
            now = time.time()
            for w in list(self._scheduler._workers.values()):
                if w.last_heartbeat > 0 and (now - w.last_heartbeat) > HEARTBEAT_TIMEOUT:
                    if w.status != "unhealthy":
                        logger.warning(f"Worker {w.worker_id} heartbeat timeout, marking unhealthy")
                        w.status = "unhealthy"
                if w.last_heartbeat > 0 and (now - w.last_heartbeat) > HEARTBEAT_TIMEOUT * 4:
                    logger.warning(f"Worker {w.worker_id} offline, deregistering")
                    self._scheduler.deregister_worker(w.worker_id)

            await asyncio.sleep(CLEANUP_INTERVAL)

    def _setup_routes(self):
        """Set up FastAPI routes for cluster management."""

        @self.router.post("/register")
        async def register_worker(req: RegisterRequest):
            self._scheduler.register_worker(
                worker_id=req.worker_id,
                host=req.host,
                port=req.port,
                gpu_memory_gb=req.gpu_memory_gb,
                models=req.models,
                mlx=req.mlx,
            )
            return {"status": "ok", "worker_id": req.worker_id}

        @self.router.post("/heartbeat")
        async def heartbeat(req: HeartbeatRequest):
            self._scheduler.update_heartbeat(
                worker_id=req.worker_id,
                status=req.status,
                loaded_models=req.loaded_models,
                gpu_memory_used_gb=req.gpu_memory_used_gb,
            )
            return {"status": "ok"}

        @self.router.post("/deregister")
        async def deregister_worker(req: DeregisterRequest):
            self._scheduler.deregister_worker(req.worker_id)
            return {"status": "ok"}

        @self.router.get("/status")
        async def cluster_status():
            return self._scheduler.get_cluster_status()

        @self.router.post("/infer")
        async def infer(req: InferenceRequest):
            return await self.route_inference(req)
