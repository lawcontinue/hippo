"""Gateway — cluster coordinator and unified API entry point.

The Gateway:
1. Accepts registration from workers
2. Maintains a Scheduler for model placement
3. Routes inference requests to the right worker
4. Provides a unified OpenAI-compatible API
5. Handles failover if a worker goes offline
"""

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hippo.cluster.discovery import DiscoveryService, NodeInfo
from hippo.cluster.scheduler import Scheduler
from hippo.cluster.transport import Transport

logger = logging.getLogger("hippo.cluster.gateway")

HEARTBEAT_TIMEOUT = 120  # seconds before marking worker unhealthy (generous for Wi-Fi)
CLEANUP_INTERVAL = 60  # seconds between worker cleanup checks


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
        await gw.start()

        # Expose REST API via FastAPI router
        app.include_router(gw.router)

        await gw.stop()
    """

    def __init__(self, port: int = 11434):
        self.port = port
        self._scheduler = Scheduler()
        self._discovery = DiscoveryService(role="gateway", port=port)
        self._transport = Transport()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        self._local_manager = None
        self._rpc_backends: dict[str, "LLamaRPCBackend"] = {}  # model_name → backend (常驻)

        # FastAPI router for cluster endpoints
        self.router = APIRouter(prefix="/cluster", tags=["cluster"])
        self._setup_routes()

    async def start(self):
        """Start the gateway."""
        logger.info(f"Starting gateway on port {self.port}")
        self._running = True
        await self._transport.start()

        # Broadcast via mDNS (async to avoid blocking event loop)
        await self._discovery.start_broadcast_async()

        # Start worker cleanup loop
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Browse for workers via mDNS — but do NOT double-register:
        # mDNS discovery only provides awareness, HTTP /register is authoritative
        self._discovery.start_browse(self._on_node_discovered)

        logger.info("Gateway started")

    async def stop(self):
        """Stop the gateway and unload all RPC backends."""
        self._running = False

        # Unload all resident RPC backends
        for name, backend in self._rpc_backends.items():
            try:
                backend.unload()
                logger.info(f"Unloaded RPC backend for {name}")
            except Exception as e:
                logger.warning(f"Error unloading RPC backend {name}: {e}")
        self._rpc_backends.clear()

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._discovery.shutdown()
        await self._transport.stop()

        logger.info("Gateway stopped")

    def set_local_manager(self, manager):
        """Set the local ModelManager for fallback inference."""
        self._local_manager = manager

    async def route_inference(self, request: InferenceRequest) -> dict:
        """Route an inference request to the appropriate worker.

        Strategy (in order):
        1. RPC-distributed: split model across local + remote workers
        2. Single worker: route to worker that has the model
        3. Local fallback: use local ModelManager
        """
        # Strategy 1: RPC-distributed inference (only for models >= 3GB)
        # Small models are faster locally — no point splitting across network
        MODEL_SIZE_THRESHOLD_GB = 3.0
        dist = self._scheduler.compute_tensor_split(model_size_gb=MODEL_SIZE_THRESHOLD_GB)
        if dist and self._local_manager:
            return await self._rpc_distributed_inference(request, dist)

        # Strategy 2: single worker routing
        assignment = self._scheduler.get_assignment(request.model)

        if assignment:
            worker = self._scheduler.get_worker(assignment.worker_id)
            if worker and worker.status in ("healthy", "unhealthy"):
                url = f"http://{worker.host}:{worker.port}/api/generate"
                payload = {
                    "model": request.model,
                    "prompt": request.prompt,
                    "stream": False,
                    "options": {
                        "num_predict": request.max_tokens,
                        "temperature": request.temperature,
                    },
                }
                try:
                    return await self._transport.post(url, payload)
                except Exception as e:
                    logger.error(f"Worker {worker.worker_id} error: {e}")
                    worker.status = "unhealthy"

        # Strategy 3: local fallback
        if self._local_manager:
            logger.info(f"Using local fallback for model {request.model}")
            # TODO(Phase 1): call local ModelManager for fallback inference
            return {"error": "local fallback not yet implemented"}

        raise HTTPException(status_code=503, detail=f"No worker available for model {request.model}")

    async def _rpc_distributed_inference(self, request: InferenceRequest, dist: dict) -> dict:
        """Run inference using LLamaRPCBackend (常驻 — model stays loaded).

        Backend is cached per model_name, loaded once, reused across requests.
        """
        import asyncio

        from hippo.cluster.backend import LLamaRPCBackend

        rpc_workers = dist["rpc_workers"]
        tensor_split = dist["tensor_split"]
        model_name = request.model

        # Reuse cached backend (常驻模式)
        backend = self._rpc_backends.get(model_name)
        if backend and backend.is_loaded:
            logger.debug(f"RPC backend cache hit for {model_name}")
        else:
            # First time: create + load
            logger.info(
                f"RPC distributed inference: loading model={model_name}, "
                f"workers={len(rpc_workers)}, split={tensor_split}"
            )

            if not self._local_manager:
                raise HTTPException(status_code=503, detail="No local manager configured")

            # Resolve model path via ModelManager's internal method
            model_path = self._local_manager._resolve_model_path(model_name)
            if not model_path:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            model_path = str(model_path)

            backend = LLamaRPCBackend(
                rpc_workers=rpc_workers,
                tensor_split=tensor_split,
                n_ctx=2048,
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, backend.load, model_path)

            # Cache it (常驻)
            self._rpc_backends[model_name] = backend
            logger.info(f"RPC backend for {model_name} loaded and cached (常驻)")

        # Generate using cached backend
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: backend.generate(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            )

            return {
                "model": model_name,
                "response": result["choices"][0]["text"],
                "done": True,
                "distributed": True,
                "rpc_workers": len(rpc_workers),
                "tensor_split": tensor_split,
            }
        except Exception as e:
            logger.exception(f"RPC distributed inference failed for {model_name}")
            # Remove broken backend from cache so next request retries
            self._rpc_backends.pop(model_name, None)
            raise HTTPException(status_code=500, detail=str(e))

    def get_cluster_status(self) -> dict:
        """Return cluster status."""
        return self._scheduler.get_cluster_status()

    def _on_node_discovered(self, event: str, node: NodeInfo):
        """Handle discovered nodes via mDNS.

        mDNS discovery provides visibility only. Actual registration
        happens via HTTP POST /cluster/register from the worker,
        which is authoritative and carries full capabilities info.
        """
        if event == "added" and node.role == "worker":
            logger.info(
                f"mDNS: Worker {node.name} seen at {node.host}:{node.port}, "
                f"awaiting HTTP registration"
            )
        elif event == "removed":
            # mDNS removal is a signal the worker is gone
            worker = self._scheduler.get_worker(node.name)
            if worker:
                logger.warning(f"mDNS: Worker {node.name} disappeared, marking offline")
                worker.status = "offline"

    async def _cleanup_loop(self):
        """Periodically check for unhealthy workers."""
        while self._running:
            now = time.time()
            status = self._scheduler.get_cluster_status()
            for w_detail in status["workers_detail"]:
                wid = w_detail["id"]
                worker = self._scheduler.get_worker(wid)
                if not worker or worker.last_heartbeat == 0:
                    continue
                elapsed = now - worker.last_heartbeat
                if elapsed > HEARTBEAT_TIMEOUT and worker.status != "unhealthy":
                    logger.warning(f"Worker {wid} heartbeat timeout ({elapsed:.0f}s), marking unhealthy")
                    worker.status = "unhealthy"
                if elapsed > HEARTBEAT_TIMEOUT * 4:
                    logger.warning(f"Worker {wid} offline (no heartbeat for {elapsed:.0f}s), deregistering")
                    self._scheduler.deregister_worker(wid)

            await asyncio.sleep(CLEANUP_INTERVAL)

    def _setup_routes(self):
        """Set up FastAPI routes for cluster management.

        TODO(Phase 2): Add authentication (API key or mTLS) to prevent
        unauthorized registration on shared networks.
        """

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
            logger.info(f"Worker {req.worker_id} registered via HTTP from {req.host}")
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
