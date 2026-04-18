"""Worker node — registers with Gateway, executes inference tasks.

A Worker:
1. Starts its local inference engine (MLX on Mac, llama.cpp on others)
2. Discovers the Gateway via mDNS
3. Registers itself (reports GPU/CPU, available models)
4. Sends heartbeat every 15s
5. Accepts inference requests from the Gateway
"""

import os
import time
import uuid
import logging
import asyncio
import platform
from typing import Optional
from dataclasses import dataclass

import aiohttp

from hippo.cluster.discovery import DiscoveryService, NodeInfo

logger = logging.getLogger("hippo.cluster.worker")

HEARTBEAT_INTERVAL = 15  # seconds
REGISTRATION_TIMEOUT = 30  # seconds


@dataclass
class WorkerConfig:
    """Worker node configuration."""
    gateway_host: Optional[str] = None  # override mDNS discovery
    gateway_port: int = 11434
    worker_port: int = 11435  # port this worker listens on
    heartbeat_interval: int = HEARTBEAT_INTERVAL
    gpu_memory_gb: float = 0.0  # 0 = auto-detect
    models_dir: str = "~/.hippo/models"
    mlx_preferred: bool = True  # prefer MLX on Apple Silicon


class WorkerService:
    """Worker node service for Hippo cluster.

    Lifecycle:
        worker = WorkerService(WorkerConfig())
        await worker.start()  # discovers gateway, registers
        # ... runs until stopped
        await worker.stop()
    """

    def __init__(self, config: WorkerConfig):
        self.config = config
        self._id = f"worker-{uuid.uuid4().hex[:8]}"
        self._discovery: Optional[DiscoveryService] = None
        self._gateway_url: Optional[str] = None
        self._registered = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._models: list[str] = []
        self._gpu_memory_gb = config.gpu_memory_gb
        self._gpu_memory_used_gb: float = 0.0
        self._loaded_models: list[str] = []

    async def start(self):
        """Start the worker: auto-detect resources, find gateway, register."""
        logger.info(f"Starting worker {self._id}")
        self._running = True

        # Auto-detect capabilities
        self._detect_capabilities()

        # Find gateway
        if self.config.gateway_host:
            self._gateway_url = f"http://{self.config.gateway_host}:{self.config.gateway_port}"
            logger.info(f"Using configured gateway: {self._gateway_url}")
        else:
            await self._discover_gateway()

        if not self._gateway_url:
            logger.warning("No gateway found, running in standalone mode")
            return

        # Register via HTTP (authoritative)
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        await self._register()

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Also broadcast via mDNS so other tools can discover us
        node_info = NodeInfo(
            name=self._id,
            host=DiscoveryService._get_local_ip(),
            port=self.config.worker_port,
            role="worker",
            gpu_memory_gb=self._gpu_memory_gb,
            models=self._models,
        )
        self._discovery = DiscoveryService(
            role="worker", port=self.config.worker_port, node_info=node_info
        )
        self._discovery.start_broadcast()

        logger.info(f"Worker {self._id} started and registered")

    async def stop(self):
        """Graceful shutdown."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Deregister from gateway
        if self._gateway_url and self._session:
            try:
                await self._session.post(
                    f"{self._gateway_url}/cluster/deregister",
                    json={"worker_id": self._id},
                )
            except Exception:
                pass

        if self._session:
            await self._session.close()

        if self._discovery:
            self._discovery.shutdown()

        logger.info(f"Worker {self._id} stopped")

    def _detect_capabilities(self):
        """Auto-detect GPU memory and available models."""
        system = platform.system()
        processor = platform.processor()

        # Detect Apple Silicon MLX
        if system == "Darwin" and ("arm" in processor.lower() or "apple" in processor.lower()):
            logger.info("Detected Apple Silicon — MLX mode")
            self.config.mlx_preferred = True
            if self._gpu_memory_gb == 0:
                try:
                    import subprocess
                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True
                    )
                    total_mem_gb = int(result.stdout.strip()) / (1024**3)
                    self._gpu_memory_gb = max(0, total_mem_gb - 3.0)
                except Exception:
                    self._gpu_memory_gb = 13.0
        else:
            self.config.mlx_preferred = False
            if self._gpu_memory_gb == 0:
                self._gpu_memory_gb = 12.0

        # Scan available models
        models_dir = os.path.expanduser(self.config.models_dir)
        if os.path.isdir(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith((".gguf", ".bin")):
                    model_name = os.path.splitext(f)[0]
                    self._models.append(model_name)

        logger.info(
            f"Capabilities: gpu_mem={self._gpu_memory_gb:.1f}GB, "
            f"mlx={self.config.mlx_preferred}, models={self._models}"
        )

    async def _discover_gateway(self):
        """Use mDNS to find a gateway on the LAN."""
        logger.info("Discovering gateway via mDNS...")
        gateway_found = asyncio.Event()

        disc = DiscoveryService(role="worker", port=self.config.worker_port)

        def on_node(event: str, node: NodeInfo):
            if event == "added" and node.role == "gateway":
                self._gateway_url = f"http://{node.host}:{node.port}"
                logger.info(f"Found gateway at {self._gateway_url}")
                gateway_found.set()

        disc.start_browse(on_node)

        try:
            await asyncio.wait_for(gateway_found.wait(), timeout=REGISTRATION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Gateway discovery timed out")
        finally:
            disc.shutdown()

    async def _register(self):
        """Register this worker with the gateway."""
        if not self._gateway_url or not self._session:
            return

        payload = {
            "worker_id": self._id,
            "host": DiscoveryService._get_local_ip(),
            "port": self.config.worker_port,
            "gpu_memory_gb": self._gpu_memory_gb,
            "mlx": self.config.mlx_preferred,
            "models": self._models,
            "platform": platform.system(),
            "cpu_cores": os.cpu_count() or 4,
        }

        try:
            async with self._session.post(
                f"{self._gateway_url}/cluster/register", json=payload
            ) as resp:
                if resp.status == 200:
                    self._registered = True
                    logger.info(f"Registered with gateway {self._gateway_url}")
                else:
                    text = await resp.text()
                    logger.error(f"Registration failed: {resp.status} {text}")
        except Exception as e:
            logger.error(f"Registration error: {e}")

    async def _heartbeat_loop(self):
        """Periodic heartbeat to gateway.

        TODO(Phase 1): Track actual loaded_models and gpu_memory_used_gb
        from the local ModelManager instead of using placeholder values.
        """
        while self._running:
            try:
                if self._session and self._gateway_url:
                    async with self._session.post(
                        f"{self._gateway_url}/cluster/heartbeat",
                        json={
                            "worker_id": self._id,
                            "status": "healthy",
                            "loaded_models": self._loaded_models,
                            "gpu_memory_used_gb": self._gpu_memory_used_gb,
                        },
                    ) as resp:
                        if resp.status != 200:
                            logger.warning(f"Heartbeat failed: {resp.status}")
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")

            await asyncio.sleep(self.config.heartbeat_interval)

    def update_loaded_models(self, models: list[str], memory_used_gb: float):
        """Update tracked loaded models and memory usage.

        Called by the local inference engine when models are loaded/unloaded.
        """
        self._loaded_models = models
        self._gpu_memory_used_gb = memory_used_gb
