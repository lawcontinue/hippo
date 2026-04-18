"""Scheduler — assigns models to workers based on resources.

Policies:
1. Best-fit: assign model to smallest worker that can hold it
2. Spread: distribute models across workers evenly
3. Locality: prefer workers that already have the model loaded
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("hippo.cluster.scheduler")


@dataclass
class ModelAssignment:
    """A model assigned to a worker."""
    model: str
    worker_id: str
    size_gb: float
    status: str = "assigned"  # assigned, loading, ready, failed


@dataclass
class WorkerState:
    """Tracked state of a worker node."""
    worker_id: str
    host: str
    port: int
    gpu_memory_gb: float
    gpu_memory_used_gb: float = 0.0
    mlx: bool = False
    models: list[str] = field(default_factory=list)
    loaded_models: list[str] = field(default_factory=list)
    status: str = "healthy"  # healthy, unhealthy, offline
    last_heartbeat: float = 0.0


class Scheduler:
    """Schedule model placement across cluster workers.

    Simple policy for Phase 0:
    - Each model lives on exactly one worker (no sharding yet)
    - Assign to the worker with enough memory that has fewest models
    """

    def __init__(self):
        self._workers: dict[str, WorkerState] = {}
        self._assignments: dict[str, ModelAssignment] = {}  # model -> assignment

    def register_worker(self, worker_id: str, host: str, port: int,
                        gpu_memory_gb: float, models: list[str],
                        mlx: bool = False, **kwargs):
        """Register a new worker."""
        self._workers[worker_id] = WorkerState(
            worker_id=worker_id,
            host=host,
            port=port,
            gpu_memory_gb=gpu_memory_gb,
            models=models,
            mlx=mlx,
        )
        logger.info(f"Worker registered: {worker_id} ({gpu_memory_gb}GB, {len(models)} models)")
        # Re-schedule assignments
        self._reschedule()

    def deregister_worker(self, worker_id: str):
        """Remove a worker and reassign its models."""
        if worker_id in self._workers:
            del self._workers[worker_id]
            # Remove assignments for this worker
            failed = [m for m, a in self._assignments.items() if a.worker_id == worker_id]
            for model in failed:
                del self._assignments[model]
            if failed:
                logger.warning(f"Worker {worker_id} removed, models orphaned: {failed}")
                self._reschedule()

    def update_heartbeat(self, worker_id: str, status: str = "healthy",
                         loaded_models: list[str] | None = None,
                         gpu_memory_used_gb: float = 0.0):
        """Update worker status from heartbeat."""
        import time
        if worker_id in self._workers:
            w = self._workers[worker_id]
            w.status = status
            w.last_heartbeat = time.time()
            w.loaded_models = loaded_models or []
            w.gpu_memory_used_gb = gpu_memory_used_gb

    def get_assignment(self, model: str) -> Optional[ModelAssignment]:
        """Get the worker assigned to a model."""
        return self._assignments.get(model)

    def get_worker(self, worker_id: str) -> Optional[WorkerState]:
        """Get a worker by ID."""
        return self._workers.get(worker_id)

    def find_worker_for_model(self, model: str, model_size_gb: float) -> Optional[WorkerState]:
        """Find the best worker for a model.

        Policy: pick the worker with enough free memory and fewest assigned models.
        """
        candidates = []
        for w in self._workers.values():
            if w.status != "healthy":
                continue
            free = w.gpu_memory_gb - w.gpu_memory_used_gb
            if free >= model_size_gb:
                assigned_count = sum(
                    1 for a in self._assignments.values()
                    if a.worker_id == w.worker_id
                )
                candidates.append((w, free, assigned_count))

        if not candidates:
            logger.warning(f"No worker can fit model {model} ({model_size_gb}GB)")
            return None

        # Sort by: fewest assignments first, then most free memory
        candidates.sort(key=lambda x: (x[2], -x[1]))
        return candidates[0][0]

    def get_cluster_status(self) -> dict:
        """Return cluster status summary."""
        total_gpu = sum(w.gpu_memory_gb for w in self._workers.values())
        used_gpu = sum(w.gpu_memory_used_gb for w in self._workers.values())
        healthy = sum(1 for w in self._workers.values() if w.status == "healthy")

        return {
            "workers": len(self._workers),
            "healthy_workers": healthy,
            "total_gpu_memory_gb": total_gpu,
            "used_gpu_memory_gb": used_gpu,
            "assignments": len(self._assignments),
            "workers_detail": [
                {
                    "id": w.worker_id,
                    "host": w.host,
                    "status": w.status,
                    "gpu_total_gb": w.gpu_memory_gb,
                    "gpu_used_gb": w.gpu_memory_used_gb,
                    "loaded_models": w.loaded_models,
                    "mlx": w.mlx,
                }
                for w in self._workers.values()
            ],
        }

    def _reschedule(self):
        """Re-schedule all model assignments across workers."""
        # For Phase 0: each model on one worker, no sharding
        # Simply clear and reassign based on worker model lists
        new_assignments = {}
        for w in self._workers.values():
            if w.status != "healthy":
                continue
            for model in w.models:
                if model not in new_assignments:
                    # Estimate model size (rough: 1B params ≈ 2GB in GGUF Q4)
                    new_assignments[model] = ModelAssignment(
                        model=model,
                        worker_id=w.worker_id,
                        size_gb=0,  # unknown for now
                    )
        self._assignments = new_assignments
        logger.info(f"Scheduled {len(new_assignments)} model assignments across {len(self._workers)} workers")
