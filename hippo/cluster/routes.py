"""FastAPI router for Hippo cluster management endpoints."""

import logging
from fastapi import APIRouter, Request

logger = logging.getLogger("hippo.cluster.routes")

router = APIRouter(prefix="/api/cluster", tags=["cluster"])


@router.post("/register")
async def register_worker(request: Request):
    """Register a Worker node with the Gateway."""
    gateway = request.app.state.cluster_gateway
    if not gateway:
        return {"error": "Not running as gateway"}
    payload = await request.json()
    return gateway.register_worker(payload)


@router.post("/deregister")
async def deregister_worker(request: Request):
    """Deregister a Worker node."""
    gateway = request.app.state.cluster_gateway
    if not gateway:
        return {"error": "Not running as gateway"}
    payload = await request.json()
    return gateway.deregister_worker(payload.get("name", ""))


@router.post("/heartbeat")
async def worker_heartbeat(request: Request):
    """Handle heartbeat from a Worker."""
    gateway = request.app.state.cluster_gateway
    if not gateway:
        return {"error": "Not running as gateway"}
    payload = await request.json()
    return gateway.handle_heartbeat(payload)


@router.get("/status")
async def cluster_status(request: Request):
    """Get cluster status (workers, models, health)."""
    gateway = request.app.state.cluster_gateway
    if not gateway:
        return {"role": "standalone", "workers": []}
    return gateway.get_status()


@router.get("/capacity")
async def cluster_capacity(request: Request):
    """Get cluster capacity (total memory, models)."""
    gateway = request.app.state.cluster_gateway
    scheduler = request.app.state.cluster_scheduler
    if not gateway or not scheduler:
        return {"role": "standalone"}
    return scheduler.get_cluster_capacity()


@router.post("/schedule")
async def schedule_model(request: Request):
    """Get scheduling decision for a model."""
    scheduler = request.app.state.cluster_scheduler
    if not scheduler:
        return {"error": "Scheduler not available"}
    payload = await request.json()
    model = payload.get("model", "")
    placement = scheduler.schedule(model)
    if placement:
        return {
            "worker": placement.worker_name,
            "host": placement.worker_host,
            "model": placement.model,
            "reason": placement.reason,
        }
    return {"error": f"No worker available for model '{model}'"}
