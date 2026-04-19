"""Hippo Cluster — distributed LLM inference across home devices."""

# Lazy imports to avoid hard dependency on zeroconf at package level
__all__ = [
    "DiscoveryService",
    "WorkerService",
    "Scheduler",
    "GatewayService",
    "Transport",
    "LocalBackend",
    "LLamaRPCBackend",
    "InferenceBackend",
]


def __getattr__(name):
    """Lazy import cluster components."""
    if name == "DiscoveryService":
        from hippo.cluster.discovery import DiscoveryService
        return DiscoveryService
    elif name == "WorkerService":
        from hippo.cluster.worker import WorkerService
        return WorkerService
    elif name == "Scheduler":
        from hippo.cluster.scheduler import Scheduler
        return Scheduler
    elif name == "GatewayService":
        from hippo.cluster.gateway import GatewayService
        return GatewayService
    elif name == "Transport":
        from hippo.cluster.transport import Transport
        return Transport
    elif name == "LocalBackend":
        from hippo.cluster.backend import LocalBackend
        return LocalBackend
    elif name == "LLamaRPCBackend":
        from hippo.cluster.backend import LLamaRPCBackend
        return LLamaRPCBackend
    elif name == "InferenceBackend":
        from hippo.cluster.backend import InferenceBackend
        return InferenceBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
