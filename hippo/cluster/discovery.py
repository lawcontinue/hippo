"""mDNS-based service discovery for Hippo cluster nodes.

Uses zeroconf to broadcast and discover Hippo instances on the LAN.
- Gateway broadcasts _hippo._tcp.local.
- Workers discover Gateways and register.
"""

import logging
import socket
import asyncio
from typing import Callable, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from zeroconf import Zeroconf, ServiceBrowser, ServiceInfo, ServiceStateChange
from zeroconf import IPVersion

logger = logging.getLogger("hippo.cluster.discovery")

SERVICE_TYPE = "_hippo._tcp.local."
SERVICE_NAME = "hippo-cluster"


@dataclass
class NodeInfo:
    """Information about a discovered cluster node."""
    name: str
    host: str
    port: int
    role: str = "worker"  # "gateway" or "worker"
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 0
    models: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)


class DiscoveryService:
    """mDNS discovery for Hippo cluster nodes.

    Usage:
        # Gateway mode: broadcast presence
        disc = DiscoveryService(role="gateway", port=11434)
        disc.start_broadcast()
        disc.start_browse(on_node_found)

        # Worker mode: discover gateways
        disc = DiscoveryService(role="worker", port=11435)
        disc.start_browse(on_gateway_found)
    """

    def __init__(
        self,
        role: str = "worker",
        port: int = 11434,
        node_info: Optional[NodeInfo] = None,
    ):
        self.role = role
        self.port = port
        self.node_info = node_info
        self._zeroconf: Optional[Zeroconf] = None
        self._browser: Optional[ServiceBrowser] = None
        self._service_info: Optional[ServiceInfo] = None
        self._discovered: dict[str, NodeInfo] = {}
        self._on_change: Optional[Callable] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hippo-mdns")

    def start_broadcast(self, extra_props: Optional[dict] = None):
        """Broadcast this node's presence via mDNS (blocking, call from thread)."""
        if self._zeroconf is None:
            self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)

        props = {
            "role": self.role,
            "port": str(self.port),
        }
        if self.node_info:
            props["gpu_memory_gb"] = str(self.node_info.gpu_memory_gb)
            props["cpu_cores"] = str(self.node_info.cpu_cores)
            if self.node_info.models:
                props["models"] = ",".join(self.node_info.models)
        if extra_props:
            props.update(extra_props)

        # Get local IP
        local_ip = self._get_local_ip()

        instance_name = f"{SERVICE_NAME}-{self.role}"
        self._service_info = ServiceInfo(
            SERVICE_TYPE,
            f"{instance_name}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(local_ip)],
            port=self.port,
            properties=props,
        )

        self._zeroconf.register_service(self._service_info)
        logger.info(f"Broadcasting as {instance_name} at {local_ip}:{self.port}")

    async def start_broadcast_async(self, extra_props: Optional[dict] = None):
        """Async wrapper — runs start_broadcast in a thread to avoid blocking event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self.start_broadcast, extra_props)

    def stop_broadcast(self):
        """Stop broadcasting."""
        if self._zeroconf and self._service_info:
            self._zeroconf.unregister_service(self._service_info)
            logger.info("Stopped broadcasting")

    def start_browse(self, on_change: Callable[[str, NodeInfo], None]):
        """Browse for other Hippo nodes on the LAN.

        Args:
            on_change: callback(service_name, NodeInfo) called on add/remove.
        """
        if self._zeroconf is None:
            self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)

        self._on_change = on_change

        def _on_service_state_change(
            zeroconf: Zeroconf,
            service_type: str,
            name: str,
            state_change: ServiceStateChange,
        ):
            if state_change == ServiceStateChange.Added:
                info = zeroconf.get_service_info(service_type, name)
                if info:
                    node = self._parse_service_info(name, info)
                    if node:
                        self._discovered[name] = node
                        logger.info(f"Discovered node: {node.name} ({node.role}) at {node.host}:{node.port}")
                        if self._on_change:
                            self._on_change("added", node)

            elif state_change == ServiceStateChange.Removed:
                node = self._discovered.pop(name, None)
                if node and self._on_change:
                    logger.info(f"Node left: {node.name}")
                    self._on_change("removed", node)

        self._browser = ServiceBrowser(
            self._zeroconf, SERVICE_TYPE, handlers=[_on_service_state_change]
        )
        logger.info("Started browsing for Hippo nodes")

    def stop_browse(self):
        """Stop browsing."""
        if self._browser:
            self._browser.cancel()
            self._browser = None

    def get_discovered_nodes(self) -> dict[str, NodeInfo]:
        """Return currently discovered nodes."""
        return dict(self._discovered)

    def shutdown(self):
        """Full cleanup."""
        self.stop_browse()
        self.stop_broadcast()
        if self._zeroconf:
            self._zeroconf.close()
            self._zeroconf = None

    @staticmethod
    def _parse_service_info(name: str, info: ServiceInfo) -> Optional[NodeInfo]:
        """Parse zeroconf ServiceInfo into NodeInfo."""
        try:
            addresses = info.parsed_addresses()
            if not addresses:
                return None

            props = {}
            if info.properties:
                for k, v in info.properties.items():
                    key = k.decode("utf-8") if isinstance(k, bytes) else k
                    val = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    props[key] = val

            models_str = props.get("models", "")
            models = [m.strip() for m in models_str.split(",") if m.strip()]

            return NodeInfo(
                name=name.replace(f".{SERVICE_TYPE}", ""),
                host=addresses[0],
                port=info.port,
                role=props.get("role", "worker"),
                gpu_memory_gb=float(props.get("gpu_memory_gb", 0)),
                cpu_cores=int(props.get("cpu_cores", 0)),
                models=models,
                properties=props,
            )
        except Exception as e:
            logger.warning(f"Failed to parse service info for {name}: {e}")
            return None

    @staticmethod
    def _get_local_ip() -> str:
        """Get the local LAN IP address."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
