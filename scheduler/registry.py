"""Node registry — track available worker nodes."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NodeInfo:
    name: str
    url: str  # e.g. "http://192.168.1.100:8000"
    tags: List[str] = field(default_factory=list)  # e.g. ["gpu", "windows", "jimeng"]
    status: str = "unknown"  # online / offline / unknown
    last_check: float = 0.0
    health: dict = field(default_factory=dict)

    @property
    def is_online(self) -> bool:
        return self.status == "online"


class NodeRegistry:
    def __init__(self):
        self._nodes: Dict[str, NodeInfo] = {}
        self._lock = threading.Lock()

    def register(self, name: str, url: str, tags: Optional[List[str]] = None) -> NodeInfo:
        with self._lock:
            node = NodeInfo(name=name, url=url.rstrip("/"), tags=tags or [])
            self._nodes[name] = node
            return node

    def get(self, name: str) -> Optional[NodeInfo]:
        return self._nodes.get(name)

    def all_nodes(self) -> List[NodeInfo]:
        return list(self._nodes.values())

    def online_nodes(self) -> List[NodeInfo]:
        return [n for n in self._nodes.values() if n.is_online]

    def find_by_tag(self, tag: str) -> List[NodeInfo]:
        return [n for n in self.online_nodes() if tag in n.tags]

    def check_health(self, node: NodeInfo) -> bool:
        """Ping node /health endpoint. Timeout = still online (may be busy)."""
        try:
            req = urllib.request.Request(f"{node.url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                node.health = json.loads(resp.read())
                node.status = "online"
                node.last_check = time.time()
                return True
        except Exception:
            # Don't mark offline on timeout — node may be busy
            if node.status == "online":
                node.last_check = time.time()
            else:
                node.status = "offline"
                node.last_check = time.time()
            return node.status == "online"

    def check_all(self) -> Dict[str, bool]:
        results = {}
        with self._lock:
            nodes = list(self._nodes.values())
        for node in nodes:
            results[node.name] = self.check_health(node)
        return results
