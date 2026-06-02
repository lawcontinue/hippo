"""Scheduler — assign tasks to nodes, execute, collect results."""

from __future__ import annotations

import json
import logging
import time
import threading
import urllib.request
import uuid
from typing import Any, Dict, Optional

from .queue import Task, TaskQueue, TaskStatus
from .registry import NodeRegistry

logger = logging.getLogger("scheduler")


class Scheduler:
    def __init__(self, registry: NodeRegistry, max_concurrent: int = 5):
        self.registry = registry
        self.queue = TaskQueue()
        self.max_concurrent = max_concurrent
        self._running: Dict[str, threading.Thread] = {}
        self._stop = threading.Event()

    def submit(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        target_node: Optional[str] = None,
    ) -> Task:
        task = Task(
            id=uuid.uuid4().hex[:8],
            type=task_type,
            payload=payload,
            priority=priority,
            target_node=target_node,
        )
        self.queue.put(task)
        return task

    def _pick_node(self, task: Task) -> Optional[str]:
        """Pick best node for a task."""
        if task.target_node:
            node = self.registry.get(task.target_node)
            return task.target_node if node and node.is_online else None

        online = self.registry.online_nodes()
        if not online:
            return None

        # Tag-based routing
        tag_map = {
            "image": "gpu",
            "embed": "gpu",
            "jimeng": "jimeng",
        }
        preferred_tag = tag_map.get(task.type)
        if preferred_tag:
            tagged = self.registry.find_by_tag(preferred_tag)
            if tagged:
                return tagged[0].name

        # Round-robin: pick least-loaded node
        return online[0].name

    def _execute(self, task: Task) -> None:
        # Retry up to 3 times with backoff
        for attempt in range(3):
            node_name = self._pick_node(task)
            if node_name:
                break
            self._stop.wait(2 * (attempt + 1))
            if self._stop.is_set():
                self.queue.complete(task.id, error="scheduler stopping")
                return
        else:
            self.queue.complete(task.id, error="no available node after 3 retries")
            logger.warning("Task %s: no available node after retries", task.id)
            return

        node = self.registry.get(node_name)
        if not node:
            self.queue.complete(task.id, error=f"node {node_name} not found")
            return

        logger.info("Task %s → node %s (%s)", task.id, node_name, task.type)

        try:
            # Route to appropriate endpoint based on task type
            if task.type == "chat":
                endpoint = f"{node.url}/chat"
                data = json.dumps(task.payload).encode()
            elif task.type == "embed":
                endpoint = f"{node.url}/embed"
                data = json.dumps(task.payload).encode()
            elif task.type == "pipeline":
                endpoint = f"{node.url}/pipeline/run"
                data = json.dumps(task.payload).encode()
            else:
                endpoint = f"{node.url}/chat"
                data = json.dumps(task.payload).encode()

            req = urllib.request.Request(
                endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            self.queue.complete(task.id, result=result)
            logger.info("Task %s done (%.1fs)", task.id, task.duration or 0)
        except Exception as e:
            self.queue.complete(task.id, error=str(e))
            logger.error("Task %s failed: %s", task.id, e)

    def _worker(self) -> None:
        while not self._stop.is_set():
            active = sum(1 for t in self.queue._tasks.values() if t.status == TaskStatus.RUNNING)
            if active >= self.max_concurrent:
                self._stop.wait(1)
                continue

            task = self.queue.pop()
            if not task:
                self._stop.wait(1)
                continue

            t = threading.Thread(target=self._execute, args=(task,), daemon=True)
            self._running[task.id] = t
            t.start()

    def start(self) -> None:
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        logger.info("Scheduler started (max_concurrent=%d)", self.max_concurrent)

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> Dict[str, Any]:
        nodes = {n.name: {"url": n.url, "status": n.status, "tags": n.tags}
                 for n in self.registry.all_nodes()}
        return {
            "queue": self.queue.stats(),
            "nodes": nodes,
            "max_concurrent": self.max_concurrent,
        }
