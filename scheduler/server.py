"""Scheduler HTTP server — REST API for task submission and monitoring."""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict
from urllib.parse import urlparse

from .registry import NodeRegistry
from .scheduler import Scheduler

logger = logging.getLogger("scheduler")

# Default node config (env overrides)
DEFAULT_NODES = {
    "5060ti": {"url": "http://192.168.1.100:8000", "tags": ["gpu", "windows", "jimeng"]},
    "r1": {"url": "http://192.168.1.36:8000", "tags": ["mac", "minimax", "gemma", "long-running"]},
}


def create_scheduler() -> Scheduler:
    """Build scheduler with nodes from env or defaults."""
    registry = NodeRegistry()
    node_config = os.environ.get("SCHEDULER_NODES", "")
    if node_config:
        # JSON format: {"name":{"url":"...","tags":[...]}}
        for name, cfg in json.loads(node_config).items():
            registry.register(name, cfg["url"], cfg.get("tags"))
    else:
        for name, cfg in DEFAULT_NODES.items():
            registry.register(name, cfg["url"], cfg.get("tags"))

    max_concurrent = int(os.environ.get("SCHEDULER_MAX_CONCURRENT", "5"))
    sched = Scheduler(registry, max_concurrent=max_concurrent)

    # Initial health check
    registry.check_all()
    return sched


class SchedulerHandler(BaseHTTPRequestHandler):
    sched: Scheduler  # set by run_server

    def _json_response(self, code: int, data: Any) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/status":
            self._json_response(200, self.sched.status())
        elif path == "/tasks":
            tasks = self.sched.queue.recent()
            self._json_response(200, [
                {"id": t.id, "type": t.type, "status": t.status.value,
                 "duration": t.duration, "error": t.error}
                for t in tasks
            ])
        elif path == "/nodes":
            nodes = self.sched.registry.check_all()
            self._json_response(200, nodes)
        elif path.startswith("/tasks/"):
            tid = path.split("/")[-1]
            task = self.sched.queue.get(tid)
            if task:
                self._json_response(200, {
                    "id": task.id, "type": task.type, "status": task.status.value,
                    "result": task.result, "error": task.error, "duration": task.duration,
                })
            else:
                self._json_response(404, {"error": "task not found"})
        elif path == "/health":
            self._json_response(200, {"status": "ok"})
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path

        if path == "/tasks":
            body = self._read_body()
            task = self.sched.submit(
                task_type=body.get("type", "chat"),
                payload=body.get("payload", {}),
                priority=body.get("priority", 0),
                target_node=body.get("target_node"),
            )
            self._json_response(201, {
                "id": task.id, "status": task.status.value, "type": task.type,
            })
        else:
            self._json_response(404, {"error": "not found"})

    def log_message(self, format, *args) -> None:
        logger.info(format, *args)


def run_server(host: str = "0.0.0.0", port: int = 8090) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    sched = create_scheduler()
    sched.start()
    SchedulerHandler.sched = sched

    # Background health check every 60s
    def health_loop():
        while True:
            threading.Event().wait(60)
            sched.registry.check_all()

    threading.Thread(target=health_loop, daemon=True).start()

    server = HTTPServer((host, port), SchedulerHandler)
    logger.info("Scheduler server listening on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sched.stop()
        server.server_close()
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    run_server()
