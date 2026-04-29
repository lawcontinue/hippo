"""Audit logging — JSONL-based request audit trail.

Every API call is logged with timestamp, model, endpoint, latency,
token counts, and status. Designed for compliance auditing.

Usage:
    hippo serve --audit-log ~/.hippo/audit.jsonl

Note: This provides basic audit logging. Production compliance (SOC 2, ISO 27001)
may require additional measures such as log signing, centralized log collection,
and tamper-evident storage.
"""

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    """Thread-safe JSONL audit logger with in-memory counters."""

    def __init__(self, path: str | Path | None = None):
        self._enabled = path is not None
        self._path = Path(path) if path else None
        self._lock = threading.Lock()

        # In-memory counters for fast stats (P2-2 fix)
        self._total_requests = 0
        self._total_errors = 0
        self._models_used: dict[str, int] = {}

        if self._enabled:
            self._path = Path(os.path.expanduser(str(self._path)))
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log(
        self,
        method: str,
        path: str,
        status: int,
        latency_ms: float,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        client_ip: str | None = None,
        api_key: str | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ):
        """Log an API call to the audit trail."""
        if not self._enabled:
            return

        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "path": path,
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "queue_time_ms": round(extra.get("queue_time_ms", 0), 2) if extra else 0,
            "ttft_ms": round(extra.get("ttft_ms", 0), 2) if extra else 0,
            "generation_time_ms": round(extra.get("generation_time_ms", 0), 2) if extra else 0,
            "mean_itl_ms": round(extra.get("mean_itl_ms", 0), 2) if extra else 0,
        }

        if model:
            entry["model"] = model
        if prompt_tokens is not None:
            entry["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            entry["completion_tokens"] = completion_tokens
        if client_ip:
            entry["client_ip"] = client_ip
        if api_key:
            # P1-1 fix: 32 hex chars (128 bits) for collision resistance
            entry["api_key_hash"] = f"sha256:{hashlib.sha256(api_key.encode()).hexdigest()[:32]}"
        if error:
            # P2-4 fix: only log error type, not full message
            entry["error_type"] = type(error).__name__ if isinstance(error, Exception) else str(error)[:100]
        if extra:
            entry.update(extra)

        with self._lock:
            self._total_requests += 1
            if status >= 400:
                self._total_errors += 1
            if model:
                self._models_used[model] = self._models_used.get(model, 0) + 1

            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def rotate(self, max_size_mb: float = 100):
        """Rotate audit log if it exceeds max size."""
        if not self._enabled:
            return

        with self._lock:
            if not self._path.exists():
                return
            size_mb = self._path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                backup = self._path.with_suffix(
                    f".{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl"
                )
                self._path.rename(backup)
                # Reset in-memory counters after rotation
                self._total_requests = 0
                self._total_errors = 0
                self._models_used.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get audit log statistics (O(1) via in-memory counters)."""
        if not self._enabled:
            return {"enabled": False}

        file_size_mb = 0.0
        if self._path.exists():
            try:
                file_size_mb = round(self._path.stat().st_size / (1024 * 1024), 2)
            except OSError:
                pass

        return {
            "enabled": True,
            "path": str(self._path),
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": round(
                self._total_errors / max(self._total_requests, 1) * 100, 2
            ),
            "models_used": dict(self._models_used),
            "file_size_mb": file_size_mb,
        }
