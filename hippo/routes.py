"""Route configuration — external intent-to-model mapping.

Loads route definitions from ``~/.hippo/routes.json`` or the path
specified by ``HIPPO_ROUTES_PATH``. Each route maps an intent to
a model name and optional parameters.

Example routes.json:

    {
      "routes": [
        {
          "intent": "code",
          "model": "deepseek-r1-llama-8b",
          "priority": 1,
          "params": {"temperature": 0.3}
        },
        {
          "intent": "chat",
          "model": "gemma3-tools-12b",
          "priority": 2,
          "params": {"temperature": 0.7}
        },
        {
          "intent": "embed",
          "model": "nomic-embed-text-v1.5",
          "priority": 1
        }
      ],
      "default": {
        "model": "deepseek-r1-llama-8b",
        "params": {}
      }
    }

Reference: ollama/ollama#15619 (multi-model tier flags)
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hippo")

DEFAULT_ROUTES_PATH = Path("~/.hippo/routes.json").expanduser()


def _routes_path() -> Path:
    """Get routes file path from env or default."""
    env = os.environ.get("HIPPO_ROUTES_PATH")
    return Path(env).expanduser() if env else DEFAULT_ROUTES_PATH


class RouteEntry:
    """A single route mapping: intent → model + params."""

    __slots__ = ("intent", "model", "priority", "params")

    def __init__(
        self,
        intent: str,
        model: str,
        priority: int = 1,
        params: dict | None = None,
    ):
        self.intent = intent
        self.model = model
        self.priority = priority
        self.params = params or {}


class RouteConfig:
    """Manages external route configuration.

    Falls back gracefully: if no routes.json exists, all lookups return None
    and the caller uses its default model selection logic.
    """

    def __init__(self, path: Path | None = None):
        self._path = path or _routes_path()
        self._routes: dict[str, list[RouteEntry]] = {}
        self._default: Optional[RouteEntry] = None
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Load routes from JSON file. Returns True if loaded successfully.

        Thread-safe: uses double-checked locking to avoid redundant loads.
        """
        if self._loaded:
            return True

        with self._lock:
            # Double-check after acquiring lock
            if self._loaded:
                return True

            if not self._path.exists():
                logger.debug("No routes config at %s, using defaults", self._path)
                self._loaded = True
                return False

            try:
                with open(self._path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load routes from %s: %s", self._path, e)
                return False

            for entry in data.get("routes", []):
                # Validate required fields
                if not isinstance(entry, dict):
                    logger.warning("Skipping invalid route entry (not a dict): %s", entry)
                    continue
                if "intent" not in entry or "model" not in entry:
                    logger.warning("Skipping route entry missing 'intent' or 'model': %s", entry)
                    continue
                r = RouteEntry(
                    intent=entry["intent"],
                    model=entry["model"],
                    priority=entry.get("priority", 1),
                    params=entry.get("params", {}),
                )
                self._routes.setdefault(r.intent, []).append(r)
                self._routes[r.intent].sort(key=lambda x: x.priority)

            default = data.get("default")
            if default:
                self._default = RouteEntry(
                    intent="default",
                    model=default["model"],
                    params=default.get("params", {}),
                )

            self._loaded = True
            logger.info(
                "Loaded %d routes from %s (default: %s)",
                sum(len(v) for v in self._routes.values()),
                self._path,
                self._default.model if self._default else "none",
            )
            return True

    def resolve(self, intent: str) -> Optional[RouteEntry]:
        """Resolve an intent to a route entry. Returns None if no match."""
        self.load()  # lazy load on first use

        entries = self._routes.get(intent)
        if entries:
            return entries[0]  # highest priority first
        return None

    def resolve_default(self) -> Optional[RouteEntry]:
        """Get the default route entry."""
        self.load()
        return self._default

    def list_routes(self) -> list[dict]:
        """List all configured routes as dicts."""
        self.load()
        result = []
        for intent, entries in self._routes.items():
            for e in entries:
                result.append({
                    "intent": e.intent,
                    "model": e.model,
                    "priority": e.priority,
                    "params": e.params,
                })
        if self._default:
            result.append({
                "intent": "default",
                "model": self._default.model,
                "params": self._default.params,
            })
        return result

    def reload(self) -> bool:
        """Force reload routes from disk."""
        self._loaded = False
        self._routes.clear()
        self._default = None
        return self.load()
