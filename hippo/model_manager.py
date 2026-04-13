"""Model lifecycle manager — load, unload, auto-cleanup."""

import time
import logging
import threading
from pathlib import Path
from typing import Optional

from llama_cpp import Llama

from hippo.config import HippoConfig

logger = logging.getLogger("hippo")


class ModelManager:
    """Manages loaded models with lazy loading and auto-unload."""

    def __init__(self, config: HippoConfig):
        self.config = config
        self._loaded: dict[str, Llama] = {}
        self._last_used: dict[str, float] = {}
        self._lock = threading.Lock()
        self._model_locks: dict[str, threading.Lock] = {}
        self._stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._family_cache: dict[str, str] = {}

    def _get_model_lock(self, name: str) -> threading.Lock:
        """Get or create a per-model lock."""
        with self._lock:
            if name not in self._model_locks:
                self._model_locks[name] = threading.Lock()
            return self._model_locks[name]

    def start_cleanup_thread(self):
        """Start the background auto-unload thread."""
        if self.config.idle_timeout > 0:
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop, daemon=True
            )
            self._cleanup_thread.start()
            logger.info("Auto-unload thread started (timeout=%ds)", self.config.idle_timeout)

    def stop_cleanup_thread(self):
        """Stop the background thread."""
        self._stop_event.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

    def _cleanup_loop(self):
        """Periodically check and unload idle models."""
        while not self._stop_event.is_set():
            self._stop_event.wait(30)
            if self._stop_event.is_set():
                break
            self._unload_idle()

    def _unload_idle(self):
        """Unload models that have been idle longer than the timeout."""
        now = time.time()
        timeout = self.config.idle_timeout
        to_unload = []

        with self._lock:
            for name, last in self._last_used.items():
                if now - last > timeout:
                    to_unload.append(name)

        for name in to_unload:
            # Re-check last_used under lock before unloading (race condition fix)
            with self._lock:
                if name not in self._last_used:
                    continue  # already unloaded
                if time.time() - self._last_used[name] <= timeout:
                    continue  # recently used, skip
                # Mark for unload by removing tracking entries while holding lock
                self._last_used.pop(name, None)
                llama = self._loaded.pop(name, None)

            if llama is not None:
                del llama
                logger.info("Auto-unloaded model '%s' (idle %ds)", name, timeout)

    def _resolve_model_path(self, name: str) -> Optional[Path]:
        """Find a GGUF file for the given model name."""
        models_dir = self.config.models_dir

        # Direct filename match
        direct = models_dir / name
        if direct.exists() and direct.suffix == ".gguf":
            return direct

        # Strip tag (e.g., "llama3.2:3b" → "llama3.2")
        base_name = name.split(":")[0]

        # Search for matching GGUF files
        candidates = list(models_dir.glob(f"{base_name}*.gguf"))
        if candidates:
            for c in candidates:
                if c.stem == base_name:
                    return c
            return candidates[0]

        # Also check subdirectories
        candidates = list(models_dir.rglob(f"{base_name}*.gguf"))
        if candidates:
            return candidates[0]

        return None

    def _detect_family(self, model_path: Path) -> str:
        """Detect model family from GGUF metadata (cached)."""
        key = model_path.stem
        if key in self._family_cache:
            return self._family_cache[key]
        try:
            llama = Llama(model_path=str(model_path), n_ctx=1, verbose=False)
            meta = llama.metadata()
            arch = meta.get("general.architecture", "")
            del llama
            family = arch if arch else "unknown"
        except Exception:
            family = "unknown"
        self._family_cache[key] = family
        return family

    def get(self, name: str) -> Llama:
        """Get a loaded model, loading it lazily if needed (thread-safe with per-model lock)."""
        # Fast path: already loaded
        with self._lock:
            if name in self._loaded:
                self._last_used[name] = time.time()
                return self._loaded[name]

        # Per-model lock prevents concurrent loading of the same model
        model_lock = self._get_model_lock(name)
        with model_lock:
            # Double-check after acquiring model lock
            with self._lock:
                if name in self._loaded:
                    self._last_used[name] = time.time()
                    return self._loaded[name]

            return self._load(name)

    def _load(self, name: str) -> Llama:
        """Load a GGUF model into memory."""
        model_path = self._resolve_model_path(name)
        if model_path is None:
            raise FileNotFoundError(
                f"Model '{name}' not found in {self.config.models_dir}. "
                f"Run: hippo pull {name}"
            )

        # LRU eviction if memory limit is set
        self._evict_if_needed(model_path)

        logger.info("Loading model '%s' from %s ...", name, model_path)
        start = time.time()

        llama = Llama(
            model_path=str(model_path),
            n_ctx=self.config.defaults.n_ctx,
            n_gpu_layers=self.config.defaults.n_gpu_layers,
            verbose=False,
        )

        elapsed = time.time() - start
        logger.info("Model '%s' loaded in %.1fs", name, elapsed)

        with self._lock:
            self._loaded[name] = llama
            self._last_used[name] = time.time()

        return llama

    def _evict_if_needed(self, incoming_path: Path):
        """Evict least-recently-used models if memory limit would be exceeded."""
        limit_gb = self.config.max_memory_gb
        if limit_gb <= 0:
            return

        incoming_gb = incoming_path.stat().st_size / (1024 ** 3)

        # Calculate current memory usage from loaded models
        with self._lock:
            loaded_names = list(self._loaded.keys())

        current_gb = 0.0
        for n in loaded_names:
            p = self._resolve_model_path(n)
            if p:
                current_gb += p.stat().st_size / (1024 ** 3)

        while current_gb + incoming_gb > limit_gb and loaded_names:
            # Find LRU
            with self._lock:
                lru_name = min(loaded_names, key=lambda n: self._last_used.get(n, 0))
            self.unload(lru_name)
            p = self._resolve_model_path(lru_name)
            if p:
                current_gb -= p.stat().st_size / (1024 ** 3)
            loaded_names.remove(lru_name)
            logger.info("LRU evicted '%s' to free memory", lru_name)

    def unload(self, name: str) -> bool:
        """Unload a model from memory. Returns True if it was loaded."""
        with self._lock:
            llama = self._loaded.pop(name, None)
            self._last_used.pop(name, None)

        if llama is not None:
            del llama
            logger.info("Unloaded model '%s'", name)
            return True
        return False

    def list_loaded(self) -> list[dict]:
        """List currently loaded models with last-used timestamps."""
        with self._lock:
            return [{"name": n, "last_used": self._last_used.get(n, 0)} for n in self._loaded]

    def list_available(self) -> list[dict]:
        """List all available GGUF models on disk."""
        models = []
        for p in self.config.models_dir.rglob("*.gguf"):
            rel = p.relative_to(self.config.models_dir)
            name = rel.stem
            size_gb = p.stat().st_size / (1024 ** 3)
            with self._lock:
                loaded = name in self._loaded
            models.append({
                "name": name,
                "path": str(rel),
                "size_gb": round(size_gb, 2),
                "loaded": loaded,
            })
        return models

    def unload_all(self):
        """Unload all models."""
        with self._lock:
            names = list(self._loaded.keys())

        for name in names:
            self.unload(name)
