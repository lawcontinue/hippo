"""Model lifecycle manager — load, unload, auto-cleanup."""

import time
import logging
import threading
from pathlib import Path
from typing import Optional

from llama_cpp import Llama

from hippo.config import HippoConfig
from hippo import metrics

logger = logging.getLogger("hippo")


class ModelManager:
    """Manages loaded models with lazy loading and auto-unload.

    Uses LFRU (Least-Frequent-Recently-Used) eviction: tracks both
    access frequency and recency. High-frequency models are retained
    longer than low-frequency ones, even if accessed less recently.
    """

    def __init__(self, config: HippoConfig):
        self.config = config
        self._loaded: dict[str, Llama] = {}
        self._last_used: dict[str, float] = {}
        self._access_count: dict[str, int] = {}  # LFRU: frequency tracking
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
            self._decay_access_counts()

    def _decay_access_counts(self):
        """Decay access counts by 10% each cleanup cycle to maintain LFRU differentiation.

        Without decay, long-running models accumulate counts that make
        frequency comparisons meaningless over time.
        """
        with self._lock:
            for name in self._access_count:
                self._access_count[name] = int(self._access_count[name] * 0.9)
                if self._access_count[name] < 1:
                    self._access_count[name] = 1

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
        """Find a GGUF file for the given model name.

        Supports multi-file GGUF shards (e.g., model-00001-of-00005.gguf).
        When shards are detected, returns the first shard for llama-cpp.
        """
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
            # Multi-file shard detection: model-00001-of-NNNNN.gguf
            shard = self._pick_first_shard(candidates)
            if shard:
                return shard
            return candidates[0]

        # Also check subdirectories
        candidates = list(models_dir.rglob(f"{base_name}*.gguf"))
        if candidates:
            shard = self._pick_first_shard(candidates)
            if shard:
                return shard
            return candidates[0]

        return None

    @staticmethod
    def _pick_first_shard(files: list[Path]) -> Optional[Path]:
        """From a list of GGUF files, detect multi-file shards and return the first one.

        Shard naming convention: ``<base>-00001-of-<total>.gguf``
        Returns None if no shard pattern is found.
        """
        import re

        shard_pattern = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
        shards = []
        for f in files:
            m = shard_pattern.search(f.name)
            if m:
                idx = int(m.group(1))
                shards.append((idx, f))

        if not shards:
            return None

        shards.sort(key=lambda x: x[0])
        return shards[0][1]

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
                self._access_count[name] = self._access_count.get(name, 0) + 1
                return self._loaded[name]

        # Per-model lock prevents concurrent loading of the same model
        model_lock = self._get_model_lock(name)
        with model_lock:
            # Double-check after acquiring model lock
            with self._lock:
                if name in self._loaded:
                    self._last_used[name] = time.time()
                    self._access_count[name] = self._access_count.get(name, 0) + 1
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

        # Detect embedding models (nomic-embed, bge-m3, etc.)
        is_embedding_model = any(
            keyword in name.lower()
            for keyword in ["embed", "bge", "e5", "retrieval"]
        )

        # Detect reranker models — need logits_all=True for logprob scoring
        is_reranker_model = any(
            keyword in name.lower()
            for keyword in ["rerank", "re-rank", "cross-encoder", "ms-marco", "minilm"]
        )

        llama = Llama(
            model_path=str(model_path),
            n_ctx=self.config.defaults.n_ctx,
            n_gpu_layers=self.config.defaults.n_gpu_layers,
            verbose=False,
            embedding=is_embedding_model,  # Enable embedding for embedding models
            logits_all=is_reranker_model,  # Enable logprobs for reranker scoring
        )

        elapsed = time.time() - start
        logger.info("Model '%s' loaded in %.1fs", name, elapsed)

        with self._lock:
            self._loaded[name] = llama
            self._last_used[name] = time.time()
            self._access_count[name] = self._access_count.get(name, 0) + 1

        # Metrics: record model load
        metrics.models_loaded.labels(model=name).set(1)

        # Metrics: record memory usage
        if model_path.exists():
            memory_bytes = model_path.stat().st_size
            metrics.memory_usage_bytes.labels(model=name).set(memory_bytes)

        return llama

    def _evict_if_needed(self, incoming_path: Path):
        """Evict least-frequent-recently-used models if memory limit would be exceeded.

        LFRU strategy: score = frequency / recency_age. Lower score = evict first.
        High-frequency models survive longer than low-frequency ones.
        """
        limit_gb = self.config.max_memory_gb
        if limit_gb <= 0:
            return

        incoming_gb = incoming_path.stat().st_size / (1024 ** 3)

        with self._lock:
            loaded_names = list(self._loaded.keys())

        current_gb = 0.0
        for n in loaded_names:
            p = self._resolve_model_path(n)
            if p:
                current_gb += p.stat().st_size / (1024 ** 3)

        now = time.time()
        while current_gb + incoming_gb > limit_gb and loaded_names:
            # LFRU score: freq / age. Lower = evict first.
            with self._lock:
                def _lfru_score(n):
                    freq = max(self._access_count.get(n, 1), 1)
                    age = max(now - self._last_used.get(n, now), 1.0)
                    return freq / age
                victim = min(loaded_names, key=_lfru_score)
                victim_freq = self._access_count.get(victim, 0)
            self.unload(victim)
            p = self._resolve_model_path(victim)
            if p:
                current_gb -= p.stat().st_size / (1024 ** 3)
            loaded_names.remove(victim)
            logger.info("LFRU evicted '%s' (freq=%d) to free memory", victim, victim_freq)

    def unload(self, name: str) -> bool:
        """Unload a model from memory. Returns True if it was loaded."""
        with self._lock:
            llama = self._loaded.pop(name, None)
            self._last_used.pop(name, None)
            # Note: keep _access_count for LFRU history (respects past popularity)

        if llama is not None:
            # Metrics: remove model load
            metrics.models_loaded.labels(model=name).set(0)
            # Metrics: remove memory usage
            metrics.memory_usage_bytes.labels(model=name).set(0)

            del llama
            logger.info("Unloaded model '%s'", name)
            return True
        return False

    def list_loaded(self) -> list[dict]:
        """List currently loaded models with last-used timestamps and access counts."""
        with self._lock:
            return [
                {
                    "name": n,
                    "last_used": self._last_used.get(n, 0),
                    "access_count": self._access_count.get(n, 0),
                }
                for n in self._loaded
            ]

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

    def prewarm(self, model_names: list[str] | None = None):
        """Pre-warm models by running a short inference to fill KV cache.

        If model_names is None, pre-warms all currently loaded models.
        Models must already be loaded (via get() or explicit load).
        """
        if model_names is None:
            with self._lock:
                model_names = list(self._loaded.keys())

        # Check memory budget before pre-warming
        limit_gb = self.config.max_memory_gb
        if limit_gb > 0 and model_names:
            total_gb = 0.0
            for name in model_names:
                p = self._resolve_model_path(name)
                if p:
                    total_gb += p.stat().st_size / (1024 ** 3)
            if total_gb > limit_gb:
                logger.warning(
                    "Pre-warm total %.1f GB exceeds limit %.1f GB, skipping some models",
                    total_gb, limit_gb,
                )
                # Only pre-warm models that fit
                fitted = []
                used = 0.0
                for name in model_names:
                    p = self._resolve_model_path(name)
                    size = p.stat().st_size / (1024 ** 3) if p else 0
                    if used + size <= limit_gb:
                        fitted.append(name)
                        used += size
                model_names = fitted

        for name in model_names:
            with self._lock:
                llama = self._loaded.get(name)

            if llama is None:
                logger.warning("Pre-warm: model '%s' not loaded, skipping", name)
                continue

            try:
                start = time.time()
                # Short inference to populate KV cache and GPU memory
                llama.create_completion(
                    prompt=" ",
                    max_tokens=1,
                    temperature=0.0,
                )
                elapsed = time.time() - start
                logger.info("Pre-warmed model '%s' in %.2fs", name, elapsed)
            except Exception as e:
                logger.warning("Pre-warm failed for '%s': %s", name, e)

    def unload_all(self):
        """Unload all loaded models."""
        with self._lock:
            for name, llama in list(self._loaded.items()):
                try:
                    del llama
                except Exception:
                    pass
            self._loaded.clear()
            self._last_used.clear()
            self._access_count.clear()
            logger.info("All models unloaded")
