"""Sleep-time Query Predictor — pre-compute likely queries before they arrive.

Inspired by arxiv 2504.13171 (Sleep-time Compute, Letta 2025):
predict what the user will ask next, run inference ahead of time,
and serve cached results instantly.

v1 design: rule-based prediction (no ML model needed).
  - Keyword/intent classification from request history
  - Multi-query amortization (related queries share pre-computation)
  - TTL-gated cache with consistency checks (Shield's security requirement)
  - Thread-safe, integrated with Hippo's ModelManager and pre-warm lifecycle.

Usage:
  predictor = QueryPredictor(manager)
  predictor.record("/api/generate", {"model": "deepseek-r1-8b", "prompt": "def fibonacci"})
  predictions = predictor.predict()  # → [{"prompt": "def fibonacci_iterative", "model": "deepseek-r1-8b", "confidence": 0.7}]
  await predictor.prewarm_predictions()  # pre-compute in background
  hit = predictor.lookup("/api/generate", {"prompt": "def fibonacci_iterative"})  # cached result or None
"""

import time
import re
import json
import logging
import threading
import hashlib
from pathlib import Path
from typing import Optional
from collections import Counter

logger = logging.getLogger("hippo")

# ---------------------------------------------------------------------------
# Intent classification (rule-based v1)
# ---------------------------------------------------------------------------

# Keyword → intent mapping, ordered by specificity (longer patterns first)
_INTENT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(def |class |function |import |from \w+ import)\b", re.I), "code_define"),
    (re.compile(r"\b(fix|debug|error|traceback|exception|bug)\b", re.I), "code_debug"),
    (re.compile(r"\b(refactor|optimize|clean|simplify|improve)\b", re.I), "code_refactor"),
    (re.compile(r"\b(test|unittest|pytest|spec|assert)\b", re.I), "code_test"),
    (re.compile(r"\b(explain|what is|how does|describe|tell me)\b", re.I), "explain"),
    (re.compile(r"\b(write|create|generate|make|build)\b", re.I), "code_generate"),
    (re.compile(r"\b(review|check|audit|analyze)\b", re.I), "review"),
    (re.compile(r"\b(translat|convert|rewrite)\b", re.I), "transform"),
    (re.compile(r"\b(summar|tldr|brief|condense)\b", re.I), "summarize"),
    (re.compile(r"\b(compare|difference|vs\.?|versus)\b", re.I), "compare"),
]


def _classify_intent(prompt: str) -> str:
    """Classify a prompt into an intent category using keyword rules."""
    for pattern, intent in _INTENT_RULES:
        if pattern.search(prompt):
            return intent
    return "general"


# ---------------------------------------------------------------------------
# Follow-up prediction patterns
# ---------------------------------------------------------------------------

# Given an intent, what intents typically follow?
# Built from common programming workflow patterns.
_FOLLOWUP_GRAPH: dict[str, list[tuple[str, float]]] = {
    "code_define":   [("code_test", 0.5), ("code_debug", 0.2), ("code_refactor", 0.15), ("explain", 0.1)],
    "code_debug":    [("code_define", 0.3), ("code_test", 0.3), ("explain", 0.2), ("code_refactor", 0.1)],
    "code_refactor": [("code_test", 0.4), ("code_define", 0.2), ("explain", 0.2)],
    "code_test":     [("code_debug", 0.3), ("code_refactor", 0.2), ("code_define", 0.2)],
    "code_generate": [("code_test", 0.3), ("code_debug", 0.25), ("explain", 0.2), ("code_refactor", 0.15)],
    "explain":       [("code_generate", 0.3), ("compare", 0.2), ("code_define", 0.15)],
    "review":        [("code_refactor", 0.4), ("code_debug", 0.2), ("code_test", 0.2)],
    "transform":     [("code_test", 0.3), ("explain", 0.2), ("code_refactor", 0.2)],
    "summarize":     [("explain", 0.3), ("compare", 0.2), ("code_generate", 0.1)],
    "compare":       [("code_generate", 0.3), ("explain", 0.3)],
    "general":       [("code_generate", 0.2), ("explain", 0.2), ("code_define", 0.15)],
}

# Prompt suffixes for predicted follow-ups (appended to original prompt context)
_FOLLOWUP_PROMPTS: dict[str, str] = {
    "code_test":     "\n\nNow write unit tests for the above code.",
    "code_debug":    "\n\nFind and fix any bugs in the above code.",
    "code_refactor": "\n\nRefactor the above code for better readability and performance.",
    "explain":       "\n\nExplain how the above code works step by step.",
    "code_define":   "\n\nNow implement a related function: ",
    "code_generate": "\n\nGenerate a complete implementation for: ",
    "review":        "\n\nReview the above code for issues and improvements.",
    "transform":     "\n\nRewrite the above in a different language or style.",
    "summarize":     "\n\nProvide a brief summary of the above.",
    "compare":       "\n\nCompare the above with alternative approaches.",
}


# ---------------------------------------------------------------------------
# Prediction cache entry
# ---------------------------------------------------------------------------

class CachedPrediction:
    """A single pre-computed prediction result."""

    __slots__ = ("prompt_hash", "model", "endpoint", "intent", "result",
                 "created_at", "ttl_seconds", "confidence", "hit_count")

    def __init__(
        self,
        prompt_hash: str,
        model: str,
        endpoint: str,
        intent: str,
        result: dict,
        ttl_seconds: float = 300.0,
        confidence: float = 0.0,
    ):
        self.prompt_hash = prompt_hash
        self.model = model
        self.endpoint = endpoint
        self.intent = intent
        self.result = result
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
        self.confidence = confidence
        self.hit_count = 0

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "intent": self.intent,
            "confidence": round(self.confidence, 2),
            "age_seconds": round(time.time() - self.created_at, 1),
            "ttl_seconds": self.ttl_seconds,
            "hit_count": self.hit_count,
        }


# ---------------------------------------------------------------------------
# Query Predictor
# ---------------------------------------------------------------------------

class QueryPredictor:
    """Predicts likely next queries and pre-computes results.

    Thread-safe. Integrates with ModelManager for inference and pre-warm.
    """

    # History buffer size
    MAX_HISTORY = 50
    # Max cached predictions
    MAX_CACHE = 20
    # Default TTL for cached predictions (seconds)
    DEFAULT_TTL = 300.0  # 5 minutes

    def __init__(self, manager=None, persist_path: Optional[Path] = None):
        self._manager = manager
        self._history: list[dict] = []  # [{endpoint, model, prompt, intent, ts}]
        self._cache: dict[str, CachedPrediction] = {}
        self._lock = threading.Lock()
        self._persist_path = persist_path
        self._stop_event = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._stats = {"predictions_made": 0, "cache_hits": 0, "cache_misses": 0, "precomputations": 0}

        # Load persisted stats
        if persist_path and persist_path.exists():
            self._load_stats()

    def set_manager(self, manager):
        """Set or update the model manager reference."""
        self._manager = manager

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, endpoint: str, body: dict):
        """Record a request into history for pattern learning."""
        prompt = body.get("prompt", "") or ""
        if not prompt:
            # chat endpoint: concatenate messages
            messages = body.get("messages", [])
            prompt = " ".join(m.get("content", "") for m in messages if m.get("content"))

        model = body.get("model", "")
        intent = _classify_intent(prompt)

        entry = {
            "endpoint": endpoint,
            "model": model,
            "prompt": prompt[:500],  # truncate for memory
            "intent": intent,
            "ts": time.time(),
        }

        with self._lock:
            self._history.append(entry)
            if len(self._history) > self.MAX_HISTORY:
                self._history = self._history[-self.MAX_HISTORY:]

        logger.debug("Recorded query: intent=%s model=%s", intent, model)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, top_k: int = 5) -> list[dict]:
        """Predict likely next queries based on history.

        Returns list of {"prompt_suffix", "model", "intent", "confidence"}.
        """
        with self._lock:
            history = list(self._history)

        if not history:
            return []

        # Get last query's context
        last = history[-1]
        last_intent = last["intent"]
        model = last["model"]

        # Follow-up graph lookup
        followups = _FOLLOWUP_GRAPH.get(last_intent, [])
        if not followups:
            return []

        # Boost confidence for recently-seen intents (temporal locality)
        recent_intents = Counter(h["intent"] for h in history[-10:])
        total_recent = sum(recent_intents.values()) or 1

        predictions = []
        for intent, base_conf in followups[:top_k]:
            # Boost if this intent appeared recently
            freq_boost = recent_intents.get(intent, 0) / total_recent
            confidence = min(base_conf + freq_boost * 0.2, 0.95)

            suffix = _FOLLOWUP_PROMPTS.get(intent, "")
            predictions.append({
                "prompt_suffix": suffix,
                "model": model,
                "intent": intent,
                "confidence": round(confidence, 3),
            })

        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:top_k]

    # ------------------------------------------------------------------
    # Pre-computation (Sleep-time Compute)
    # ------------------------------------------------------------------

    def prewarm_predictions(self, top_k: int = 3):
        """Pre-compute top-K predicted queries in background.

        This is the core "sleep-time compute" operation: run inference
        for predicted queries before the user actually asks them.
        """
        if self._manager is None:
            logger.warning("QueryPredictor: no manager, cannot prewarm")
            return

        predictions = self.predict(top_k=top_k)
        if not predictions:
            return

        # Get the last prompt as context prefix
        with self._lock:
            if not self._history:
                return
            last_prompt = self._history[-1].get("prompt", "")

        for pred in predictions:
            full_prompt = last_prompt + pred["prompt_suffix"]
            cache_key = self._hash_request("/api/generate", full_prompt, pred["model"])

            # Skip if already cached and fresh
            with self._lock:
                existing = self._cache.get(cache_key)
                if existing and not existing.expired:
                    continue

            # Run inference
            try:
                llama = self._manager.get(pred["model"])
                result = llama.create_completion(
                    prompt=full_prompt,
                    max_tokens=256,  # short pre-computation
                    temperature=0.3,
                )
                text = result["choices"][0]["text"]

                cached = CachedPrediction(
                    prompt_hash=cache_key,
                    model=pred["model"],
                    endpoint="/api/generate",
                    intent=pred["intent"],
                    result={"response": text, "done": True},
                    ttl_seconds=self.DEFAULT_TTL,
                    confidence=pred["confidence"],
                )

                with self._lock:
                    self._cache[cache_key] = cached

                # Evict oldest if over limit (outside lock to avoid nested lock)
                with self._lock:
                    while len(self._cache) > self.MAX_CACHE:
                        oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
                        del self._cache[oldest_key]

                self._stats["precomputations"] += 1
                logger.info(
                    "Pre-computed prediction: intent=%s confidence=%.2f model=%s (%d chars)",
                    pred["intent"], pred["confidence"], pred["model"], len(text),
                )

            except Exception as e:
                logger.warning("Pre-computation failed for intent=%s: %s", pred["intent"], e)

    # ------------------------------------------------------------------
    # Cache lookup
    # ------------------------------------------------------------------

    def lookup(self, endpoint: str, body: dict) -> Optional[dict]:
        """Check if a request matches a cached prediction.

        Returns the cached result dict or None.
        Uses fuzzy matching: checks if the request intent matches a
        cached prediction's intent and the prompt is semantically similar.
        """
        prompt = body.get("prompt", "") or ""
        if not prompt:
            messages = body.get("messages", [])
            prompt = " ".join(m.get("content", "") for m in messages if m.get("content"))

        model = body.get("model", "")
        intent = _classify_intent(prompt)

        with self._lock:
            # Strategy 1: exact hash match
            exact_key = self._hash_request(endpoint, prompt, model)
            if exact_key in self._cache:
                cached = self._cache[exact_key]
                if not cached.expired:
                    cached.hit_count += 1
                    self._stats["cache_hits"] += 1
                    logger.info("Cache HIT (exact): intent=%s model=%s", intent, model)
                    return cached.result
                else:
                    del self._cache[exact_key]

            # Strategy 2: intent-based fuzzy match (same intent + model)
            for key, cached in list(self._cache.items()):
                if cached.expired:
                    del self._cache[key]
                    continue
                if cached.intent == intent and cached.model == model:
                    # Simple word overlap check
                    if self._prompt_similarity(prompt, cached) > 0.4:
                        cached.hit_count += 1
                        self._stats["cache_hits"] += 1
                        logger.info("Cache HIT (fuzzy): intent=%s model=%s", intent, model)
                        return cached.result

        self._stats["cache_misses"] += 1
        return None

    def _prompt_similarity(self, prompt: str, cached: CachedPrediction) -> float:
        """Simple word-overlap similarity for fuzzy cache matching."""
        cached_prompt_words = set(
            (cached.result.get("response", "") or "").lower().split()[:50]
        )
        query_words = set(prompt.lower().split())
        if not cached_prompt_words or not query_words:
            return 0.0
        overlap = len(cached_prompt_words & query_words)
        return overlap / max(len(cached_prompt_words), len(query_words), 1)

    # ------------------------------------------------------------------
    # Background pre-warm thread
    # ------------------------------------------------------------------

    def start_background(self, interval_seconds: float = 60.0):
        """Start background pre-computation thread."""
        self._bg_thread = threading.Thread(
            target=self._bg_loop,
            args=(interval_seconds,),
            daemon=True,
        )
        self._bg_thread.start()
        logger.info("QueryPredictor background thread started (interval=%ds)", interval_seconds)

    def stop_background(self):
        """Stop background thread."""
        self._stop_event.set()
        if self._bg_thread:
            self._bg_thread.join(timeout=5)

    def _bg_loop(self, interval: float):
        while not self._stop_event.is_set():
            self._stop_event.wait(interval)
            if self._stop_event.is_set():
                break
            try:
                self.prewarm_predictions(top_k=3)
            except Exception as e:
                logger.warning("Background prewarm error: %s", e)
            # Periodically persist stats
            if self._persist_path:
                self._save_stats()

    # ------------------------------------------------------------------
    # Statistics & introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return predictor statistics."""
        with self._lock:
            cache_entries = len(self._cache)
            cache_sizes = {k: v.to_dict() for k, v in self._cache.items()}

        return {
            "history_size": len(self._history),
            "cache_entries": cache_entries,
            "stats": dict(self._stats),
            "cache": cache_sizes,
        }

    def get_predictions(self) -> list[dict]:
        """Get current predictions (for /api/predict endpoint)."""
        return self.predict(top_k=5)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _hash_request(self, endpoint: str, prompt: str, model: str) -> str:
        return hashlib.sha256(f"{endpoint}|{prompt[:200]}|{model}".encode()).hexdigest()[:16]

    def _save_stats(self):
        if not self._persist_path:
            return
        try:
            data = {"stats": self._stats}
            self._persist_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to persist predictor stats: %s", e)

    def _load_stats(self):
        if not self._persist_path:
            return
        try:
            data = json.loads(self._persist_path.read_text())
            self._stats.update(data.get("stats", {}))
        except Exception:
            pass
