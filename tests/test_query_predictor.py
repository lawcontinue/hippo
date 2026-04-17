"""Tests for Query Predictor (Sleep-time Compute)."""

import time
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from hippo.query_predictor import (
    QueryPredictor,
    CachedPrediction,
    _classify_intent,
    _FOLLOWUP_GRAPH,
    _FOLLOWUP_PROMPTS,
)


# ── Intent classification ──────────────────────────────────────────────

class TestIntentClassification:
    def test_code_define(self):
        assert _classify_intent("def fibonacci(n):") == "code_define"
        assert _classify_intent("class MyModel:") == "code_define"
        assert _classify_intent("import os") == "code_define"

    def test_code_debug(self):
        assert _classify_intent("fix the error in my code") == "code_debug"
        assert _classify_intent("debug this traceback") == "code_debug"

    def test_code_test(self):
        assert _classify_intent("write a test for this function") == "code_test"
        assert _classify_intent("pytest this module") == "code_test"

    def test_code_refactor(self):
        assert _classify_intent("refactor this for readability") == "code_refactor"
        assert _classify_intent("optimize the performance") == "code_refactor"

    def test_explain(self):
        assert _classify_intent("explain how this works") == "explain"
        assert _classify_intent("what is a decorator?") == "explain"

    def test_code_generate(self):
        assert _classify_intent("write a web scraper") == "code_generate"
        assert _classify_intent("create a REST API") == "code_generate"

    def test_review(self):
        assert _classify_intent("review my pull request") == "review"
        assert _classify_intent("audit the codebase") == "review"

    def test_general(self):
        assert _classify_intent("hello world") == "general"
        assert _classify_intent("") == "general"


# ── Recording ──────────────────────────────────────────────────────────

class TestRecording:
    def test_record_generate(self):
        qp = QueryPredictor()
        qp.record("/api/generate", {"model": "deepseek-r1", "prompt": "def foo():"})
        assert len(qp._history) == 1
        assert qp._history[0]["intent"] == "code_define"

    def test_record_chat(self):
        qp = QueryPredictor()
        qp.record("/api/chat", {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "debug this error"}],
        })
        assert qp._history[0]["intent"] == "code_debug"

    def test_history_truncation(self):
        qp = QueryPredictor()
        for i in range(60):
            qp.record("/api/generate", {"model": "m", "prompt": f"prompt {i}"})
        assert len(qp._history) == 50  # MAX_HISTORY


# ── Prediction ─────────────────────────────────────────────────────────

class TestPrediction:
    def test_predict_after_code_define(self):
        qp = QueryPredictor()
        qp.record("/api/generate", {"model": "deepseek-r1", "prompt": "def fibonacci(n):"})
        preds = qp.predict()
        assert len(preds) > 0
        # code_test should be top follow-up for code_define
        intents = [p["intent"] for p in preds]
        assert "code_test" in intents

    def test_predict_empty_history(self):
        qp = QueryPredictor()
        assert qp.predict() == []

    def test_predict_confidence_range(self):
        qp = QueryPredictor()
        qp.record("/api/generate", {"model": "deepseek-r1", "prompt": "def foo():"})
        for p in qp.predict():
            assert 0.0 <= p["confidence"] <= 1.0

    def test_predict_uses_model_from_last(self):
        qp = QueryPredictor()
        qp.record("/api/generate", {"model": "gemma3-12b", "prompt": "write a function"})
        preds = qp.predict()
        assert all(p["model"] == "gemma3-12b" for p in preds)


# ── Caching ────────────────────────────────────────────────────────────

class TestCaching:
    def test_cache_lookup_exact(self):
        qp = QueryPredictor()
        # Manually inject a cached prediction
        key = qp._hash_request("/api/generate", "def foo():", "deepseek-r1")
        qp._cache[key] = CachedPrediction(
            prompt_hash=key,
            model="deepseek-r1",
            endpoint="/api/generate",
            intent="code_define",
            result={"response": "def foo(): pass", "done": True},
            ttl_seconds=300,
            confidence=0.8,
        )
        hit = qp.lookup("/api/generate", {"prompt": "def foo():", "model": "deepseek-r1"})
        assert hit is not None
        assert hit["response"] == "def foo(): pass"

    def test_cache_miss(self):
        qp = QueryPredictor()
        hit = qp.lookup("/api/generate", {"prompt": "something random", "model": "m"})
        assert hit is None

    def test_cache_expiry(self):
        qp = QueryPredictor()
        key = qp._hash_request("/api/generate", "old prompt", "m")
        qp._cache[key] = CachedPrediction(
            prompt_hash=key, model="m", endpoint="/api/generate",
            intent="general", result={"response": "old"},
            ttl_seconds=-1,  # already expired
            confidence=0.5,
        )
        hit = qp.lookup("/api/generate", {"prompt": "old prompt", "model": "m"})
        assert hit is None  # expired


# ── Pre-computation ────────────────────────────────────────────────────

class TestPrecomputation:
    def test_prewarm_no_manager(self):
        qp = QueryPredictor(manager=None)
        qp.record("/api/generate", {"model": "m", "prompt": "def test():"})
        # Should not crash
        qp.prewarm_predictions()

    def test_prewarm_with_mock_manager(self):
        mock_llama = MagicMock()
        mock_llama.create_completion.return_value = {
            "choices": [{"text": "def test_foo(): pass"}]
        }
        mock_manager = MagicMock()
        mock_manager.get.return_value = mock_llama

        qp = QueryPredictor(manager=mock_manager)
        qp.record("/api/generate", {"model": "deepseek-r1", "prompt": "def foo():"})
        qp.prewarm_predictions(top_k=2)

        # Should have pre-computed and cached
        assert len(qp._cache) > 0
        assert qp._stats["precomputations"] > 0

    def test_prewarm_skips_already_cached(self):
        mock_llama = MagicMock()
        mock_llama.create_completion.return_value = {"choices": [{"text": "result"}]}
        mock_manager = MagicMock()
        mock_manager.get.return_value = mock_llama

        qp = QueryPredictor(manager=mock_manager)
        qp.record("/api/generate", {"model": "deepseek-r1", "prompt": "def foo():"})

        # First prewarm
        qp.prewarm_predictions(top_k=1)
        call_count_1 = mock_llama.create_completion.call_count

        # Second prewarm (should skip cached)
        qp.prewarm_predictions(top_k=1)
        assert mock_llama.create_completion.call_count == call_count_1  # no new calls

    def test_cache_eviction(self):
        qp = QueryPredictor()
        mock_llama = MagicMock()
        mock_llama.create_completion.return_value = {"choices": [{"text": "result"}]}
        mock_manager = MagicMock()
        mock_manager.get.return_value = mock_llama
        qp._manager = mock_manager

        # Fill cache via repeated prewarm cycles
        for i in range(15):
            qp._history = [{"prompt": f"prompt_{i}", "intent": "code_define", "model": "m"}]
            qp.prewarm_predictions(top_k=3)

        # Cache should not exceed MAX_CACHE after each prewarm cycle
        # (eviction happens per-entry insertion)
        assert len(qp._cache) <= qp.MAX_CACHE


# ── Statistics ─────────────────────────────────────────────────────────

class TestStatistics:
    def test_get_stats(self):
        qp = QueryPredictor()
        qp.record("/api/generate", {"model": "m", "prompt": "def x():"})
        stats = qp.get_stats()
        assert "history_size" in stats
        assert "cache_entries" in stats
        assert "stats" in stats
        assert stats["history_size"] == 1

    def test_cache_hit_miss_tracking(self):
        qp = QueryPredictor()
        # Miss
        qp.lookup("/api/generate", {"prompt": "miss", "model": "m"})
        assert qp._stats["cache_misses"] == 1

        # Hit
        key = qp._hash_request("/api/generate", "hit", "m")
        qp._cache[key] = CachedPrediction(
            prompt_hash=key, model="m", endpoint="/api/generate",
            intent="general", result={"response": "hit"},
            ttl_seconds=300, confidence=0.5,
        )
        qp.lookup("/api/generate", {"prompt": "hit", "model": "m"})
        assert qp._stats["cache_hits"] == 1


# ── Persistence ────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_load_stats(self, tmp_path):
        stats_file = tmp_path / "predictor_stats.json"
        qp = QueryPredictor(persist_path=stats_file)
        qp._stats["precomputations"] = 42
        qp._save_stats()

        qp2 = QueryPredictor(persist_path=stats_file)
        qp2._load_stats()
        assert qp2._stats["precomputations"] == 42


# ── Background thread ─────────────────────────────────────────────────

class TestBackground:
    def test_start_stop(self):
        qp = QueryPredictor()
        qp.start_background(interval_seconds=1)
        time.sleep(0.5)
        assert qp._bg_thread.is_alive()
        qp.stop_background()
        time.sleep(0.5)
        assert not qp._bg_thread.is_alive()


# ── P0 fix regression tests ────────────────────────────────────────────

class TestP0Fixes:
    def test_p0_1_stats_thread_safe(self):
        """P0-1 fix: _inc_stat must be thread-safe."""
        import threading
        qp = QueryPredictor()
        n = 100

        def inc_many():
            for _ in range(n):
                qp._inc_stat("cache_hits")

        threads = [threading.Thread(target=inc_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert qp._stats["cache_hits"] == n * 5  # 500

    def test_p0_2_fuzzy_match_uses_source_prompt(self):
        """P0-2 fix: fuzzy match compares source_prompt, not response."""
        qp = QueryPredictor()
        key = "test_key"
        # Cache with source_prompt "def fibonacci" but response "1 2 3 4 5"
        qp._cache[key] = CachedPrediction(
            prompt_hash=key, model="m", endpoint="/api/generate",
            intent="code_define",
            result={"response": "1 2 3 4 5 6 7 8 9 10"},
            ttl_seconds=300, confidence=0.8,
            source_prompt="def fibonacci(n): return result",
        )
        # Query with similar words to source_prompt, NOT to response
        # "def fibonacci(n): return result" shares "def fibonacci return result" with query
        hit = qp.lookup("/api/generate", {"prompt": "def fibonacci(n): return result", "model": "m"})
        assert hit is not None  # fuzzy match should hit on source_prompt overlap
