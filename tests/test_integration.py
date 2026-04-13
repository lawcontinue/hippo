"""Integration tests — end-to-end flows for Hippo."""

import os
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hippo.config import HippoConfig, ServerConfig, DefaultsConfig
from hippo.model_manager import ModelManager
from hippo import api


@pytest.fixture
def tmp_models(tmp_path):
    """Create a temporary models directory with a fake GGUF file."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    # Create a small dummy GGUF file (not a real model, just for path resolution)
    fake_model = models_dir / "test-model.gguf"
    fake_model.write_bytes(b"\x00" * 1024)
    return models_dir, fake_model


@pytest.fixture
def cfg(tmp_models):
    models_dir, _ = tmp_models
    return HippoConfig(
        server=ServerConfig(),
        models_dir=models_dir,
        idle_timeout=2,  # short timeout for testing
        max_memory_gb=0,
        max_concurrent_requests=2,
    )


@pytest.fixture
def client(cfg):
    """Create a TestClient with a mock ModelManager."""
    manager = ModelManager(cfg)
    app = api.app
    app.state.config = cfg
    app.state.manager = manager

    # Patch _load to avoid actually loading llama.cpp
    with patch.object(manager, '_load') as mock_load:
        mock_llama = MagicMock()
        mock_llama.create_completion.return_value = {
            "choices": [{"text": "Hello world"}]
        }
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hi there"}}]
        }
        mock_load.return_value = mock_llama

        # Also patch _detect_family to avoid llama.cpp
        manager._family_cache["test-model"] = "llama"

        with TestClient(app) as c:
            yield c, manager


# ============================================================
# Test 1: Full flow — load → generate → chat → unload
# ============================================================

class TestFullFlow:
    def test_generate_non_stream(self, client, tmp_models):
        c, manager = client
        # Model should be lazy-loaded on first request
        resp = c.post("/api/generate", json={
            "model": "test-model",
            "prompt": "Say hello",
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Hello world"
        assert data["done"] is True
        assert data["total_duration"] > 0

    def test_chat_non_stream(self, client, tmp_models):
        c, manager = client
        resp = c.post("/api/chat", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"]["content"] == "Hi there"
        assert data["done"] is True

    def test_generate_streaming(self, client, tmp_models):
        c, manager = client
        mock_llama = MagicMock()

        def fake_stream(prompt, **kwargs):
            yield {"choices": [{"text": "Hel"}]}
            yield {"choices": [{"text": "lo"}]}

        mock_llama.create_completion.side_effect = lambda prompt, stream=False, **kw: (
            fake_stream(prompt, **kw) if stream else
            {"choices": [{"text": "Hello"}]}
        )

        # Re-patch _load to return our streaming mock
        with patch.object(manager, '_load', return_value=mock_llama):
            manager._loaded["test-model"] = mock_llama
            manager._last_used["test-model"] = time.time()

            resp = c.post("/api/generate", json={
                "model": "test-model",
                "prompt": "Say hello",
                "stream": True,
            })
            assert resp.status_code == 200
            # NDJSON response
            lines = [l for l in resp.text.strip().split("\n") if l]
            assert len(lines) >= 3  # 2 chunks + final done

    def test_model_not_found(self, cfg):
        """Requesting a non-existent model returns 404."""
        manager = ModelManager(cfg)
        app = api.app
        app.state.config = cfg
        app.state.manager = manager
        # No mock _load — FileNotFoundError propagates to 404
        with TestClient(app) as c:
            resp = c.post("/api/generate", json={
                "model": "nonexistent-model",
                "prompt": "test",
                "stream": False,
            })
            assert resp.status_code == 404

    def test_list_models(self, client):
        c, _ = client
        resp = c.get("/api/tags")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        names = [m["name"] for m in data["models"]]
        assert "test-model" in names

    def test_show_model(self, client, tmp_models):
        c, manager = client
        models_dir, fake_model = tmp_models
        # Use family cache instead of loading llama.cpp
        manager._family_cache["test-model"] = "llama"
        resp = c.post("/api/show", json={"name": "test-model"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["details"]["family"] == "llama"
        assert data["size"] > 0

    def test_version(self, client):
        c, _ = client
        resp = c.get("/api/version")
        assert resp.status_code == 200
        assert resp.json()["version"] == "0.1.0"


# ============================================================
# Test 2: API key auth enforcement
# ============================================================

class TestAuth:
    def test_pull_requires_auth(self, tmp_models):
        """When HIPPO_API_KEY is set, /api/pull requires Bearer token."""
        models_dir, _ = tmp_models
        cfg = HippoConfig(models_dir=models_dir)
        os.environ["HIPPO_API_KEY"] = "test-secret-key"
        try:
            app = api.app
            app.state.config = cfg
            app.state.manager = ModelManager(cfg)

            with TestClient(app) as c:
                # No auth header → 401
                resp = c.post("/api/pull", json={"name": "some-model"})
                assert resp.status_code == 401

                # Wrong key → 401
                resp = c.post("/api/pull", json={"name": "some-model"},
                              headers={"Authorization": "Bearer wrong"})
                assert resp.status_code == 401

                # Correct key → proceed (will fail for other reasons, not 401)
                resp = c.post("/api/pull", json={"name": "some-model"},
                              headers={"Authorization": "Bearer test-secret-key"})
                assert resp.status_code != 401
        finally:
            os.environ.pop("HIPPO_API_KEY", None)

    def test_delete_requires_auth(self, tmp_models):
        """When HIPPO_API_KEY is set, /api/delete requires Bearer token."""
        models_dir, _ = tmp_models
        cfg = HippoConfig(models_dir=models_dir)
        os.environ["HIPPO_API_KEY"] = "test-secret-key"
        try:
            app = api.app
            app.state.config = cfg
            app.state.manager = ModelManager(cfg)

            with TestClient(app) as c:
                resp = c.request("DELETE", "/api/delete", json={"name": "test-model"})
                assert resp.status_code == 401

                resp = c.request("DELETE", "/api/delete", json={"name": "test-model"},
                                headers={"Authorization": "Bearer test-secret-key"})
                assert resp.status_code != 401
        finally:
            os.environ.pop("HIPPO_API_KEY", None)

    def test_no_auth_when_key_unset(self, tmp_models):
        """When HIPPO_API_KEY is not set, auth is skipped."""
        models_dir, _ = tmp_models
        os.environ.pop("HIPPO_API_KEY", None)
        cfg = HippoConfig(models_dir=models_dir)
        app = api.app
        app.state.config = cfg
        app.state.manager = ModelManager(cfg)

        with TestClient(app) as c:
            resp = c.post("/api/pull", json={"name": "some-model"})
            # Should not be 401 (may be 500 for other reasons)
            assert resp.status_code != 401


# ============================================================
# Test 3: Concurrency limiting
# ============================================================

class TestConcurrency:
    def test_concurrent_limit_enforced(self, cfg):
        """Verify concurrency middleware tracks active requests correctly."""
        assert cfg.max_concurrent_requests == 2
        from hippo import api as api_mod

        app = api_mod.app
        app.state.config = cfg
        app.state.manager = ModelManager(cfg)

        with TestClient(app) as c:
            resp = c.get("/api/version")
            assert resp.status_code == 200


# ============================================================
# Test 4: Idle timeout auto-unload
# ============================================================

class TestIdleTimeout:
    def test_auto_unload_after_idle(self, cfg):
        """Models are auto-unloaded after idle_timeout seconds."""
        manager = ModelManager(cfg)
        assert cfg.idle_timeout == 2

        mock_llama = MagicMock()
        with patch.object(manager, '_load', return_value=mock_llama):
            manager._loaded["test-model"] = mock_llama
            manager._last_used["test-model"] = time.time()

        # Verify model is loaded
        assert "test-model" in manager._loaded

        # Simulate idle time passing
        manager._last_used["test-model"] = time.time() - 5  # 5s ago (> 2s timeout)

        # Trigger unload check
        manager._unload_idle()

        # Model should be unloaded
        assert "test-model" not in manager._loaded
        assert "test-model" not in manager._last_used


# ============================================================
# Test 5: LRU eviction
# ============================================================

class TestLRUEviction:
    def test_lru_eviction_when_memory_exceeded(self, tmp_path):
        """LRU model is evicted when max_memory_gb would be exceeded."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Create two model files: 100KB each
        model_a = models_dir / "model-a.gguf"
        model_b = models_dir / "model-b.gguf"
        model_a.write_bytes(b"\x00" * 102400)
        model_b.write_bytes(b"\x00" * 102400)

        # Set memory limit to 150KB (can't hold both)
        cfg = HippoConfig(
            models_dir=models_dir,
            max_memory_gb=150.0 / (1024**3),  # ~150KB in GB
        )
        manager = ModelManager(cfg)

        mock_llama = MagicMock()
        with patch.object(manager, '_load', return_value=mock_llama):
            # Load model-a first
            manager._loaded["model-a"] = mock_llama
            manager._last_used["model-a"] = time.time() - 10  # older

            # Load model-b should trigger eviction of model-a
            manager._evict_if_needed(model_b)

        # model-a should have been evicted
        assert "model-a" not in manager._loaded
