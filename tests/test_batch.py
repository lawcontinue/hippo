"""Tests for batch inference endpoints."""
import pytest
from fastapi.testclient import TestClient

from hippo import api
from hippo.config import HippoConfig
from hippo.model_manager import ModelManager


@pytest.fixture
def client(tmp_path):
    cfg = HippoConfig()
    cfg.models_dir = tmp_path
    mgr = ModelManager(cfg)
    api.app.state.config = cfg
    api.app.state.manager = mgr
    return TestClient(api.app)


def test_batch_generate_empty_requests(client):
    """Test batch generate with empty requests list."""
    resp = client.post("/api/batch/generate", json={"requests": []})
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []
    assert data["success_count"] == 0
    assert data["error_count"] == 0


def test_batch_generate_model_not_found(client):
    """Test batch generate with non-existent model."""
    resp = client.post(
        "/api/batch/generate",
        json={
            "requests": [
                {"model": "nonexistent", "prompt": "hello"},
                {"model": "nonexistent2", "prompt": "world"}
            ],
            "max_concurrent": 2
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    assert data["success_count"] == 0
    assert data["error_count"] == 2
    # All requests should fail with model not found
    for result in data["results"]:
        assert result["status"] == "error"
        assert "not found" in result["error"]


def test_batch_generate_exceeds_max_size(client, monkeypatch):
    """Test batch generate exceeds maximum batch size."""
    # Override batch max_size to a small value for testing
    monkeypatch.setenv("HIPPO_BATCH_MAX_SIZE", "2")

    # Create a large batch
    requests = [{"model": "test", "prompt": f"prompt{i}"} for i in range(10)]

    resp = client.post("/api/batch/generate", json={"requests": requests})
    assert resp.status_code == 400
    assert "exceeds maximum" in resp.json()["error"]


def test_batch_generate_max_concurrent_exceeds_limit(client):
    """Test batch generate max_concurrent exceeds limit."""
    resp = client.post(
        "/api/batch/generate",
        json={
            "requests": [{"model": "test", "prompt": "hello"}],
            "max_concurrent": 100  # Exceeds maximum of 16
        }
    )
    assert resp.status_code == 400
    assert "exceeds maximum 16" in resp.json()["error"]


def test_batch_chat_empty_requests(client):
    """Test batch chat with empty requests list."""
    resp = client.post("/api/batch/chat", json={"requests": []})
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []
    assert data["success_count"] == 0
    assert data["error_count"] == 0


def test_batch_chat_model_not_found(client):
    """Test batch chat with non-existent model."""
    resp = client.post(
        "/api/batch/chat",
        json={
            "requests": [
                {"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
                {"model": "nonexistent2", "messages": [{"role": "user", "content": "bye"}]}
            ],
            "max_concurrent": 2
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    assert data["success_count"] == 0
    assert data["error_count"] == 2
    # All requests should fail with model not found
    for result in data["results"]:
        assert result["status"] == "error"
        assert "not found" in result["error"]


def test_batch_generate_requires_auth(monkeypatch, client):
    """Test batch generate requires authentication."""
    monkeypatch.setenv("HIPPO_API_KEY", "secret123")
    resp = client.post("/api/batch/generate", json={"requests": [{"model": "test", "prompt": "hi"}]})
    assert resp.status_code == 401


def test_batch_chat_requires_auth(monkeypatch, client):
    """Test batch chat requires authentication."""
    monkeypatch.setenv("HIPPO_API_KEY", "secret123")
    resp = client.post("/api/batch/chat", json={"requests": [{"model": "test", "messages": []}]})
    assert resp.status_code == 401


def test_batch_generate_with_auth(monkeypatch, client):
    """Test batch generate with correct authentication."""
    monkeypatch.setenv("HIPPO_API_KEY", "secret123")
    resp = client.post(
        "/api/batch/generate",
        json={"requests": [{"model": "nonexistent", "prompt": "hi"}]},
        headers={"Authorization": "Bearer secret123"},
    )
    # Auth passes, will fail on model not found
    assert resp.status_code == 200


def test_batch_chat_with_auth(monkeypatch, client):
    """Test batch chat with correct authentication."""
    monkeypatch.setenv("HIPPO_API_KEY", "secret123")
    resp = client.post(
        "/api/batch/chat",
        json={"requests": [{"model": "nonexistent", "messages": []}]},
        headers={"Authorization": "Bearer secret123"},
    )
    # Auth passes, will fail on model not found
    assert resp.status_code == 200


def test_batch_response_structure(client):
    """Test batch response has correct structure."""
    resp = client.post(
        "/api/batch/generate",
        json={
            "requests": [
                {"model": "test", "prompt": "hello"},
                {"model": "test", "prompt": "world"}
            ]
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    # Verify response structure
    assert "results" in data
    assert "total_duration_ns" in data
    assert "success_count" in data
    assert "error_count" in data
    # Verify results structure
    for result in data["results"]:
        assert "response" in result
        assert "status" in result
        assert result["status"] in ["success", "error"]
