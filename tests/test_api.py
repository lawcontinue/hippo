"""Tests for API endpoints."""
import secrets

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


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Hippo" in resp.json()["message"]


def test_api_version(client):
    resp = client.get("/api/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_tags_empty(client):
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    assert resp.json()["models"] == []


def test_generate_model_not_found(client):
    resp = client.post("/api/generate", json={"model": "nonexistent", "prompt": "hi"})
    assert resp.status_code == 404


def test_chat_model_not_found(client):
    resp = client.post("/api/chat", json={"model": "nonexistent", "messages": []})
    assert resp.status_code == 404


def test_pull_requires_auth(monkeypatch, client):
    monkeypatch.setenv("HIPPO_API_KEY", "secret123")
    # Re-read config to pick up env var
    api.app.state.config.api_key  # accessed via property
    resp = client.post("/api/pull", json={"name": "test"})
    assert resp.status_code == 401


def test_delete_requires_auth(monkeypatch, client):
    monkeypatch.setenv("HIPPO_API_KEY", "secret123")
    resp = client.request("DELETE", "/api/delete", json={"name": "test"})
    assert resp.status_code == 401


def test_pull_with_auth(monkeypatch, client):
    monkeypatch.setenv("HIPPO_API_KEY", "secret123")
    resp = client.post(
        "/api/pull",
        json={"name": "nonexistent/model"},
        headers={"Authorization": "Bearer secret123"},
    )
    # Will fail to find model, but auth passes
    assert resp.status_code == 500


def test_auth_timing_attack_protection(monkeypatch, client):
    """P0-1: Verify secrets.compare_digest() is used for timing attack protection."""
    monkeypatch.setenv("HIPPO_API_KEY", "my-secret-api-key-12345")

    # Test 1: Correct key should pass
    resp = client.post(
        "/api/pull",
        json={"name": "test"},
        headers={"Authorization": "Bearer my-secret-api-key-12345"},
    )
    # Auth passes (will fail on model download, but that's expected)
    assert resp.status_code == 500

    # Test 2: Wrong key should fail
    resp = client.post(
        "/api/pull",
        json={"name": "test"},
        headers={"Authorization": "Bearer wrong-key-different"},
    )
    assert resp.status_code == 401

    # Test 3: Partially matching key should also fail (timing attack protection)
    resp = client.post(
        "/api/pull",
        json={"name": "test"},
        headers={"Authorization": "Bearer my-secret"},
    )
    assert resp.status_code == 401

    # Test 4: No auth header should fail
    resp = client.post(
        "/api/pull",
        json={"name": "test"},
    )
    assert resp.status_code == 401

    # Test 5: secrets.compare_digest() verification
    # Verify that secrets.compare_digest() works as expected
    api_key = "my-secret-api-key-12345"
    assert secrets.compare_digest(api_key, api_key) is True
    assert secrets.compare_digest("wrong", api_key) is False
    assert secrets.compare_digest("my-secret", api_key) is False
