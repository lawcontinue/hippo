"""Tests for API endpoints."""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from hippo.config import HippoConfig
from hippo.model_manager import ModelManager
from hippo import api


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
