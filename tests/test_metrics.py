"""Tests for Prometheus metrics."""

import pytest


def test_metrics_module_import():
    """Test that metrics module can be imported."""
    from hippo import metrics
    assert hasattr(metrics, 'inference_requests_total')
    assert hasattr(metrics, 'inference_duration_seconds')
    assert hasattr(metrics, 'models_loaded')
    assert hasattr(metrics, 'memory_usage_bytes')


def test_metrics_router_import():
    """Test that metrics router can be imported."""
    from hippo.routers import metrics as metrics_router
    assert hasattr(metrics_router, 'router')


def test_metrics_config():
    """Test that config has metrics settings."""
    from hippo.config import HippoConfig, load_config

    config = load_config()
    assert hasattr(config, 'metrics_enabled')
    assert hasattr(config, 'metrics_path')
    assert config.metrics_enabled is True
    assert config.metrics_path == "/metrics"


def test_metrics_endpoint_registration():
    """Test that metrics router is registered."""
    from hippo.api import create_app
    from hippo.config import HippoConfig
    from hippo.model_manager import ModelManager

    config = HippoConfig()
    manager = ModelManager(config)
    app = create_app(config, manager)

    # Check that /metrics route is registered
    routes = [route.path for route in app.routes]
    assert "/metrics" in routes
