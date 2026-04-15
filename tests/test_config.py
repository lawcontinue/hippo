"""Tests for config module."""
import os
import tempfile
from pathlib import Path
from hippo.config import HippoConfig, load_config


def test_default_config():
    cfg = HippoConfig()
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 11434
    assert cfg.idle_timeout == 300
    assert cfg.defaults.repeat_penalty == 1.1
    assert cfg.defaults.n_ctx == 4096


def test_load_config_from_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "test_config.yaml"
        cfg_path.write_text("server:\n  host: 0.0.0.0\n  port: 8080\nidle_timeout: 60\ndefaults:\n  temperature: 0.5\n  repeat_penalty: 1.2\n")
        cfg = load_config(cfg_path)
        assert cfg.server.host == "0.0.0.0"
        assert cfg.server.port == 8080
        assert cfg.idle_timeout == 60
        assert cfg.defaults.temperature == 0.5
        assert cfg.defaults.repeat_penalty == 1.2


def test_api_key_from_env():
    cfg = HippoConfig()
    assert cfg.api_key is None
    os.environ["HIPPO_API_KEY"] = "test123"
    try:
        assert cfg.api_key == "test123"
    finally:
        del os.environ["HIPPO_API_KEY"]


def test_models_dir_created():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.yaml"
        cfg_path.write_text(f"models:\n  dir: {tmp}/models_test\n")
        load_config(cfg_path)
        assert (Path(tmp) / "models_test").exists()


def test_max_concurrent_requests_default():
    cfg = HippoConfig()
    assert cfg.max_concurrent_requests == 0
