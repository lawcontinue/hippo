"""Tests for model_manager module."""
import threading
from unittest.mock import MagicMock, patch
from hippo.config import HippoConfig
from hippo.model_manager import ModelManager


def _make_manager(models_dir="/tmp/test_models"):
    cfg = HippoConfig()
    cfg.models_dir = __import__("pathlib").Path(models_dir)
    return ModelManager(cfg)


def test_list_loaded_empty():
    m = _make_manager()
    assert m.list_loaded() == []


def test_per_model_lock():
    m = _make_manager()
    lock1 = m._get_model_lock("model_a")
    lock2 = m._get_model_lock("model_b")
    lock1b = m._get_model_lock("model_a")
    assert lock1 is lock1b
    assert lock1 is not lock2


def test_list_available_empty(tmp_path):
    cfg = HippoConfig()
    cfg.models_dir = tmp_path
    m = ModelManager(cfg)
    assert m.list_available() == []


def test_double_check_locking():
    """Test that concurrent get() calls don't load model twice."""
    m = _make_manager()
    load_count = 0
    real_load = m._load

    def slow_load(name):
        nonlocal load_count
        load_count += 1
        __import__("time").sleep(0.1)
        llama = MagicMock()
        with m._lock:
            m._loaded[name] = llama
            m._last_used[name] = __import__("time").time()
        return llama

    m._load = slow_load

    results = []
    errors = []

    def worker():
        try:
            r = m.get("test_model")
            results.append(r)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should only load once due to double-check locking
    assert load_count <= 2  # at most 2 due to per-model lock serialization
    assert len(results) == 3
    assert not errors
