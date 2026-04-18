"""Tests for Hippo cluster module — Phase 0 validation."""

import pytest
from hippo.cluster.discovery import DiscoveryService, NodeInfo
from hippo.cluster.scheduler import Scheduler
from hippo.cluster.worker import WorkerService, WorkerConfig


class TestScheduler:
    """Unit tests for the Scheduler."""

    def test_register_worker(self):
        s = Scheduler()
        s.register_worker(
            worker_id="w1",
            host="192.168.1.10",
            port=11435,
            gpu_memory_gb=13.0,
            models=["deepseek-r1-8b"],
            mlx=True,
        )
        assert "w1" in s._workers
        assert s._workers["w1"].gpu_memory_gb == 13.0
        assert s._workers["w1"].mlx is True

    def test_deregister_worker(self):
        s = Scheduler()
        s.register_worker("w1", "192.168.1.10", 11435, 13.0, ["model-a"])
        s.deregister_worker("w1")
        assert "w1" not in s._workers

    def test_deregister_orphaned_models(self):
        s = Scheduler()
        s.register_worker("w1", "192.168.1.10", 11435, 13.0, ["model-a"])
        assert s.get_assignment("model-a") is not None
        s.deregister_worker("w1")
        assert s.get_assignment("model-a") is None

    def test_find_worker_for_model(self):
        s = Scheduler()
        s.register_worker("w1", "192.168.1.10", 11435, 13.0, ["model-a"])
        s.register_worker("w2", "192.168.1.11", 11435, 8.0, ["model-b"])

        w = s.find_worker_for_model("model-a", 5.0)
        assert w is not None
        assert w.worker_id == "w1"

    def test_find_worker_insufficient_memory(self):
        s = Scheduler()
        s.register_worker("w1", "192.168.1.10", 11435, 4.0, [])

        w = s.find_worker_for_model("big-model", 10.0)
        assert w is None

    def test_heartbeat(self):
        s = Scheduler()
        s.register_worker("w1", "192.168.1.10", 11435, 13.0, [])
        s.update_heartbeat("w1", "healthy", ["model-a"], 5.0)

        w = s._workers["w1"]
        assert w.status == "healthy"
        assert w.loaded_models == ["model-a"]
        assert w.gpu_memory_used_gb == 5.0

    def test_cluster_status(self):
        s = Scheduler()
        s.register_worker("w1", "192.168.1.10", 11435, 13.0, ["a"])
        s.register_worker("w2", "192.168.1.11", 11435, 8.0, ["b"])

        status = s.get_cluster_status()
        assert status["workers"] == 2
        assert status["healthy_workers"] == 2
        assert status["total_gpu_memory_gb"] == 21.0

    def test_get_worker(self):
        s = Scheduler()
        s.register_worker("w1", "192.168.1.10", 11435, 13.0, ["a"])
        w = s.get_worker("w1")
        assert w is not None
        assert w.host == "192.168.1.10"
        assert s.get_worker("nonexistent") is None

    def test_unhealthy_worker_excluded(self):
        s = Scheduler()
        s.register_worker("w1", "192.168.1.10", 11435, 13.0, [])
        s._workers["w1"].status = "unhealthy"
        w = s.find_worker_for_model("model", 5.0)
        assert w is None


class TestDiscoveryService:
    """Test mDNS discovery (local only, no network required)."""

    def test_get_local_ip(self):
        ip = DiscoveryService._get_local_ip()
        assert ip != "0.0.0.0"
        assert "." in ip

    def test_parse_service_info_with_none(self):
        result = DiscoveryService._parse_service_info("test", None)
        assert result is None

    def test_shutdown_without_start(self):
        """Shutdown should not raise if nothing was started."""
        disc = DiscoveryService(role="worker")
        disc.shutdown()  # should not raise


class TestWorkerConfig:
    """Test WorkerConfig defaults."""

    def test_defaults(self):
        cfg = WorkerConfig()
        assert cfg.worker_port == 11435
        assert cfg.heartbeat_interval == 15
        assert cfg.mlx_preferred is True

    def test_custom_config(self):
        cfg = WorkerConfig(
            gateway_host="192.168.1.10",
            gpu_memory_gb=8.0,
        )
        assert cfg.gateway_host == "192.168.1.10"
        assert cfg.gpu_memory_gb == 8.0


class TestWorkerService:
    """Test WorkerService (no network)."""

    def test_update_loaded_models(self):
        ws = WorkerService(WorkerConfig())
        assert ws._loaded_models == []
        assert ws._gpu_memory_used_gb == 0.0

        ws.update_loaded_models(["model-a", "model-b"], 5.2)
        assert ws._loaded_models == ["model-a", "model-b"]
        assert ws._gpu_memory_used_gb == 5.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
