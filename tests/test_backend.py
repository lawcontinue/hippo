"""Tests for LLamaRPCBackend and distributed inference."""

import pytest

# We test the logic without actually loading models


class TestAutoTensorSplit:
    """Test automatic tensor split calculation."""

    def test_single_worker(self):
        from hippo.cluster.backend import LLamaRPCBackend
        split = LLamaRPCBackend.auto_tensor_split(1)
        assert len(split) == 2  # local + 1 worker
        assert abs(sum(split) - 1.0) < 0.01
        assert split[0] == split[1]  # equal split

    def test_two_workers(self):
        from hippo.cluster.backend import LLamaRPCBackend
        split = LLamaRPCBackend.auto_tensor_split(2)
        assert len(split) == 3  # local + 2 workers
        assert abs(sum(split) - 1.0) < 0.01
        assert split[0] == pytest.approx(1/3, abs=0.01)

    def test_five_workers(self):
        from hippo.cluster.backend import LLamaRPCBackend
        split = LLamaRPCBackend.auto_tensor_split(5)
        assert len(split) == 6  # local + 5 workers
        assert abs(sum(split) - 1.0) < 0.01

    def test_zero_workers(self):
        from hippo.cluster.backend import LLamaRPCBackend
        split = LLamaRPCBackend.auto_tensor_split(0)
        assert len(split) == 1  # local only
        assert split[0] == 1.0


class TestLocalBackend:
    """Test LocalBackend protocol compliance."""

    def test_implements_protocol(self):
        from hippo.cluster.backend import InferenceBackend, LocalBackend
        backend = LocalBackend()
        assert isinstance(backend, InferenceBackend)


class TestLLamaRPCBackendInit:
    """Test LLamaRPCBackend initialization."""

    def test_init_defaults(self):
        from hippo.cluster.backend import LLamaRPCBackend
        backend = LLamaRPCBackend(rpc_workers=["192.168.1.10:50052"])
        assert backend.rpc_workers == ["192.168.1.10:50052"]
        assert backend.tensor_split is None  # auto
        assert backend.n_gpu_layers == 99
        assert backend.n_ctx == 4096

    def test_init_custom_split(self):
        from hippo.cluster.backend import LLamaRPCBackend
        backend = LLamaRPCBackend(
            rpc_workers=["10.0.0.1:50052", "10.0.0.2:50052"],
            tensor_split=[0.4, 0.3, 0.3],
        )
        assert backend.tensor_split == [0.4, 0.3, 0.3]

    def test_not_loaded_error(self):
        from hippo.cluster.backend import LLamaRPCBackend
        backend = LLamaRPCBackend(rpc_workers=["localhost:50052"])
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.generate("test")


class TestSchedulerTensorSplit:
    """Test Scheduler.compute_tensor_split."""

    def test_no_workers(self):
        from hippo.cluster.scheduler import Scheduler
        s = Scheduler()
        assert s.compute_tensor_split(5.0) is None

    def test_single_worker_sufficient(self):
        from hippo.cluster.scheduler import Scheduler
        s = Scheduler()
        s.register_worker("w1", "192.168.1.10", 50052, gpu_memory_gb=16.0, models=[])
        result = s.compute_tensor_split(5.0)
        assert result is not None
        assert len(result["rpc_workers"]) == 1
        assert result["rpc_workers"][0] == "192.168.1.10:50052"
        assert abs(sum(result["tensor_split"]) - 1.0) < 0.01

    def test_insufficient_memory(self):
        """Should return None if total cluster memory < model_size."""
        from hippo.cluster.scheduler import Scheduler
        s = Scheduler()
        s.register_worker("w1", "192.168.1.10", 50052, gpu_memory_gb=2.0, models=[])
        result = s.compute_tensor_split(50.0)  # 50GB needed, ~10GB available
        assert result is None

    def test_two_workers(self):
        from hippo.cluster.scheduler import Scheduler
        s = Scheduler()
        s.register_worker("w1", "192.168.1.10", 50052, gpu_memory_gb=16.0, models=[])
        s.register_worker("w2", "192.168.1.11", 50052, gpu_memory_gb=8.0, models=[])
        result = s.compute_tensor_split(5.0)
        assert result is not None
        assert len(result["rpc_workers"]) == 2
        assert len(result["tensor_split"]) == 3  # local + 2 workers


class TestGetWorkersInfo:
    """Test Scheduler.get_workers_info."""

    def test_empty(self):
        from hippo.cluster.scheduler import Scheduler
        s = Scheduler()
        assert s.get_workers_info() == []

    def test_with_workers(self):
        from hippo.cluster.scheduler import Scheduler
        s = Scheduler()
        s.register_worker("w1", "10.0.0.1", 50052, gpu_memory_gb=16.0, models=["llama"])
        info = s.get_workers_info()
        assert len(info) == 1
        assert info[0]["worker_id"] == "w1"
        assert info[0]["gpu_free_gb"] == 16.0
