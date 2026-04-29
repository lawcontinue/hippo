"""InferenceBackend abstraction + LLamaRPCBackend.

Provides a unified interface for local and distributed inference,
with automatic RPC worker management and tensor splitting.

Distributed mode: model is resident (常驻), not unloaded after each request.
"""

import logging
import os
import subprocess
import time
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger("hippo.cluster.backend")


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol for inference backends."""

    def load(self, model_path: str, **kwargs) -> None: ...
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, **kwargs) -> dict: ...
    def unload(self) -> None: ...


class LocalBackend:
    """Local inference using llama-cpp-python (no RPC)."""

    def __init__(self, n_gpu_layers: int = 99, n_ctx: int = 4096):
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._model = None

    def load(self, model_path: str, **kwargs):
        from llama_cpp import Llama
        logger.info(f"LocalBackend: loading {model_path}")
        self._model = Llama(
            model_path=model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=kwargs.get("n_ctx", self.n_ctx),
            verbose=kwargs.get("verbose", False),
        )
        logger.info("LocalBackend: model loaded")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, **kwargs) -> dict:
        if not self._model:
            raise RuntimeError("Model not loaded")
        return self._model(prompt, max_tokens=max_tokens, temperature=temperature)

    def get_llama(self):
        """Return the underlying Llama object for streaming access."""
        return self._model

    def unload(self):
        if self._model:
            del self._model
            self._model = None


class LLamaRPCBackend:
    """Distributed inference via llama.cpp RPC. Model is resident (常驻).

    Manages rpc-server subprocess(es) and distributes model layers
    across local GPU + remote RPC workers using tensor_split.

    Lifecycle:
        load() → [generate()/chat()]* → unload()
        Model stays in memory between requests (常驻模式).
        unload() only called on shutdown or explicit model switch.

    Usage:
        backend = LLamaRPCBackend(
            rpc_workers=["192.168.1.10:50052", "192.168.1.11:50052"],
        )
        backend.load(model_path)
        result = backend.generate("Hello", max_tokens=64)  # uses loaded model
        result2 = backend.generate("World", max_tokens=64)  # same model, no reload
        backend.unload()  # only on shutdown
    """

    def __init__(
        self,
        rpc_workers: list[str],
        tensor_split: Optional[list[float]] = None,
        n_gpu_layers: int = 99,
        n_ctx: int = 4096,
        rpc_binary: Optional[str] = None,
        start_local_rpc: bool = False,
        local_rpc_port: int = 50052,
    ):
        """
        Args:
            rpc_workers: List of "host:port" RPC endpoints (remote).
            tensor_split: How to split layers across devices.
                         E.g., [0.6, 0.4] = 60% local, 40% remote.
                         If None, auto-calculated based on worker count.
            n_gpu_layers: GPU layers to offload.
            n_ctx: Context window size.
            rpc_binary: Path to llama-rpc-server binary.
            start_local_rpc: Whether to start a local rpc-server subprocess.
            local_rpc_port: Port for local rpc-server (if start_local_rpc).
        """
        self.rpc_workers = rpc_workers
        self.tensor_split = tensor_split
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._rpc_binary = rpc_binary or self._find_rpc_binary()
        self._start_local_rpc = start_local_rpc
        self._local_rpc_port = local_rpc_port
        self._model = None
        self._model_path: Optional[str] = None
        self._local_rpc_process: Optional[subprocess.Popen] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_model_path(self) -> Optional[str]:
        return self._model_path

    @staticmethod
    def _find_rpc_binary() -> Optional[str]:
        """Find rpc-server binary."""
        candidates = [
            os.path.expanduser("~/.hippo/bin/rpc-server"),
            "/usr/local/bin/rpc-server",
            "/opt/homebrew/bin/rpc-server",
        ]
        # Check absolute paths first
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        # Check PATH
        try:
            result = subprocess.run(
                ["which", "rpc-server"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _start_rpc_server(self, host: str = "0.0.0.0", port: int = 50052) -> subprocess.Popen:
        """Start a local rpc-server subprocess."""
        if not self._rpc_binary:
            raise RuntimeError("rpc-server binary not found. Install to ~/.hippo/bin/rpc-server")

        logger.info(f"Starting local rpc-server on {host}:{port}")
        proc = subprocess.Popen(
            [self._rpc_binary, "-H", host, "-p", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait briefly to confirm startup
        time.sleep(2)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode()
            raise RuntimeError(f"rpc-server failed to start: {stderr}")

        logger.info(f"Local rpc-server running (PID {proc.pid})")
        return proc

    def _stop_rpc_server(self):
        """Stop the local rpc-server subprocess."""
        if self._local_rpc_process and self._local_rpc_process.poll() is None:
            self._local_rpc_process.terminate()
            try:
                self._local_rpc_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._local_rpc_process.kill()
            logger.info("Local rpc-server stopped")

    @staticmethod
    def auto_tensor_split(worker_count: int) -> list[float]:
        """Auto-calculate tensor split for equal distribution.

        Returns a list that sums to 1.0, one entry per device
        (local + each remote worker).
        """
        total = worker_count + 1  # +1 for local GPU
        return [1.0 / total] * total

    def load(self, model_path: str, **kwargs):
        """Load model with RPC distribution (常驻 — stays in memory).

        If start_local_rpc=True, starts a local rpc-server first,
        then connects to all workers (local + remote).

        If the same model is already loaded, this is a no-op.
        """
        # No-op if same model already loaded
        if self._model is not None and self._model_path == model_path:
            logger.info(f"LLamaRPCBackend: {os.path.basename(model_path)} already loaded, skip")
            return

        # Unload previous model if any
        if self._model is not None:
            self.unload()

        from llama_cpp import Llama

        all_workers = list(self.rpc_workers)

        # Optionally start local rpc-server
        if self._start_local_rpc:
            self._local_rpc_process = self._start_rpc_server(
                port=self._local_rpc_port
            )
            all_workers.insert(0, f"127.0.0.1:{self._local_rpc_port}")

        if not all_workers:
            raise ValueError("No RPC workers configured")

        # Auto tensor split if not specified
        ts = self.tensor_split
        if ts is None:
            ts = self.auto_tensor_split(len(all_workers))

        logger.info(
            f"LLamaRPCBackend: loading {os.path.basename(model_path)} "
            f"with {len(all_workers)} RPC workers, tensor_split={ts}"
        )

        self._model = Llama(
            model_path=model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=kwargs.get("n_ctx", self.n_ctx),
            rpc_servers=all_workers,
            tensor_split=ts,
            verbose=kwargs.get("verbose", False),
        )
        self._model_path = model_path
        logger.info(f"LLamaRPCBackend: model loaded (常驻) across {len(all_workers)} RPC workers")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, **kwargs) -> dict:
        """Generate text using distributed inference."""
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model(prompt, max_tokens=max_tokens, temperature=temperature)

    def generate_stream(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, **kwargs):
        """Stream text using distributed inference."""
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model.create_completion(
            prompt=prompt, max_tokens=max_tokens,
            temperature=temperature, stream=True, **kwargs
        )

    def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.7, **kwargs) -> dict:
        """Chat completion using distributed inference."""
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model.create_chat_completion(
            messages=messages, max_tokens=max_tokens,
            temperature=temperature, **kwargs
        )

    def chat_stream(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.7, **kwargs):
        """Stream chat using distributed inference."""
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model.create_chat_completion(
            messages=messages, max_tokens=max_tokens,
            temperature=temperature, stream=True, **kwargs
        )

    def get_llama(self):
        """Return the underlying Llama object."""
        return self._model

    def unload(self):
        """Unload model and stop local rpc-server.

        Only call on shutdown or explicit model switch.
        Distributed mode: model is 常驻, so this is rare.
        """
        if self._model:
            del self._model
            self._model = None
            self._model_path = None
        self._stop_rpc_server()
