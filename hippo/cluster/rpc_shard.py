"""Hippo Cluster — llama.cpp RPC 分片推理 PoC

利用 llama-cpp-python 的 RPC 支持，将模型的 tensor 计算
offload 到远程 Worker。

原理：
  主节点加载模型，通过 rpc:// backend 将部分层的计算
  发送到远程 Worker 执行。Worker 不需要加载完整模型。

Worker 端：运行 llama-rpc-server（监听端口）
主节点：Llama(..., rpc_servers=["worker_ip:port"], tensor_split=[...])
"""

import os
import logging
import subprocess
import time
import signal
import sys
from typing import Optional

logger = logging.getLogger("hippo.cluster.rpc")


def find_rpc_server_binary() -> Optional[str]:
    """Find llama-rpc-server binary."""
    # Check common locations
    candidates = [
        "llama-rpc-server",
        os.path.expanduser("~/.local/bin/llama-rpc-server"),
        "/usr/local/bin/llama-rpc-server",
        "/opt/homebrew/bin/llama-rpc-server",
    ]
    for c in candidates:
        try:
            result = subprocess.run(
                ["which", c], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            continue
    return None


def build_rpc_server_from_source(install_dir: str = "~/.hippo/bin"):
    """Build llama-rpc-server from llama.cpp source.

    This is needed because the Python wheel doesn't include the binary.
    """
    install_dir = os.path.expanduser(install_dir)
    os.makedirs(install_dir, exist_ok=True)

    build_dir = "/tmp/llama.cpp-build"
    if not os.path.exists(build_dir):
        logger.info("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", build_dir],
            check=True,
        )

    logger.info("Building llama-rpc-server...")
    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DGGML_METAL=ON -DGGML_RPC=ON"

    subprocess.run(
        ["cmake", "-B", "build", "-DGGML_METAL=ON", "-DGGML_RPC=ON"],
        cwd=build_dir,
        env=env,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", "build", "--config", "Release", "-j", str(os.cpu_count())],
        cwd=build_dir,
        check=True,
    )

    # Copy binary
    src = os.path.join(build_dir, "build", "bin", "llama-rpc-server")
    dst = os.path.join(install_dir, "llama-rpc-server")
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
        logger.info(f"Installed llama-rpc-server to {dst}")
        return dst
    else:
        logger.error(f"Binary not found at {src}")
        return None


class RPCWorker:
    """Run llama-rpc-server on a Worker node.

    The RPC server listens for tensor computation requests from
    the main node and executes them locally using Metal/CPU.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 50052):
        self.host = host
        self.port = port
        self._process: Optional[subprocess.Popen] = None
        self._binary: Optional[str] = None

    def start(self):
        """Start the RPC server."""
        self._binary = find_rpc_server_binary()
        if not self._binary:
            logger.info("llama-rpc-server not found, building from source...")
            self._binary = build_rpc_server_from_source()
            if not self._binary:
                raise RuntimeError("Cannot find or build llama-rpc-server")

        logger.info(f"Starting RPC server: {self._binary} -H {self.host} -p {self.port}")
        self._process = subprocess.Popen(
            [self._binary, "-H", self.host, "-p", str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Give it a moment to start
        time.sleep(2)
        if self._process.poll() is not None:
            stderr = self._process.stderr.read().decode()
            raise RuntimeError(f"RPC server failed to start: {stderr}")
        logger.info(f"RPC server running on {self.host}:{self.port}")

    def stop(self):
        """Stop the RPC server."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=5)
            logger.info("RPC server stopped")

    @property
    def url(self) -> str:
        return f"{self.host}:{self.port}"


class RPCShardedInference:
    """Run sharded inference across multiple RPC workers.

    Uses llama-cpp-python's built-in RPC support:
      Llama(model_path, rpc_servers=[...], tensor_split=[...])

    tensor_split controls how much of the model each device handles.
    E.g., [0.6, 0.4] means 60% on local, 40% on remote.
    """

    def __init__(
        self,
        model_path: str,
        rpc_workers: list[str],  # ["host:port", ...]
        tensor_split: Optional[list[float]] = None,
        n_ctx: int = 4096,
    ):
        self.model_path = model_path
        self.rpc_workers = rpc_workers
        self.tensor_split = tensor_split
        self.n_ctx = n_ctx
        self._model = None

    def load(self):
        """Load the model with RPC distribution."""
        from llama_cpp import Llama

        kwargs = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "rpc_servers": self.rpc_workers,
            "verbose": True,
        }
        if self.tensor_split:
            kwargs["tensor_split"] = self.tensor_split

        logger.info(f"Loading model with RPC workers: {self.rpc_workers}")
        self._model = Llama(**kwargs)
        logger.info("Model loaded with RPC distribution")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate text using distributed inference."""
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")

        result = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result["choices"][0]["text"]

    def unload(self):
        """Unload the model."""
        if self._model:
            del self._model
            self._model = None


# --- CLI ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hippo RPC Worker")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=50052, help="Listen port")
    parser.add_argument("--build", action="store_true", help="Build rpc-server from source")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.build:
        path = build_rpc_server_from_source()
        print(f"Built: {path}")
    else:
        worker = RPCWorker(host=args.host, port=args.port)
        try:
            worker.start()
            print(f"RPC Worker running on {worker.url}. Press Ctrl+C to stop.")
            worker._process.wait()
        except KeyboardInterrupt:
            worker.stop()
