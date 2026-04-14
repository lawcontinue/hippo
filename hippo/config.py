"""Configuration loader for Hippo."""

import os
from pathlib import Path
from dataclasses import dataclass, field

import yaml


DEFAULT_CONFIG_PATH = Path("~/.hippo/config.yaml").expanduser()
DEFAULT_MODELS_DIR = Path("~/.hippo/models").expanduser()


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 11434
    ssl_enabled: bool = False
    ssl_cert_path: str | None = None
    ssl_key_path: str | None = None


@dataclass
class DefaultsConfig:
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    temperature: float = 0.7
    max_tokens: int = 2048
    repeat_penalty: float = 1.1


@dataclass
class BatchConfig:
    """Batch inference configuration."""
    max_size: int = 32  # maximum batch size
    max_concurrent: int = 4  # maximum concurrent requests in a batch
    timeout_seconds: int = 120  # per-request timeout


@dataclass
class HippoConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    models_dir: Path = field(default_factory=lambda: DEFAULT_MODELS_DIR)
    idle_timeout: int = 300  # seconds before auto-unload
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    max_concurrent_requests: int = 0  # 0 = unlimited
    max_memory_gb: float = 0  # 0 = no limit
    metrics_enabled: bool = True  # enable/disable /metrics endpoint
    metrics_path: str = "/metrics"  # metrics endpoint path
    batch: BatchConfig = field(default_factory=BatchConfig)  # batch inference config

    @property
    def api_key(self) -> str | None:
        return os.environ.get("HIPPO_API_KEY")

    @property
    def batch_max_size(self) -> int:
        """Batch max size from config or environment variable."""
        return int(os.environ.get("HIPPO_BATCH_MAX_SIZE", self.batch.max_size))


def load_config(path: Path | None = None) -> HippoConfig:
    """Load config from YAML file, falling back to defaults."""
    cfg = HippoConfig()
    path = path or DEFAULT_CONFIG_PATH

    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        if "server" in data:
            for k, v in data["server"].items():
                if hasattr(cfg.server, k):
                    setattr(cfg.server, k, v)
                    # 特殊处理布尔值
                    if k == "ssl_enabled" and isinstance(v, str):
                        cfg.server.ssl_enabled = v.lower() in ("true", "1", "yes")

        if "models" in data and "dir" in data["models"]:
            cfg.models_dir = Path(data["models"]["dir"]).expanduser()

        if "idle_timeout" in data:
            cfg.idle_timeout = int(data["idle_timeout"])

        if "max_concurrent_requests" in data:
            cfg.max_concurrent_requests = int(data["max_concurrent_requests"])

        if "max_memory_gb" in data:
            cfg.max_memory_gb = float(data["max_memory_gb"])

        if "defaults" in data:
            for k, v in data["defaults"].items():
                if hasattr(cfg.defaults, k):
                    setattr(cfg.defaults, k, v)

        if "batch" in data:
            for k, v in data["batch"].items():
                if hasattr(cfg.batch, k):
                    setattr(cfg.batch, k, v)

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    return cfg
