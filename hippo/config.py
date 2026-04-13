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


@dataclass
class DefaultsConfig:
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    temperature: float = 0.7
    max_tokens: int = 2048
    repeat_penalty: float = 1.1


@dataclass
class HippoConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    models_dir: Path = field(default_factory=lambda: DEFAULT_MODELS_DIR)
    idle_timeout: int = 300  # seconds before auto-unload
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    max_concurrent_requests: int = 0  # 0 = unlimited
    max_memory_gb: float = 0  # 0 = no limit

    @property
    def api_key(self) -> str | None:
        return os.environ.get("HIPPO_API_KEY")


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

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    return cfg
