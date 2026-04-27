"""
cluster.py — 集群管理和设备发现

管理双 Mac Mini 集群：
- 设备发现（通过配置或 mDNS）
- 内存查询
- 分片分配
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from .shard import ShardMetadata, split_model, memory_weighted_split
except ImportError:
    from shard import ShardMetadata, split_model, memory_weighted_split


@dataclass
class DeviceInfo:
    """集群中的一个设备"""
    name: str
    host: str                    # IP 地址
    port: int = 50052
    total_memory_gb: float = 16.0
    available_memory_gb: float = 0.0
    python_version: str = ""
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "total_memory_gb": self.total_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "python_version": self.python_version,
            "rank": self.rank,
        }


class Cluster:
    """管理 Hippo Pipeline 集群"""

    def __init__(self, config_path: Optional[str | Path] = None):
        self.devices: list[DeviceInfo] = []
        self.config_path = Path(config_path) if config_path else None
        if self.config_path and self.config_path.exists():
            self._load_config()

    def _load_config(self):
        """从配置文件加载集群信息"""
        with open(self.config_path) as f:
            data = json.load(f)
        self.devices = [
            DeviceInfo(**d) for d in data.get("devices", [])
        ]

    def save_config(self):
        """保存集群配置"""
        if self.config_path is None:
            return
        data = {"devices": [d.to_dict() for d in self.devices]}
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_device(self, device: DeviceInfo):
        device.rank = len(self.devices)
        self.devices.append(device)

    @property
    def world_size(self) -> int:
        return len(self.devices)

    def plan_shards(self, model_id: str, n_layers: int,
                    memory_weighted: bool = True) -> list[ShardMetadata]:
        """
        为集群规划模型分片。
        
        Args:
            model_id: 模型 ID
            n_layers: 模型总层数
            memory_weighted: 是否按内存权重分配
        
        Returns:
            每台设备的 ShardMetadata
        """
        if memory_weighted and all(d.available_memory_gb > 0 for d in self.devices):
            return memory_weighted_split(
                model_id, n_layers,
                [d.available_memory_gb for d in self.devices]
            )
        return split_model(model_id, n_layers, self.world_size)


def get_local_memory_gb() -> float:
    """获取本机可用内存（GB）"""
    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        )
        page_size = 16384
        free = inactive = speculative = 0
        for line in result.stdout.split("\n"):
            if "Pages free:" in line and "swap" not in line:
                free = int(line.split()[-1].rstrip("."))
            elif "Pages inactive:" in line:
                inactive = int(line.split()[-1].rstrip("."))
            elif "Pages speculative:" in line:
                speculative = int(line.split()[-1].rstrip("."))
        return (free + inactive + speculative) * page_size / (1024 ** 3)
    except Exception:
        return 0.0


# 预定义我们的双 Mac Mini 集群
DEFAULT_CLUSTER = Cluster()
DEFAULT_CLUSTER.add_device(DeviceInfo(
    name="mac-mini-1",
    host="169.254.248.46",   # Thunderbolt
    port=50052,
    total_memory_gb=16.0,
))
DEFAULT_CLUSTER.add_device(DeviceInfo(
    name="mac-mini-2",
    host="169.254.100.105",  # Thunderbolt
    port=50052,
    total_memory_gb=16.0,
))
