"""
test_pipeline.py — Pipeline 模块单元测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from shard import ShardMetadata, split_model, memory_weighted_split
from cluster import Cluster, DeviceInfo, get_local_memory_gb


def test_uniform_split():
    """均匀分片测试"""
    shards = split_model("test-model", n_layers=64, world_size=2)
    assert len(shards) == 2
    assert shards[0].start_layer == 0
    assert shards[0].end_layer == 32
    assert shards[1].start_layer == 32
    assert shards[1].end_layer == 64
    assert shards[0].is_first
    assert not shards[0].is_last
    assert not shards[1].is_first
    assert shards[1].is_last
    print("✅ test_uniform_split passed")


def test_memory_weighted_split():
    """内存权重分片测试"""
    shards = memory_weighted_split(
        "test-model",
        n_layers=64,
        device_memories=[10.0, 12.0],  # 第二台内存多 20%
    )
    assert len(shards) == 2
    # 第二台应该分到更多层
    assert shards[1].n_local_layers >= shards[0].n_local_layers
    # 总层数 = 64
    assert shards[0].n_local_layers + shards[1].n_local_layers == 64
    print(f"  Device 0: {shards[0].n_local_layers} layers")
    print(f"  Device 1: {shards[1].n_local_layers} layers")
    print("✅ test_memory_weighted_split passed")


def test_shard_metadata():
    """ShardMetadata 属性测试"""
    shard = ShardMetadata(
        model_id="Qwen/Qwen2.5-14B",
        start_layer=10,
        end_layer=30,
        n_layers=40,
        device_rank=1,
        world_size=2,
    )
    assert shard.n_local_layers == 20
    assert not shard.is_first
    assert shard.is_last
    print(f"  {shard}")
    print("✅ test_shard_metadata passed")


def test_cluster_plan():
    """集群分片规划测试"""
    cluster = Cluster()
    cluster.add_device(DeviceInfo(
        name="mac-1", host="192.168.1.10",
        available_memory_gb=8.0,
    ))
    cluster.add_device(DeviceInfo(
        name="mac-2", host="192.168.1.11",
        available_memory_gb=10.0,
    ))

    shards = cluster.plan_shards("Qwen/Qwen2.5-14B", n_layers=40)
    assert len(shards) == 2
    assert shards[0].n_local_layers + shards[1].n_local_layers == 40
    # 第二台内存更多，应该分到更多层
    assert shards[1].n_local_layers >= shards[0].n_local_layers
    print(f"  {shards[0]}")
    print(f"  {shards[1]}")
    print("✅ test_cluster_plan passed")


def test_get_local_memory():
    """本地内存查询测试"""
    mem = get_local_memory_gb()
    assert mem > 0
    print(f"  Local available memory: {mem:.1f} GB")
    print("✅ test_get_local_memory passed")


if __name__ == "__main__":
    test_uniform_split()
    test_memory_weighted_split()
    test_shard_metadata()
    test_cluster_plan()
    test_get_local_memory()
    print("\n🎉 All tests passed!")
