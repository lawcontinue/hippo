"""Hippo MLX Distributed — 两台 Mac Mini 分布式推理 PoC

测试 MLX ring backend 的分布式通信。

用法:
  Mac Mini #1: MLX_RANK=0 MLX_HOSTFILE=/tmp/hostfile.txt python3 mlx_distributed_test.py
  Mac Mini #2: MLX_RANK=1 MLX_HOSTFILE=/tmp/hostfile.txt python3 mlx_distributed_test.py

hostfile.txt 内容:
  192.168.1.10
  192.168.1.11
"""

import os

import mlx.core as mx


def test_distributed():
    rank = os.environ.get("MLX_RANK", "0")
    print(f"[Rank {rank}] Initializing MLX distributed...")

    try:
        group = mx.distributed.init(backend="ring", strict=True)
        print(f"[Rank {rank}] ✅ Connected! rank={group.rank()}, size={group.size()}")
    except Exception as e:
        print(f"[Rank {rank}] ❌ Failed: {e}")
        print(f"[Rank {rank}] Falling back to singleton group...")
        group = mx.distributed.init(backend="any", strict=False)
        print(f"[Rank {rank}] Singleton: rank={group.rank()}, size={group.size()}")

    # Test all_reduce (sum)
    print(f"[Rank {rank}] Testing all_sum...")
    t = mx.ones((3,))
    result = mx.distributed.all_sum(t, group=group)
    print(f"[Rank {rank}] all_sum result: {result}")

    # Test send/recv
    if group.size() == 2:
        if group.rank() == 0:
            data = mx.array([42.0, 43.0, 44.0])
            print(f"[Rank 0] Sending: {data}")
            mx.distributed.send(data, dst=1, group=group)
            print("[Rank 0] ✅ Sent!")
        else:
            buf = mx.zeros((3,))
            print("[Rank 1] Waiting to receive...")
            mx.distributed.recv(buf, src=0, group=group)
            print(f"[Rank 1] ✅ Received: {buf}")

    print(f"[Rank {rank}] Done!")


if __name__ == "__main__":
    test_distributed()
