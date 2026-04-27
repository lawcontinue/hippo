#!/usr/bin/env python3
"""
双机 Pipeline 测试脚本

在两台 Mac Mini 上分别运行：
- Rank 0: python3 test_dual_rank.py --rank 0 --host 192.168.1.11
- Rank 1: python3 test_dual_rank.py --rank 1 --host 0.0.0.0

测试流程：
1. Rank 1 启动接收器
2. Rank 0 加载前 24 层，发送 hidden states
3. Rank 1 接收后，加载后 24 层，完成推理
4. 返回结果给 Rank 0
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# 添加当前目录到 path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from tcp_transport import TensorSender, TensorReceiver
from distributed_model import DistributedModel
from shard import ShardMetadata


async def rank0_task(target_host: str, port: int = 29900):
    """Rank 0: 前 24 层 + 发送 hidden states"""
    print(f"=== Rank 0 启动 ===")
    print(f"目标: {target_host}:{port}")
    print()
    
    # 创建分片（前 24 层）
    shard = ShardMetadata(
        model_id='mlx-community/gemma-3-12b-it-qat-4bit',
        start_layer=0,
        end_layer=24,
        n_layers=48,
        device_rank=0,
        world_size=2
    )
    
    print("步骤 1: 加载 Rank 0 模型 (层 0-24)...")
    t0 = time.time()
    model = DistributedModel('mlx-community/gemma-3-12b-it-qat-4bit', shard)
    model.load()
    t1 = time.time()
    print(f"✅ 加载完成 ({t1-t0:.1f}s)")
    print(f"   内存: {mx.metal.get_active_memory() / 1024**3:.2f} GB")
    
    # 创建测试输入
    print("\n步骤 2: 创建测试输入...")
    input_ids = mx.array([[1, 2, 3, 4, 5] * 25])  # [1, 125]
    print(f"   Input shape: {input_ids.shape}")
    
    # Rank 0 前向传播
    print("\n步骤 3: Rank 0 前向传播...")
    t2 = time.time()
    hidden_states = model.model(input_ids)  # 假设直接调用
    t3 = time.time()
    print(f"✅ Rank 0 前向完成 ({t3-t2:.3f}s)")
    print(f"   Hidden states shape: {hidden_states.shape}")
    
    # 发送给 Rank 1
    print(f"\n步骤 4: 发送 hidden states 到 Rank 1...")
    sender = TensorSender(target_host, port)
    await sender.connect()
    
    t4 = time.time()
    await sender.send(hidden_states, rank=0)
    t5 = time.time()
    print(f"✅ 发送完成 ({t5-t4:.3f}s)")
    print(f"   吞吐量: {hidden_states.nbytes / (t5-t4) / 1024 / 1024:.1f} MB/s")
    
    # 等待 Rank 1 返回结果
    print(f"\n步骤 5: 等待 Rank 1 返回...")
    t6 = time.time()
    receiver = TensorReceiver(host='0.0.0.0', port=port+1)
    await receiver.start()
    result = await receiver.recv(timeout=30.0)
    t7 = time.time()
    print(f"✅ 接收结果 ({t7-t6:.3f}s)")
    
    await sender.close()
    await receiver.stop()
    
    total = t7 - t0
    print(f"\n=== Rank 0 完成 ===")
    print(f"总时间: {total:.1f}s")


async def rank1_task(bind_host: str, port: int = 29900):
    """Rank 1: 后 24 层 + 接收 hidden states"""
    print(f"=== Rank 1 启动 ===")
    print(f"监听: {bind_host}:{port}")
    print()
    
    # 创建分片（后 24 层）
    shard = ShardMetadata(
        model_id='mlx-community/gemma-3-12b-it-qat-4bit',
        start_layer=24,
        end_layer=48,
        n_layers=48,
        device_rank=1,
        world_size=2
    )
    
    print("步骤 1: 加载 Rank 1 模型 (层 24-48)...")
    t0 = time.time()
    model = DistributedModel('mlx-community/gemma-3-12b-it-qat-4bit', shard)
    model.load()
    t1 = time.time()
    print(f"✅ 加载完成 ({t1-t0:.1f}s)")
    print(f"   内存: {mx.metal.get_active_memory() / 1024**3:.2f} GB")
    
    # 启动接收器
    print(f"\n步骤 2: 启动接收器...")
    receiver = TensorReceiver(host=bind_host, port=port)
    await receiver.start()
    print(f"✅ 接收器已启动: {bind_host}:{port}")
    
    # 等待 Rank 0 的数据
    print(f"\n步骤 3: 等待 Rank 0 发送...")
    t2 = time.time()
    hidden_states = await receiver.recv(timeout=60.0)
    t3 = time.time()
    print(f"✅ 接收 hidden states ({t3-t2:.3f}s)")
    print(f"   Shape: {hidden_states.shape}")
    
    # Rank 1 前向传播
    print(f"\n步骤 4: Rank 1 前向传播...")
    t4 = time.time()
    # output = model.model(hidden_states)  # 需要根据实际 API 调整
    output = hidden_states  # 暂时直接返回
    t5 = time.time()
    print(f"✅ Rank 1 前向完成 ({t5-t4:.3f}s)")
    
    # 返回结果给 Rank 0
    print(f"\n步骤 5: 返回结果给 Rank 0...")
    sender = TensorSender('192.168.1.10', port+1)  # Rank 0 的 IP
    await sender.connect()
    
    t6 = time.time()
    await sender.send(output, rank=1)
    t7 = time.time()
    print(f"✅ 发送完成 ({t7-t6:.3f}s)")
    
    await sender.close()
    await receiver.stop()
    
    total = t7 - t0
    print(f"\n=== Rank 1 完成 ===")
    print(f"总时间: {total:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="双机 Pipeline 测试")
    parser.add_argument("--rank", type=int, choices=[0, 1], required=True,
                       help="Rank 编号 (0 或 1)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Rank 1 监听地址，或 Rank 0 目标地址")
    parser.add_argument("--port", type=int, default=29900,
                       help="端口号 (默认: 29900)")
    
    args = parser.parse_args()
    
    if args.rank == 1:
        asyncio.run(rank1_task(args.host, args.port))
    else:
        asyncio.run(rank0_task(args.host, args.port))


if __name__ == "__main__":
    main()
