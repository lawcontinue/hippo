"""
test_e2e_single.py — 单机双 rank 端到端测试

在单台 Mac 上模拟双设备 Pipeline Parallelism：
- 启动 mx.distributed（2 个进程）
- 每个 rank 加载模型的不同层
- 验证 Pipeline 通信正常
"""

import sys
import os
import subprocess

# 单机双进程测试
TEST_SCRIPT = '''
import mlx.core as mx
import mlx_lm

# 初始化分布式
mx.distributed.init(backend="gloo")
group = mx.distributed.new_group([0, 1])
rank = group.rank()
world_size = group.size()
print(f"Rank {rank}/{world_size} initialized")

# 加载小模型测试
from mlx_lm import load
model_id = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
model, tokenizer = load(model_id)

# 获取层数
inner = model.model if hasattr(model, 'model') else model
layers = list(inner.layers) if hasattr(inner, 'layers') else []
n_layers = len(layers)
print(f"Rank {rank}: Model has {n_layers} layers")

# 分片：前半 / 后半
mid = n_layers // 2
if rank == 0:
    local = layers[:mid]
    print(f"Rank 0: {len(local)} layers [0:{mid}]")
else:
    local = layers[mid:]
    print(f"Rank 1: {len(local)} layers [{mid}:{n_layers}]")

# 测试通信：rank 0 发送 tensor，rank 1 接收
import numpy as np
if rank == 0:
    t = mx.array([1.0, 2.0, 3.0])
    sent = mx.distributed.send(t, 1, group=group)
    mx.eval(sent)
    print(f"Rank 0: sent tensor")
else:
    t = mx.distributed.recv_like(mx.array([0.0, 0.0, 0.0]), 0, group=group)
    mx.eval(t)
    print(f"Rank 1: received tensor = {t}")

# 测试 all_gather
result = mx.distributed.all_gather(mx.array([float(rank)]), group=group)
mx.eval(result)
print(f"Rank {rank}: all_gather = {result}")

print(f"Rank {rank}: DONE")
'''

if __name__ == "__main__":
    print("🧪 Single-machine dual-rank Pipeline test")
    print("=" * 50)
    
    # 使用 torchrun 或手动启动
    # MLX distributed 需要 MPI 或 gloo 后端
    # 简单测试：直接用 multiprocessing
    
    import multiprocessing
    
    def run_rank(rank):
        env = os.environ.copy()
        env["RANK"] = str(rank)
        env["WORLD_SIZE"] = "2"
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = "29500"
        
        # 执行测试脚本
        exec_globals = {"__name__": f"rank_{rank}", "mx": None}
        # 简化：直接 import
        import mlx.core as mx
        
        mx.distributed.init(backend="gloo")
        group = mx.distributed.new_group([0, 1])
        r = group.rank()
        ws = group.size()
        print(f"[Process {rank}] Rank {r}/{ws}")
        
        # 测试通信
        if r == 0:
            t = mx.array([1.0, 2.0, 3.0])
            mx.distributed.send(t, 1, group=group)
            mx.eval(t)
            print(f"[Process {rank}] Sent tensor")
        else:
            t = mx.distributed.recv_like(mx.array([0.0, 0.0, 0.0]), 0, group=group)
            mx.eval(t)
            print(f"[Process {rank}] Received: {t}")
        
        print(f"[Process {rank}] DONE")
    
    # 启动两个进程
    p0 = multiprocessing.Process(target=run_rank, args=(0,))
    p1 = multiprocessing.Process(target=run_rank, args=(1,))
    
    p0.start()
    p1.start()
    
    p0.join(timeout=30)
    p1.join(timeout=30)
    
    if p0.exitcode == 0 and p1.exitcode == 0:
        print("✅ Both ranks completed successfully!")
    else:
        print(f"❌ Failed: p0={p0.exitcode}, p1={p1.exitcode}")
