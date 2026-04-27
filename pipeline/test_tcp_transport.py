"""
test_tcp_transport.py — TCP tensor 传输测试

单机模拟双设备 Pipeline：
  rank 0 → send hidden states → rank 1 → send result back

验证：
1. 编码/解码正确性
2. TCP 收发正确性
3. Pipeline 传输正确性
4. 延迟基准
"""

import asyncio
import sys
import time
from pathlib import Path

# 添加父目录到 path
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx

from tcp_transport import (
    TensorFrame,
    encode_tensor,
    decode_tensor,
    frame_to_mlx,
    TensorReceiver,
    TensorSender,
    PipelineTransport,
)


def test_encode_decode():
    """测试 1: 编码/解码正确性"""
    print("=== Test 1: Encode/Decode ===")

    for dtype in [mx.float32, mx.float16, mx.int32]:
        tensor = mx.ones((2, 4), dtype=dtype)
        data = encode_tensor(tensor, rank=0)
        frame = decode_tensor(data)
        result = frame_to_mlx(frame)

        assert frame.rank == 0
        assert frame.shape == [2, 4]
        assert frame.dtype == dtype
        assert mx.allclose(result, tensor), f"Mismatch for {dtype}"
        print(f"  ✅ {dtype}: shape={frame.shape}, nbytes={frame.nbytes}")

    print("  ✅ All encode/decode tests passed\n")


async def test_tcp_send_recv():
    """测试 2: TCP 收发"""
    print("=== Test 2: TCP Send/Recv ===")

    receiver = TensorReceiver(host="127.0.0.1", port=29500)
    await receiver.start()

    sender = TensorSender(host="127.0.0.1", port=29500)
    await sender.connect()

    # 发送一个 float32 tensor
    original = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=mx.float32)
    await sender.send(original, rank=0)

    frame = await receiver.recv(timeout=5.0)
    result = frame_to_mlx(frame)

    assert frame.rank == 0
    assert frame.shape == [3, 2]
    assert mx.allclose(result, original)
    print(f"  ✅ Sent {original.shape} → received {result.shape}")
    print(f"  ✅ Values match: {mx.allclose(result, original)}")

    await sender.close()
    await receiver.stop()
    print()


async def test_pipeline_transport():
    """测试 3: Pipeline 双向传输（模拟 rank 0 → rank 1）"""
    print("=== Test 3: Pipeline Transport (rank 0 → rank 1) ===")

    # 创建两个 transport（同一台机器，不同端口）
    t0 = PipelineTransport(rank=0, world_size=2, port=29600)
    t1 = PipelineTransport(rank=1, world_size=2, port=29600)

    # 启动（rank 0 发给 rank 1）
    await t1.start()  # 先启动 receiver
    await t0.start(next_host="127.0.0.1")

    # rank 0 发送 hidden states
    hidden = mx.random.normal((1, 8, 64), dtype=mx.float16)
    t0_send = asyncio.create_task(t0.send_next(hidden))

    # rank 1 接收
    received = await t1.recv_prev(timeout=5.0)
    await t0_send

    print(f"  ✅ Rank 0 sent: {hidden.shape}")
    print(f"  ✅ Rank 1 received: {received.shape}")
    print(f"  ✅ Values match: {mx.allclose(hidden, received, atol=1e-3)}")

    print(f"  Stats T0: {t0.stats}")
    print(f"  Stats T1: {t1.stats}")

    await t0.stop()
    await t1.stop()
    print()


async def test_latency():
    """测试 4: 延迟基准"""
    print("=== Test 4: Latency Benchmark ===")

    receiver = TensorReceiver(host="127.0.0.1", port=29700)
    await receiver.start()

    sender = TensorSender(host="127.0.0.1", port=29700)
    await sender.connect()

    # 模拟不同大小的 hidden states
    sizes = [
        (1, 8, 64, mx.float16),      # 小 (1KB)
        (1, 128, 1024, mx.float16),   # 中 (256KB)
        (1, 512, 4096, mx.float16),   # 大 (4MB)
        (8, 512, 4096, mx.float16),   # XL (32MB)
    ]

    for shape in sizes:
        dtype = shape[-1]
        tensor_shape = shape[:-1]
        tensor = mx.random.normal(tensor_shape, dtype=dtype)
        nbytes = 1
        for s in tensor_shape:
            nbytes *= s
        nbytes *= 2  # float16

        # 发送
        t0 = time.perf_counter()
        await sender.send(tensor, rank=0)
        frame = await receiver.recv(timeout=10.0)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        throughput_gbps = (frame.nbytes / 1e9) / (t1 - t0) if t1 > t0 else 0

        print(f"  {tensor_shape} ({frame.nbytes/1024:.0f}KB): "
              f"{latency_ms:.2f}ms, {throughput_gbps:.2f} GB/s")

    await sender.close()
    await receiver.stop()
    print()


async def main():
    print("🧪 Hippo TCP Transport Tests\n")

    # 同步测试
    test_encode_decode()

    # 异步测试
    await test_tcp_send_recv()
    await test_pipeline_transport()
    await test_latency()

    print("✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
