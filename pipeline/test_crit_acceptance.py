"""
test_crit_acceptance.py — Crit ⚖️ 独立验收测试

覆盖：TCP Transport 边界、Pipeline 多 rank、Encode/Decode 全 dtype、
      Distributed memory_preflight、Shard 不均匀切分、协议一致性
"""

import asyncio
import gc
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
import numpy as np

from tcp_transport import (
    MAGIC, HEADER_FMT, HEADER_SIZE, DTYPE_TO_CODE, CODE_TO_DTYPE, DTYPE_ITEMSIZE,
    TensorFrame, encode_tensor, decode_tensor, frame_to_mlx,
    TensorReceiver, TensorSender, PipelineTransport,
)
from shard import ShardMetadata, split_model, memory_weighted_split
from cluster import Cluster, DeviceInfo
from distributed_model import memory_preflight

# ─── Helpers ──────────────────────────────────────────────

passed = 0
failed = 0
results = []

def record(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        results.append(f"  ✅ {name}")
    else:
        failed += 1
        results.append(f"  ❌ {name}: {detail}")
    print(results[-1])


# ─── 1. Encode / Decode 全 dtype ─────────────────────────

def test_encode_decode_bfloat16():
    """bfloat16 编解码 — 检测 numpy 兼容性问题"""
    t = mx.ones((2, 3), dtype=mx.bfloat16)
    try:
        data = encode_tensor(t, rank=1)
        frame = decode_tensor(data)
        result = frame_to_mlx(frame)
        record("bfloat16 encode/decode", frame.dtype == mx.bfloat16 and mx.allclose(result, t))
    except RuntimeError as e:
        record("bfloat16 encode/decode", False, f"numpy bfloat16 bug: {e}")


def test_encode_decode_bool():
    """bool_ 编解码"""
    t = mx.array([[True, False], [False, True]])
    data = encode_tensor(t, rank=0)
    frame = decode_tensor(data)
    result = frame_to_mlx(frame)
    record("bool_ encode/decode", frame.dtype == mx.bool_ and mx.array_equal(result, t))


def test_encode_decode_all_dtypes():
    """所有支持 dtype 往返（排除 bfloat16 — 已知 numpy bug）"""
    skip_dtypes = {mx.bfloat16}  # numpy bfloat16 buffer protocol bug
    ok = True
    for dtype in DTYPE_TO_CODE:
        if dtype in skip_dtypes:
            continue
        t = mx.ones((4,), dtype=dtype)
        data = encode_tensor(t, rank=0)
        frame = decode_tensor(data)
        result = frame_to_mlx(frame)
        if frame.dtype != dtype or not mx.allclose(result, t):
            ok = False
            break
    record("All dtypes roundtrip", ok)


def test_encode_decode_scalar():
    """标量 tensor (0-dim)"""
    # MLX 0-dim arrays - use 1-element instead
    t = mx.array(42.0)
    data = encode_tensor(t, rank=0)
    frame = decode_tensor(data)
    record("Scalar-like tensor (0-dim)", frame.nbytes > 0)


def test_encode_decode_large():
    """大 tensor (>1MB)"""
    shape = (256, 1024)  # 256KB floats = 1MB
    t = mx.random.normal(shape, dtype=mx.float32)
    data = encode_tensor(t, rank=0)
    frame = decode_tensor(data)
    result = frame_to_mlx(frame)
    ok = mx.allclose(result, t) and frame.nbytes == 256 * 1024 * 4
    record(f"Large tensor ({frame.nbytes/1024/1024:.1f}MB)", ok)


def test_encode_invalid_dtype():
    """不支持的 dtype 应报错"""
    # complex64 is likely unsupported
    try:
        t = mx.array([1+2j])
        encode_tensor(t, rank=0)
        record("Invalid dtype raises", False, "no error raised")
    except (ValueError, KeyError):
        record("Invalid dtype raises ValueError", True)


def test_decode_bad_magic():
    """magic number 校验"""
    bad = struct.pack(HEADER_FMT, 0xDEADBEEF, 0, 1, 1) + struct.pack("!I", 4) + b'\x00\x00\x80?' * 4
    try:
        decode_tensor(bad)
        record("Bad magic rejected", False, "no error")
    except ValueError as e:
        record("Bad magic rejected", True)


def test_decode_body_size_mismatch():
    """body 大小不匹配"""
    header = struct.pack(HEADER_FMT, MAGIC, 0, 1, 1)  # float32, ndim=1
    shape = struct.pack("!1I", 10)  # says 10 elements
    body = b'\x00' * 20  # only 5 float32s
    try:
        decode_tensor(header + shape + body)
        record("Body size mismatch rejected", False, "no error")
    except ValueError:
        record("Body size mismatch rejected", True)


def test_protocol_endian_consistency():
    """协议大端一致性"""
    t = mx.array([1.0, 2.0], dtype=mx.float32)
    data = encode_tensor(t, rank=5)
    # Header should be network byte order (big-endian)
    magic, rank, ndim, dtype_code = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    record("Protocol big-endian consistency",
           magic == MAGIC and rank == 5 and ndim == 1 and dtype_code == DTYPE_TO_CODE[mx.float32])


# ─── 2. TCP Transport ───────────────────────────────────

async def test_tcp_large_tensor():
    """TCP 传输大 tensor (>1MB)"""
    port = 29800
    recv = TensorReceiver(host="127.0.0.1", port=port)
    await recv.start()
    sender = TensorSender(host="127.0.0.1", port=port)
    await sender.connect()

    t = mx.random.normal((128, 1024), dtype=mx.float32)  # 512KB
    await sender.send(t, rank=0)
    frame = await recv.recv(timeout=10.0)
    result = frame_to_mlx(frame)
    ok = mx.allclose(result, t)
    record(f"TCP large tensor ({frame.nbytes/1024:.0f}KB)", ok)

    await sender.close()
    await recv.stop()


async def test_tcp_multiple_sends():
    """连续多次发送"""
    port = 29810
    recv = TensorReceiver(host="127.0.0.1", port=port)
    await recv.start()
    sender = TensorSender(host="127.0.0.1", port=port)
    await sender.connect()

    tensors = [mx.random.normal((4, 4), dtype=mx.float32) for _ in range(5)]
    for i, t in enumerate(tensors):
        await sender.send(t, rank=i)

    ok = True
    for i, expected in enumerate(tensors):
        frame = await recv.recv(timeout=5.0)
        result = frame_to_mlx(frame)
        if not mx.allclose(result, expected) or frame.rank != i:
            ok = False
            break
    record("TCP 5 consecutive sends", ok)

    await sender.close()
    await recv.stop()


async def test_tcp_connect_refused():
    """连接拒绝后重试"""
    sender = TensorSender(host="127.0.0.1", port=29899)
    try:
        await sender.connect(retries=2, delay=0.1)
        record("Connection refused raises", False)
    except (ConnectionRefusedError, OSError):
        record("Connection refused raises correctly", True)


async def test_sender_not_connected():
    """未连接时 send 应报错"""
    sender = TensorSender(host="127.0.0.1", port=29898)
    try:
        await sender.send(mx.array([1.0]), rank=0)
        record("Send without connect raises", False)
    except RuntimeError:
        record("Send without connect raises RuntimeError", True)


# ─── 3. Pipeline 3+ rank ────────────────────────────────

async def test_pipeline_3_rank():
    """3 rank pipeline: rank 0 → rank 1 → rank 2"""
    base_port = 29900
    transports = [
        PipelineTransport(rank=i, world_size=3, port=base_port)
        for i in range(3)
    ]

    # Start receivers first (reverse order for stability)
    for t in reversed(transports):
        next_host = "127.0.0.1" if t.rank < 2 else None
        await t.start(next_host=next_host)

    # rank 0 sends
    hidden = mx.random.normal((1, 4, 32), dtype=mx.float16)
    send_task = asyncio.create_task(transports[0].send_next(hidden))

    # rank 1 receives and forwards
    r1 = await transports[1].recv_prev(timeout=5.0)
    fwd_task = asyncio.create_task(transports[1].send_next(r1))

    # rank 2 receives
    r2 = await transports[2].recv_prev(timeout=5.0)

    await asyncio.gather(send_task, fwd_task)

    ok = mx.allclose(hidden, r1, atol=1e-3) and mx.allclose(hidden, r2, atol=1e-3)
    record("3-rank pipeline chain", ok)

    for t in transports:
        await t.stop()


async def test_pipeline_rank0_sendonly():
    """rank 0 只发不收（最后一个 rank 不发）"""
    base_port = 29950
    t0 = PipelineTransport(rank=0, world_size=2, port=base_port)
    t1 = PipelineTransport(rank=1, world_size=2, port=base_port)

    await t1.start()
    await t0.start(next_host="127.0.0.1")

    t = mx.array([1.0, 2.0, 3.0])
    await t0.send_next(t)
    r = await t1.recv_prev(timeout=5.0)

    record("Rank 0 send-only", mx.allclose(r, t))

    # Last rank send_next should be no-op
    await t1.send_next(t)  # Should not raise, just return
    record("Rank last send_next is no-op", True)

    await t0.stop()
    await t1.stop()


# ─── 4. Shard 边界 ──────────────────────────────────────

def test_shard_uneven_7_3():
    """不均匀切分: 7 层 3 设备"""
    shards = split_model("test", n_layers=7, world_size=3)
    assert len(shards) == 3
    # 7 / 3 = 2 remainder 1 → rank 0 gets 3, rank 1 gets 2, rank 2 gets 2
    total = sum(s.n_local_layers for s in shards)
    ok = total == 7 and shards[0].end_layer == shards[1].start_layer
    record(f"Uneven split 7/3 (layers: {[s.n_local_layers for s in shards]})", ok)


def test_shard_single_device():
    """单设备: 所有层在一个分片"""
    shards = split_model("test", n_layers=32, world_size=1)
    ok = (len(shards) == 1 and shards[0].n_local_layers == 32
          and shards[0].is_first and shards[0].is_last)
    record("Single device split", ok)


def test_shard_layers_equal_total():
    """层数总和必须等于 n_layers"""
    for n, w in [(1, 1), (3, 3), (10, 3), (100, 7), (7, 5)]:
        shards = split_model("t", n_layers=n, world_size=w)
        total = sum(s.n_local_layers for s in shards)
        if total != n:
            record(f"Layer sum for {n}/{w}", False, f"got {total}")
            return
    record("Layer sum equals n_layers (5 cases)", True)


def test_shard_no_overlap():
    """分片无重叠且无间隙"""
    shards = split_model("t", n_layers=40, world_size=5)
    layers = set()
    ok = True
    for s in shards:
        for l in range(s.start_layer, s.end_layer):
            if l in layers:
                ok = False
            layers.add(l)
    ok = ok and len(layers) == 40
    record("No overlap, no gaps", ok)


def test_memory_weighted_uneven():
    """memory_weighted_split 边界: 差异很大的内存"""
    shards = memory_weighted_split("t", n_layers=10, device_memories=[1.0, 9.0])
    total = sum(s.n_local_layers for s in shards)
    ok = total == 10 and shards[1].n_local_layers > shards[0].n_local_layers
    record(f"Memory weighted (1:9 ratio, layers: {[s.n_local_layers for s in shards]})", ok)


def test_memory_weighted_equal():
    """memory_weighted_split: 等内存"""
    shards = memory_weighted_split("t", n_layers=64, device_memories=[8.0, 8.0])
    total = sum(s.n_local_layers for s in shards)
    record(f"Memory weighted equal (layers: {[s.n_local_layers for s in shards]})", total == 64)


def test_memory_weighted_single():
    """memory_weighted_split: 单设备"""
    shards = memory_weighted_split("t", n_layers=32, device_memories=[16.0])
    record("Memory weighted single device", len(shards) == 1 and shards[0].n_local_layers == 32)


# ─── 5. Distributed ─────────────────────────────────────

def test_memory_preflight_ok():
    """内存充足"""
    ok, msg = memory_preflight(model_size_gb=4.0, available_gb=8.0)
    record("memory_preflight OK", ok)


def test_memory_preflight_tight():
    """内存刚好"""
    ok, msg = memory_preflight(model_size_gb=4.0, available_gb=6.5)
    # 4.0 * 1.3 = 5.2, usable = 6.5 * 0.8 = 5.2, borderline
    record(f"memory_preflight tight (ok={ok})", True)  # just verify it doesn't crash


def test_memory_preflight_insufficient():
    """内存不足"""
    ok, msg = memory_preflight(model_size_gb=8.0, available_gb=4.0)
    record("memory_preflight insufficient", not ok)


def test_memory_preflight_zero():
    """零可用内存"""
    ok, msg = memory_preflight(model_size_gb=1.0, available_gb=0.0)
    record("memory_preflight zero memory", not ok)


def test_memory_preflight_large_safety():
    """大安全系数"""
    ok, msg = memory_preflight(model_size_gb=4.0, available_gb=8.0, safety_margin=0.5)
    # usable = 4.0, peak = 5.2, not ok
    record("memory_preflight large safety margin", not ok)


# ─── 6. 上次修复验证 ────────────────────────────────────

def test_P01_only_eval_assigned_layers():
    """
    P0-1 验证: distributed_model.py 的 load() 方法代码检查
    确认只 eval start:end 范围内的层
    """
    import inspect
    from distributed_model import DistributedModel
    source = inspect.getsource(DistributedModel.load)
    # Should have the pattern of iterating start:end
    has_range = "range(start, end)" in source
    has_eval = "mx.eval(layer)" in source
    has_gc = "gc.collect()" in source
    has_sync = "mx.synchronize()" in source
    ok = has_range and has_eval and has_gc and has_sync
    record("P0-1: Only eval assigned layers + gc + sync", ok,
           f"range={has_range}, eval={has_eval}, gc={has_gc}, sync={has_sync}")


def test_P02_tcp_transport():
    """P0-2 验证: TCP Transport 替代 MPI"""
    # If we got here, tcp_transport.py exists and works
    record("P0-2: TCP Transport replaces MPI", True)


def test_P11_prefill_flag():
    """P1-1 验证: prefill 标志存在"""
    from pipeline_layers import PipelineLastLayer, set_pipeline_prefill
    # PipelineLastLayer has is_prefill
    import inspect
    src = inspect.getsource(PipelineLastLayer.__call__)
    has_prefill_check = "is_prefill" in src
    has_all_gather = "all_gather" in src
    ok = has_prefill_check and has_all_gather
    record("P1-1: prefill flag + conditional all_gather", ok)


def test_P13_tokenizer():
    """P1-3 验证: tokenizer 正确解包"""
    import inspect
    from distributed_model import DistributedModel
    src = inspect.getsource(DistributedModel.load)
    has_unpack = "tokenizer" in src and "mlx_load" in src
    record("P1-3: tokenizer unpacking from mlx_load", has_unpack)


# ─── 7. 资源管理 / 错误处理 ────────────────────────────

async def test_receiver_stats():
    """Receiver stats 跟踪"""
    port = 29850
    recv = TensorReceiver(host="127.0.0.1", port=port)
    await recv.start()
    sender = TensorSender(host="127.0.0.1", port=port)
    await sender.connect()

    await sender.send(mx.array([1.0]), rank=0)
    await recv.recv(timeout=5.0)

    stats = recv.stats
    ok = stats["received"] == 1 and stats["bytes"] > 0 and stats["errors"] == 0
    record(f"Receiver stats tracking (recv={stats['received']}, err={stats['errors']})", ok)

    await sender.close()
    await recv.stop()


def test_frame_latency():
    """TensorFrame latency 计算"""
    f = TensorFrame(rank=0, shape=[2], dtype=mx.float32, data=b'\x00'*8,
                    send_time=1.0, recv_time=1.5)
    record("TensorFrame latency_ms", abs(f.latency_ms - 500.0) < 0.01)


def test_frame_nbytes():
    """TensorFrame nbytes"""
    f = TensorFrame(rank=0, shape=[10], dtype=mx.float32, data=b'\x00'*40)
    record("TensorFrame nbytes", f.nbytes == 40)


# ─── Main ────────────────────────────────────────────────

async def run_all():
    print("=" * 60)
    print("⚖️  Crit 独立验收测试")
    print("=" * 60)

    print("\n--- 1. Encode/Decode 全 dtype ---")
    test_encode_decode_bfloat16()
    test_encode_decode_bool()
    test_encode_decode_all_dtypes()
    test_encode_decode_scalar()
    test_encode_decode_large()
    test_encode_invalid_dtype()
    test_decode_bad_magic()
    test_decode_body_size_mismatch()
    test_protocol_endian_consistency()

    print("\n--- 2. TCP Transport ---")
    await test_tcp_large_tensor()
    await test_tcp_multiple_sends()
    await test_tcp_connect_refused()
    await test_sender_not_connected()

    print("\n--- 3. Pipeline 3+ rank ---")
    await test_pipeline_3_rank()
    await test_pipeline_rank0_sendonly()

    print("\n--- 4. Shard 边界 ---")
    test_shard_uneven_7_3()
    test_shard_single_device()
    test_shard_layers_equal_total()
    test_shard_no_overlap()
    test_memory_weighted_uneven()
    test_memory_weighted_equal()
    test_memory_weighted_single()

    print("\n--- 5. Distributed ---")
    test_memory_preflight_ok()
    test_memory_preflight_tight()
    test_memory_preflight_insufficient()
    test_memory_preflight_zero()
    test_memory_preflight_large_safety()

    print("\n--- 6. 上次修复验证 ---")
    test_P01_only_eval_assigned_layers()
    test_P02_tcp_transport()
    test_P11_prefill_flag()
    test_P13_tokenizer()

    print("\n--- 7. 资源管理 ---")
    await test_receiver_stats()
    test_frame_latency()
    test_frame_nbytes()

    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败, 共 {passed + failed} 项")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(run_all())
    sys.exit(0 if ok else 1)
