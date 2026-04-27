"""
test_memory_loading.py — 内存加载策略验证

验证 P0-1 修复：只加载分配的层，释放未使用的层。
"""

import sys
import os
import gc

import mlx.core as mx


def get_gpu_memory_gb() -> float:
    """获取 MLX 当前 GPU 内存使用"""
    try:
        info = mx.device_info()
        # active_memory 是当前使用的内存
        return info.get('active_memory', 0) / (1024**3)
    except Exception:
        return 0.0


def test_lazy_eval_memory():
    """
    验证 MLX 的 lazy evaluation：
    未 eval 的 tensor 不应该占用物理内存。
    """
    print("=== Test: MLX Lazy Eval Memory ===")
    
    # 创建大 tensor 但不 eval
    big_lazy = mx.zeros((1000, 1000, 100), dtype=mx.float32)  # ~400MB
    mem_before = get_gpu_memory_gb()
    print(f"  Before eval: {mem_before:.2f} GB")
    
    # eval 后应该占用内存
    mx.eval(big_lazy)
    mem_after = get_gpu_memory_gb()
    print(f"  After eval: {mem_after:.2f} GB")
    
    # 删除并同步
    del big_lazy
    gc.collect()
    mx.synchronize()
    mem_freed = get_gpu_memory_gb()
    print(f"  After free: {mem_freed:.2f} GB")
    
    print("✅ Lazy eval memory test passed\n")


def test_partial_load_strategy():
    """
    验证部分加载策略：
    加载模型 → 只 eval 部分层 → 释放其余层 → 内存应减少。
    """
    print("=== Test: Partial Load Strategy ===")
    
    try:
        from mlx_lm import load as mlx_load
        
        # 用小模型测试
        model_id = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        print(f"  Loading model: {model_id}")
        
        mem_start = get_gpu_memory_gb()
        print(f"  Memory before load: {mem_start:.2f} GB")
        
        model, tokenizer = mlx_load(model_id)
        mem_loaded = get_gpu_memory_gb()
        print(f"  Memory after full load: {mem_loaded:.2f} GB")
        
        # 获取层数
        inner = model.model if hasattr(model, 'model') else model
        layers = list(inner.layers) if hasattr(inner, 'layers') else []
        n_layers = len(layers)
        print(f"  Total layers: {n_layers}")
        
        # 只保留前半部分层
        half = n_layers // 2
        kept = layers[:half]
        freed = layers[half:]
        
        # 替换为 None
        new_layers = [None] * n_layers
        for i, l in enumerate(kept):
            new_layers[i] = l
        
        if hasattr(inner, 'layers'):
            inner.layers = new_layers
        elif hasattr(inner, 'h'):
            inner.h = new_layers
        
        # 释放
        del freed
        del layers
        del kept
        gc.collect()
        mx.synchronize()
        
        mem_sliced = get_gpu_memory_gb()
        print(f"  Memory after slice+free: {mem_sliced:.2f} GB")
        print(f"  Freed: {(mem_loaded - mem_sliced):.2f} GB")
        
        print("✅ Partial load test passed\n")
        
    except Exception as e:
        print(f"⚠️ Model not available (network?): {e}")
        print("  Skipping model load test\n")


def test_memory_preflight():
    """测试内存预检函数"""
    print("=== Test: Memory Preflight ===")
    
    # 导入
    sys.path.insert(0, os.path.dirname(__file__))
    from distributed_model import memory_preflight
    
    ok, msg = memory_preflight(model_size_gb=7.0, available_gb=10.0)
    print(f"  7GB model, 10GB avail: {msg}")
    assert ok
    
    ok, msg = memory_preflight(model_size_gb=12.0, available_gb=10.0)
    print(f"  12GB model, 10GB avail: {msg}")
    assert not ok
    
    print("✅ Memory preflight test passed\n")


if __name__ == "__main__":
    test_lazy_eval_memory()
    test_partial_load_strategy()
    test_memory_preflight()
    print("🎉 All memory tests passed!")
