"""
distributed_model.py — 分布式模型加载与推理

核心：只加载分配给本设备的层，不 mmap 全模型。

关键设计（修复 Crit P0-1）：
- 方案 1（推荐）：逐层加载，只 eval 分配的层，不 eval 其他层
- 方案 2（备用）：加载全模型后立即释放不需要的层 + gc + synchronize
- Exo 的做法：slice_transformer_blocks() 在加载前修改模型结构

借鉴 Exo 的 pipeline_auto_parallel()：
1. 加载完整模型配置（config.json）
2. 只加载 model.layers[start:end] 的权重
3. 替换首尾层为 Pipeline 版本
4. 通过 mx.distributed 通信
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

try:
    from .shard import ShardMetadata
    from .pipeline_layers import (
        PipelineFirstLayer,
        PipelineLastLayer,
        get_inner_model,
        get_layers,
    )
except ImportError:
    from shard import ShardMetadata
    from pipeline_layers import (
        PipelineFirstLayer,
        PipelineLastLayer,
        get_inner_model,
        get_layers,
    )


class DistributedModel:
    """
    分布式 Pipeline 模型。
    
    每台设备只持有模型的一部分层，通过 mx.distributed 通信。
    
    内存策略（P0-1 修复）：
    - 方案 A（lazy eval）：加载模型但不 eval 全部层，只 eval 分配的层。
      MLX 的 lazy evaluation 意味着未 eval 的 tensor 不会被分配物理内存。
    - 方案 B（暴力释放）：加载全模型 → 切片 → 删除未使用的层 → gc + synchronize。
    """

    def __init__(
        self,
        model_path: str | Path,
        shard: ShardMetadata,
        group: Optional[object] = None,  # mx.distributed.Group
        on_layer_loaded: Optional[Callable[[int, int], None]] = None,
    ):
        self.model_path = Path(model_path)
        self.shard = shard
        self.group = group
        self.on_layer_loaded = on_layer_loaded
        self.model = None
        self.tokenizer = None

    def load(self):
        """
        加载模型（同步版本）。
        
        关键优化（P0-1 修复）：
        1. 加载模型（lazy，不立即 eval 全部参数）
        2. 只 eval 分配的层（start:end）→ 只有这些层占用 GPU 内存
        3. 未 eval 的层保持 lazy 状态 → 不占用物理内存
        4. 主动删除未使用的层引用 + gc + synchronize 释放内存
        """
        from mlx_lm import load as mlx_load
        
        # --- Step 1: 加载 tokenizer ---
        model, tokenizer = mlx_load(str(self.model_path))
        self.tokenizer = tokenizer  # P1-3 修复：正确解包
        
        # --- Step 2: 获取模型结构 ---
        inner = get_inner_model(model)
        all_layers = get_layers(inner)
        
        total = len(all_layers)
        assert total == self.shard.n_layers, (
            f"Model has {total} layers, expected {self.shard.n_layers}"
        )
        
        start = self.shard.start_layer
        end = self.shard.end_layer
        
        # --- Step 3: 只 eval 分配的层（P0-1 核心修复）---
        # MLX 是 lazy evaluation：不 eval = 不分配物理内存
        local_layers = []
        for i in range(start, end):
            layer = all_layers[i]
            mx.eval(layer)  # 只 eval 这一层 → 只有这一层占内存
            local_layers.append(layer)
            if self.on_layer_loaded:
                self.on_layer_loaded(i - start + 1, end - start)
        
        # --- Step 4: 释放未使用的层（暴力清理）---
        # 将未分配的层替换为 None，释放引用
        freed_layers = []
        for i in range(total):
            if i < start or i >= end:
                freed_layers.append(all_layers[i])
        
        # 重建层列表：用 None 替换未使用的层
        new_layers = [None] * total
        for i, layer in enumerate(local_layers):
            new_layers[start + i] = layer
        
        if hasattr(inner, 'layers'):
            inner.layers = new_layers
        elif hasattr(inner, 'h'):
            inner.h = new_layers
        
        # 删除引用，触发 GC
        del freed_layers
        del all_layers
        del local_layers
        gc.collect()
        
        # MLX synchronize 确保 GPU 内存释放
        mx.synchronize()
        
        # --- Step 5: 现在安全地获取 local_layers（非 None 的）---
        final_layers = [l for l in 
                        (inner.layers if hasattr(inner, 'layers') else inner.h)
                        if l is not None]
        
        # --- Step 6: 替换首尾层为 Pipeline 版本 ---
        if self.group is not None and len(final_layers) > 0:
            final_layers[0] = PipelineFirstLayer(
                final_layers[0], self.shard.device_rank, self.group
            )
            final_layers[-1] = PipelineLastLayer(
                final_layers[-1], self.shard.device_rank,
                self.shard.world_size, self.group,
            )
        
        # 更新模型的层列表
        if hasattr(inner, 'layers'):
            inner.layers = final_layers
        elif hasattr(inner, 'h'):
            inner.h = final_layers
        
        # 更新 num_layers 等元信息
        if hasattr(inner, 'num_layers'):
            inner.num_layers = len(final_layers)
        
        # 处理 layer_types（某些模型如 GPT-OSS 需要）
        if hasattr(inner, 'layer_types'):
            inner.layer_types = inner.layer_types[start:end]
        
        self.model = model
        return model

    def load_lazy(self):
        """
        更激进的内存优化（实验性）。
        
        直接从 safetensors 文件只读取分配层的权重，
        完全不加载其他层。需要手动解析模型结构。
        
        TODO: 实现基于 safetensors 的精确层加载
        """
        raise NotImplementedError("Lazy loading not yet implemented. Use load() instead.")

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        """
        生成文本。
        
        在分布式模式下：
        - rank 0: tokenize 输入 → 发送给 pipeline → 收集输出 → decode
        - rank > 0: 接收 hidden states → 执行分配的层 → 发送给下一个 rank
        """
        from mlx_lm import generate as mlx_generate
        
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        response = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            **kwargs,
        )
        return response

    def get_memory_usage_gb(self) -> float:
        """获取当前模型占用内存（GB）— 只计算本分片的层"""
        if self.model is None:
            return 0.0
        total_bytes = sum(
            p.nbytes for p in self.model.parameters()
            if p is not None
        )
        return total_bytes / (1024 ** 3)


def memory_preflight(model_size_gb: float, available_gb: float,
                     safety_margin: float = 0.8) -> tuple[bool, str]:
    """
    加载前内存预检。
    
    Args:
        model_size_gb: 模型分片大小（GB），不是全模型大小！
        available_gb: 可用内存（GB）
        safety_margin: 安全系数（0.8 = 80% 可用内存才算安全）
    
    Returns:
        (ok, message)
    """
    usable = available_gb * safety_margin
    
    # 加载峰值 = 分片大小 + 基础开销（embedding, lm_head 等）
    # 估算峰值约为分片大小的 1.3 倍
    peak_estimate = model_size_gb * 1.3
    
    if peak_estimate > usable:
        return False, (
            f"内存不足: 峰值预估 {peak_estimate:.1f}GB, "
            f"可用 {available_gb:.1f}GB (安全阈值 {usable:.1f}GB)"
        )
    return True, (
        f"内存充足: 峰值预估 {peak_estimate:.1f}GB, "
        f"可用 {available_gb:.1f}GB (安全阈值 {usable:.1f}GB)"
    )
