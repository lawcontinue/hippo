"""
pipeline_layers.py — Pipeline First/Last Layer 封装

借鉴 Exo 的 PipelineFirstLayer + PipelineLastLayer，
实现层间数据传递（send/recv）和结果收集（all_gather）。

核心思路：
- PipelineFirstLayer: 如果不是 rank 0，从前一个 rank 接收输入 tensor
- PipelineLastLayer: 执行计算后，发送给下一个 rank，并 all_gather 收集结果
"""

from __future__ import annotations

from typing import Protocol, cast

import mlx.core as mx


class LayerCallable(Protocol):
    """兼容 mlx.nn.Module 的层协议"""
    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array: ...


class PipelineFirstLayer:
    """
    Pipeline 的第一层。
    
    - rank 0: 直接处理输入（tokenize 后的 embeddings）
    - rank > 0: 从前一个 rank 接收 hidden states
    """

    def __init__(self, original_layer: LayerCallable, rank: int,
                 group: mx.distributed.Group):
        self.original_layer = original_layer
        self.rank = rank
        self.group = group

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self.rank != 0:
            # 从前一个 rank 接收 hidden states
            mx.eval(x)
            x = mx.distributed.recv_like(x, self.rank - 1, group=self.group)
            mx.eval(x)
        return self.original_layer(x, *args, **kwargs)


class PipelineLastLayer:
    """
    Pipeline 的最后一层。
    
    - 执行层计算
    - 如果不是最后一个 rank，发送给下一个 rank
    - decode 阶段：all_gather 收集所有 rank 的输出
    - prefill 阶段：跳过 all_gather（减少通信开销）
    """

    def __init__(self, original_layer: LayerCallable, rank: int,
                 world_size: int, group: mx.distributed.Group):
        self.original_layer = original_layer
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.is_prefill: bool = False  # P1-1 修复：prefill 标志

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        output = self.original_layer(x, *args, **kwargs)
        mx.eval(output)

        # 发送给下一个 rank
        if self.rank != self.world_size - 1:
            mx.distributed.send(output, self.rank + 1, group=self.group)
            mx.eval(output)

        # P1-1 修复：decode 阶段才 all_gather
        # prefill 阶段跳过，减少通信开销
        if not self.is_prefill:
            gathered = mx.distributed.all_gather(output, group=self.group)
            mx.eval(gathered)
            # 取最后一个 rank 的输出（包含完整推理结果）
            return gathered[-output.shape[0]:]

        return output


def get_inner_model(model):
    """获取模型的内部 model 层列表容器"""
    for attr in ('model', 'transformer', 'backbone', 'language_model'):
        inner = getattr(model, attr, None)
        if inner is not None:
            # language_model 可能还需要再深入一层
            if attr == 'language_model':
                inner2 = getattr(inner, 'model', None)
                if inner2 is not None:
                    inner = inner2
            return inner
    raise ValueError("Model must have 'model', 'transformer', or 'backbone' attribute")


def get_layers(inner_model) -> list:
    """获取模型的层列表"""
    if hasattr(inner_model, 'layers'):
        return list(inner_model.layers)
    elif hasattr(inner_model, 'h'):
        return list(inner_model.h)
    raise ValueError("Inner model must have 'layers' or 'h' attribute")


def set_pipeline_prefill(model, is_prefill: bool) -> None:
    """设置所有 Pipeline 层的 prefill 状态"""
    inner = get_inner_model(model)
    layers = get_layers(inner)
    for layer in layers:
        if isinstance(layer, PipelineLastLayer):
            layer.is_prefill = is_prefill
