"""
acceleration.py — Hippo 统一加速抽象层（AccelerationStrategy）

来源：熔炉 #33（2026-05-23）
设计者：Code + Aria + 刻菲斯
理念：分端服务（DFlash for Mac / MTP for NVIDIA）+ 统一接口 + 自动策略选择
法学隐喻：最密切联系原则——系统自动判断哪个加速方案与当前场景最密切联系

架构：
  AccelerationStrategy（抽象基类）
  ├── DFlashStrategy    — Apple Silicon, KV injection
  ├── MTPStrategy       — NVIDIA GPU, vLLM speculative decoding
  ├── PipelineStrategy  — 双机分片, Thunderbolt
  └── NoneStrategy      — 无加速（基线）

  AccelerationOrchestrator — 自动策略选择引擎
    三维匹配：hardware × language × task_type
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ─── Enums ───────────────────────────────────────────────


class Hardware(str, Enum):
    APPLE_SILICON = "apple_silicon"
    NVIDIA_GPU = "nvidia_gpu"
    UNKNOWN = "unknown"


class Language(str, Enum):
    CHINESE = "chinese"
    ENGLISH = "english"
    CODE = "code"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    CHAT = "chat"
    CODE_GEN = "code_gen"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"


class StrategyName(str, Enum):
    DFLASH = "dflash"
    MTP = "mtp"
    PIPELINE = "pipeline"
    NONE = "none"


# ─── Data Classes ───────────────────────────────────────


@dataclass(frozen=True)
class AccelerationContext:
    """推理请求的上下文信息，用于策略选择。"""

    hardware: Hardware = Hardware.UNKNOWN
    language: Language = Language.UNKNOWN
    task_type: TaskType = TaskType.UNKNOWN
    model_name: str = ""
    batch_size: int = 1
    prompt_length: int = 0

    # DFlash 中文减速的教训（熔炉 #33, 雅典娜贡献）
    # 中文场景 DFlash 0.81×，应避免使用
    CHINESE_PENALTY_DEVICES: frozenset[Hardware] = frozenset({Hardware.APPLE_SILICON})


@dataclass
class AccelerationResult:
    """加速策略选择结果。"""

    strategy: StrategyName
    reason: str
    estimated_speedup: float = 1.0
    confidence: float = 1.0  # 0-1, 策略置信度

    @property
    def config_hint(self) -> dict:
        """返回给推理引擎的配置提示。"""
        hints = {
            StrategyName.DFLASH: {"block_size": 16, "enable_thinking": False},
            StrategyName.MTP: {"num_speculative_tokens": 1, "method": "deepseek_mtp"},
            StrategyName.PIPELINE: {"thunderbolt": True, "shard_count": 2},
            StrategyName.NONE: {},
        }
        return hints.get(self.strategy, {})


# ─── Strategy Interface ────────────────────────────────


class AccelerationStrategy(ABC):
    """加速策略抽象基类。"""

    @property
    @abstractmethod
    def name(self) -> StrategyName:
        ...

    @abstractmethod
    def is_available(self, ctx: AccelerationContext) -> bool:
        """当前上下文是否可用此策略。"""
        ...

    @abstractmethod
    def estimate_speedup(self, ctx: AccelerationContext) -> float:
        """估算加速倍数。"""
        ...

    @abstractmethod
    def suitability_score(self, ctx: AccelerationContext) -> float:
        """适配度评分（0-1），用于策略比较。"""
        ...


class DFlashStrategy(AccelerationStrategy):
    """DFlash：基于 KV injection 的自投机解码。

    约束（ADR-163, ADR-174）：
    - 仅 Apple Silicon
    - 16GB+ 内存
    - 中文场景减速（0.81×）
    - 不可与 Pipeline 叠加（OOM）
    """

    @property
    def name(self) -> StrategyName:
        return StrategyName.DFLASH

    def is_available(self, ctx: AccelerationContext) -> bool:
        return ctx.hardware == Hardware.APPLE_SILICON

    def estimate_speedup(self, ctx: AccelerationContext) -> float:
        base = 4.08
        if ctx.language == Language.CHINESE:
            return 0.81  # 实测减速（ADR-174）
        if ctx.language == Language.CODE:
            return 7.7  # 代码场景峰值
        if ctx.task_type == TaskType.CODE_GEN:
            return 7.7
        return base

    def suitability_score(self, ctx: AccelerationContext) -> float:
        if not self.is_available(ctx):
            return 0.0
        if ctx.language == Language.CHINESE:
            return 0.1  # 几乎不适合
        if ctx.language == Language.CODE or ctx.task_type == TaskType.CODE_GEN:
            return 0.95  # 最佳场景
        if ctx.language == Language.ENGLISH:
            return 0.8
        return 0.6  # 混合/未知


class MTPStrategy(AccelerationStrategy):
    """MTP：Multi-Token Prediction 投机解码。

    约束：
    - 需要模型原生支持（DeepSeek-V3, Gemma 4, MiMo）
    - NVIDIA GPU（vLLM）
    - batch=1 时 MoE 模型收益极低
    """

    # 支持 MTP 的模型列表（可通过 register_mtp_model 扩展）
    _mtp_models: set[str] = {"deepseek-v3", "deepseek-r1", "gemma-4", "mimo-7b"}

    @classmethod
    def register_mtp_model(cls, model_name: str) -> None:
        """注册新的 MTP 兼容模型。"""
        cls._mtp_models.add(model_name.lower())

    @property
    def name(self) -> StrategyName:
        return StrategyName.MTP

    def is_available(self, ctx: AccelerationContext) -> bool:
        if ctx.hardware != Hardware.NVIDIA_GPU:
            return False
        model_lower = ctx.model_name.lower()
        return any(m in model_lower for m in self._mtp_models)

    def estimate_speedup(self, ctx: AccelerationContext) -> float:
        # DeepSeek-V3 MTP: 1.8×, FastMTP: 2.03×, Gemma 4: ~2×
        if ctx.batch_size == 1 and "moe" in ctx.model_name.lower():
            return 1.1  # 几乎无加速（Reddit 实测）
        return 2.0  # 保守估计

    def suitability_score(self, ctx: AccelerationContext) -> float:
        if not self.is_available(ctx):
            return 0.0
        if ctx.batch_size == 1 and "moe" in ctx.model_name.lower():
            return 0.2
        if ctx.language == Language.CHINESE:
            return 0.85  # MTP 多语言优势
        if ctx.language == Language.ENGLISH:
            return 0.7
        return 0.7


class PipelineStrategy(AccelerationStrategy):
    """Pipeline Parallel：双机分片推理。

    约束（ADR-163, ADR-174）：
    - 仅 27B+ 模型有收益（12B 单机更快）
    - 不可与 DFlash 叠加（OOM）
    - Thunderbolt 直连 ~1ms 延迟
    """

    # 需要 Pipeline 的模型大小阈值（参数量 B）
    MIN_MODEL_SIZE_B = 27

    @property
    def name(self) -> StrategyName:
        return StrategyName.PIPELINE

    def is_available(self, ctx: AccelerationContext) -> bool:
        return ctx.hardware == Hardware.APPLE_SILICON

    def estimate_speedup(self, ctx: AccelerationContext) -> float:
        return 48.0  # Thunderbolt 双机（ADR-174）

    def suitability_score(self, ctx: AccelerationContext) -> float:
        if not self.is_available(ctx):
            return 0.0
        # 尝试从模型名推断大小
        model_lower = ctx.model_name.lower()
        size_match = re.search(r"(\d+)b", model_lower)
        if size_match:
            size = int(size_match.group(1))
            if size < self.MIN_MODEL_SIZE_B:
                return 0.0  # 小模型不需要
            return 0.99  # 大模型强烈推荐 Pipeline
        # 未知大小时给予中等评分
        return 0.5


class NoneStrategy(AccelerationStrategy):
    """无加速（基线）。"""

    @property
    def name(self) -> StrategyName:
        return StrategyName.NONE

    def is_available(self, ctx: AccelerationContext) -> bool:
        return True

    def estimate_speedup(self, ctx: AccelerationContext) -> float:
        return 1.0

    def suitability_score(self, ctx: AccelerationContext) -> float:
        return 0.3  # 永远可用但不是最优


# ─── Orchestrator ───────────────────────────────────────


class AccelerationOrchestrator:
    """加速策略自动选择引擎。

    核心逻辑：最密切联系原则
    对每个可用策略评分，选最高分者。
    置信度 = 最高分 / (最高分 + 次高分)（分数越接近越不确定）
    """

    def __init__(self) -> None:
        self._strategies: list[AccelerationStrategy] = [
            DFlashStrategy(),
            MTPStrategy(),
            PipelineStrategy(),
            NoneStrategy(),
        ]

    def register_strategy(self, strategy: AccelerationStrategy) -> None:
        """注册自定义策略。"""
        self._strategies.append(strategy)
        logger.info("Registered strategy: %s", strategy.name.value)

    def select(self, ctx: AccelerationContext) -> AccelerationResult:
        """自动选择最优策略。"""
        scored: list[tuple[AccelerationStrategy, float]] = []

        for s in self._strategies:
            if s.is_available(ctx):
                score = s.suitability_score(ctx)
                speedup = s.estimate_speedup(ctx)
                scored.append((s, score))
                logger.debug(
                    "Strategy %s: score=%.2f, speedup=%.2fx",
                    s.name.value, score, speedup,
                )

        if not scored:
            return AccelerationResult(
                strategy=StrategyName.NONE,
                reason="No strategy available",
                estimated_speedup=1.0,
                confidence=1.0,
            )

        # 按分数降序排列
        scored.sort(key=lambda x: x[1], reverse=True)
        best_strat, best_score = scored[0]
        best_speedup = best_strat.estimate_speedup(ctx)

        # 置信度计算
        if len(scored) > 1:
            second_score = scored[1][1]
            if best_score + second_score > 0:
                confidence = best_score / (best_score + second_score)
            else:
                confidence = 1.0
        else:
            confidence = 1.0

        # 如果最优策略的 speedup < 1.0，回退到 None
        if best_speedup < 1.0 and best_strat.name != StrategyName.NONE:
            logger.warning(
                "Strategy %s would slow down (speedup=%.2fx), falling back to none",
                best_strat.name.value, best_speedup,
            )
            return AccelerationResult(
                strategy=StrategyName.NONE,
                reason=f"{best_strat.name.value} would slow down (speedup={best_speedup:.2f}x)",
                estimated_speedup=1.0,
                confidence=0.9,
            )

        reason = self._build_reason(ctx, best_strat, best_score)
        return AccelerationResult(
            strategy=best_strat.name,
            reason=reason,
            estimated_speedup=best_speedup,
            confidence=round(confidence, 2),
        )

    def _build_reason(
        self,
        ctx: AccelerationContext,
        strategy: AccelerationStrategy,
        score: float,
    ) -> str:
        """构建人类可读的选择原因。"""
        parts = [
            f"hardware={ctx.hardware.value}",
            f"language={ctx.language.value}",
            f"task={ctx.task_type.value}",
            f"→ {strategy.name.value} (score={score:.2f})",
        ]
        return " | ".join(parts)

    def select_with_override(
        self,
        ctx: AccelerationContext,
        force: StrategyName,
    ) -> AccelerationResult:
        """强制指定策略（用户 override --acceleration=dflash 等）。"""
        for s in self._strategies:
            if s.name == force:
                if not s.is_available(ctx):
                    return AccelerationResult(
                        strategy=StrategyName.NONE,
                        reason=f"Requested {force.value} but not available for {ctx.hardware.value}",
                        estimated_speedup=1.0,
                        confidence=0.0,
                    )
                speedup = s.estimate_speedup(ctx)
                if speedup < 1.0:
                    logger.warning(
                        "User override %s would slow down (speedup=%.2fx) — proceeding with override",
                        force.value, speedup,
                    )
                return AccelerationResult(
                    strategy=force,
                    reason=f"User override: {force.value}" + (f" (WARNING: speedup={speedup:.2f}x)" if speedup < 1.0 else ""),
                    estimated_speedup=speedup,
                    confidence=1.0,
                )
        return AccelerationResult(
            strategy=StrategyName.NONE,
            reason=f"Unknown strategy: {force.value}",
            estimated_speedup=1.0,
            confidence=0.0,
        )


# ─── Convenience ───────────────────────────────────────


def detect_language(text: str) -> Language:
    """简单语言检测（基于 Unicode 范围）。"""
    if not text:
        return Language.UNKNOWN

    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_chars = len(re.findall(r"[a-zA-Z]", text))
    code_indicators = len(re.findall(r"(def |class |import |function |=>|\{|\}|```)", text, re.MULTILINE))

    if code_indicators >= 1:
        return Language.CODE
    if cjk > 0 and ascii_chars > 0:
        return Language.MIXED
    if cjk > ascii_chars:
        return Language.CHINESE
    if ascii_chars > cjk:
        return Language.ENGLISH
    return Language.UNKNOWN


def auto_accelerate(
    hardware: str = "unknown",
    model_name: str = "",
    text: str = "",
    task_type: str = "chat",
    batch_size: int = 1,
) -> AccelerationResult:
    """一键加速选择（--acceleration=auto 的入口函数）。

    Args:
        hardware: "apple_silicon" | "nvidia_gpu" | "unknown"
        model_name: 模型名（如 "qwen3-4b", "deepseek-v3"）
        text: 输入文本（用于语言检测）
        task_type: "chat" | "code_gen" | "summarization" | ...
        batch_size: 批大小

    Returns:
        AccelerationResult 包含选择的策略和配置提示
    """
    ctx = AccelerationContext(
        hardware=Hardware(hardware),
        language=detect_language(text),
        task_type=TaskType(task_type),
        model_name=model_name,
        batch_size=batch_size,
    )
    orch = AccelerationOrchestrator()
    return orch.select(ctx)
