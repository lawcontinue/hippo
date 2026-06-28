"""
reasoning_path_impact.py — RPIA (Reasoning Path Impact Assessment)

熔炉#111 S2 成果（卡193）+ ADR-356。
推理路径变更时的质量影响评估框架。

当 AI 服务的推理路径（thinking on/off/depth/budget）发生变更时，
必须附带 RPIA 评估报告，否则告知=形式合规实质不透明。

RPIA 四要素（卡193）:
1. 评测集 ≥200 条推理密集型任务
2. 对比维度：正确率 + 延迟 + 输出长度（三维交叉）
3. 分场景报告（不能只给总体数字）
4. 版本快照（≥50 条输出样本供独立验证）

用法:
    from reasoning_path_impact import RPIAReport, ReasoningMode, run_rpia

    # 定义变更前后的推理模式
    old = ReasoningMode(thinking=True, label="deep")
    new = ReasoningMode(thinking=False, label="quick")

    # 生成 RPIA 报告
    report = run_rpia(
        eval_set="eval/reasoning_200.jsonl",
        old_mode=old,
        new_mode=new,
        sample_size=50,
    )
    print(report.summary())
    # 是否满足告知阈值
    if report.max_regression > 0.10:
        report.flag_for_disclosure()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
import statistics


class ReasoningDepth(Enum):
    """推理深度等级（熔炉#111 + #112 审查阶梯映射）"""
    DEEP = "deep"        # thinking on, full reasoning chain
    QUICK = "quick"      # thinking off, direct output
    SKIP = "skip"        # no reasoning at all (degraded)
    BUDGETED = "budgeted"  # thinking with limited budget


@dataclass
class ReasoningMode:
    """推理路径配置"""
    thinking: bool
    label: str = ""
    budget_tokens: Optional[int] = None  # thinking budget (None=unlimited)
    depth: ReasoningDepth = field(default=ReasoningDepth.DEEP)

    def __post_init__(self):
        if not self.label:
            self.label = self.depth.value if self.thinking else "quick"
        # auto-set depth
        if self.thinking:
            if self.budget_tokens is not None:
                self.depth = ReasoningDepth.BUDGETED
            else:
                self.depth = ReasoningDepth.DEEP
        else:
            self.depth = ReasoningDepth.QUICK


@dataclass
class EvalResult:
    """单条评测结果"""
    task_id: str
    category: str          # 场景类别（math/logic/legal/code/dialogue）
    correct: bool
    latency_ms: float
    output_length: int     # 字符数
    expected_answer: str = ""
    actual_answer: str = ""


@dataclass
class CategoryBreakdown:
    """分场景报告"""
    category: str
    old_accuracy: float
    new_accuracy: float
    delta: float           # new - old（负=退化）
    old_latency: float
    new_latency: float
    sample_size: int


@dataclass
class RPIAReport:
    """
    RPIA 推理路径影响评估报告（卡193 / ADR-356）。

    没有此报告的推理路径变更告知 = 形式合规实质不透明。
    """
    old_mode: ReasoningMode
    new_mode: ReasoningMode
    total_samples: int
    category_breakdowns: List[CategoryBreakdown] = field(default_factory=list)
    snapshot_ids: List[str] = field(default_factory=list)  # 输出样本 ID
    generated_at: str = ""
    assessor: str = ""       # 评估者标识

    @property
    def max_regression(self) -> float:
        """最大场景退化幅度（正值=退化百分比）"""
        if not self.category_breakdowns:
            return 0.0
        # delta 是 new-old，退化是负 delta，我们取最大退化=最小 delta 的绝对值
        worst = min(cb.delta for cb in self.category_breakdowns)
        return abs(worst) if worst < 0 else 0.0

    @property
    def overall_delta(self) -> float:
        """总体正确率变化"""
        if not self.category_breakdowns:
            return 0.0
        total_old = sum(cb.old_accuracy * cb.sample_size for cb in self.category_breakdowns)
        total_new = sum(cb.new_accuracy * cb.sample_size for cb in self.category_breakdowns)
        total_n = sum(cb.sample_size for cb in self.category_breakdowns)
        if total_n == 0:
            return 0.0
        return (total_new - total_old) / total_n

    def summary(self) -> str:
        """生成人类可读摘要"""
        lines = [
            f"RPIA Report: {self.old_mode.label} → {self.new_mode.label}",
            f"Samples: {self.total_samples}",
            f"Overall delta: {self.overall_delta:+.1%}",
            f"Max regression: {self.max_regression:.1%}",
            "",
            "Category breakdown:",
        ]
        for cb in sorted(self.category_breakdowns, key=lambda x: x.delta):
            reg = "⚠️" if cb.delta < -0.05 else "✅"
            lines.append(
                f"  {reg} {cb.category:12s}: "
                f"{cb.old_accuracy:.1%}→{cb.new_accuracy:.1%} "
                f"({cb.delta:+.1%}, n={cb.sample_size})"
            )
        return "\n".join(lines)

    def flag_for_disclosure(self) -> bool:
        """是否需要触发告知义务（Crit三条件之一：影响不可忽略）"""
        return self.max_regression > 0.10  # 最大场景退化 >10%


def compute_breakdown(
    old_results: List[EvalResult],
    new_results: List[EvalResult],
) -> List[CategoryBreakdown]:
    """
    计算分场景报告。

    Args:
        old_results: 变更前评测结果
        new_results: 变更后评测结果
    Returns:
        分场景 breakdown 列表
    """
    categories = set(r.category for r in old_results) | set(r.category for r in new_results)
    breakdowns = []

    for cat in categories:
        old_cat = [r for r in old_results if r.category == cat]
        new_cat = [r for r in new_results if r.category == cat]
        if not old_cat or not new_cat:
            continue

        old_acc = sum(1 for r in old_cat if r.correct) / len(old_cat)
        new_acc = sum(1 for r in new_cat if r.correct) / len(new_cat)
        old_lat = statistics.mean(r.latency_ms for r in old_cat)
        new_lat = statistics.mean(r.latency_ms for r in new_cat)
        n = min(len(old_cat), len(new_cat))

        breakdowns.append(CategoryBreakdown(
            category=cat,
            old_accuracy=old_acc,
            new_accuracy=new_acc,
            delta=new_acc - old_acc,
            old_latency=old_lat,
            new_latency=new_lat,
            sample_size=n,
        ))
    return breakdowns


def run_rpia(
    old_results: List[EvalResult],
    new_results: List[EvalResult],
    old_mode: ReasoningMode,
    new_mode: ReasoningMode,
    snapshot_ids: Optional[List[str]] = None,
    assessor: str = "automated",
) -> RPIAReport:
    """
    生成 RPIA 报告。

    Args:
        old_results: 变更前评测结果（≥200 条推荐，TOOLS.md #19）
        new_results: 变更后评测结果
        old_mode: 变更前推理模式
        new_mode: 变更后推理模式
        snapshot_ids: 输出样本 ID（≥50 条，供独立验证）
        assessor: 评估者
    Returns:
        RPIAReport 实例
    """
    from datetime import datetime, timezone

    breakdowns = compute_breakdown(old_results, new_results)

    # TOOLS.md #19: 验收用评测集 >=200 条，50 条系统性高估 +11pp
    if len(new_results) < 200:
        import warnings
        warnings.warn(
            f"RPIA eval set only {len(new_results)} samples (< 200). "
            "Results may be systematically overestimated (TOOLS.md #19).",
            stacklevel=2,
        )

    return RPIAReport(
        old_mode=old_mode,
        new_mode=new_mode,
        total_samples=len(new_results),
        category_breakdowns=breakdowns,
        snapshot_ids=snapshot_ids or [],
        generated_at=datetime.now(timezone.utc).isoformat(),
        assessor=assessor,
    )
