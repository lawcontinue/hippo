"""
Multi-Dimensional Evaluation Metric Matrix.

Five dimensions:
  1. Success Rate  — binary task completion metric.
  2. Recovery Rate — recovery from injected faults.
  3. Efficiency    — resource usage (token/step/time).
  4. Safety        — safety boundary violations.
  5. Consistency   — variance across repeated runs of the same task.

Dependencies: numpy, scipy.stats.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats

# ── Dimension definitions ─────────────────────────────────

DIMENSIONS = ["success", "recovery", "efficiency", "safety", "consistency"]
DIMENSION_DOCS = {
    "success":     "成功率: 任务完成的二元指标 (0/1 均值)。1.0 = 全部成功",
    "recovery":    "恢复率: 注入故障后恢复的比例。1.0 = 全部恢复",
    "efficiency":  "效率: 归一化资源效率 (1 - used/budget)。1.0 = 几乎不耗资源",
    "safety":      "安全性: 未触发安全边界的比例。1.0 = 从不违规",
    "consistency": "一致性: 多次运行方差的归一化 (1 - std/max_std)。1.0 = 完全稳定",
}

MERGE_THRESHOLD = 0.7   # >0.7 suggests merging dimensions
WARN_THRESHOLD = 0.3    # ideal tension: inter-dimension correlation should be <0.3

# ── Data structures ───────────────────────────────────────


@dataclass
class TaskResult:
    """Result of a single task execution."""

    task_id: str
    agent_id: str
    success: bool
    recovered: bool | None = None
    tokens_used: int = 0
    steps: int = 0
    time_seconds: float = 0.0
    safety_violations: int = 0
    run_group: str | None = None


@dataclass
class DimensionScore:
    """Score for a single dimension."""

    name: str
    score: float          # 0-1 normalised
    raw_value: float
    detail: str


@dataclass
class AgentEvaluation:
    """Full evaluation for a single agent/configuration."""

    agent_id: str
    dimensions: dict[str, DimensionScore] = field(default_factory=dict)
    sample_count: int = 0

    def score_vector(self) -> np.ndarray:
        """Return dimension score vector (ordered by DIMENSIONS)."""
        return np.array([self.dimensions[d].score for d in DIMENSIONS])


@dataclass
class MatrixReport:
    """Complete evaluation report."""

    agents: dict[str, AgentEvaluation] = field(default_factory=dict)
    correlation_matrix: np.ndarray | None = None
    merge_warnings: list[str] = field(default_factory=list)
    goodhart_analysis: dict[str, str] = field(default_factory=dict)


# ── Dimension calculators ─────────────────────────────────


def _calc_success(results: list[TaskResult], budget: dict) -> DimensionScore:
    """Success rate: fraction of tasks with success=True."""
    if not results:
        return DimensionScore("success", 0.0, 0.0, "无数据")
    raw = sum(1 for r in results if r.success) / len(results)
    return DimensionScore("success", raw, raw,
        f"{raw:.1%} ({sum(1 for r in results if r.success)}/{len(results)})")


def _calc_recovery(results: list[TaskResult], budget: dict) -> DimensionScore:
    """Recovery rate: only counts tasks where faults were injected (recovered is not None)."""
    injected = [r for r in results if r.recovered is not None]
    if not injected:
        return DimensionScore("recovery", 0.0, 0.0, "无故障注入数据")
    raw = sum(1 for r in injected if r.recovered) / len(injected)
    return DimensionScore("recovery", raw, raw,
        f"{raw:.1%} ({sum(1 for r in injected if r.recovered)}/{len(injected)} 故障恢复)")


def _calc_efficiency(results: list[TaskResult], budget: dict) -> DimensionScore:
    """Efficiency: inverse-normalised resource usage. Combines token + step + time."""
    if not results:
        return DimensionScore("efficiency", 0.0, 0.0, "无数据")
    tok_b = max(budget.get("max_tokens", 1), 1)
    step_b = max(budget.get("max_steps", 1), 1)
    time_b = max(budget.get("max_time", 1.0), 1.0)
    effs = []
    for r in results:
        tok_ratio = min(r.tokens_used / tok_b, 1.0)
        step_ratio = min(r.steps / step_b, 1.0)
        time_ratio = min(r.time_seconds / time_b, 1.0)
        eff = 1.0 - (tok_ratio + step_ratio + time_ratio) / 3.0
        effs.append(eff)
    raw = float(np.mean(effs))
    return DimensionScore("efficiency", raw, raw,
        f"平均效率 {raw:.1%} (token budget={tok_b}, step budget={step_b})")


def _calc_safety(results: list[TaskResult], budget: dict) -> DimensionScore:
    """Safety: fraction of results with zero safety violations."""
    if not results:
        return DimensionScore("safety", 0.0, 0.0, "无数据")
    clean = sum(1 for r in results if r.safety_violations == 0)
    raw = clean / len(results)
    return DimensionScore("safety", raw, raw,
        f"{raw:.1%} 无违规 ({clean}/{len(results)}), "
        f"总违规 {sum(r.safety_violations for r in results)} 次")


def _calc_consistency(results: list[TaskResult], budget: dict) -> DimensionScore:
    """Consistency: inverse-normalised variance of success rate within run groups."""
    groups: dict[str, list[bool]] = {}
    for r in results:
        g = r.run_group or r.task_id
        groups.setdefault(g, []).append(r.success)
    multi_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not multi_groups:
        return DimensionScore("consistency", 0.0, 0.0, "无多次运行数据")
    group_stds = []
    for vals in multi_groups.values():
        vals_f = [float(v) for v in vals]
        if len(vals_f) >= 2:
            group_stds.append(np.std(vals_f))
    if not group_stds:
        return DimensionScore("consistency", 0.0, 0.0, "无法计算方差")
    avg_std = float(np.mean(group_stds))
    max_std = 0.5  # maximum std for binary variables
    score = 1.0 - min(avg_std / max_std, 1.0)
    return DimensionScore("consistency", score, avg_std,
        f"平均标准差 {avg_std:.3f} (跨 {len(multi_groups)} 组)")


_CALCULATORS = {
    "success": _calc_success,
    "recovery": _calc_recovery,
    "efficiency": _calc_efficiency,
    "safety": _calc_safety,
    "consistency": _calc_consistency,
}


# ── Core engine ───────────────────────────────────────────


def evaluate_agent(
    results: list[TaskResult],
    budget: dict | None = None,
    agent_id: str | None = None,
) -> AgentEvaluation:
    """
    Compute five-dimension scores for a single agent.

    Args:
        results: All task results for this agent.
        budget: Resource budget {"max_tokens": N, "max_steps": N, "max_time": S}.
        agent_id: Agent identifier (inferred from results if omitted).
    """
    budget = budget or {}
    aid = agent_id or (results[0].agent_id if results else "unknown")
    ev = AgentEvaluation(agent_id=aid, sample_count=len(results))
    for dim in DIMENSIONS:
        ev.dimensions[dim] = _CALCULATORS[dim](results, budget)
    return ev


def compute_correlation(agent_evals: list[AgentEvaluation]) -> np.ndarray:
    """
    Compute inter-dimension Spearman rank correlation matrix.

    Each agent's dimension scores form a row; correlation is computed across agents.
    Uses Spearman (rank-based): more robust for non-linear relations and small samples.
    Requires ≥ 5 agents for statistical significance; fewer triggers a warning.
    """
    n_agents = len(agent_evals)
    if n_agents < 2:
        return np.eye(len(DIMENSIONS))
    if n_agents < 5:
        warnings.warn(
            f"相关性矩阵仅基于 {n_agents} 个 Agent，统计意义有限"
            f"（建议 ≥ 5）"
        )
    matrix = np.array([ev.score_vector() for ev in agent_evals])
    n = len(DIMENSIONS)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
            else:
                if n_agents < 3:
                    r = np.corrcoef(matrix[:, i], matrix[:, j])[0, 1]
                    r = 0.0 if math.isnan(r) else r
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", sp_stats.ConstantInputWarning)
                        r, _ = sp_stats.spearmanr(matrix[:, i], matrix[:, j])
                        r = 0.0 if math.isnan(r) else r
                corr[i, j] = r
    return corr


def check_merge_warnings(corr: np.ndarray) -> list[str]:
    """Check inter-dimension correlations and return merge warnings."""
    warnings_list = []
    for i in range(len(DIMENSIONS)):
        for j in range(i + 1, len(DIMENSIONS)):
            r = corr[i, j]
            if abs(r) > MERGE_THRESHOLD:
                warnings_list.append(
                    f"⚠️ {DIMENSIONS[i]} ↔ {DIMENSIONS[j]} 相关性 {r:.2f} > {MERGE_THRESHOLD}"
                    f"，建议合并（无张力 = 无效维度）"
                )
            elif abs(r) > WARN_THRESHOLD:
                warnings_list.append(
                    f"⚡ {DIMENSIONS[i]} ↔ {DIMENSIONS[j]} 相关性 {r:.2f}"
                    f"，张力不足（理想 < {WARN_THRESHOLD}）"
                )
    return warnings_list


def analyze_goodhart_surface(
    corr: np.ndarray,
    agent_evals: list[AgentEvaluation],
) -> dict[str, str]:
    """
    Assess Goodhart attack surface: optimisable direction + tension constraints.

    If a dimension has low correlation with others (high tension),
    optimising it alone would hurt other dimensions → Goodhart is not economical.
    """
    analysis = {}
    for i, dim in enumerate(DIMENSIONS):
        other_corrs = [abs(corr[i, j]) for j in range(len(DIMENSIONS)) if j != i]
        avg_corr = float(np.mean(other_corrs)) if other_corrs else 0.0
        tension = 1.0 - avg_corr
        scores = [ev.dimensions[dim].score for ev in agent_evals]
        avg_score = float(np.mean(scores)) if scores else 0.0
        headroom = 1.0 - avg_score
        if tension > 0.7:
            risk = "低（高张力约束，单独优化不经济）"
        elif tension > 0.4:
            risk = "中（部分张力，需关注联动效应）"
        else:
            risk = "高（低张力，可被单独 hack）"
        analysis[dim] = (
            f"平均分 {avg_score:.2f}, 可优化空间 {headroom:.2f}, "
            f"平均张力 {tension:.2f}, Goodhart 风险: {risk}"
        )
    return analysis


def generate_report(
    all_results: list[TaskResult],
    budget: dict | None = None,
) -> MatrixReport:
    """
    Generate a complete evaluation report.

    Args:
        all_results: All task results across all agents.
        budget: Resource budget.
    Returns:
        MatrixReport with per-agent scores, correlation, warnings, Goodhart analysis.
    """
    budget = budget or {}
    report = MatrixReport()

    by_agent: dict[str, list[TaskResult]] = {}
    for r in all_results:
        by_agent.setdefault(r.agent_id, []).append(r)

    agent_evals = []
    for aid, results in sorted(by_agent.items()):
        ev = evaluate_agent(results, budget, aid)
        report.agents[aid] = ev
        agent_evals.append(ev)

    if len(agent_evals) >= 2:
        report.correlation_matrix = compute_correlation(agent_evals)
        report.merge_warnings = check_merge_warnings(report.correlation_matrix)
        report.goodhart_analysis = analyze_goodhart_surface(
            report.correlation_matrix, agent_evals
        )

    return report


# ── Formatting ────────────────────────────────────────────


def format_report(report: MatrixReport) -> str:
    """Generate a human-readable evaluation report."""
    lines = ["=" * 60, "多维评测指标矩阵报告", "=" * 60]

    for aid, ev in report.agents.items():
        lines.append(f"\n■ Agent: {aid} (n={ev.sample_count})")
        for dim in DIMENSIONS:
            ds = ev.dimensions[dim]
            bar = "█" * int(ds.score * 20) + "░" * (20 - int(ds.score * 20))
            lines.append(f"  {dim:12s} {ds.score:5.1%} {bar}  {ds.detail}")

    if len(report.agents) >= 2:
        lines.append(f"\n{'─' * 50}")
        lines.append("横向对比 (维度 × Agent):")
        header = f"  {'维度':12s}" + "".join(f"  {aid:>10s}" for aid in report.agents)
        lines.append(header)
        for dim in DIMENSIONS:
            row = f"  {dim:12s}"
            for aid in report.agents:
                s = report.agents[aid].dimensions[dim].score
                row += f"  {s:>10.1%}"
            lines.append(row)

    if report.correlation_matrix is not None:
        lines.append(f"\n{'─' * 50}")
        lines.append("维度间相关性矩阵 (Spearman):")
        header = f"  {'':12s}" + "".join(f"  {d:>10s}" for d in DIMENSIONS)
        lines.append(header)
        for i, dim in enumerate(DIMENSIONS):
            row = f"  {dim:12s}"
            for j in range(len(DIMENSIONS)):
                row += f"  {report.correlation_matrix[i, j]:>10.2f}"
            lines.append(row)

    if report.merge_warnings:
        lines.append(f"\n{'─' * 50}")
        lines.append("张力检查:")
        for w in report.merge_warnings:
            lines.append(f"  {w}")
    else:
        lines.append(f"\n  ✅ 所有维度间张力充足（相关性 < {WARN_THRESHOLD}）")

    if report.goodhart_analysis:
        lines.append(f"\n{'─' * 50}")
        lines.append("Goodhart 攻击面评估:")
        for dim, txt in report.goodhart_analysis.items():
            lines.append(f"  {dim:12s} {txt}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
