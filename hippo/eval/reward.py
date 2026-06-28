"""
Reward Rule Evaluator — deterministic reward shaping assessment.

Replaces LLM-based critic with rule-driven dimension scoring for reward design
evaluation.  Designed for small models that cannot run PPO: instead of training
a critic, we *evaluate* whether a reward design is sound.

Core capabilities:
  1. Deterministic multi-dimension scoring (numpy only, no LLM calls).
  2. Sparsity detection — fraction of zero-score signals (key PPO motivator).
  3. Conflict detection — Pearson correlation to flag reward-hacking patterns.
  4. Design comparison — compare two reward configs and recommend the better one.
  5. Distribution shaping — histogram stats (min/max/mean/std/percentiles).

Dependencies: numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ── Thresholds ────────────────────────────────────────────

_SPARSITY_WARN = 0.40      # >40% zero-scores → sparse reward warning
_SPARSITY_CRITICAL = 0.60  # >60% → critical sparsity
_CONFLICT_THRESHOLD = -0.5  # correlation < -0.5 → conflict warning
_LOW_SCORE_WARN = 0.30     # any dimension below this → warning


# ── Data structures ───────────────────────────────────────


@dataclass
class RewardDimension:
    """A single dimension of the reward signal."""

    name: str
    weight: float
    min_score: float = 0.0
    max_score: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")
        if self.max_score <= self.min_score:
            raise ValueError(
                f"max_score ({self.max_score}) must exceed min_score ({self.min_score})"
            )


@dataclass
class RewardSignal:
    """A complete reward signal from one episode/step."""

    agent_id: str
    dimensions: dict[str, float]
    metadata: dict = field(default_factory=dict)


@dataclass
class RewardEvaluation:
    """Result of evaluating one or more reward signals."""

    signal: RewardSignal
    total_score: float
    dimension_scores: dict[str, float]
    warnings: list[str]
    goodhart_risk: str  # "low" / "medium" / "high"


@dataclass
class RewardConfig:
    """Reward design configuration: which dimensions and their weights."""

    dimensions: list[RewardDimension] = field(default_factory=lambda: [
        RewardDimension("correctness", 0.35),
        RewardDimension("efficiency", 0.20),
        RewardDimension("safety", 0.25),
        RewardDimension("recoverability", 0.20),
    ])

    def __post_init__(self) -> None:
        total = sum(d.weight for d in self.dimensions)
        if total <= 0:
            raise ValueError("Total weight must be positive")
        # normalise weights so they sum to 1.0
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            for d in self.dimensions:
                d.weight /= total

    @property
    def dim_map(self) -> dict[str, RewardDimension]:
        return {d.name: d for d in self.dimensions}


# ── Core evaluator ────────────────────────────────────────


class RewardEvaluator:
    """
    Deterministic reward rule evaluator.

    Usage::

        config = RewardConfig()
        ev = RewardEvaluator(config)
        result = ev.evaluate(signal)
        comparison = ev.compare_designs(config_a, config_b, signals)
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    # ── Single-signal evaluation ──────────────────────────

    def evaluate(self, signal: RewardSignal) -> RewardEvaluation:
        """Score a single reward signal against the configured dimensions."""
        dimension_scores: dict[str, float] = {}
        warnings: list[str] = []

        for dim in self.config.dimensions:
            raw = signal.dimensions.get(dim.name, 0.0)
            # clamp to [min_score, max_score]
            clamped = max(dim.min_score, min(dim.max_score, raw))
            normalised = (clamped - dim.min_score) / (dim.max_score - dim.min_score)
            dimension_scores[dim.name] = normalised

            if normalised < _LOW_SCORE_WARN:
                warnings.append(
                    f"维度 '{dim.name}' 得分 {normalised:.2f} 低于警告线 {_LOW_SCORE_WARN}"
                )

        total = self._weighted_total(dimension_scores)

        goodhart = self._assess_goodhart(dimension_scores, warnings)

        return RewardEvaluation(
            signal=signal,
            total_score=total,
            dimension_scores=dimension_scores,
            warnings=warnings,
            goodhart_risk=goodhart,
        )

    # ── Batch evaluation ──────────────────────────────────

    def evaluate_batch(
        self, signals: list[RewardSignal]
    ) -> list[RewardEvaluation]:
        """Evaluate multiple reward signals."""
        return [self.evaluate(s) for s in signals]

    # ── Sparsity detection ────────────────────────────────

    def sparsity(self, signals: list[RewardSignal]) -> float:
        """
        Fraction of dimension-scores that are exactly zero across all signals.

        High sparsity (>0.4) means the reward signal is mostly silent — this is
        the core reason GLM-5.2 reverted to PPO (critic can do credit assignment
        on sparse signals; rule-based systems cannot).
        """
        if not signals:
            return 1.0
        dim_names = [d.name for d in self.config.dimensions]
        total_cells = len(signals) * len(dim_names)
        zero_cells = 0
        for s in signals:
            for name in dim_names:
                if s.dimensions.get(name, 0.0) == 0.0:
                    zero_cells += 1
        return zero_cells / total_cells if total_cells else 1.0

    def sparsity_report(self, signals: list[RewardSignal]) -> dict:
        """Detailed sparsity breakdown per dimension."""
        dim_names = [d.name for d in self.config.dimensions]
        per_dim: dict[str, float] = {}
        for name in dim_names:
            vals = [s.dimensions.get(name, 0.0) for s in signals]
            zeros = sum(1 for v in vals if v == 0.0)
            per_dim[name] = zeros / len(vals) if vals else 1.0
        overall = self.sparsity(signals)
        level = "ok"
        if overall > _SPARSITY_CRITICAL:
            level = "critical"
        elif overall > _SPARSITY_WARN:
            level = "warning"
        return {
            "overall_sparsity": overall,
            "per_dimension": per_dim,
            "level": level,
        }

    # ── Conflict detection ────────────────────────────────

    def conflict_detection(
        self, signals: list[RewardSignal]
    ) -> dict[str, float | list[str]]:
        """
        Detect inter-dimension conflicts via Pearson correlation.

        If two dimensions are strongly negatively correlated (e.g. efficiency↑
        but safety↓), the agent can hack one at the expense of the other.
        Returns a dict with conflict pairs and warnings.
        """
        dim_names = [d.name for d in self.config.dimensions]
        n_dims = len(dim_names)
        warnings: list[str] = []
        conflicts: dict[str, float] = {}

        if len(signals) < 3:
            return {"conflicts": {}, "warnings": ["样本数 <3，冲突检测不可靠"]}

        # Build matrix: rows=signals, cols=dimensions
        matrix = np.array(
            [[s.dimensions.get(name, 0.0) for name in dim_names] for s in signals]
        )

        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                col_i = matrix[:, i]
                col_j = matrix[:, j]
                # guard against zero variance
                if np.std(col_i) < 1e-9 or np.std(col_j) < 1e-9:
                    continue
                r = float(np.corrcoef(col_i, col_j)[0, 1])
                if math.isnan(r):
                    continue
                if r < _CONFLICT_THRESHOLD:
                    pair = f"{dim_names[i]} ↔ {dim_names[j]}"
                    conflicts[pair] = r
                    warnings.append(
                        f"⚠️ 冲突: {pair} 相关系数 {r:.2f} < {_CONFLICT_THRESHOLD}"
                        f" → 可能存在 reward hacking"
                    )
        return {"conflicts": conflicts, "warnings": warnings}

    # ── Design comparison ─────────────────────────────────

    def compare_designs(
        self,
        config_a: RewardConfig,
        config_b: RewardConfig,
        signals: list[RewardSignal],
    ) -> dict:
        """
        Compare two reward configs on the same signal set.

        Returns which design produces better score distribution (higher mean,
        lower variance, lower sparsity).
        """
        ev_a = RewardEvaluator(config_a)
        ev_b = RewardEvaluator(config_b)

        results_a = ev_a.evaluate_batch(signals)
        results_b = ev_b.evaluate_batch(signals)

        scores_a = [r.total_score for r in results_a]
        scores_b = [r.total_score for r in results_b]

        stats_a = self._distribution_stats(scores_a)
        stats_b = self._distribution_stats(scores_b)

        sparsity_a = ev_a.sparsity(signals)
        sparsity_b = ev_b.sparsity(signals)

        # Simple decision: higher mean, lower std, lower sparsity = better
        score_a = stats_a["mean"] - 0.3 * stats_a["std"] - 0.5 * sparsity_a
        score_b = stats_b["mean"] - 0.3 * stats_b["std"] - 0.5 * sparsity_b

        if score_a > score_b:
            recommendation = "config_a"
        elif score_b > score_a:
            recommendation = "config_b"
        else:
            recommendation = "tie"

        return {
            "config_a": {
                "stats": stats_a,
                "sparsity": sparsity_a,
                "composite_score": score_a,
            },
            "config_b": {
                "stats": stats_b,
                "sparsity": sparsity_b,
                "composite_score": score_b,
            },
            "recommendation": recommendation,
            "difference": abs(score_a - score_b),
        }

    # ── Distribution / shaping analysis ───────────────────

    def shaping_analysis(
        self, signals: list[RewardSignal]
    ) -> dict:
        """
        Return reward distribution statistics for shaping analysis.

        Computes per-dimension and overall stats: min, max, mean, std,
        percentiles (p25/p50/p75/p90).
        """
        dim_names = [d.name for d in self.config.dimensions]
        per_dim: dict[str, dict] = {}

        for name in dim_names:
            vals = [s.dimensions.get(name, 0.0) for s in signals]
            per_dim[name] = self._distribution_stats(vals)

        # Overall weighted scores
        totals = [self.evaluate(s).total_score for s in signals]
        overall = self._distribution_stats(totals)

        return {
            "per_dimension": per_dim,
            "overall": overall,
            "sample_count": len(signals),
        }

    def histogram(
        self, signals: list[RewardSignal], bins: int = 10
    ) -> dict:
        """Return histogram data for total reward scores."""
        totals = [self.evaluate(s).total_score for s in signals]
        if not totals:
            return {"bin_edges": [], "counts": [], "sample_count": 0}
        counts, edges = np.histogram(totals, bins=bins, range=(0.0, 1.0))
        return {
            "bin_edges": edges.tolist(),
            "counts": counts.tolist(),
            "sample_count": len(totals),
        }

    # ── Internal helpers ──────────────────────────────────

    def _weighted_total(self, dimension_scores: dict[str, float]) -> float:
        total = 0.0
        for dim in self.config.dimensions:
            total += dim.weight * dimension_scores.get(dim.name, 0.0)
        return total

    def _assess_goodhart(
        self, dimension_scores: dict[str, float], warnings: list[str]
    ) -> str:
        """Assess Goodhart's law risk for a single evaluation."""
        if not dimension_scores:
            return "high"
        values = list(dimension_scores.values())
        spread = max(values) - min(values)
        # If everything is zero, the reward signal is useless
        if max(values) < 1e-9:
            return "high"
        # If one dimension is near-perfect while others are low → hackable
        if spread > 0.6 and max(values) > 0.8:
            return "high"
        if spread > 0.4 or len(warnings) >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _distribution_stats(values: list[float]) -> dict:
        """Compute distribution statistics for a list of values."""
        if not values:
            return {
                "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
                "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0,
            }
        arr = np.array(values)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
        }
