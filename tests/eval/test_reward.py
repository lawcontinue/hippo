"""Tests for hippo.eval.reward — deterministic reward rule evaluator."""

import math

import numpy as np
import pytest

from hippo.eval.reward import (
    RewardConfig,
    RewardDimension,
    RewardEvaluation,
    RewardEvaluator,
    RewardSignal,
)


# ── Helpers ───────────────────────────────────────────────


def _signal(
    agent_id: str = "agent_a",
    correctness: float = 0.8,
    efficiency: float = 0.7,
    safety: float = 0.9,
    recoverability: float = 0.6,
) -> RewardSignal:
    return RewardSignal(
        agent_id=agent_id,
        dimensions={
            "correctness": correctness,
            "efficiency": efficiency,
            "safety": safety,
            "recoverability": recoverability,
        },
    )


def _default_evaluator() -> RewardEvaluator:
    return RewardEvaluator()


# ── 1. Basic scoring ──────────────────────────────────────


def test_basic_scoring():
    """Weighted total score matches manual calculation."""
    ev = _default_evaluator()
    sig = _signal(correctness=1.0, efficiency=1.0, safety=1.0, recoverability=1.0)
    result = ev.evaluate(sig)
    assert result.total_score == pytest.approx(1.0)
    assert all(v == 1.0 for v in result.dimension_scores.values())


def test_basic_scoring_all_zero():
    """All-zero signal produces zero total score."""
    ev = _default_evaluator()
    sig = _signal(correctness=0.0, efficiency=0.0, safety=0.0, recoverability=0.0)
    result = ev.evaluate(sig)
    assert result.total_score == pytest.approx(0.0)
    assert all(v == 0.0 for v in result.dimension_scores.values())


def test_basic_scoring_partial():
    """Partial scores produce correct weighted sum."""
    ev = _default_evaluator()
    sig = _signal(correctness=0.5, efficiency=0.5, safety=0.5, recoverability=0.5)
    result = ev.evaluate(sig)
    # All dimensions 0.5 → total should be 0.5 regardless of weights
    assert result.total_score == pytest.approx(0.5, abs=1e-6)


# ── 2. Weight configuration ───────────────────────────────


def test_weight_normalisation():
    """Weights not summing to 1.0 are auto-normalised."""
    dims = [
        RewardDimension("a", 2.0),
        RewardDimension("b", 2.0),
        RewardDimension("c", 1.0),
    ]
    config = RewardConfig(dimensions=dims)
    total = sum(d.weight for d in config.dimensions)
    assert total == pytest.approx(1.0)
    assert config.dimensions[0].weight == pytest.approx(0.4)
    assert config.dimensions[2].weight == pytest.approx(0.2)


def test_custom_weights_affect_total():
    """Different weight configs produce different totals for the same signal."""
    dims_a = [
        RewardDimension("x", 0.9),
        RewardDimension("y", 0.1),
    ]
    dims_b = [
        RewardDimension("x", 0.1),
        RewardDimension("y", 0.9),
    ]
    sig = RewardSignal(
        agent_id="t",
        dimensions={"x": 1.0, "y": 0.0},
    )
    ev_a = RewardEvaluator(RewardConfig(dimensions=dims_a))
    ev_b = RewardEvaluator(RewardConfig(dimensions=dims_b))
    assert ev_a.evaluate(sig).total_score > ev_b.evaluate(sig).total_score


def test_invalid_weight_raises():
    """Negative weight raises ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        RewardDimension("bad", -0.5)


# ── 3. Sparsity detection ─────────────────────────────────


def test_sparsity_all_zero():
    """All-zero signals → sparsity = 1.0."""
    ev = _default_evaluator()
    signals = [
        _signal(correctness=0, efficiency=0, safety=0, recoverability=0),
        _signal(correctness=0, efficiency=0, safety=0, recoverability=0),
    ]
    assert ev.sparsity(signals) == pytest.approx(1.0)


def test_sparsity_all_full():
    """All-max signals → sparsity = 0.0."""
    ev = _default_evaluator()
    signals = [
        _signal(correctness=1, efficiency=1, safety=1, recoverability=1),
        _signal(correctness=1, efficiency=1, safety=1, recoverability=1),
    ]
    assert ev.sparsity(signals) == pytest.approx(0.0)


def test_sparsity_report_levels():
    """Sparsity report correctly classifies severity levels."""
    ev = _default_evaluator()
    # 50% zero → warning level
    signals = [
        _signal(correctness=0, efficiency=0, safety=1, recoverability=1),
        _signal(correctness=0, efficiency=0, safety=1, recoverability=1),
    ]
    report = ev.sparsity_report(signals)
    assert report["overall_sparsity"] == pytest.approx(0.5)
    assert report["level"] == "warning"


def test_sparsity_empty_signals():
    """Empty signal list → sparsity 1.0 (vacuously sparse)."""
    ev = _default_evaluator()
    assert ev.sparsity([]) == pytest.approx(1.0)


# ── 4. Conflict detection ─────────────────────────────────


def test_conflict_detected():
    """Strong negative correlation between two dimensions → conflict flagged."""
    dims = [
        RewardDimension("efficiency", 0.5),
        RewardDimension("safety", 0.5),
    ]
    config = RewardConfig(dimensions=dims)
    ev = RewardEvaluator(config)
    # efficiency up, safety down — perfect anti-correlation
    signals = [
        RewardSignal("a", {"efficiency": 0.9, "safety": 0.1}),
        RewardSignal("a", {"efficiency": 0.8, "safety": 0.2}),
        RewardSignal("a", {"efficiency": 0.7, "safety": 0.3}),
        RewardSignal("a", {"efficiency": 0.6, "safety": 0.4}),
        RewardSignal("a", {"efficiency": 0.5, "safety": 0.5}),
    ]
    result = ev.conflict_detection(signals)
    assert len(result["conflicts"]) >= 1
    assert any("reward hacking" in w for w in result["warnings"])


def test_no_conflict_when_aligned():
    """Positively correlated dimensions → no conflict."""
    dims = [
        RewardDimension("a", 0.5),
        RewardDimension("b", 0.5),
    ]
    ev = RewardEvaluator(RewardConfig(dimensions=dims))
    signals = [
        RewardSignal("a", {"a": 0.1, "b": 0.1}),
        RewardSignal("a", {"a": 0.5, "b": 0.5}),
        RewardSignal("a", {"a": 0.9, "b": 0.9}),
        RewardSignal("a", {"a": 0.7, "b": 0.8}),
        RewardSignal("a", {"a": 0.3, "b": 0.4}),
    ]
    result = ev.conflict_detection(signals)
    assert len(result["conflicts"]) == 0


def test_conflict_insufficient_samples():
    """<3 signals → warning about unreliable conflict detection."""
    ev = _default_evaluator()
    signals = [_signal(), _signal()]
    result = ev.conflict_detection(signals)
    assert any("<3" in w for w in result["warnings"])


# ── 5. Design comparison ──────────────────────────────────


def test_compare_designs_picks_better():
    """Compare two configs on the same signals → higher mean wins."""
    dims_a = [
        RewardDimension("correctness", 0.01),
        RewardDimension("efficiency", 0.99),
    ]
    dims_b = [
        RewardDimension("correctness", 0.99),
        RewardDimension("efficiency", 0.01),
    ]
    signals = [
        RewardSignal("a", {"correctness": 0.9, "efficiency": 0.1}),
        RewardSignal("a", {"correctness": 0.8, "efficiency": 0.2}),
        RewardSignal("a", {"correctness": 0.85, "efficiency": 0.15}),
        RewardSignal("a", {"correctness": 0.95, "efficiency": 0.05}),
    ]
    ev = _default_evaluator()
    result = ev.compare_designs(
        RewardConfig(dimensions=dims_a),
        RewardConfig(dimensions=dims_b),
        signals,
    )
    # config_b weights correctness heavily, and signals have high correctness
    assert result["recommendation"] == "config_b"


def test_compare_designs_tie():
    """Identical configs produce a tie."""
    config = RewardConfig()
    ev = RewardEvaluator(config)
    signals = [_signal(), _signal(), _signal()]
    result = ev.compare_designs(config, RewardConfig(), signals)
    assert result["recommendation"] == "tie"


# ── 6. Shaping analysis & histogram ───────────────────────


def test_shaping_analysis_stats():
    """Shaping analysis returns correct distribution stats."""
    ev = _default_evaluator()
    signals = [
        _signal(correctness=0.2, efficiency=0.4, safety=0.6, recoverability=0.8),
        _signal(correctness=0.4, efficiency=0.6, safety=0.8, recoverability=1.0),
        _signal(correctness=0.1, efficiency=0.3, safety=0.5, recoverability=0.7),
        _signal(correctness=0.9, efficiency=0.9, safety=0.9, recoverability=0.9),
    ]
    shaping = ev.shaping_analysis(signals)
    assert shaping["sample_count"] == 4
    assert "per_dimension" in shaping
    assert "overall" in shaping
    overall = shaping["overall"]
    assert overall["min"] >= 0.0
    assert overall["max"] <= 1.0
    assert overall["mean"] > 0.0
    assert overall["p50"] > 0.0


def test_histogram():
    """Histogram returns correct bin structure."""
    ev = _default_evaluator()
    signals = [_signal() for _ in range(10)]
    hist = ev.histogram(signals, bins=5)
    assert len(hist["bin_edges"]) == 6  # 5 bins → 6 edges
    assert sum(hist["counts"]) == 10
    assert hist["sample_count"] == 10


def test_histogram_empty():
    """Empty signals → empty histogram."""
    ev = _default_evaluator()
    hist = ev.histogram([])
    assert hist["counts"] == []
    assert hist["sample_count"] == 0


# ── 7. Goodhart risk assessment ───────────────────────────


def test_goodhart_risk_high():
    """One dimension near-perfect, others near-zero → high Goodhart risk."""
    ev = _default_evaluator()
    sig = _signal(correctness=0.95, efficiency=0.05, safety=0.1, recoverability=0.05)
    result = ev.evaluate(sig)
    assert result.goodhart_risk == "high"


def test_goodhart_risk_low():
    """Balanced scores → low Goodhart risk."""
    ev = _default_evaluator()
    sig = _signal(correctness=0.7, efficiency=0.7, safety=0.7, recoverability=0.7)
    result = ev.evaluate(sig)
    assert result.goodhart_risk == "low"


# ── 8. Edge cases ─────────────────────────────────────────


def test_empty_dimensions_signal():
    """Signal with no matching dimensions → zero total, high risk."""
    dims = [RewardDimension("x", 1.0)]
    ev = RewardEvaluator(RewardConfig(dimensions=dims))
    sig = RewardSignal(agent_id="a", dimensions={"y": 0.9})
    result = ev.evaluate(sig)
    assert result.total_score == pytest.approx(0.0)
    assert result.goodhart_risk == "high"


def test_clamping_to_range():
    """Scores outside [min_score, max_score] are clamped."""
    dim = RewardDimension("x", 1.0, min_score=0.0, max_score=1.0)
    ev = RewardEvaluator(RewardConfig(dimensions=[dim]))
    sig = RewardSignal(agent_id="a", dimensions={"x": 5.0})
    result = ev.evaluate(sig)
    assert result.dimension_scores["x"] == pytest.approx(1.0)

    sig_low = RewardSignal(agent_id="a", dimensions={"x": -3.0})
    result_low = ev.evaluate(sig_low)
    assert result_low.dimension_scores["x"] == pytest.approx(0.0)
