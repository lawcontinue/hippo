"""Tests for hippo.eval.matrix."""

import numpy as np
import pytest

from hippo.eval.matrix import (
    TaskResult,
    DIMENSIONS,
    MERGE_THRESHOLD,
    WARN_THRESHOLD,
    evaluate_agent,
    compute_correlation,
    check_merge_warnings,
    analyze_goodhart_surface,
    generate_report,
    format_report,
)


def _make_result(
    task_id="t1", agent_id="A", success=True, recovered=None,
    tokens=1000, steps=10, time_s=5.0, violations=0, group=None,
):
    return TaskResult(
        task_id=task_id, agent_id=agent_id, success=success,
        recovered=recovered, tokens_used=tokens, steps=steps,
        time_seconds=time_s, safety_violations=violations,
        run_group=group,
    )


BUDGET = {"max_tokens": 10000, "max_steps": 50, "max_time": 60.0}


def test_success_rate_basic():
    results = [
        _make_result(task_id="t1", success=True),
        _make_result(task_id="t2", success=True),
        _make_result(task_id="t3", success=False),
        _make_result(task_id="t4", success=False),
    ]
    ev = evaluate_agent(results, BUDGET, "A")
    assert ev.dimensions["success"].score == pytest.approx(0.5)


def test_success_rate_all_pass():
    results = [_make_result(task_id=f"t{i}", success=True) for i in range(5)]
    ev = evaluate_agent(results, BUDGET)
    assert ev.dimensions["success"].score == 1.0


def test_recovery_rate_with_injection():
    results = [
        _make_result(task_id="t1", recovered=True),
        _make_result(task_id="t2", recovered=False),
        _make_result(task_id="t3", recovered=True),
        _make_result(task_id="t4", recovered=None),
    ]
    ev = evaluate_agent(results, BUDGET)
    assert ev.dimensions["recovery"].score == pytest.approx(2 / 3)


def test_recovery_rate_no_injection():
    results = [_make_result(recovered=None)]
    ev = evaluate_agent(results, BUDGET)
    assert ev.dimensions["recovery"].score == 0.0
    assert "无故障注入" in ev.dimensions["recovery"].detail


def test_efficiency_zero_resource():
    results = [_make_result(tokens=0, steps=0, time_s=0.0)]
    ev = evaluate_agent(results, BUDGET)
    assert ev.dimensions["efficiency"].score == pytest.approx(1.0)


def test_efficiency_full_resource():
    results = [_make_result(tokens=10000, steps=50, time_s=60.0)]
    ev = evaluate_agent(results, BUDGET)
    assert ev.dimensions["efficiency"].score == pytest.approx(0.0)


def test_safety_no_violations():
    results = [_make_result(violations=0) for _ in range(3)]
    ev = evaluate_agent(results, BUDGET)
    assert ev.dimensions["safety"].score == 1.0


def test_safety_with_violations():
    results = [
        _make_result(violations=0),
        _make_result(violations=2),
        _make_result(violations=0),
        _make_result(violations=1),
    ]
    ev = evaluate_agent(results, BUDGET)
    assert ev.dimensions["safety"].score == 0.5


def test_consistency_identical_runs():
    results = [_make_result(task_id="t1", success=True, group="g1") for _ in range(5)]
    ev = evaluate_agent(results, BUDGET)
    assert ev.dimensions["consistency"].score == pytest.approx(1.0)


def test_consistency_random_runs():
    results = (
        [_make_result(task_id="t1", success=True, group="g1") for _ in range(5)]
        + [_make_result(task_id="t1", success=False, group="g1") for _ in range(5)]
    )
    ev = evaluate_agent(results, BUDGET)
    assert ev.dimensions["consistency"].score < 0.5


def test_correlation_matrix_shape():
    agents_results = []
    for i in range(4):
        results = [_make_result(
            agent_id=f"A{i}",
            success=bool(i % 2),
            tokens=1000 * (i + 1),
            violations=i,
            group=f"g{i}",
        )]
        agents_results.extend(results)
    by_agent = {}
    for r in agents_results:
        by_agent.setdefault(r.agent_id, []).append(r)
    evals = [evaluate_agent(rs, BUDGET, aid) for aid, rs in by_agent.items()]
    corr = compute_correlation(evals)
    assert corr.shape == (5, 5)
    for i in range(5):
        assert corr[i, i] == pytest.approx(1.0)


def test_correlation_high():
    agent_evals = []
    for i in range(5):
        ev = evaluate_agent(
            [_make_result(success=True if i < 3 else False, violations=0 if i < 3 else 5)],
            BUDGET, f"A{i}",
        )
        agent_evals.append(ev)
    corr = compute_correlation(agent_evals)
    s_idx = DIMENSIONS.index("success")
    sf_idx = DIMENSIONS.index("safety")
    assert corr[s_idx, sf_idx] > 0.8


def test_merge_warning_triggered():
    corr = np.eye(5)
    corr[0, 1] = corr[1, 0] = 0.85
    warnings = check_merge_warnings(corr)
    assert any("建议合并" in w for w in warnings)


def test_no_warning_when_tension_ok():
    corr = np.eye(5) * 0.1
    np.fill_diagonal(corr, 1.0)
    warnings = check_merge_warnings(corr)
    assert len(warnings) == 0


def test_goodhart_analysis_output():
    agent_evals = [
        evaluate_agent([_make_result(agent_id=f"A{i}")], BUDGET, f"A{i}")
        for i in range(3)
    ]
    for i, ev in enumerate(agent_evals):
        for dim in DIMENSIONS:
            ev.dimensions[dim].score = (i + 1) / 5
    corr = np.eye(5) * 0.1
    np.fill_diagonal(corr, 1.0)
    analysis = analyze_goodhart_surface(corr, agent_evals)
    assert len(analysis) == 5
    for dim, txt in analysis.items():
        assert "Goodhart 风险" in txt


def test_generate_report_multi_agent():
    all_results = []
    for aid in ["alpha", "beta", "gamma"]:
        for tid in range(10):
            all_results.append(_make_result(
                task_id=f"t{tid}", agent_id=aid,
                success=(tid % 3 != 0),
                recovered=(True if tid % 4 == 0 else None),
                tokens=500 + tid * 200,
                steps=5 + tid,
                time_s=2.0 + tid * 0.5,
                violations=(1 if tid == 9 else 0),
                group=f"g{tid % 3}",
            ))
    report = generate_report(all_results, BUDGET)
    assert len(report.agents) == 3
    assert report.correlation_matrix is not None
    assert report.correlation_matrix.shape == (5, 5)
    text = format_report(report)
    assert "多维评测指标矩阵报告" in text
    assert "alpha" in text
    assert "相关性矩阵" in text


def test_generate_report_single_agent():
    results = [_make_result(agent_id="solo")]
    report = generate_report(results, BUDGET)
    assert len(report.agents) == 1
    assert report.correlation_matrix is None


def test_empty_results():
    ev = evaluate_agent([], BUDGET, "empty")
    for dim in DIMENSIONS:
        assert ev.dimensions[dim].score == 0.0


def test_score_vector_order():
    results = [_make_result()]
    ev = evaluate_agent(results, BUDGET)
    vec = ev.score_vector()
    assert len(vec) == 5
    for i, dim in enumerate(DIMENSIONS):
        assert vec[i] == ev.dimensions[dim].score
