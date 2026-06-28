"""
Tests for reasoning_path_impact.py + review_depth_ladder.py
熔炉#111 卡193 + 熔炉#112 卡198 / ADR-356 + ADR-357
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reasoning_path_impact import (
    ReasoningMode, ReasoningDepth, EvalResult,
    RPIAReport, CategoryBreakdown, compute_breakdown, run_rpia,
)
from review_depth_ladder import (
    ReviewLevel, classify_task, ReviewRecord, LeverageTracker,
)

L0 = ReviewLevel.L0_SIGN_ONLY
L1 = ReviewLevel.L1_READ_SUMMARY
L2 = ReviewLevel.L2_VERIFY_KEY
L3 = ReviewLevel.L3_LINE_BY_LINE
L4 = ReviewLevel.L4_NO_AI


# ---- RPIA Tests (卡193) ----

def test_reasoning_mode_auto_label():
    deep = ReasoningMode(thinking=True)
    assert deep.depth == ReasoningDepth.DEEP
    assert deep.label == "deep"
    quick = ReasoningMode(thinking=False)
    assert quick.depth == ReasoningDepth.QUICK
    assert quick.label == "quick"
    budgeted = ReasoningMode(thinking=True, budget_tokens=1024)
    assert budgeted.depth == ReasoningDepth.BUDGETED


def test_compute_breakdown_no_regression():
    old = [
        EvalResult("t1", "math", True, 1000, 100),
        EvalResult("t2", "math", True, 1100, 110),
        EvalResult("t3", "logic", True, 900, 90),
    ]
    new = [
        EvalResult("t1", "math", True, 500, 100),
        EvalResult("t2", "math", True, 550, 110),
        EvalResult("t3", "logic", True, 450, 90),
    ]
    breakdowns = compute_breakdown(old, new)
    assert len(breakdowns) == 2
    for cb in breakdowns:
        assert cb.delta == 0.0


def test_compute_breakdown_with_regression():
    old = [
        EvalResult("t1", "math", True, 1000, 100),
        EvalResult("t2", "math", True, 1100, 110),
        EvalResult("t3", "logic", True, 900, 90),
        EvalResult("t4", "logic", True, 950, 95),
    ]
    new = [
        EvalResult("t1", "math", False, 500, 100),
        EvalResult("t2", "math", True, 550, 110),
        EvalResult("t3", "logic", True, 450, 90),
        EvalResult("t4", "logic", True, 475, 95),
    ]
    breakdowns = compute_breakdown(old, new)
    math_cb = next(cb for cb in breakdowns if cb.category == "math")
    logic_cb = next(cb for cb in breakdowns if cb.category == "logic")
    assert math_cb.delta == -0.5
    assert logic_cb.delta == 0.0


def test_rpia_max_regression():
    old = [
        EvalResult("t1", "math", True, 3000, 200),
        EvalResult("t2", "dialogue", True, 1000, 50),
    ]
    new = [
        EvalResult("t1", "math", False, 1000, 200),
        EvalResult("t2", "dialogue", True, 1000, 50),
    ]
    report = run_rpia(old, new, ReasoningMode(True), ReasoningMode(False))
    assert report.max_regression == 1.0
    assert report.flag_for_disclosure() is True


def test_rpia_no_disclosure_needed():
    old = [EvalResult(f"t{i}", "dialogue", True, 1000, 50) for i in range(10)]
    new = [EvalResult(f"t{i}", "dialogue", True, 500, 50) for i in range(10)]
    report = run_rpia(old, new, ReasoningMode(True), ReasoningMode(False))
    assert report.max_regression == 0.0
    assert report.flag_for_disclosure() is False


def test_rpia_summary_format():
    old = [EvalResult("t1", "math", True, 3000, 100)]
    new = [EvalResult("t1", "math", False, 1000, 100)]
    report = run_rpia(old, new, ReasoningMode(True, "deep"), ReasoningMode(False, "quick"))
    s = report.summary()
    assert "deep" in s
    assert "quick" in s
    assert "math" in s


# ---- Review Depth Ladder Tests (卡198) ----

def test_review_level_ordering():
    assert L0 < L1
    assert L1 < L2
    assert L2 < L3
    assert L3 < L4


def test_classify_task_high_stakes():
    assert classify_task("high", "legal_opinion") == L3
    assert classify_task("critical", "court_filing") == L4
    assert classify_task("high", "medical_diagnosis") == L3


def test_classify_task_medium_stakes():
    assert classify_task("medium", "contract_draft") == L2
    assert classify_task("medium", "legal_research") == L2


def test_classify_task_low_stakes():
    assert classify_task("low", "information_query") == L1
    assert classify_task("low", "summary") == L1


def test_classify_task_default():
    assert classify_task("medium") == L2
    assert classify_task("high") == L3


def test_review_record_validate_ok():
    record = ReviewRecord("t1", L2, "lawyer_zhang")
    assert record.validate(min_level=L2) is True


def test_review_record_validate_insufficient():
    record = ReviewRecord("t1", L1, "lawyer_zhang")
    assert record.validate(min_level=L2) is False


def test_review_record_l0_never_valid():
    record = ReviewRecord("t1", L0, "lawyer_zhang")
    assert record.validate() is False
    assert record.is_form_compliance_only() is True


def test_review_record_l1_form_compliance():
    record = ReviewRecord("t1", L1, "lawyer_zhang")
    assert record.is_form_compliance_only() is True


def test_leverage_tracker():
    tracker = LeverageTracker(reviewer="zhang", max_warning_ratio=5, max_safe_ratio=10)
    assert tracker.review_count == 0
    for i in range(3):
        tracker.add_review(ReviewRecord(f"t{i}", L2, "zhang"))
    assert tracker.current_leverage == 3
    assert tracker.status == "safe"
    for i in range(5):
        tracker.add_review(ReviewRecord(f"t{i+3}", L2, "zhang"))
    assert tracker.current_leverage == 8
    assert "approaching" in tracker.status
    for i in range(5):
        tracker.add_review(ReviewRecord(f"t{i+8}", L2, "zhang"))
    assert tracker.current_leverage == 13
    assert "over" in tracker.status


def test_leverage_summary():
    tracker = LeverageTracker(reviewer="lawyer_li")
    s = tracker.summary()
    assert "lawyer_li" in s
    assert "daily" in s
