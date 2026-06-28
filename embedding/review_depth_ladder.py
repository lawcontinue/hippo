"""
review_depth_ladder.py - 审查深度阶梯 L0-L4

熔炉#112 S4 成果(卡198) + ADR-357.
Python 3.14 compat: no @property inside Enum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Dict, List, Optional


class ReviewLevel(IntEnum):
    """审查深度阶梯(卡198 / ADR-357)."""
    L0_SIGN_ONLY = 0
    L1_READ_SUMMARY = 1
    L2_VERIFY_KEY = 2
    L3_LINE_BY_LINE = 3
    L4_NO_AI = 4


_LEVEL_DESCRIPTIONS = {
    0: "sign only (trust AI fully)",
    1: "read summary + conclusion",
    2: "verify key arguments + citations",
    3: "line-by-line + independent cross-check",
    4: "no AI used",
}


def level_description(level: ReviewLevel) -> str:
    return _LEVEL_DESCRIPTIONS.get(level.value, "unknown")


def is_review(level: ReviewLevel) -> bool:
    return level.value > 0


_MIN_REVIEW_LEVEL: Dict[str, ReviewLevel] = {
    "legal_opinion": ReviewLevel.L3_LINE_BY_LINE,
    "court_filing": ReviewLevel.L4_NO_AI,
    "medical_diagnosis": ReviewLevel.L3_LINE_BY_LINE,
    "surgery_plan": ReviewLevel.L4_NO_AI,
    "financial_audit": ReviewLevel.L3_LINE_BY_LINE,
    "contract_final": ReviewLevel.L3_LINE_BY_LINE,
    "regulatory_filing": ReviewLevel.L3_LINE_BY_LINE,
    "contract_draft": ReviewLevel.L2_VERIFY_KEY,
    "due_diligence": ReviewLevel.L2_VERIFY_KEY,
    "legal_research": ReviewLevel.L2_VERIFY_KEY,
    "compliance_check": ReviewLevel.L2_VERIFY_KEY,
    "medical_brief": ReviewLevel.L2_VERIFY_KEY,
    "tax_analysis": ReviewLevel.L2_VERIFY_KEY,
    "information_query": ReviewLevel.L1_READ_SUMMARY,
    "summary": ReviewLevel.L1_READ_SUMMARY,
    "draft_email": ReviewLevel.L1_READ_SUMMARY,
    "brainstorm": ReviewLevel.L1_READ_SUMMARY,
}

DEFAULT_MIN = ReviewLevel.L2_VERIFY_KEY


def classify_task(stakes: str, output_type: str = "") -> ReviewLevel:
    if output_type and output_type in _MIN_REVIEW_LEVEL:
        return _MIN_REVIEW_LEVEL[output_type]
    s = stakes.lower().strip()
    if s in ("critical", "high"):
        return ReviewLevel.L3_LINE_BY_LINE
    elif s == "medium":
        return ReviewLevel.L2_VERIFY_KEY
    return DEFAULT_MIN


@dataclass
class ReviewRecord:
    """审查行为记录(审计追溯)."""
    task_id: str
    level_used: ReviewLevel
    reviewer: str
    ai_source: str = ""
    ai_output_id: str = ""
    reviewed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    items_reviewed: int = 0
    items_flagged: int = 0
    notes: str = ""

    def validate(self, min_level: Optional[ReviewLevel] = None) -> bool:
        if min_level is None:
            min_level = DEFAULT_MIN
        if self.level_used < min_level:
            return False
        return self.level_used != ReviewLevel.L0_SIGN_ONLY

    def is_form_compliance_only(self) -> bool:
        return self.level_used.value <= 1


@dataclass
class LeverageTracker:
    """责任杠杆追踪器(卡197 / ADR-357). 技术天花板 1:500, 保险精算天花板 1:50-100."""
    reviewer: str
    period: str = "daily"
    max_safe_ratio: int = 100
    max_warning_ratio: int = 50
    _records: List[ReviewRecord] = field(default_factory=list)

    def add_review(self, record: ReviewRecord):
        self._records.append(record)

    @property
    def review_count(self) -> int:
        return len(self._records)

    @property
    def current_leverage(self) -> int:
        return len(self._records)

    @property
    def status(self) -> str:
        lev = self.current_leverage
        if lev == 0:
            return "idle"
        if lev <= self.max_warning_ratio:
            return "safe"
        if lev <= self.max_safe_ratio:
            return "approaching limit"
        return "over limit"

    def summary(self) -> str:
        return (
            f"Leverage Tracker [{self.reviewer}] {self.period}: "
            f"{self.current_leverage} reviews -> {self.status} "
            f"(safe<={self.max_warning_ratio}, hard<={self.max_safe_ratio})"
        )
