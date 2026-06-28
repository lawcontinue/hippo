"""
Eval Fusion: Rule-based safety net + LLM enhancement layer.

Decision matrix:
  Rule fail              → REJECT (deterministic safety net)
  Rule pass + LLM < lo   → NEEDS_REVIEW (LLM uncertain)
  Rule pass + LLM ≥ hi   → PASS (dual confirmation)
  Rule pass + lo≤LLM<hi  → NEEDS_REVIEW (gray zone)
  LLM unavailable         → NEEDS_REVIEW (degrade safely, never auto-pass)
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ─── Verdict ───

class Verdict(str, Enum):
    PASS = "pass"
    NEEDS_REVIEW = "needs_review"
    REJECT = "reject"


@dataclass
class RuleResult:
    """Result of a single rule check."""

    name: str
    passed: bool
    reason: str = ""


@dataclass
class LLMResult:
    """Score from the LLM enhancement layer."""

    score: float          # 0.0-1.0
    confidence: float     # 0.0-1.0
    rationale: str = ""
    available: bool = True


@dataclass
class FusionResult:
    """Final decision from the fusion layer."""

    verdict: Verdict
    rule_results: list[RuleResult]
    llm_result: LLMResult | None
    reason: str
    challenged_rules: list[str] = field(default_factory=list)


# ─── Rule layer ───

@dataclass
class RuleConfig:
    """Rule configuration — auditable and adjustable."""

    min_length: int = 50
    max_length: int = 100_000
    banned_phrases: list[str] = field(default_factory=lambda: [
        "综上所述", "总而言之", "不可否认", "众所周知",
        "显而易见", "毋庸置疑", "由此可见",
    ])
    max_similarity: float = 0.85
    require_sections: list[str] = field(default_factory=list)


def _max_ngram_similarity(text: str, references: list[str], n: int = 3) -> float:
    """Compute the highest n-gram Jaccard similarity between *text* and *references*."""

    def ngrams(s: str) -> set[str]:
        chars = list(s)
        return {" ".join(chars[i:i + n]) for i in range(len(chars) - n + 1)} if len(chars) >= n else {s}

    if not references:
        return 0.0
    tg = ngrams(text)
    best = 0.0
    for ref in references:
        rg = ngrams(ref)
        if not tg or not rg:
            continue
        sim = len(tg & rg) / len(tg | rg)
        best = max(best, sim)
    return best


def check_rules(
    text: str,
    config: RuleConfig,
    references: list[str] | None = None,
) -> list[RuleResult]:
    """
    Rule layer: deterministic checks, each producing pass/fail with a reason.

    Each rule is independent and auditable.
    """
    results: list[RuleResult] = []

    # R1: length
    length = len(text)
    if length < config.min_length:
        results.append(RuleResult("length", False, f"长度 {length} < {config.min_length}"))
    elif length > config.max_length:
        results.append(RuleResult("length", False, f"长度 {length} > {config.max_length}"))
    else:
        results.append(RuleResult("length", True))

    # R2: cliché / banned phrases
    found = [p for p in config.banned_phrases if p in text]
    results.append(RuleResult(
        "cliche", not found,
        f"命中: {found}" if found else "",
    ))

    # R3: similarity (dedup against reference set)
    if references:
        sim = _max_ngram_similarity(text, references)
        results.append(RuleResult(
            "similarity", sim <= config.max_similarity,
            f"相似度 {sim:.3f} > {config.max_similarity}" if sim > config.max_similarity else f"相似度 {sim:.3f}",
        ))
    else:
        results.append(RuleResult("similarity", True, "无参考集，跳过"))

    # R4: required sections
    if config.require_sections:
        missing = [s for s in config.require_sections if s not in text]
        results.append(RuleResult(
            "sections", not missing,
            f"缺少: {missing}" if missing else "",
        ))
    else:
        results.append(RuleResult("sections", True))

    # R5: compliance (basic structure — non-empty, has sentence terminators or line breaks)
    has_structure = bool(re.search(r'[。.!?;！？；]', text)) or '\n' in text
    results.append(RuleResult(
        "compliance", has_structure,
        "无句号/换行/标点结构" if not has_structure else "",
    ))

    return results


# ─── LLM enhancement layer (interface) ───

class LLMJudge(ABC):
    """
    Abstract interface for the LLM enhancement layer.

    Implementations must guarantee score ∈ [0, 1], confidence ∈ [0, 1].
    """

    @abstractmethod
    def evaluate(self, text: str, context: dict[str, Any] | None = None) -> LLMResult:
        ...


class DummyLLMJudge(LLMJudge):
    """Placeholder — never calls a model, returns unavailable. For tests and degraded mode."""

    def evaluate(self, text: str, context: dict[str, Any] | None = None) -> LLMResult:
        return LLMResult(score=0.0, confidence=0.0, available=False, rationale="LLM not configured")


# ─── Challenge log ───

class ChallengeLog:
    """
    Track how often each rule is challenged.

    - Rule fails but LLM scores high → rule may be too strict.
    - Rule passes but LLM scores low  → rule may be too lenient.
    """

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else None
        self._data: dict[str, dict[str, int]] = {}
        if self._path and self._path.exists():
            self._load()

    def record(self, rule_name: str, direction: str) -> None:
        """Record a challenge. *direction* is 'over_strict' or 'over_lenient'."""
        entry = self._data.setdefault(rule_name, {"over_strict": 0, "over_lenient": 0})
        entry[direction] = entry.get(direction, 0) + 1
        if self._path:
            self._save()

    def get(self, rule_name: str) -> dict[str, int]:
        return self._data.get(rule_name, {"over_strict": 0, "over_lenient": 0})

    def suspicious_rules(self, threshold: int = 3) -> list[str]:
        """Rules challenged ≥ *threshold* times — candidates for revision."""
        return [
            name for name, counts in self._data.items()
            if max(counts.get("over_strict", 0), counts.get("over_lenient", 0)) >= threshold
        ]

    def summary(self) -> dict[str, dict[str, int]]:
        return dict(self._data)

    def _save(self) -> None:
        if self._path:
            self._path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2))

    def _load(self) -> None:
        try:
            self._data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            self._data = {}


# ─── Fusion layer ───

@dataclass
class FusionConfig:
    """Fusion layer configuration."""

    llm_lo: float = 0.4
    llm_hi: float = 0.7
    challenge_threshold: int = 3


def evaluate(
    text: str,
    rule_config: RuleConfig | None = None,
    llm_judge: LLMJudge | None = None,
    references: list[str] | None = None,
    fusion_config: FusionConfig | None = None,
    challenge_log: ChallengeLog | None = None,
    context: dict[str, Any] | None = None,
) -> FusionResult:
    """
    Fusion evaluation main entry point.

    Decision logic:
      1. Rule layer runs first → any fail = REJECT (safety net)
      2. All rules pass → LLM enhancement layer scores
      3. LLM ≥ hi → PASS; LLM < lo → NEEDS_REVIEW; gray zone → NEEDS_REVIEW
      4. LLM unavailable → NEEDS_REVIEW (degrade, never auto-pass)
      5. Rule / LLM contradiction → logged to challenge log
    """
    rc = rule_config or RuleConfig()
    fc = fusion_config or FusionConfig()
    judge = llm_judge or DummyLLMJudge()

    # 1. Rule layer
    rule_results = check_rules(text, rc, references)
    failed = [r for r in rule_results if not r.passed]

    if failed:
        llm_res = LLMResult(score=0, confidence=0, available=False)
        challenged = []
        if challenge_log:
            llm_res = judge.evaluate(text, context)
            if llm_res.available and llm_res.score >= fc.llm_hi:
                for r in failed:
                    challenge_log.record(r.name, "over_strict")
                    challenged.append(r.name)
        return FusionResult(
            verdict=Verdict.REJECT,
            rule_results=rule_results,
            llm_result=llm_res if llm_res.available else None,
            reason=f"规则层拒绝: {', '.join(r.name for r in failed)}",
            challenged_rules=challenged,
        )

    # 2. LLM enhancement layer
    llm_res = judge.evaluate(text, context)

    if not llm_res.available:
        return FusionResult(
            verdict=Verdict.NEEDS_REVIEW,
            rule_results=rule_results,
            llm_result=None,
            reason="LLM 增强层不可用，降级为人工审查",
        )

    # 3. Fusion decision
    if llm_res.score >= fc.llm_hi:
        verdict = Verdict.PASS
        reason = f"规则全通过 + LLM score={llm_res.score:.2f}≥{fc.llm_hi}"
    elif llm_res.score < fc.llm_lo:
        if challenge_log:
            for r in rule_results:
                challenge_log.record(r.name, "over_lenient")
        verdict = Verdict.NEEDS_REVIEW
        reason = f"规则通过但 LLM score={llm_res.score:.2f}<{fc.llm_lo}，可能规则过松"
    else:
        verdict = Verdict.NEEDS_REVIEW
        reason = f"灰区: LLM score={llm_res.score:.2f} ∈ [{fc.llm_lo}, {fc.llm_hi})"

    return FusionResult(
        verdict=verdict,
        rule_results=rule_results,
        llm_result=llm_res,
        reason=reason,
    )
