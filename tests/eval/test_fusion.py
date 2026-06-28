"""Tests for hippo.eval.fusion."""

import os
import tempfile

import pytest

from hippo.eval.fusion import (
    ChallengeLog,
    DummyLLMJudge,
    FusionConfig,
    LLMJudge,
    LLMResult,
    RuleConfig,
    RuleResult,
    Verdict,
    check_rules,
    evaluate,
)


class FixedLLMJudge(LLMJudge):
    """Returns a fixed score for testing."""

    def __init__(self, score: float, confidence: float = 0.9, available: bool = True):
        self._score = score
        self._confidence = confidence
        self._available = available

    def evaluate(self, text, context=None):
        if not self._available:
            return LLMResult(score=0, confidence=0, available=False)
        return LLMResult(score=self._score, confidence=self._confidence, rationale="test")


# ═══ Rule layer tests ═══


class TestRuleLayer:

    def test_pass_all_rules(self):
        text = "这是一段足够长的测试文本，包含了正常的中文标点。它讨论了评测系统的设计原则。" * 3
        results = check_rules(text, RuleConfig(min_length=50))
        assert all(r.passed for r in results)

    def test_fail_length_short(self):
        results = check_rules("太短了。", RuleConfig(min_length=50))
        length_result = next(r for r in results if r.name == "length")
        assert not length_result.passed
        assert "50" in length_result.reason

    def test_fail_length_long(self):
        results = check_rules("测试。" * 60_000, RuleConfig(max_length=100_000))
        length_result = next(r for r in results if r.name == "length")
        assert not length_result.passed

    def test_fail_cliche(self):
        text = "综上所述，这段文本太短。" * 5
        results = check_rules(text, RuleConfig(min_length=10))
        cliche_result = next(r for r in results if r.name == "cliche")
        assert not cliche_result.passed
        assert "综上所述" in cliche_result.reason

    def test_pass_cliche_no_banned(self):
        text = "这段文本使用了原创表述，没有套话。它讨论了具体的技术方案。" * 3
        results = check_rules(text, RuleConfig(min_length=10))
        cliche_result = next(r for r in results if r.name == "cliche")
        assert cliche_result.passed

    def test_fail_similarity(self):
        ref = "评测系统的架构设计需要考虑规则层和增强层的协同工作"
        text = ref + "。" * 20
        results = check_rules(text, RuleConfig(min_length=10), references=[ref])
        sim_result = next(r for r in results if r.name == "similarity")
        assert not sim_result.passed

    def test_pass_similarity_no_refs(self):
        text = "任意文本。" * 10
        results = check_rules(text, RuleConfig(min_length=10))
        sim_result = next(r for r in results if r.name == "similarity")
        assert sim_result.passed

    def test_fail_sections(self):
        text = "这段文本没有要求的章节标记。" * 5
        results = check_rules(text, RuleConfig(min_length=10, require_sections=["## 引言"]))
        sec_result = next(r for r in results if r.name == "sections")
        assert not sec_result.passed
        assert "## 引言" in sec_result.reason

    def test_fail_compliance_no_punctuation(self):
        text = "a" * 100
        results = check_rules(text, RuleConfig(min_length=10))
        comp_result = next(r for r in results if r.name == "compliance")
        assert not comp_result.passed


# ═══ Fusion layer tests ═══


class TestFusionLayer:

    _GOOD_TEXT = (
        "这是一段足够长的、包含原创观点的测试文本。"
        "它讨论了评测架构的设计原则和实践考量。"
        "核心观点是规则层和增强层的互补融合优于单一系统。"
    ) * 2

    def test_rule_fail_rejects(self):
        result = evaluate("短。", RuleConfig(min_length=50), llm_judge=FixedLLMJudge(0.95))
        assert result.verdict == Verdict.REJECT
        assert "规则层拒绝" in result.reason

    def test_rule_pass_llm_high_pass(self):
        result = evaluate(
            self._GOOD_TEXT,
            RuleConfig(min_length=50),
            llm_judge=FixedLLMJudge(0.85),
        )
        assert result.verdict == Verdict.PASS

    def test_rule_pass_llm_low_review(self):
        result = evaluate(
            self._GOOD_TEXT,
            RuleConfig(min_length=50),
            llm_judge=FixedLLMJudge(0.2),
        )
        assert result.verdict == Verdict.NEEDS_REVIEW
        assert "规则过松" in result.reason

    def test_rule_pass_llm_gray_review(self):
        result = evaluate(
            self._GOOD_TEXT,
            RuleConfig(min_length=50),
            llm_judge=FixedLLMJudge(0.5),
            fusion_config=FusionConfig(llm_lo=0.4, llm_hi=0.7),
        )
        assert result.verdict == Verdict.NEEDS_REVIEW
        assert "灰区" in result.reason

    def test_llm_unavailable_review(self):
        result = evaluate(
            self._GOOD_TEXT,
            RuleConfig(min_length=50),
            llm_judge=FixedLLMJudge(0, available=False),
        )
        assert result.verdict == Verdict.NEEDS_REVIEW
        assert result.llm_result is None

    def test_dummy_judge_defaults_review(self):
        result = evaluate(self._GOOD_TEXT, RuleConfig(min_length=50))
        assert result.verdict == Verdict.NEEDS_REVIEW


# ═══ Challenge log tests ═══


class TestChallengeLog:

    def test_record_and_get(self):
        log = ChallengeLog()
        log.record("length", "over_strict")
        log.record("length", "over_strict")
        assert log.get("length")["over_strict"] == 2

    def test_suspicious_rules(self):
        log = ChallengeLog()
        for _ in range(3):
            log.record("cliche", "over_lenient")
        assert "cliche" in log.suspicious_rules(threshold=3)

    def test_not_suspicious_below_threshold(self):
        log = ChallengeLog()
        log.record("length", "over_strict")
        log.record("length", "over_lenient")
        assert log.suspicious_rules(threshold=3) == []

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "challenge.json")
            log1 = ChallengeLog(path)
            log1.record("similarity", "over_strict")
            log1.record("similarity", "over_strict")

            log2 = ChallengeLog(path)
            assert log2.get("similarity")["over_strict"] == 2

    def test_challenge_logged_on_contradiction(self):
        log = ChallengeLog()
        result = evaluate(
            "短。",
            RuleConfig(min_length=50),
            llm_judge=FixedLLMJudge(0.9),
            challenge_log=log,
        )
        assert result.verdict == Verdict.REJECT
        assert "length" in result.challenged_rules
        assert log.get("length")["over_strict"] == 1

    def test_challenge_logged_over_lenient(self):
        log = ChallengeLog()
        good_text = (
            "这是一段足够长的原创文本，包含中文标点。"
            "讨论评测系统的架构设计和实现方案。"
        ) * 3
        evaluate(
            good_text,
            RuleConfig(min_length=50),
            llm_judge=FixedLLMJudge(0.1),
            challenge_log=log,
        )
        summary = log.summary()
        assert any(v.get("over_lenient", 0) > 0 for v in summary.values())


# ═══ Edge cases ═══


class TestEdgeCases:

    def test_empty_text_rejected(self):
        result = evaluate("", RuleConfig(min_length=1))
        assert result.verdict == Verdict.REJECT
        failed_names = {r.name for r in result.rule_results if not r.passed}
        assert "length" in failed_names

    def test_custom_thresholds(self):
        text = "正常文本，有标点。" * 10
        result = evaluate(
            text,
            RuleConfig(min_length=10),
            llm_judge=FixedLLMJudge(0.8),
            fusion_config=FusionConfig(llm_lo=0.3, llm_hi=0.9),
        )
        assert result.verdict == Verdict.NEEDS_REVIEW

    def test_custom_banned_phrases(self):
        config = RuleConfig(min_length=10, banned_phrases=["禁止词"])
        results = check_rules("这段文本包含禁止词，但长度足够。" * 3, config)
        cliche = next(r for r in results if r.name == "cliche")
        assert not cliche.passed
        assert "禁止词" in cliche.reason
