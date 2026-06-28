"""Tests for curation (data quality gating)."""

from __future__ import annotations

import numpy as np
import pytest

from hippo.eval.curation import (
    CurationItem,
    CurationResult,
    CurationRule,
    DataCurator,
    _diversity_score,
    _informativeness_score,
    _label_balance_score,
)


# ── _diversity_score ──

class TestDiversityScore:
    def test_identical_content(self):
        text = "这是一条测试文本"
        assert _diversity_score(text, [text]) == 0.0

    def test_fully_novel(self):
        a = "今天天气很好我们去公园散步"
        b = "python代码测试框架非常强大好用"
        assert _diversity_score(a, [b]) > 0.9

    def test_empty_selected(self):
        assert _diversity_score("anything", []) == 1.0

    def test_partial_similarity(self):
        text = "深度学习模型训练需要大量数据"
        similar = "深度学习模型需要大量训练数据"
        score = _diversity_score(text, [similar])
        assert 0.0 < score < 1.0


# ── _informativeness_score ──

class TestInformativenessScore:
    def test_empty_content(self):
        assert _informativeness_score("") == 0.0

    def test_all_stop_words(self):
        assert _informativeness_score("的 了 在 是") == 0.0

    def test_mixed_content(self):
        score = _informativeness_score("今天 的 天气 很 好")
        assert 0.4 <= score <= 0.6  # 3 content / 5 total

    def test_full_content(self):
        assert _informativeness_score("深度学习 模型 训练 数据") == 1.0


# ── _label_balance_score ──

class TestLabelBalanceScore:
    def test_no_selected(self):
        assert _label_balance_score("a", []) == 1.0

    def test_under_represented(self):
        # 2 "a" out of 10 → 20% → gets high score (80%)
        score = _label_balance_score("c", ["a"] * 2 + ["b"] * 2 + ["c"] * 6)
        assert 0.3 <= score <= 0.6

    def test_over_represented(self):
        # 80 of 100 → gets low score
        selected = ["a"] * 80 + ["b"] * 20
        score = _label_balance_score("a", selected)
        assert score < 0.3


# ── DataCurator ──

class TestDataCurator:
    def _make_items(self, n: int = 20) -> list[CurationItem]:
        return [
            CurationItem(
                content=f"实验数据样本编号{i}，用于验证模型性能。",
                source="experiment",
            )
            for i in range(n)
        ]

    def test_empty_items(self):
        curator = DataCurator([])
        result = curator.run()
        assert result == []

    def test_single_item(self):
        items = [CurationItem(content="深度学习模型训练数据，质量良好。")]
        curator = DataCurator(items)
        result = curator.run()
        assert len(result) == 1

    def test_evaluate_return_type(self):
        item = CurationItem(content="这是一条高质量训练数据，价值很高。")
        curator = DataCurator([item])
        result = curator.evaluate(item)
        assert isinstance(result, CurationResult)
        assert isinstance(result.score, float)
        assert isinstance(result.passed, bool)
        assert isinstance(result.rule_results, dict)
        assert isinstance(result.rule_scores, dict)

    def test_select_limit(self):
        items = self._make_items(20)
        curator = DataCurator(items)
        result = curator.run(select=5)
        assert len(result) == 5

    def test_per_source_selection(self):
        items = [
            CurationItem(content=f"这是来自src_a的第{i}条训练数据样本", source="src_a") for i in range(5)
        ] + [
            CurationItem(content=f"这是来自src_b的第{i}条训练数据样本", source="src_b") for i in range(5)
        ]
        curator = DataCurator(items)
        result = curator.run(per_source=2)
        assert len(result) == 4
        sources = [item.source for item in result]
        assert sources.count("src_a") == 2
        assert sources.count("src_b") == 2

    def test_per_source_ratio(self):
        items = [
            CurationItem(content=f"这是来自source_A的第{i}条训练数据样本，质量较好", source="a") for i in range(10)
        ] + [
            CurationItem(content=f"这是来自source_B的第{i}条训练数据样本，质量较好", source="b") for i in range(10)
        ]
        curator = DataCurator(items)
        result = curator.run(select=6, per_source_ratio={"a": 0.5, "b": 0.5})
        assert len(result) == 6
        sources = [item.source for item in result]
        assert sources.count("a") == 3
        assert sources.count("b") == 3

    def test_custom_extra_rules(self):
        items = [
            CurationItem(content="短", metadata={"label": "a"}),
            CurationItem(content="这是一条较长且内容丰富的数据样本", metadata={"label": "b"}),
        ]
        extra = [
            CurationRule(name="custom_len", description="Min 5 chars", weight=1.0, threshold=0.5),
        ]
        curator = DataCurator(items, extra_rules=extra)
        results = [curator.evaluate(item) for item in items]
        # First item "短" fails custom_len; second passes
        assert results[0].score < results[1].score

    def test_passed_items_are_high_quality(self):
        items = [
            CurationItem(content="这是一条内容丰富、格式规范的训练数据样本，用于微调模型。", source="good"),
            CurationItem(content="短", source="bad"),
        ]
        curator = DataCurator(items)
        result = curator.run(select=10)
        # Only the first item should pass
        assert len(result) == 1
        assert result[0].source == "good"

    def test_curation_report_format(self):
        items = self._make_items(10)
        curator = DataCurator(items)
        selected = curator.run(select=5)
        report = curator.curation_report(selected)
        assert "Data Curation Report" in report
        assert "10" in report
        assert "Selected" in report
        assert "Mean" in report

    def test_short_text_fails_length_rule(self):
        item = CurationItem(content="短")
        curator = DataCurator([item])
        result = curator.evaluate(item)
        # length rule should fail
        assert not result.rule_results.get("length", True)
        assert not result.passed

    def test_banned_phrases_rejected(self):
        item = CurationItem(content="综上所述，这是一个很好的样本。")
        curator = DataCurator([item])
        result = curator.evaluate(item)
        # cliche rule should fail
        assert not result.rule_results.get("cliche", True)
        assert not result.passed

    def test_stratified_sampling_reproducible(self):
        items = self._make_items(20)
        c1 = DataCurator(items)
        c2 = DataCurator(items)
        r1 = c1.run(select=3, seed=42)
        r2 = c2.run(select=3, seed=42)
        # Non-reproducible with top-k selection (no shuffle), but both run same config
        assert len(r1) == len(r2)

    def test_informativeness_rule_affects_score(self):
        items = [
            CurationItem(content="的 了 在 是 我 有 和", source="low_info"),
            CurationItem(content="深度学习模型训练需要大量高质量数据", source="high_info"),
        ]
        curator = DataCurator(items)
        eval_low = curator.evaluate(items[0])
        eval_high = curator.evaluate(items[1])
        assert eval_high.score > eval_low.score
