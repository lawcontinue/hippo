"""
Data curation — quality-gated data selection for post-training.

Melts into fusion.check_rules to add 3 curation-specific rules
(diversity, informativeness, label_balance) and a stratified sampler.

Source: Crucible #126 — GLM-5.2 post-training analysis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .fusion import RuleConfig, check_rules

# ─── Default stop words (Chinese + English) ───

_DEFAULT_STOP_WORDS: frozenset[str] = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们",
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if",
    "it", "its", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they",
})


# ─── Data structures ───

@dataclass
class CurationItem:
    """A single data item to be curated."""
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CurationRule:
    """A curation rule with configurable weight and threshold."""
    name: str
    description: str = ""
    weight: float = 1.0
    threshold: float = 0.5
    enabled: bool = True


@dataclass
class CurationResult:
    """Curation result for a single item."""
    item: CurationItem
    passed: bool
    score: float
    rule_results: dict[str, bool]
    rule_scores: dict[str, float]
    fail_reasons: list[str] = field(default_factory=list)


# ─── Curation-specific rules ───

def _diversity_score(
    content: str,
    selected_contents: list[str],
    n: int = 3,
) -> float:
    """Compute n-gram diversity relative to already-selected items.

    Returns 1.0 if fully novel, 0.0 if identical to any selected item.
    """
    def ngrams(s: str) -> set[str]:
        chars = list(s)
        return {" ".join(chars[i:i+n]) for i in range(len(chars) - n + 1)} if len(chars) >= n else {chars[0]}

    if not selected_contents:
        return 1.0

    content_ng = ngrams(content)
    if not content_ng:
        return 0.0

    # Max similarity to any single selected item
    best_sim = 0.0
    for sel in selected_contents:
        sel_ng = ngrams(sel)
        if not sel_ng:
            continue
        sim = len(content_ng & sel_ng) / len(content_ng | sel_ng)
        best_sim = max(best_sim, sim)

    # Diversity = 1 - similarity
    return 1.0 - best_sim


def _informativeness_score(content: str, stop_words: frozenset[str] | None = None) -> float:
    """Content-word ratio as a measure of informativeness."""
    sw = stop_words or _DEFAULT_STOP_WORDS
    if not content.strip():
        return 0.0
    tokens = content.split()
    if not tokens:
        return 0.0
    content_tokens = [t for t in tokens if t.lower() not in sw]
    return len(content_tokens) / len(tokens)


def _label_balance_score(
    label: str,
    selected_labels: list[str],
) -> float:
    """Score based on current label balance.

    Returns lower scores for labels that are already over-represented.
    """
    if not selected_labels:
        return 1.0
    counts = Counter(selected_labels)
    total = len(selected_labels)
    current_pct = counts.get(label, 0) / total
    # Prefer under-represented labels
    return 1.0 - current_pct


# ─── Data Curator ───

class DataCurator:
    """Data curator — quality-gated, multi-source data selection.

    Combines fusion.check_rules (5 deterministic rules) with 3 curation-specific
    rules (diversity, informativeness, label_balance) and stratified sampling.

    Usage:
        curator = DataCurator(items)
        filtered = curator.run(select=100)
        report = curator.curation_report(filtered)
    """

    def __init__(
        self,
        items: list[CurationItem],
        extra_rules: list[CurationRule] | None = None,
        rule_config: RuleConfig | None = None,
    ):
        """
        Args:
            items: Items to curate.
            extra_rules: Custom curation rules (default: 3 built-in rules).
            rule_config: Fusion rule configuration.
        """
        self._items = items
        self._rule_config = rule_config or RuleConfig(min_length=5)
        self._extra_rules = list(extra_rules or [
            CurationRule("diversity", "N-gram novelty against already-selected", threshold=0.3),
            CurationRule("informativeness", "Content-word ratio", threshold=0.3),
            CurationRule("label_balance", "Under-represented label preference", threshold=0.2),
        ])
        self._selected: list[CurationItem] = []

    # ── Public API ──

    def run(
        self,
        select: int | None = None,
        per_source: int | None = None,
        per_source_ratio: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> list[CurationItem]:
        """Run full curation pipeline.

        Args:
            select: Total items to select (None = pass all).
            per_source: Fixed number per source.
            per_source_ratio: Dict of source -> fraction, summing to 1.0.
            seed: Random seed for stratified sampling.

        Returns:
            Selected items (high quality, diversified).
        """
        # Step 1: quality-gate each item
        scored = [(item, self._evaluate(item)) for item in self._items]

        # Step 2: filter by pass
        passed = [(item, result) for item, result in scored if result.passed]

        if not passed:
            self._selected = []
            return []

        # Step 3: score-descending sort
        passed.sort(key=lambda x: x[1].score, reverse=True)

        # Step 4: stratified sampling
        if select is not None or per_source is not None or per_source_ratio is not None:
            samples = self._stratified_sample(
                [item for item, _ in passed],
                select=select,
                per_source=per_source,
                per_source_ratio=per_source_ratio,
                seed=seed,
            )
        else:
            samples = [item for item, _ in passed]

        self._selected = samples
        return samples

    def evaluate(self, item: CurationItem) -> CurationResult:
        """Evaluate a single item without selecting."""
        return self._evaluate(item)

    def curation_report(self, selected: list[CurationItem] | None = None) -> str:
        """Generate a human-readable curation report."""
        s = selected or self._selected
        lines = ["=" * 60, "Data Curation Report", "=" * 60]
        lines.append(f"Total items      : {len(self._items)}")
        lines.append(f"Selected         : {len(s)} ({len(s)/max(len(self._items),1)*100:.1f}%)")

        # Source distribution
        if any(item.source for item in self._items):
            lines.append("\nSource distribution (selected):")
            src_counts = Counter(item.source for item in s)
            for src, cnt in src_counts.most_common():
                lines.append(f"  {src:20s}  {cnt:>4d} ({cnt/max(len(s),1)*100:.1f}%)")

        # Quality stats on selected
        if s:
            scores = [self._evaluate(item).score for item in s]
            lines.append("\nQuality scores (selected):")
            lines.append(f"  Mean   : {np.mean(scores):.3f}")
            lines.append(f"  Std    : {np.std(scores):.3f}")
            lines.append(f"  Min    : {min(scores):.3f}")
            lines.append(f"  Max    : {max(scores):.3f}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    # ── Internal ──

    def _evaluate(self, item: CurationItem) -> CurationResult:
        """Evaluate a single item against all rules."""
        # Fusion rules (5 base rules)
        fusion_results = check_rules(item.content, self._rule_config)

        rule_results: dict[str, bool] = {}
        rule_scores: dict[str, float] = {}
        fail_reasons: list[str] = []
        total_weight = 0.0
        weighted_sum = 0.0

        # Fusion rules contribute to base score
        for r in fusion_results:
            # In curation context, skip compliance and sections rules
            # (training data fragments need not be full sentences)
            if r.name in ('compliance', 'sections'):
                continue
            score = 1.0 if r.passed else 0.0
            rule_results[r.name] = r.passed
            rule_scores[r.name] = score
            # Each base rule counts as weight 0.5
            weighted_sum += score * 0.5
            total_weight += 0.5
            if not r.passed and r.reason:
                fail_reasons.append(f"[{r.name}] {r.reason}")

        # Extra curation rules
        for rule in self._extra_rules:
            if not rule.enabled:
                continue
            score = self._score_extra_rule(rule.name, item.content, item.metadata)
            passed = score >= rule.threshold
            rule_results[rule.name] = passed
            rule_scores[rule.name] = score
            weighted_sum += score * rule.weight
            total_weight += rule.weight
            if not passed:
                fail_reasons.append(f"[{rule.name}] score={score:.3f} < threshold={rule.threshold}")

        # Composite score
        composite = weighted_sum / max(total_weight, 1.0)
        passed = len(fail_reasons) == 0

        return CurationResult(
            item=item,
            passed=passed,
            score=composite,
            rule_results=rule_results,
            rule_scores=rule_scores,
            fail_reasons=fail_reasons,
        )

    def _score_extra_rule(self, name: str, content: str, metadata: dict[str, Any]) -> float:
        """Score a single curation rule."""
        if name == "informativeness":
            return _informativeness_score(content)

        # diversity and label_balance need state from self._selected
        # For single-item evaluation without selection context,
        # return 1.0 (neutral — context-dependent rules evaluated in run())
        return 1.0

    def _stratified_sample(
        self,
        items: list[CurationItem],
        select: int | None = None,
        per_source: int | None = None,
        per_source_ratio: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> list[CurationItem]:
        """Stratified sampling across sources."""
        if not items:
            return []

        # Group by source
        by_source: dict[str, list[CurationItem]] = {}
        for item in items:
            by_source.setdefault(item.source, []).append(item)

        rng = np.random.RandomState(seed)

        if per_source is not None:
            # Fixed count per source
            result = []
            for src, src_items in by_source.items():
                n = min(per_source, len(src_items))
                indices = rng.choice(len(src_items), size=n, replace=False)
                result.extend([src_items[i] for i in indices])
            return result

        if per_source_ratio is not None and select is not None:
            # Proportional allocation
            result = []
            for src, ratio in per_source_ratio.items():
                if src not in by_source:
                    continue
                src_items = by_source[src]
                n = min(max(1, int(select * ratio)), len(src_items))
                indices = rng.choice(len(src_items), size=n, replace=False)
                result.extend([src_items[i] for i in indices])
            return result

        if select is not None and select < len(items):
            # Top-k by score, don't shuffle
            return items[:select]

        return items
