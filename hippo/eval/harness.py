"""
Evaluation harness — integrates all eval modules into a unified interface.

Provides three subcommands:
  fusion: Rule + LLM fusion evaluation on text.
  matrix: Multi-dimensional metric matrix analysis from task results.
  full:   Both fusion and matrix in one call.
"""

from __future__ import annotations

import json
from pathlib import Path

from .fusion import DummyLLMJudge, FusionConfig, RuleConfig
from .fusion import evaluate as fusion_evaluate
from .matrix import DIMENSIONS, TaskResult, generate_report


def run_fusion(text: str, ref_file: str | None = None) -> dict:
    """Rule-based + LLM enhancement fusion evaluation."""
    references: list[str] = []
    if ref_file:
        ref_path = Path(ref_file)
        if ref_path.exists():
            references = [ref_path.read_text(encoding="utf-8")]

    rc = RuleConfig()
    fc = FusionConfig()
    judge = DummyLLMJudge()
    result = fusion_evaluate(
        text, rule_config=rc, fusion_config=fc,
        llm_judge=judge, references=references or None,
    )
    return {
        "verdict": result.verdict.value,
        "reason": result.reason,
        "challenged_rules": result.challenged_rules,
        "rule_pass": [r.name for r in result.rule_results if r.passed],
        "rule_fail": [r.name for r in result.rule_results if not r.passed],
        "llm_score": result.llm_result.score if result.llm_result else None,
        "llm_confidence": result.llm_result.confidence if result.llm_result else None,
    }


def run_matrix(results_file: str) -> dict:
    """
    Multi-dimensional metric matrix analysis.

    Expected JSON format:
        {"agents": [{"name": "a", "results": [
            {"task_id": "t1", "agent_id": "a", "success": true, ...}
        ]}]}
    """
    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)

    all_results: list[TaskResult] = []
    for agent_data in data.get("agents", []):
        results = [TaskResult(**r) for r in agent_data.get("results", [])]
        all_results.extend(results)

    if not all_results:
        return {"error": "No results found"}

    report = generate_report(all_results)
    scores = {}
    for aid, ev in report.agents.items():
        scores[aid] = ev.score_vector().tolist()

    return {
        "scores": scores,
        "dimensions": DIMENSIONS,
        "correlation_method": "Spearman",
        "merge_warnings": report.merge_warnings,
        "goodhart_risk": report.goodhart_analysis,
    }


def run_full(text: str, results_file: str | None = None, ref_file: str | None = None) -> dict:
    """One-shot mode: fusion evaluation + multi-dimensional analysis."""
    output = {"fusion": run_fusion(text, ref_file)}
    if results_file:
        output["matrix"] = run_matrix(results_file)
    return output


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation harness")
    sub = parser.add_subparsers(dest="command")

    p_fusion = sub.add_parser("fusion", help="Rule + LLM fusion evaluation")
    p_fusion.add_argument("--text", required=True, help="Text to evaluate")
    p_fusion.add_argument("--ref-file", help="Reference file path (for dedup)")

    p_matrix = sub.add_parser("matrix", help="Multi-dimensional metric matrix analysis")
    p_matrix.add_argument("--results", required=True, help="Eval results JSON file")

    p_full = sub.add_parser("full", help="One-shot mode")
    p_full.add_argument("--text", required=True)
    p_full.add_argument("--results", help="Multi-dim eval results JSON")
    p_full.add_argument("--ref-file", help="Reference file")

    args = parser.parse_args()
    if args.command == "fusion":
        print(json.dumps(run_fusion(args.text, args.ref_file), indent=2, ensure_ascii=False))
    elif args.command == "matrix":
        print(json.dumps(run_matrix(args.results), indent=2, ensure_ascii=False))
    elif args.command == "full":
        print(json.dumps(run_full(args.text, args.results, args.ref_file), indent=2, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
