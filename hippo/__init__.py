"""Hippo — local embedding + hybrid search for AI agents."""

from .safety_guard import SafetyGuard

# Optional eval subpackage — requires scipy for matrix module.
try:
    from .eval import (
        DIMENSIONS,
        ChallengeLog,
        DriftDetector,
        DriftResult,
        DriftSeverity,
        FaultInjector,
        FaultType,
        FusionConfig,
        FusionResult,
        InjectionStrategy,
        LLMJudge,
        MatrixReport,
        RecoveryEvaluator,
        RuleConfig,
        TaskResult,
        Verdict,
        compute_correlation,
        evaluate,
        evaluate_agent,
        generate_report,
    )
except ImportError:
    pass

__all__ = [
    "SafetyGuard",
    # eval
    "evaluate",
    "Verdict",
    "RuleConfig",
    "FusionConfig",
    "FusionResult",
    "LLMJudge",
    "ChallengeLog",
    "DriftDetector",
    "DriftResult",
    "DriftSeverity",
    "generate_report",
    "evaluate_agent",
    "compute_correlation",
    "DIMENSIONS",
    "MatrixReport",
    "TaskResult",
    "FaultInjector",
    "FaultType",
    "InjectionStrategy",
    "RecoveryEvaluator",
]
