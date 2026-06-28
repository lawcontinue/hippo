"""
hippo.eval — Evaluation toolkit for AI agent quality assessment.

Modules:
  fusion:  Rule-based safety net + LLM enhancement fusion evaluation.
  drift:   Distribution drift detection (KL/JS divergence).
  matrix:  Multi-dimensional metric matrix (5 dimensions, Goodhart analysis).
  chaos:   Fault injection framework for chaos engineering.
  reward:  Deterministic reward rule evaluator (sparsity, conflict, Goodhart).
  harness: Unified CLI integrating all modules.
"""

from .chaos import FaultInjector, FaultType, InjectionStrategy, RecoveryEvaluator
from .curation import CurationItem, CurationResult, CurationRule, DataCurator
from .drift import DriftDetector, DriftResult, DriftSeverity
from .fusion import (
    ChallengeLog,
    FusionConfig,
    FusionResult,
    LLMJudge,
    RuleConfig,
    Verdict,
    evaluate,
)
from .matrix import (
    DIMENSIONS,
    MatrixReport,
    TaskResult,
    compute_correlation,
    evaluate_agent,
    generate_report,
)
from .matrix import (
    format_report as format_matrix_report,
)
from .reward import (
    RewardConfig,
    RewardDimension,
    RewardEvaluation,
    RewardEvaluator,
    RewardSignal,
)

__all__ = [
    # fusion
    "evaluate",
    "Verdict",
    "RuleConfig",
    "FusionConfig",
    "FusionResult",
    "LLMJudge",
    "ChallengeLog",
    # drift
    "DriftDetector",
    "DriftResult",
    "DriftSeverity",
    # matrix
    "generate_report",
    "evaluate_agent",
    "compute_correlation",
    "DIMENSIONS",
    "MatrixReport",
    "TaskResult",
    "format_matrix_report",
    # reward
    "RewardEvaluator",
    "RewardSignal",
    "RewardEvaluation",
    "RewardConfig",
    "RewardDimension",
    # curation
    "DataCurator",
    "CurationItem",
    "CurationResult",
    "CurationRule",
    # chaos
    "FaultInjector",
    "FaultType",
    "InjectionStrategy",
    "RecoveryEvaluator",
]
