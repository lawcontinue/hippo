"""
Quick start: Evaluation toolkit for AI agent quality assessment.

Shows the 4 most useful eval patterns:
  1. Rule + LLM fusion evaluation
  2. Distribution drift detection
  3. Reward design assessment
  4. Chaos fault injection

Run: python3 examples/quickstart_eval.py
"""

from hippo.eval import (
    DriftDetector,
    FaultInjector,
    FaultType,
    InjectionStrategy,
    RewardConfig,
    RewardDimension,
    RewardEvaluator,
    RewardSignal,
    evaluate,
)

# ── 1. Fusion: rule-based quality gate ──────────────────
# Evaluate a piece of text against deterministic rules (no LLM needed).
# Good for batch filtering — only borderline cases go to LLM.

text = "This is a well-structured response with specific data points: 42ms latency."
result = evaluate(text)
print("=== Fusion Evaluation ===")
print(f"  Verdict: {result.verdict.value}")
print(f"  Reason:  {result.reason}")
print(f"  Rules passed: {sum(1 for r in result.rule_results if r.passed)}/{len(result.rule_results)}")
print()

# ── 2. Drift: distribution shift detection ───────────────
# Detect if your production traffic has drifted from the eval set
# using KL/JS divergence.

import numpy as np

reference = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.2, 0.18, 0.22, 0.12, 0.28])
production = np.array([0.35, 0.4, 0.38, 0.45, 0.5, 0.42, 0.36, 0.48, 0.33, 0.44])

detector = DriftDetector()
detector.fit_reference(reference)
drift = detector.detect(production)

print("=== Drift Detection ===")
print(f"  Drifted: {drift.is_drifted}")
print(f"  Severity: {drift.severity.value}")
print(f"  KL divergence: {drift.kl_divergence:.4f}")
print(f"  JS divergence: {drift.js_divergence:.4f}")
print()

# ── 3. Reward: reward design assessment ──────────────────
# Evaluate whether your reward function design is sound:
# sparsity, conflict between dimensions, Goodhart risk.

config = RewardConfig(dimensions=[
    RewardDimension(name="accuracy", weight=0.5),
    RewardDimension(name="fluency", weight=0.3),
    RewardDimension(name="safety", weight=0.2),
])

signals = [
    RewardSignal(agent_id="agent_a", dimensions={"accuracy": 0.8, "fluency": 0.7, "safety": 0.9}),
    RewardSignal(agent_id="agent_a", dimensions={"accuracy": 0.6, "fluency": 0.5, "safety": 0.8}),
    RewardSignal(agent_id="agent_a", dimensions={"accuracy": 0.9, "fluency": 0.8, "safety": 0.95}),
    RewardSignal(agent_id="agent_a", dimensions={"accuracy": 0.0, "fluency": 0.0, "safety": 0.5}),
    RewardSignal(agent_id="agent_a", dimensions={"accuracy": 0.7, "fluency": 0.6, "safety": 0.85}),
]

evaluator = RewardEvaluator(config)
evaluation = evaluator.evaluate_batch(signals)

print("=== Reward Design Assessment ===")
for i, ev in enumerate(evaluation):
    print(f"  Signal {i}: total={ev.total_score:.3f}")

sparsity = evaluator.sparsity_report(signals)
print(f"  Sparsity: {sparsity['overall_sparsity']:.1%} ({sparsity['level']})")
print(f"  Per dimension: {sparsity['per_dimension']}")

conflicts = evaluator.conflict_detection(signals)
print(f"  Conflicts: {conflicts if conflicts else 'none'}")
print()

# ── 4. Chaos: fault injection ────────────────────────────
# Simulate network timeouts, tool errors, and malformed inputs
# to test your agent's resilience.

injector = FaultInjector.create_enabled(InjectionStrategy.KNOWN)

from hippo.eval.chaos import TaskStep

original_step = TaskStep(
    step_id="s1",
    description="Call weather API",
    tool="weather_api",
    params={"city": "Beijing"},
)

# Inject a network timeout
injected = injector.inject(original_step, FaultType.NETWORK_TIMEOUT)
print("=== Chaos Injection ===")
print(f"  Original tool: {original_step.tool}")
print(f"  Injected fault: {injected.fault_type.value}")
print(f"  Agent observes: {injected.simulate_agent_observation()}")
print()

print("✅ All eval examples completed. See hippo/eval/README.md for details.")
