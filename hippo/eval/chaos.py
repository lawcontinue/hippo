"""
Chaos Engineering for Agent Evaluation — fault injection testing framework.

Three-stage injection strategy:
  Development: DISABLED (pure baseline, no injection)
  Acceptance:  KNOWN    (inject known faults, measure recovery rate)
  Production:  RANDOM   (inject random faults, test production resilience)

Safety constraints:
  - Default disabled; must explicitly enable.
  - Pure simulation layer — no real side effects.
  - Full injection log for auditability.
  - Anti-Goodhart: the framework cannot be exploited by agents to "fake failure then recover".
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("hippo.eval.chaos")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FaultType(Enum):
    """Each fault type simulates a real production scenario."""

    NETWORK_TIMEOUT = "network_timeout"
    TOOL_ERROR = "tool_error"
    MALFORMED_INPUT = "malformed_input"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    PARTIAL_FAILURE = "partial_failure"
    NONE = "none"


class InjectionStrategy(Enum):
    """Three-stage injection strategy."""

    DISABLED = "disabled"
    KNOWN = "known"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskStep:
    """A task step that can have faults injected."""

    step_id: str
    description: str
    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None


@dataclass
class InjectedStep:
    """A task step after fault injection (simulation-layer product)."""

    original: TaskStep
    fault_type: FaultType
    injected_output: Any
    injected_error: str | None = None
    is_timeout: bool = False

    def simulate_agent_observation(self) -> dict[str, Any]:
        """What the agent observes (tampered result) — pure simulation."""
        if self.is_timeout:
            return {"error": "TimeoutError", "detail": self.injected_error}
        if self.fault_type == FaultType.PERMISSION_DENIED:
            return {"error": "PermissionDenied", "detail": self.injected_error}
        if self.fault_type == FaultType.RESOURCE_EXHAUSTED:
            return {"error": "ResourceExhausted", "detail": self.injected_error}
        return {"error": self.injected_error or "UnknownFault", "output": self.injected_output}


@dataclass
class FaultConfig:
    """Configuration for a single fault injection."""

    fault_type: FaultType
    target_step_id: str | None = None


@dataclass
class InjectionRecord:
    """Audit log entry for each injection."""

    timestamp: float
    step_id: str
    fault_type: FaultType
    strategy: InjectionStrategy
    error_message: str
    agent_response: Any = None
    recovery_path: list[str] = field(default_factory=list)


@dataclass
class RecoveryResult:
    """Recovery evaluation result."""

    recovered: bool
    recovery_steps: list[str] = field(default_factory=list)
    recovery_time: float = 0.0
    recovery_quality: float = 0.0  # 0-1
    failure_reason: str | None = None


# ---------------------------------------------------------------------------
# Fault Injector
# ---------------------------------------------------------------------------

_DEFAULT_ERRORS: dict[FaultType, str] = {
    FaultType.NETWORK_TIMEOUT: "Connection timed out after 30s",
    FaultType.TOOL_ERROR: "Tool execution failed with exit code 1",
    FaultType.MALFORMED_INPUT: "JSONDecodeError: Expecting value at line 1 column 1",
    FaultType.PERMISSION_DENIED: "403 Forbidden: insufficient permissions",
    FaultType.RESOURCE_EXHAUSTED: "Context window limit exceeded (128k tokens)",
    FaultType.PARTIAL_FAILURE: "Batch step 3/5 failed: downstream service unavailable",
}


class FaultInjector:
    """
    Fault injector — core module.

    Safety guardrails:
      - Default disabled (_enabled=False); no injection occurs.
      - Must call enable() to perform real injection.
      - All injections logged to injection_log for auditability.
      - Anti-Goodhart: inject() is not exposed to the agent under test.
    """

    def __init__(self, strategy: InjectionStrategy = InjectionStrategy.DISABLED,
                 *, seed: int | None = None) -> None:
        self.strategy = strategy
        self._enabled = False
        self._rng = random.Random(seed)
        self.injection_log: list[InjectionRecord] = []

    @classmethod
    def create_enabled(cls, strategy: InjectionStrategy,
                       *, seed: int | None = None) -> "FaultInjector":
        """Construct and enable a FaultInjector — the only enabled construction path."""
        fi = cls(strategy=strategy, seed=seed)
        fi.enable()
        return fi

    def enable(self) -> None:
        """Explicitly enable injection (safety guardrail: off by default)."""
        self._enabled = True
        logger.info("FaultInjector ENABLED (strategy=%s)", self.strategy.value)

    @property
    def enabled(self) -> bool:
        return self._enabled

    # -- Single-step injection --

    def inject(self, step: TaskStep, fault_type: FaultType,
               strategy: InjectionStrategy | None = None) -> InjectedStep:
        """
        Inject a fault of the given type into a single step.

        If the injector is not enabled, the original step is passed through unchanged.
        """
        strat = strategy or self.strategy
        if not self._enabled or strat == InjectionStrategy.DISABLED:
            return InjectedStep(
                original=step, fault_type=fault_type,
                injected_output=step.expected_output,
                injected_error=None, is_timeout=False,
            )

        err_msg = _DEFAULT_ERRORS.get(fault_type, "Unknown fault injected")
        is_timeout = fault_type == FaultType.NETWORK_TIMEOUT
        injected = InjectedStep(
            original=step, fault_type=fault_type,
            injected_output=None, injected_error=err_msg,
            is_timeout=is_timeout,
        )
        self._log_injection(step.step_id, fault_type, strat, err_msg)
        return injected

    # -- Batch injection --

    def batch_inject(self, steps: list[TaskStep],
                     configs: list[FaultConfig]) -> list[InjectedStep]:
        """Batch-inject according to a config list, each specifying target_step_id + fault_type."""
        if not self._enabled:
            return [injected_passthrough(s) for s in steps]

        cfg_by_id: dict[str, FaultConfig] = {
            c.target_step_id: c for c in configs if c.target_step_id
        }
        result: list[InjectedStep] = []
        for step in steps:
            cfg = cfg_by_id.get(step.step_id)
            if cfg:
                result.append(self.inject(step, cfg.fault_type))
            else:
                result.append(injected_passthrough(step))
        return result

    # -- Random injection --

    def random_inject(self, steps: list[TaskStep],
                      fault_types: list[FaultType],
                      probability: float = 0.3) -> list[InjectedStep]:
        """Inject random fault types at a given probability per step."""
        if not self._enabled or self.strategy == InjectionStrategy.DISABLED:
            return [injected_passthrough(s) for s in steps]

        result: list[InjectedStep] = []
        for step in steps:
            if self._rng.random() < probability and fault_types:
                ft = self._rng.choice(fault_types)
                result.append(self.inject(step, ft, InjectionStrategy.RANDOM))
            else:
                result.append(injected_passthrough(step))
        return result

    # -- Internal --

    def _log_injection(self, step_id: str, ft: FaultType,
                       strat: InjectionStrategy, msg: str) -> None:
        record = InjectionRecord(
            timestamp=time.time(), step_id=step_id, fault_type=ft,
            strategy=strat, error_message=msg,
        )
        self.injection_log.append(record)
        logger.debug("Injected %s into step=%s strategy=%s", ft.value, step_id, strat.value)

    def get_log(self) -> list[InjectionRecord]:
        """Return the full injection audit log."""
        return list(self.injection_log)


def injected_passthrough(step: TaskStep) -> InjectedStep:
    """Pass through a step without injecting any fault."""
    return InjectedStep(
        original=step, fault_type=FaultType.NONE,
        injected_output=step.expected_output, injected_error=None, is_timeout=False,
    )


# ---------------------------------------------------------------------------
# Recovery Evaluator
# ---------------------------------------------------------------------------

_RECOVERY_SIGNALS = {
    "retry", "fallback", "alternative", "recover", "reconnect",
    "re-request", "degrade", "graceful", "resume", "circuit",
    "重新", "重试", "回退", "备选", "恢复", "降级", "优雅",
}

_SUSPICIOUS_SIGNALS = {
    "intentional", "expected", "planned", "deliberately",
    "故意的", "预期的", "计划好的",
}


class RecoveryEvaluator:
    """
    Recovery capability evaluator.

    Uses deterministic rules (not LLM scoring) to evaluate agent recovery behaviour.
    """

    def evaluate_recovery(
        self,
        original_task: TaskStep,
        injected_task: InjectedStep,
        agent_response: str | dict[str, Any],
    ) -> RecoveryResult:
        """
        Evaluate an agent's recovery from an injected fault.

        Args:
            original_task: The original task step.
            injected_task: The step after fault injection.
            agent_response: The agent's actual response (text or dict).

        Returns:
            RecoveryResult
        """
        response_text = agent_response if isinstance(agent_response, str) else str(agent_response)
        response_lower = response_text.lower()

        suspicious = _detect_goodhart(response_lower)
        if suspicious:
            return RecoveryResult(
                recovered=False, recovery_quality=0.0,
                failure_reason=f"Goodhart suspicion: {suspicious}",
            )

        recovery_steps = _extract_recovery_steps(response_lower)
        recovered = len(recovery_steps) > 0

        if not recovered:
            return RecoveryResult(
                recovered=False, recovery_quality=0.0,
                failure_reason="No recovery behavior detected in response",
            )

        quality = _score_recovery(recovery_steps, response_lower, original_task)

        return RecoveryResult(
            recovered=True,
            recovery_steps=recovery_steps,
            recovery_time=_estimate_recovery_time(response_lower),
            recovery_quality=quality,
        )

    def evaluate_batch(
        self,
        cases: list[tuple[TaskStep, InjectedStep, str | dict]],
    ) -> list[RecoveryResult]:
        """Batch-evaluate recovery."""
        return [self.evaluate_recovery(orig, inj, resp)
                for orig, inj, resp in cases]


def _detect_goodhart(text: str) -> str | None:
    """Detect Goodhart attack signals."""
    for sig in _SUSPICIOUS_SIGNALS:
        if sig in text:
            return sig
    return None


def _extract_recovery_steps(text: str) -> list[str]:
    """Extract recovery steps from agent response (deterministic keyword match)."""
    found = []
    for signal in _RECOVERY_SIGNALS:
        if signal in text:
            found.append(signal)
    return found


def _score_recovery(steps: list[str], full_text: str, original: TaskStep) -> float:
    """
    Deterministic scoring: recovery step diversity + goal achievement.

    Does not use LLM scoring.
    """
    diversity = min(len(set(steps)) / 3.0, 1.0) * 0.4

    target_words = [w for w in original.description.lower().split() if len(w) > 3]
    target_hits = sum(1 for w in target_words if w in full_text)
    target_score = min(target_hits / max(len(target_words), 1), 1.0) * 0.4

    length = len(full_text)
    if 50 <= length <= 2000:
        length_score = 0.2
    elif length > 2000:
        length_score = 0.1
    else:
        length_score = 0.05

    return round(diversity + target_score + length_score, 2)


def _estimate_recovery_time(text: str) -> float:
    """Roughly estimate recovery time from time cues in the response."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*(ms|sec|s|min)\b', text)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "ms":
        return val / 1000.0
    if unit == "min":
        return val * 60.0
    return val
