"""Tests for hippo.eval.chaos."""

import pytest

from hippo.eval.chaos import (
    FaultType, InjectionStrategy, TaskStep, InjectedStep,
    FaultConfig, RecoveryResult, FaultInjector, RecoveryEvaluator,
    injected_passthrough,
)


@pytest.fixture
def sample_steps():
    return [
        TaskStep(step_id="s1", description="Fetch user data", tool="api",
                 params={"url": "/users/123"}, expected_output={"id": 123, "name": "Alice"}),
        TaskStep(step_id="s2", description="Parse response", tool="json",
                 params={}, expected_output={"parsed": True}),
        TaskStep(step_id="s3", description="Save to database", tool="db",
                 params={"table": "users"}, expected_output={"saved": True}),
    ]


@pytest.fixture
def enabled_injector():
    return FaultInjector.create_enabled(InjectionStrategy.KNOWN)


@pytest.fixture
def disabled_injector():
    return FaultInjector(InjectionStrategy.DISABLED)


class TestSafetyGuardrail:

    def test_dry_run_default_does_not_inject(self, sample_steps, disabled_injector):
        step = sample_steps[0]
        result = disabled_injector.inject(step, FaultType.NETWORK_TIMEOUT)
        assert result.injected_error is None
        assert result.injected_output == step.expected_output

    def test_must_enable_before_injection(self, sample_steps):
        fi = FaultInjector(InjectionStrategy.KNOWN)
        result = fi.inject(sample_steps[0], FaultType.TOOL_ERROR)
        assert result.injected_error is None
        assert len(fi.injection_log) == 0

    def test_injection_log_is_complete(self, sample_steps, enabled_injector):
        enabled_injector.inject(sample_steps[0], FaultType.NETWORK_TIMEOUT)
        log = enabled_injector.get_log()
        assert len(log) == 1
        assert log[0].fault_type == FaultType.NETWORK_TIMEOUT
        assert log[0].step_id == "s1"
        assert log[0].strategy == InjectionStrategy.KNOWN


class TestFaultTypes:

    @pytest.mark.parametrize("fault_type,expected_error_substr", [
        (FaultType.NETWORK_TIMEOUT, "timed out"),
        (FaultType.TOOL_ERROR, "failed"),
        (FaultType.MALFORMED_INPUT, "JSONDecodeError"),
        (FaultType.PERMISSION_DENIED, "403"),
        (FaultType.RESOURCE_EXHAUSTED, "limit exceeded"),
        (FaultType.PARTIAL_FAILURE, "3/5 failed"),
    ])
    def test_fault_type_produces_correct_error(
        self, sample_steps, enabled_injector, fault_type, expected_error_substr
    ):
        result = enabled_injector.inject(sample_steps[0], fault_type)
        assert result.injected_error is not None
        assert expected_error_substr.lower() in result.injected_error.lower()

    def test_network_timeout_sets_is_timeout_flag(self, sample_steps, enabled_injector):
        result = enabled_injector.inject(sample_steps[0], FaultType.NETWORK_TIMEOUT)
        assert result.is_timeout is True

    def test_non_timeout_does_not_set_is_timeout(self, sample_steps, enabled_injector):
        result = enabled_injector.inject(sample_steps[0], FaultType.TOOL_ERROR)
        assert result.is_timeout is False

    def test_simulate_agent_observation_permission(self, sample_steps, enabled_injector):
        result = enabled_injector.inject(sample_steps[0], FaultType.PERMISSION_DENIED)
        obs = result.simulate_agent_observation()
        assert obs["error"] == "PermissionDenied"

    def test_simulate_agent_observation_timeout(self, sample_steps, enabled_injector):
        result = enabled_injector.inject(sample_steps[0], FaultType.NETWORK_TIMEOUT)
        obs = result.simulate_agent_observation()
        assert obs["error"] == "TimeoutError"

    def test_custom_error_message_in_config(self, sample_steps):
        fi = FaultInjector.create_enabled(InjectionStrategy.KNOWN)
        step = sample_steps[0]
        result = fi.inject(step, FaultType.PERMISSION_DENIED)
        assert "403" in result.injected_error


class TestInjectionStrategies:

    def test_disabled_strategy_passthrough(self, sample_steps):
        fi = FaultInjector.create_enabled(InjectionStrategy.DISABLED)
        fi.enable()
        result = fi.inject(sample_steps[0], FaultType.TOOL_ERROR)
        assert result.injected_error is None

    def test_known_strategy_injects_at_specified_location(self, sample_steps):
        fi = FaultInjector.create_enabled(InjectionStrategy.KNOWN)
        configs = [FaultConfig(fault_type=FaultType.TOOL_ERROR, target_step_id="s2")]
        results = fi.batch_inject(sample_steps, configs)
        assert results[0].injected_error is None
        assert results[1].injected_error is not None
        assert results[2].injected_error is None

    def test_random_strategy_uses_probability(self, sample_steps):
        fi = FaultInjector.create_enabled(InjectionStrategy.RANDOM, seed=42)
        results = fi.random_inject(sample_steps, [FaultType.TOOL_ERROR], probability=0.0)
        assert all(r.injected_error is None for r in results)

        fi2 = FaultInjector.create_enabled(InjectionStrategy.RANDOM, seed=42)
        results2 = fi2.random_inject(sample_steps, [FaultType.TOOL_ERROR], probability=1.0)
        assert all(r.injected_error is not None for r in results2)


class TestRecoveryEvaluator:

    def test_successful_recovery_detected(self):
        original = TaskStep("s1", "Fetch user data from API", "api", expected_output={"ok": True})
        injected = InjectedStep(original, FaultType.NETWORK_TIMEOUT, None, "timed out", True)
        response = "I detected a timeout. I will retry the API call with exponential backoff."
        evaluator = RecoveryEvaluator()
        result = evaluator.evaluate_recovery(original, injected, response)
        assert result.recovered is True
        assert "retry" in result.recovery_steps
        assert result.recovery_quality > 0

    def test_no_recovery_behavior_detected(self):
        original = TaskStep("s1", "Fetch user data", "api")
        injected = InjectedStep(original, FaultType.TOOL_ERROR, None, "failed", False)
        response = "The task failed."
        evaluator = RecoveryEvaluator()
        result = evaluator.evaluate_recovery(original, injected, response)
        assert result.recovered is False
        assert "No recovery" in (result.failure_reason or "")

    def test_goodhart_attack_detected(self):
        original = TaskStep("s1", "Fetch data", "api")
        injected = InjectedStep(original, FaultType.TOOL_ERROR, None, "failed", False)
        response = "This was an intentional failure. I will now recover to show my ability."
        evaluator = RecoveryEvaluator()
        result = evaluator.evaluate_recovery(original, injected, response)
        assert result.recovered is False
        assert "Goodhart" in (result.failure_reason or "")

    def test_recovery_quality_score_range(self):
        original = TaskStep("s1", "Fetch user data from API", "api")
        injected = InjectedStep(original, FaultType.NETWORK_TIMEOUT, None, "timeout", True)
        response = "Timeout detected. I will retry with fallback alternative and reconnect."
        evaluator = RecoveryEvaluator()
        result = evaluator.evaluate_recovery(original, injected, response)
        assert 0.0 <= result.recovery_quality <= 1.0

    def test_batch_evaluation(self):
        evaluator = RecoveryEvaluator()
        orig = TaskStep("s1", "Fetch data", "api")
        inj = InjectedStep(orig, FaultType.TOOL_ERROR, None, "error", False)
        cases = [
            (orig, inj, "I will retry the request."),
            (orig, inj, "Failed."),
            (orig, inj, "Reconnecting with fallback."),
        ]
        results = evaluator.evaluate_batch(cases)
        assert len(results) == 3
        assert results[0].recovered is True
        assert results[1].recovered is False
        assert results[2].recovered is True


class TestEdgeCases:

    def test_empty_steps_list(self):
        fi = FaultInjector.create_enabled(InjectionStrategy.KNOWN)
        results = fi.batch_inject([], [])
        assert results == []

    def test_random_inject_empty_fault_types(self, sample_steps):
        fi = FaultInjector.create_enabled(InjectionStrategy.RANDOM, seed=1)
        results = fi.random_inject(sample_steps, [], probability=1.0)
        assert all(r.injected_error is None for r in results)

    def test_passthrough_preserves_expected_output(self, sample_steps):
        step = sample_steps[0]
        result = injected_passthrough(step)
        assert result.injected_output == step.expected_output
        assert result.injected_error is None

    def test_batch_inject_config_targets_missing_step(self, sample_steps):
        fi = FaultInjector.create_enabled(InjectionStrategy.KNOWN)
        configs = [FaultConfig(fault_type=FaultType.TOOL_ERROR, target_step_id="nonexistent")]
        results = fi.batch_inject(sample_steps, configs)
        assert len(results) == 3
        assert all(r.injected_error is None for r in results)
