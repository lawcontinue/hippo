"""
test_acceleration.py — AccelerationStrategy 抽象层测试

来源：熔炉 #33（2026-05-23）
覆盖：策略可用性、加速估算、适配度评分、自动选择、语言检测、安全回退
"""

import pytest
from acceleration import (
    AccelerationContext,
    AccelerationOrchestrator,
    AccelerationResult,
    DFlashStrategy,
    Hardware,
    Language,
    MTPStrategy,
    NoneStrategy,
    PipelineStrategy,
    StrategyName,
    TaskType,
    auto_accelerate,
    detect_language,
)


# ─── Fixtures ───────────────────────────────────────────


@pytest.fixture
def dflash() -> DFlashStrategy:
    return DFlashStrategy()


@pytest.fixture
def mtp() -> MTPStrategy:
    return MTPStrategy()


@pytest.fixture
def pipeline() -> PipelineStrategy:
    return PipelineStrategy()


@pytest.fixture
def none_strat() -> NoneStrategy:
    return NoneStrategy()


@pytest.fixture
def orch() -> AccelerationOrchestrator:
    return AccelerationOrchestrator()


# ─── DFlash Tests ───────────────────────────────────────


class TestDFlash:
    def test_available_on_apple_silicon(self, dflash: DFlashStrategy):
        ctx = AccelerationContext(hardware=Hardware.APPLE_SILICON)
        assert dflash.is_available(ctx) is True

    def test_not_available_on_nvidia(self, dflash: DFlashStrategy):
        ctx = AccelerationContext(hardware=Hardware.NVIDIA_GPU)
        assert dflash.is_available(ctx) is False

    def test_not_available_on_unknown(self, dflash: DFlashStrategy):
        ctx = AccelerationContext(hardware=Hardware.UNKNOWN)
        assert dflash.is_available(ctx) is False

    def test_speedup_chinese_penalty(self, dflash: DFlashStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            language=Language.CHINESE,
        )
        assert dflash.estimate_speedup(ctx) == 0.81

    def test_speedup_code_peak(self, dflash: DFlashStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            language=Language.CODE,
        )
        assert dflash.estimate_speedup(ctx) == 7.7

    def test_speedup_mixed_average(self, dflash: DFlashStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            language=Language.MIXED,
        )
        assert dflash.estimate_speedup(ctx) == 4.08

    def test_suitability_chinese_low(self, dflash: DFlashStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            language=Language.CHINESE,
        )
        assert dflash.suitability_score(ctx) == 0.1

    def test_suitability_code_high(self, dflash: DFlashStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            language=Language.CODE,
        )
        assert dflash.suitability_score(ctx) == 0.95

    def test_suitability_unavailable_zero(self, dflash: DFlashStrategy):
        ctx = AccelerationContext(hardware=Hardware.NVIDIA_GPU)
        assert dflash.suitability_score(ctx) == 0.0


# ─── MTP Tests ──────────────────────────────────────────


class TestMTP:
    def test_available_deepseek_on_nvidia(self, mtp: MTPStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.NVIDIA_GPU,
            model_name="deepseek-v3-0324",
        )
        assert mtp.is_available(ctx) is True

    def test_available_gemma4_on_nvidia(self, mtp: MTPStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.NVIDIA_GPU,
            model_name="gemma-4-31b",
        )
        assert mtp.is_available(ctx) is True

    def test_not_available_on_apple_silicon(self, mtp: MTPStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            model_name="deepseek-v3",
        )
        assert mtp.is_available(ctx) is False

    def test_not_available_unsupported_model(self, mtp: MTPStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.NVIDIA_GPU,
            model_name="qwen3-4b",
        )
        assert mtp.is_available(ctx) is False

    def test_speedup_moe_batch1_low(self, mtp: MTPStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.NVIDIA_GPU,
            model_name="deepseek-v3-moe",
            batch_size=1,
        )
        assert mtp.estimate_speedup(ctx) == 1.1

    def test_speedup_normal(self, mtp: MTPStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.NVIDIA_GPU,
            model_name="deepseek-v3",
            batch_size=4,
        )
        assert mtp.estimate_speedup(ctx) == 2.0

    def test_suitability_chinese_high(self, mtp: MTPStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.NVIDIA_GPU,
            model_name="deepseek-v3",
            language=Language.CHINESE,
        )
        assert mtp.suitability_score(ctx) == 0.85


# ─── Pipeline Tests ─────────────────────────────────────


class TestPipeline:
    def test_available_on_apple_silicon(self, pipeline: PipelineStrategy):
        ctx = AccelerationContext(hardware=Hardware.APPLE_SILICON)
        assert pipeline.is_available(ctx) is True

    def test_not_available_on_nvidia(self, pipeline: PipelineStrategy):
        ctx = AccelerationContext(hardware=Hardware.NVIDIA_GPU)
        assert pipeline.is_available(ctx) is False

    def test_suitability_small_model_zero(self, pipeline: PipelineStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            model_name="qwen3-4b",
        )
        assert pipeline.suitability_score(ctx) == 0.0

    def test_suitability_large_model(self, pipeline: PipelineStrategy):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            model_name="qwen3-72b",
        )
        assert pipeline.suitability_score(ctx) > 0.0

    def test_speedup_48x(self, pipeline: PipelineStrategy):
        ctx = AccelerationContext(hardware=Hardware.APPLE_SILICON)
        assert pipeline.estimate_speedup(ctx) == 48.0


# ─── None Strategy Tests ────────────────────────────────


class TestNoneStrategy:
    def test_always_available(self, none_strat: NoneStrategy):
        for hw in Hardware:
            ctx = AccelerationContext(hardware=hw)
            assert none_strat.is_available(ctx) is True

    def test_speedup_always_1(self, none_strat: NoneStrategy):
        ctx = AccelerationContext()
        assert none_strat.estimate_speedup(ctx) == 1.0

    def test_suitability_low(self, none_strat: NoneStrategy):
        ctx = AccelerationContext()
        assert none_strat.suitability_score(ctx) == 0.3


# ─── Orchestrator Tests ────────────────────────────────


class TestOrchestrator:
    def test_mac_code_prefers_dflash(self, orch: AccelerationOrchestrator):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            language=Language.CODE,
            task_type=TaskType.CODE_GEN,
            model_name="qwen3-4b",
        )
        result = orch.select(ctx)
        assert result.strategy == StrategyName.DFLASH
        assert result.estimated_speedup == 7.7

    def test_nvidia_deepseek_prefers_mtp(self, orch: AccelerationOrchestrator):
        ctx = AccelerationContext(
            hardware=Hardware.NVIDIA_GPU,
            language=Language.CHINESE,
            model_name="deepseek-v3",
            batch_size=4,
        )
        result = orch.select(ctx)
        assert result.strategy == StrategyName.MTP

    def test_mac_chinese_avoids_dflash(self, orch: AccelerationOrchestrator):
        """中文场景 DFlash 0.81× → 自动回退到 None。"""
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            language=Language.CHINESE,
            model_name="qwen3-4b",
        )
        result = orch.select(ctx)
        # DFlash score=0.1 but speedup=0.81 → fallback to None
        assert result.strategy == StrategyName.NONE
        assert result.estimated_speedup == 1.0

    def test_mac_large_model_prefers_pipeline(self, orch: AccelerationOrchestrator):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            language=Language.ENGLISH,
            model_name="qwen3-72b",
        )
        result = orch.select(ctx)
        assert result.strategy == StrategyName.PIPELINE

    def test_unknown_hardware_fallback_none(self, orch: AccelerationOrchestrator):
        ctx = AccelerationContext(
            hardware=Hardware.UNKNOWN,
            model_name="qwen3-4b",
        )
        result = orch.select(ctx)
        assert result.strategy == StrategyName.NONE

    def test_nvidia_unsupported_model_fallback_none(self, orch: AccelerationOrchestrator):
        ctx = AccelerationContext(
            hardware=Hardware.NVIDIA_GPU,
            model_name="qwen3-4b",
        )
        result = orch.select(ctx)
        assert result.strategy == StrategyName.NONE

    def test_override_dflash(self, orch: AccelerationOrchestrator):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            model_name="qwen3-4b",
        )
        result = orch.select_with_override(ctx, StrategyName.DFLASH)
        assert result.strategy == StrategyName.DFLASH
        assert "override" in result.reason.lower()

    def test_override_unavailable_fallback(self, orch: AccelerationOrchestrator):
        ctx = AccelerationContext(
            hardware=Hardware.NVIDIA_GPU,
            model_name="qwen3-4b",
        )
        result = orch.select_with_override(ctx, StrategyName.DFLASH)
        assert result.strategy == StrategyName.NONE
        assert result.confidence == 0.0

    def test_confidence_high_when_clear_winner(self, orch: AccelerationOrchestrator):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            language=Language.CODE,
            model_name="qwen3-4b",
        )
        result = orch.select(ctx)
        assert result.confidence > 0.7

    def test_config_hint_dflash(self, orch: AccelerationOrchestrator):
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            language=Language.CODE,
            model_name="qwen3-4b",
        )
        result = orch.select(ctx)
        assert "block_size" in result.config_hint
        assert result.config_hint["block_size"] == 16

    def test_config_hint_mtp(self, orch: AccelerationOrchestrator):
        ctx = AccelerationContext(
            hardware=Hardware.NVIDIA_GPU,
            model_name="deepseek-v3",
            batch_size=4,
        )
        result = orch.select(ctx)
        assert "method" in result.config_hint

    def test_config_hint_none(self):
        result = AccelerationResult(
            strategy=StrategyName.NONE,
            reason="test",
        )
        assert result.config_hint == {}


# ─── Language Detection Tests ───────────────────────────


class TestLanguageDetection:
    def test_chinese(self):
        assert detect_language("你好，世界") == Language.CHINESE

    def test_english(self):
        assert detect_language("Hello world") == Language.ENGLISH

    def test_code(self):
        assert detect_language("def foo():\n    return 42") == Language.CODE

    def test_mixed(self):
        assert detect_language("Hello 世界") == Language.MIXED

    def test_empty(self):
        assert detect_language("") == Language.UNKNOWN

    def test_code_with_backticks(self):
        assert detect_language("```python\nprint('hi')\n```") == Language.CODE


# ─── Auto Accelerate Convenience Tests ──────────────────


class TestAutoAccelerate:
    def test_mac_code_auto(self):
        result = auto_accelerate(
            hardware="apple_silicon",
            model_name="qwen3-4b",
            text="def hello(): pass",
            task_type="code_gen",
        )
        assert result.strategy == StrategyName.DFLASH

    def test_nvidia_deepseek_auto(self):
        result = auto_accelerate(
            hardware="nvidia_gpu",
            model_name="deepseek-v3",
            text="你好世界",
            task_type="chat",
            batch_size=4,
        )
        assert result.strategy == StrategyName.MTP

    def test_chinese_mac_safety_fallback(self):
        """中文+Mac → DFlash减速 → 自动回退None。"""
        result = auto_accelerate(
            hardware="apple_silicon",
            model_name="qwen3-4b",
            text="今天天气很好",
            task_type="chat",
        )
        assert result.strategy == StrategyName.NONE
        assert result.estimated_speedup == 1.0


# ─── Boundary / Edge Case Tests ────────────────────────


class TestBoundaryConditions:
    def test_empty_model_name(self):
        result = auto_accelerate(
            hardware="apple_silicon",
            model_name="",
            text="hello",
        )
        assert result.strategy in (StrategyName.DFLASH, StrategyName.NONE)

    def test_long_text(self):
        """超长文本不应崩溃。"""
        text = "你好世界" * 5000  # ~20K chars
        result = auto_accelerate(
            hardware="apple_silicon",
            model_name="qwen3-4b",
            text=text,
        )
        assert result.strategy in (StrategyName.DFLASH, StrategyName.NONE, StrategyName.PIPELINE)

    def test_special_chars(self):
        result = auto_accelerate(
            hardware="apple_silicon",
            model_name="qwen3-4b",
            text="!@#$%^&*()_+-=[]{}|;':\",./<>?",
        )
        assert result.strategy in (StrategyName.DFLASH, StrategyName.NONE, StrategyName.PIPELINE)

    def test_none_model_name(self):
        result = auto_accelerate(
            hardware="apple_silicon",
            model_name="",
            text="hello",
        )
        assert result is not None

    def test_override_dflash_chinese_warns_slowdown(self, orch: AccelerationOrchestrator):
        """用户强制 DFlash 在中文场景应返回 speedup<1.0 且带警告。"""
        ctx = AccelerationContext(
            hardware=Hardware.APPLE_SILICON,
            language=Language.CHINESE,
            model_name="qwen3-4b",
        )
        result = orch.select_with_override(ctx, StrategyName.DFLASH)
        assert result.strategy == StrategyName.DFLASH
        assert result.estimated_speedup == 0.81
        assert "WARNING" in result.reason

    def test_mtp_model_registration(self):
        """自定义 MTP 模型注册。"""
        MTPStrategy.register_mtp_model("my-custom-mtp-model")
        mtp = MTPStrategy()
        ctx = AccelerationContext(
            hardware=Hardware.NVIDIA_GPU,
            model_name="my-custom-mtp-model",
        )
        assert mtp.is_available(ctx) is True
