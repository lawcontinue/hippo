"""Tests for hippo.eval.drift."""

import json

import numpy as np
import pytest

from hippo.eval.drift import DriftDetector, DriftResult, DriftSeverity


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def ref_data(rng):
    return rng.normal(0, 1, size=(500,))


@pytest.fixture
def detector():
    return DriftDetector(
        n_bins=20,
        alert_threshold=0.15,
        warning_threshold=0.05,
        critical_threshold=0.50,
    )


# 1. No drift: same distribution → KL ≈ 0


def test_no_drift_same_distribution(detector, rng):
    ref = rng.normal(0, 1, size=(2000,))
    detector.fit_reference(ref)
    prod = rng.normal(0, 1, size=(2000,))
    result = detector.detect(prod)

    assert result.severity in (DriftSeverity.NONE, DriftSeverity.WARNING)
    assert not result.is_drifted
    assert result.kl_divergence < 0.10


# 2. Slight drift → WARNING


def test_slight_drift_warning(detector, ref_data, rng):
    detector.fit_reference(ref_data)
    prod = rng.normal(0.5, 1, size=(500,))
    result = detector.detect(prod)

    assert result.kl_divergence > 0.01
    assert result.severity in (DriftSeverity.WARNING, DriftSeverity.ALERT)


# 3. Severe drift → ALERT or CRITICAL


def test_severe_drift_alert(detector, ref_data, rng):
    detector.fit_reference(ref_data)
    prod = rng.normal(2.0, 1, size=(500,))
    result = detector.detect(prod)

    assert result.is_drifted
    assert result.severity in (DriftSeverity.ALERT, DriftSeverity.CRITICAL)
    assert result.kl_divergence > 0.15


def test_extreme_drift_critical(detector, ref_data, rng):
    detector.fit_reference(ref_data)
    prod = rng.uniform(5, 10, size=(500,))
    result = detector.detect(prod)

    assert result.severity == DriftSeverity.CRITICAL
    assert result.kl_divergence > 0.50


# 4. Multi-dimensional features


def test_multidimensional_features(detector, rng):
    ref = rng.normal(0, 1, size=(300, 3))
    detector.fit_reference(ref)

    prod = rng.normal(0, 1, size=(300, 3))
    prod[:, 0] = rng.normal(3.0, 1, size=300)

    result = detector.detect(prod)
    assert result.is_drifted
    assert result.metadata["n_dims"] == 3
    assert result.metadata["worst_dim"] == 0


def test_max_aggregation(rng):
    ref = rng.normal(0, 1, size=(300, 3))

    det_mean = DriftDetector(aggregation="mean", n_bins=20)
    det_max = DriftDetector(aggregation="max", n_bins=20)

    det_mean.fit_reference(ref)
    det_max.fit_reference(ref)

    prod = rng.normal(0, 1, size=(300, 3))
    prod[:, 1] = rng.normal(2.5, 1, size=300)

    r_mean = det_mean.detect(prod)
    r_max = det_max.detect(prod)

    assert r_max.kl_divergence >= r_mean.kl_divergence


# 5. Stratified confidence sampling


def test_stratified_sampling_high_confidence_low_rate(detector):
    items = [
        {"id": i, "confidence": 0.95 if i < 90 else 0.3}
        for i in range(100)
    ]
    sampled = detector.stratified_sample(
        items,
        high_threshold=0.8,
        high_sample_rate=0.1,
        low_sample_rate=1.0,
        seed=42,
    )

    high_sampled = [s for s in sampled if s["confidence"] >= 0.8]
    low_sampled = [s for s in sampled if s["confidence"] < 0.8]

    assert len(high_sampled) <= 10
    assert len(low_sampled) == 10


def test_stratified_sampling_all_low_confidence(detector):
    items = [{"id": i, "confidence": 0.2} for i in range(50)]
    sampled = detector.stratified_sample(
        items, high_threshold=0.8, low_sample_rate=1.0, seed=42
    )
    assert len(sampled) == 50


# 6. Parameter validation


def test_invalid_n_bins():
    with pytest.raises(ValueError, match="n_bins"):
        DriftDetector(n_bins=1)


def test_invalid_threshold_order():
    with pytest.raises(ValueError, match="Thresholds"):
        DriftDetector(
            alert_threshold=0.1,
            warning_threshold=0.2,
        )


def test_detect_before_fit(detector, rng):
    with pytest.raises(RuntimeError, match="fit_reference"):
        detector.detect(rng.normal(0, 1, size=(100,)))


# 7. JS divergence bounded


def test_js_divergence_bounded(detector, ref_data, rng):
    detector.fit_reference(ref_data)
    prod = rng.normal(5, 2, size=(500,))
    result = detector.detect(prod)
    assert result.js_divergence <= 0.694  # ln(2) + tolerance


# 8. Serialisation


def test_result_to_dict_serializable(detector, ref_data, rng):
    detector.fit_reference(ref_data)
    result = detector.detect(rng.normal(0, 1, size=(200,)))

    d = result.to_dict()
    json_str = json.dumps(d)
    assert "kl_divergence" in d
    assert "severity" in d
