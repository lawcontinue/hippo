"""
Drift Detector — distribution drift detection between eval sets and production traffic.

Core metric: KL divergence (Kullback-Leibler Divergence).
Also computes Jensen-Shannon divergence (symmetric, bounded).

Dependencies: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class DriftSeverity(str, Enum):
    """Severity levels for drift."""

    NONE = "none"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of a single drift detection run."""

    kl_divergence: float
    js_divergence: float
    severity: DriftSeverity
    is_drifted: bool
    bin_edges: np.ndarray
    ref_hist: np.ndarray
    prod_hist: np.ndarray
    per_bin_contribution: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        flags = {
            DriftSeverity.NONE: "✅",
            DriftSeverity.WARNING: "🟡",
            DriftSeverity.ALERT: "🔴",
            DriftSeverity.CRITICAL: "⛔",
        }
        return (
            f"{flags[self.severity]} KL={self.kl_divergence:.4f} "
            f"JS={self.js_divergence:.4f} severity={self.severity.value}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kl_divergence": round(self.kl_divergence, 6),
            "js_divergence": round(self.js_divergence, 6),
            "severity": self.severity.value,
            "is_drifted": self.is_drifted,
            "per_bin_contribution": self.per_bin_contribution.tolist(),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    KL-divergence-based distribution drift detector.

    Workflow:
    1. ``fit_reference()`` — establish baseline distribution from initial eval set features.
    2. ``detect()`` — check newly sampled production features for drift.
    3. Super-threshold → trigger sampling / adversarial construction pipeline.

    Features can be any numeric vector:
    - Text embeddings (e.g. bge-small-zh 512d)
    - Scalar values like response length, confidence scores
    - Custom combined features

    Multi-dimensional handling:
    - Scalar (1d): direct binning.
    - Vector (>1d): per-dimension KL then aggregate (mean or max),
      avoiding sparse high-dimensional histograms.
    """

    def __init__(
        self,
        n_bins: int = 20,
        alert_threshold: float = 0.15,
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.50,
        eps: float = 1e-10,
        aggregation: str = "mean",
    ):
        """
        Args:
            n_bins: Number of histogram bins (more bins = more sensitive, noisier).
            alert_threshold: KL divergence alert threshold; reaching it sets is_drifted=True.
            warning_threshold: Warning threshold (logged but does not trigger pipeline).
            critical_threshold: Severe drift threshold.
            eps: Smoothing constant to avoid log(0).
            aggregation: Multi-dimension aggregation strategy ("mean" or "max").
        """
        if n_bins < 2:
            raise ValueError(f"n_bins must be ≥ 2, got {n_bins}")
        if not 0 < warning_threshold < alert_threshold < critical_threshold:
            raise ValueError(
                f"Thresholds must satisfy 0 < warning({warning_threshold}) "
                f"< alert({alert_threshold}) < critical({critical_threshold})"
            )
        self.n_bins = n_bins
        self.alert_threshold = alert_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.eps = eps
        self.aggregation = aggregation
        self._ref_features: np.ndarray | None = None
        self._bin_edges: list[np.ndarray] | None = None

    # ---- Core methods ----

    def fit_reference(self, features: np.ndarray) -> None:
        """
        Fit baseline distribution from reference data (initial eval set).

        Args:
            features: shape (n_samples,) or (n_samples, n_dims)
        """
        features = self._ensure_2d(features)
        self._ref_features = features
        self._bin_edges = []
        for d in range(features.shape[1]):
            col = features[:, d]
            col_min, col_max = col.min(), col.max()
            if col_min == col_max:
                col_min -= 0.5
                col_max += 0.5
            edges = np.linspace(col_min, col_max + self.eps, self.n_bins + 1)
            self._bin_edges.append(edges)

    def detect(
        self,
        prod_features: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> DriftResult:
        """
        Detect whether production traffic has drifted from the reference distribution.

        Args:
            prod_features: shape (n_samples,) or (n_samples, n_dims)
            sample_weight: optional sample weights (reserved for future use).

        Returns:
            DriftResult
        """
        if self._ref_features is None or self._bin_edges is None:
            raise RuntimeError("Must call fit_reference() before detect()")

        prod_features = self._ensure_2d(prod_features)
        ref = self._ref_features

        if prod_features.shape[1] != ref.shape[1]:
            raise ValueError(
                f"Dimension mismatch: ref has {ref.shape[1]} dims, "
                f"prod has {prod_features.shape[1]} dims"
            )

        n_dims = ref.shape[1]
        kl_per_dim: list[float] = []
        js_per_dim: list[float] = []
        contributions_per_dim: list[np.ndarray] = []
        ref_hists: list[np.ndarray] = []
        prod_hists: list[np.ndarray] = []

        for d in range(n_dims):
            ref_col = ref[:, d]
            prod_col = prod_features[:, d]
            edges = self._bin_edges[d]

            ref_hist, _ = np.histogram(ref_col, bins=edges)
            prod_hist, _ = np.histogram(prod_col, bins=edges)

            ref_p = ref_hist.astype(float) / max(ref_hist.sum(), 1)
            prod_q = prod_hist.astype(float) / max(prod_hist.sum(), 1)

            ref_p = ref_p + self.eps
            prod_q = prod_q + self.eps
            ref_p = ref_p / ref_p.sum()
            prod_q = prod_q / prod_q.sum()

            kl = float(np.sum(ref_p * np.log(ref_p / prod_q)))
            js = self._js_divergence(ref_p, prod_q)
            contribution = ref_p * np.log(ref_p / prod_q)

            kl_per_dim.append(kl)
            js_per_dim.append(js)
            contributions_per_dim.append(contribution)
            ref_hists.append(ref_hist)
            prod_hists.append(prod_hist)

        if self.aggregation == "max":
            kl_agg = max(kl_per_dim)
            js_agg = max(js_per_dim)
            worst_dim = int(np.argmax(kl_per_dim))
        else:
            kl_agg = float(np.mean(kl_per_dim))
            js_agg = float(np.mean(js_per_dim))
            worst_dim = int(np.argmax(kl_per_dim))

        severity = self._classify(kl_agg)
        is_drifted = severity in (DriftSeverity.ALERT, DriftSeverity.CRITICAL)

        return DriftResult(
            kl_divergence=kl_agg,
            js_divergence=js_agg,
            severity=severity,
            is_drifted=is_drifted,
            bin_edges=self._bin_edges[worst_dim],
            ref_hist=ref_hists[worst_dim],
            prod_hist=prod_hists[worst_dim],
            per_bin_contribution=contributions_per_dim[worst_dim],
            metadata={
                "n_dims": n_dims,
                "worst_dim": worst_dim,
                "kl_per_dim": kl_per_dim,
                "js_per_dim": js_per_dim,
                "n_ref_samples": int(ref.shape[0]),
                "n_prod_samples": int(prod_features.shape[0]),
                "aggregation": self.aggregation,
            },
        )

    # ---- Stratified confidence sampling ----

    def stratified_sample(
        self,
        items: list[dict[str, Any]],
        confidence_key: str = "confidence",
        high_threshold: float = 0.8,
        high_sample_rate: float = 0.05,
        low_sample_rate: float = 1.0,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Stratified sampling by confidence (high-confidence sampled less frequently).

        Args:
            items: Production traffic items, each containing a confidence field.
            confidence_key: Field name for confidence.
            high_threshold: High/low confidence boundary.
            high_sample_rate: Sampling rate for high-confidence items.
            low_sample_rate: Sampling rate for low-confidence items.
            seed: Random seed.

        Returns:
            Sampled subset.
        """
        rng = np.random.RandomState(seed)
        high = []
        low = []
        for item in items:
            conf = item.get(confidence_key, 0.0)
            if conf >= high_threshold:
                high.append(item)
            else:
                low.append(item)

        n_high_sample = max(1, int(len(high) * high_sample_rate)) if high else 0
        n_low_sample = max(1, int(len(low) * low_sample_rate)) if low else 0

        sampled_high = rng.choice(
            len(high), size=min(n_high_sample, len(high)), replace=False
        ) if high else []
        sampled_low = rng.choice(
            len(low), size=min(n_low_sample, len(low)), replace=False
        ) if low else []

        return [high[i] for i in sampled_high] + [low[i] for i in sampled_low]

    # ---- Internal utilities ----

    @staticmethod
    def _ensure_2d(features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 1d or 2d array, got {arr.ndim}d")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input features contain NaN or Inf values")
        return arr

    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence (symmetric, bounded [0, ln2])."""
        m = 0.5 * (p + q)
        kl_pm = float(np.sum(p * np.log(p / m)))
        kl_qm = float(np.sum(q * np.log(q / m)))
        return 0.5 * (kl_pm + kl_qm)

    def _classify(self, kl: float) -> DriftSeverity:
        if kl >= self.critical_threshold:
            return DriftSeverity.CRITICAL
        if kl >= self.alert_threshold:
            return DriftSeverity.ALERT
        if kl >= self.warning_threshold:
            return DriftSeverity.WARNING
        return DriftSeverity.NONE
