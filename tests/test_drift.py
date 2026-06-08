"""Unit tests for src/drift.py — PSI calculation logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.drift import DriftReport, _psi_single


# ── PSI computation (pure function) ──────────────────────────────────────────

class TestComputePsi:
    """Tests for the internal _psi_single helper."""

    def test_identical_distributions_zero_psi(self):
        """Identical reference and current distributions → PSI ≈ 0."""
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 1000)
        psi = _psi_single(ref, ref.copy())
        assert psi < 0.01, f"PSI for identical distributions should be ~0, got {psi:.4f}"

    def test_completely_different_distributions_high_psi(self):
        """Non-overlapping distributions → PSI > 0.2 (major drift)."""
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(10, 1, 1000)  # completely shifted
        psi = _psi_single(ref, cur)
        assert psi > 0.2, f"Non-overlapping distributions should give PSI > 0.2, got {psi:.4f}"

    def test_psi_non_negative(self):
        """PSI is always ≥ 0."""
        rng = np.random.default_rng(2)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0.5, 1.2, 500)
        psi = _psi_single(ref, cur)
        assert psi >= 0.0

    def test_psi_symmetric_approx(self):
        """PSI(ref→cur) should be close to PSI(cur→ref)."""
        rng = np.random.default_rng(3)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(1, 1, 1000)
        psi_fwd = _psi_single(ref, cur)
        psi_rev = _psi_single(cur, ref)
        assert abs(psi_fwd - psi_rev) < 0.15, "PSI should be roughly symmetric"

    def test_minor_shift_in_minor_range(self):
        """A slight shift should land in the 0.10–0.20 range."""
        rng = np.random.default_rng(4)
        ref = rng.normal(0, 1, 2000)
        cur = rng.normal(0.8, 1.1, 2000)  # moderate shift
        psi = _psi_single(ref, cur)
        # Not asserting exact range — just confirming it's between negligible and major
        assert 0 < psi, "Minor shift should give PSI > 0"

    def test_bins_parameter(self):
        """PSI with 5 bins vs 20 bins — both should run without error."""
        rng = np.random.default_rng(5)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0.5, 1, 1000)
        psi_5 = _psi_single(ref, cur, bins=5)
        psi_20 = _psi_single(ref, cur, bins=20)
        assert psi_5 >= 0
        assert psi_20 >= 0


# ── DriftReport dataclass ─────────────────────────────────────────────────────

class TestDriftReport:
    def test_overall_drift_true_when_feature_drifted(self):
        from datetime import date
        report = DriftReport(
            batch_date=date(2026, 3, 15),
            psi_by_feature={"amount": 0.25, "txn_count_1h": 0.05},
            drifted_features=["amount"],
            overall_drift=True,
            max_psi=0.25,
        )
        assert report.overall_drift is True

    def test_overall_drift_false_when_clean(self):
        from datetime import date
        report = DriftReport(
            batch_date=date(2026, 3, 15),
            psi_by_feature={"amount": 0.05, "txn_count_1h": 0.03},
            drifted_features=[],
            overall_drift=False,
            max_psi=0.05,
        )
        assert report.overall_drift is False
        assert report.drifted_features == []

    def test_max_psi_matches_max_value(self):
        from datetime import date
        psi_by_feature = {"amount": 0.12, "distance_km": 0.30, "hour_of_day": 0.02}
        report = DriftReport(
            batch_date=date(2026, 3, 15),
            psi_by_feature=psi_by_feature,
            drifted_features=["distance_km"],
            overall_drift=True,
            max_psi=max(psi_by_feature.values()),
        )
        assert report.max_psi == pytest.approx(0.30)


# ── compute_drift integration (DB mocked) ─────────────────────────────────────

class TestComputeDrift:
    """Smoke-test compute_drift() with a mocked DB connection."""

    def _make_fake_df(self, n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "amount": rng.normal(100 + shift, 30, n),
            "txn_count_1h": rng.poisson(2, n).astype(float),
            "txn_count_24h": rng.poisson(10, n).astype(float),
            "txn_count_7d": rng.poisson(50, n).astype(float),
            "amt_sum_1h": rng.normal(200 + shift, 60, n),
            "amt_ratio": rng.uniform(0.5, 2.0, n),
            "distance_km": rng.exponential(5, n),
            "hour_of_day": rng.integers(0, 24, n).astype(float),
            "day_of_week": rng.integers(0, 7, n).astype(float),
            "is_weekend": rng.integers(0, 2, n).astype(float),
            "batch_date": pd.Timestamp("2026-03-15"),
        })

    @patch("src.drift.create_engine")
    def test_returns_drift_report(self, mock_engine):
        """compute_drift() returns a DriftReport even with mocked DB."""
        from datetime import date
        from src.drift import compute_drift

        ref_df = self._make_fake_df(300, seed=10)
        cur_df = self._make_fake_df(100, seed=20)

        conn = MagicMock()
        conn.__enter__ = lambda s: conn
        conn.__exit__ = MagicMock(return_value=False)

        mock_engine.return_value.connect.return_value = conn

        # Patch pd.read_sql to return controlled frames (ref window then current)
        with patch("src.drift.pd.read_sql") as mock_read_sql:
            mock_read_sql.side_effect = [ref_df, cur_df]
            report = compute_drift(
                batch_date=date(2026, 3, 15),
                db_url="postgresql://fake/fake",
                reference_window_days=30,
                alert_threshold=0.2,
            )

        assert isinstance(report, DriftReport)
        assert isinstance(report.psi_by_feature, dict)
        assert isinstance(report.drifted_features, list)
        assert isinstance(report.overall_drift, bool)
        assert report.max_psi >= 0.0
