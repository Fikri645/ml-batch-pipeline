"""
Distribution drift detection using Population Stability Index (PSI).

PSI convention used here (Siddiqi 2006):
    PSI < 0.10  → negligible drift
    0.10–0.20   → minor drift (monitor)
    PSI > 0.20  → major drift (alert / retrain)

Reference distribution: rolling 30-day window of historical batch summaries.
Current distribution: today's batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.config import (
    DATABASE_URL,
    FEATURE_COLUMNS,
    PSI_ALERT_THRESHOLD,
    REFERENCE_WINDOW_DAYS,
)

logger = logging.getLogger(__name__)

N_BINS = 10  # number of histogram bins for PSI


@dataclass
class DriftReport:
    """Result of a PSI drift check for a single batch date."""

    batch_date: date
    psi_by_feature: Dict[str, float]
    drifted_features: List[str]
    overall_drift: bool
    max_psi: float

    def summary(self) -> str:
        status = "⚠️  DRIFT DETECTED" if self.overall_drift else "✅  No drift"
        lines = [f"[{self.batch_date}] {status} | max_psi={self.max_psi:.4f}"]
        for feat, psi in sorted(self.psi_by_feature.items(), key=lambda x: -x[1]):
            flag = " ← ALERT" if psi > PSI_ALERT_THRESHOLD else ""
            lines.append(f"  {feat:<30} PSI={psi:.4f}{flag}")
        return "\n".join(lines)


# ── PSI computation ───────────────────────────────────────────────────────────


def _psi_single(expected: np.ndarray, actual: np.ndarray, bins: int = N_BINS) -> float:
    """
    Compute PSI between two 1-D arrays.

    Both arrays are bucketed into *bins* equal-width intervals derived from
    the *expected* (reference) distribution.
    """
    # Guard against degenerate distributions
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    if np.std(expected) < 1e-9:
        return 0.0

    breakpoints = np.linspace(np.min(expected), np.max(expected), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual, bins=breakpoints)[0]

    # Replace zeros to avoid log(0)
    exp_pct = (exp_counts / len(expected)).clip(1e-6)
    act_pct = (act_counts / len(actual)).clip(1e-6)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_feature_window(
    start_date: date,
    end_date: date,
    db_url: str,
) -> pd.DataFrame:
    """Load feature rows from the mart for a date range."""
    engine = create_engine(db_url)
    query = text(
        """
        SELECT amount, txn_count_1h, txn_count_24h, txn_count_7d,
               amt_sum_1h, amt_ratio, distance_km, hour_of_day,
               day_of_week, is_weekend, batch_date
        FROM marts.fct_transaction_features
        WHERE batch_date BETWEEN :start AND :end
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"start": str(start_date), "end": str(end_date)})


# ── Public API ────────────────────────────────────────────────────────────────


def compute_drift(
    batch_date: date,
    db_url: str = DATABASE_URL,
    reference_window_days: int = REFERENCE_WINDOW_DAYS,
    alert_threshold: float = PSI_ALERT_THRESHOLD,
) -> DriftReport:
    """
    Compute feature-level PSI between *batch_date* and the preceding
    *reference_window_days* days.

    Parameters
    ----------
    batch_date : date
        Current batch to evaluate.
    db_url : str
        SQLAlchemy database URL.
    reference_window_days : int
        Number of days used as the reference distribution.
    alert_threshold : float
        PSI value above which a feature is flagged as drifted.

    Returns
    -------
    DriftReport
    """
    ref_end = batch_date - timedelta(days=1)
    ref_start = ref_end - timedelta(days=reference_window_days - 1)

    logger.info("Loading reference window %s → %s", ref_start, ref_end)
    ref_df = _load_feature_window(ref_start, ref_end, db_url)

    logger.info("Loading current batch %s", batch_date)
    curr_df = _load_feature_window(batch_date, batch_date, db_url)

    if ref_df.empty:
        logger.warning("No reference data available — skipping drift check.")
        return DriftReport(
            batch_date=batch_date,
            psi_by_feature={},
            drifted_features=[],
            overall_drift=False,
            max_psi=0.0,
        )

    numeric_features = [c for c in FEATURE_COLUMNS if c in ref_df.columns]
    psi_results: Dict[str, float] = {}

    for feat in numeric_features:
        psi = _psi_single(
            ref_df[feat].dropna().values,
            curr_df[feat].dropna().values,
        )
        psi_results[feat] = psi

    drifted = [f for f, p in psi_results.items() if p > alert_threshold]
    max_psi = max(psi_results.values()) if psi_results else 0.0

    report = DriftReport(
        batch_date=batch_date,
        psi_by_feature=psi_results,
        drifted_features=drifted,
        overall_drift=len(drifted) > 0,
        max_psi=max_psi,
    )

    logger.info("%s", report.summary())
    _persist_drift_report(report, db_url)
    return report


def _persist_drift_report(report: DriftReport, db_url: str) -> None:
    """Write PSI results to `audit.drift_reports`."""
    if not report.psi_by_feature:
        return
    rows = [
        {
            "batch_date": str(report.batch_date),
            "feature_name": feat,
            "psi": psi,
            "is_drifted": psi > PSI_ALERT_THRESHOLD,
            "checked_at": pd.Timestamp.utcnow().isoformat(),
        }
        for feat, psi in report.psi_by_feature.items()
    ]
    df = pd.DataFrame(rows)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM audit.drift_reports WHERE batch_date = :d"),
            {"d": str(report.batch_date)},
        )
    df.to_sql(
        "drift_reports",
        engine,
        schema="audit",
        if_exists="append",
        index=False,
    )
