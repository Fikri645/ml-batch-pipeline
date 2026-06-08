"""
Alerting and audit logging for the batch scoring pipeline.

Alerts are written to:
  1. Python logger (always)
  2. audit.pipeline_runs table (always)
  3. Slack webhook (optional — set SLACK_WEBHOOK_URL in .env)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
import pandas as pd
import requests
from sqlalchemy import create_engine

from src.config import DATABASE_URL
from src.drift import DriftReport

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


# ── Run summary ───────────────────────────────────────────────────────────────


@dataclass
class RunSummary:
    """Structured summary of a single pipeline run."""

    batch_date: date
    n_transactions: int
    n_fraud_flagged: int
    fraud_rate: float
    drift_detected: bool
    drifted_features: list[str]
    max_psi: float
    run_status: str  # "success" | "drift_alert" | "error"
    run_at: datetime = None

    def __post_init__(self):
        if self.run_at is None:
            self.run_at = datetime.utcnow()


# ── Persistence ───────────────────────────────────────────────────────────────


def log_run(summary: RunSummary, db_url: str = DATABASE_URL) -> None:
    """Persist a RunSummary to `audit.pipeline_runs`."""
    row = {
        "batch_date": str(summary.batch_date),
        "n_transactions": summary.n_transactions,
        "n_fraud_flagged": summary.n_fraud_flagged,
        "fraud_rate": round(summary.fraud_rate, 6),
        "drift_detected": summary.drift_detected,
        "drifted_features": json.dumps(summary.drifted_features),
        "max_psi": round(summary.max_psi, 6),
        "run_status": summary.run_status,
        "run_at": summary.run_at.isoformat(),
    }
    engine = create_engine(db_url)
    pd.DataFrame([row]).to_sql(
        "pipeline_runs",
        engine,
        schema="audit",
        if_exists="append",
        index=False,
    )
    logger.info("Run logged for %s — status=%s", summary.batch_date, summary.run_status)


# ── Slack notification ────────────────────────────────────────────────────────


def send_drift_alert(report: DriftReport, webhook_url: str = SLACK_WEBHOOK_URL) -> None:
    """
    Post a drift alert to Slack (if SLACK_WEBHOOK_URL is configured).

    The message summarises which features drifted and the max PSI value.
    Falls back to logger warning if no webhook is configured.
    """
    msg_text = (
        f":warning: *Drift alert* — batch `{report.batch_date}`\n"
        f"Max PSI: `{report.max_psi:.4f}`\n"
        f"Drifted features: `{', '.join(report.drifted_features) or 'none'}`"
    )

    if not webhook_url:
        logger.warning(
            "SLACK_WEBHOOK_URL not configured — drift alert logged only.\n%s",
            msg_text,
        )
        return

    try:
        resp = requests.post(
            webhook_url,
            json={"text": msg_text},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Drift alert sent to Slack for %s.", report.batch_date)
    except requests.RequestException as exc:
        logger.error("Failed to send Slack alert: %s", exc)


def send_clean_run_notification(
    summary: RunSummary,
    webhook_url: str = SLACK_WEBHOOK_URL,
) -> None:
    """
    Post a clean-run summary to Slack (if configured).
    """
    msg_text = (
        f":white_check_mark: *Clean run* — batch `{summary.batch_date}`\n"
        f"Transactions: `{summary.n_transactions}` | "
        f"Fraud flagged: `{summary.n_fraud_flagged}` ({summary.fraud_rate:.2%})"
    )

    if not webhook_url:
        logger.info("Clean run summary (no Slack):\n%s", msg_text)
        return

    try:
        resp = requests.post(
            webhook_url,
            json={"text": msg_text},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to send clean-run notification: %s", exc)
