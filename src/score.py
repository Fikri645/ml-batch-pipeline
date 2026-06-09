"""
Batch scoring module.

Loads the pre-trained LightGBM fraud model and scores a batch of transactions
whose features have already been computed by the dbt marts layer.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DATABASE_URL, FEATURE_COLUMNS, MODEL_PATH

logger = logging.getLogger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model(model_path=MODEL_PATH):
    """Load the joblib-serialised LightGBM pipeline. Accepts str or Path."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Run `python scripts/train_model.py` first."
        )
    model = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)
    return model


# ── Feature loading ───────────────────────────────────────────────────────────


def load_features(batch_date: date, db_url: str = DATABASE_URL) -> pd.DataFrame:
    """
    Read features for *batch_date* from the dbt mart table.

    The mart `marts.fct_transaction_features` is populated by dbt before
    this function is called in the Airflow DAG.
    """
    engine = create_engine(db_url)
    query = text(
        """
        SELECT
            transaction_id,
            card_number,
            batch_date,
            amount,
            txn_count_1h,
            txn_count_24h,
            txn_count_7d,
            amt_sum_1h,
            amt_ratio,
            distance_km,
            hour_of_day,
            day_of_week,
            is_weekend,
            is_fraud
        FROM marts.fct_transaction_features
        WHERE batch_date = :batch_date
        ORDER BY transaction_time
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"batch_date": str(batch_date)})
    logger.info("Loaded %d feature rows for %s", len(df), batch_date)
    return df


# ── Scoring ───────────────────────────────────────────────────────────────────


def score_batch(
    batch_date: date,
    model_path: Path = MODEL_PATH,
    db_url: str = DATABASE_URL,
    fraud_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    End-to-end batch scoring for *batch_date*.

    1. Load model artifact.
    2. Load features from the dbt mart.
    3. Predict fraud probability.
    4. Persist predictions to `scores.daily_predictions`.
    5. Return the scored DataFrame.

    Parameters
    ----------
    batch_date : date
        The date whose transactions should be scored.
    model_path : Path
        Path to the joblib model artifact.
    db_url : str
        SQLAlchemy database URL.
    fraud_threshold : float
        Decision boundary for the binary fraud flag.

    Returns
    -------
    pd.DataFrame with columns: transaction_id, fraud_probability, predicted_fraud.
    """
    model = load_model(model_path)
    features_df = load_features(batch_date, db_url)

    if features_df.empty:
        logger.warning("No features found for %s — skipping scoring.", batch_date)
        return pd.DataFrame()

    X = features_df[FEATURE_COLUMNS].fillna(0)
    proba = model.predict_proba(X)[:, 1]

    scores_df = features_df[["transaction_id", "card_number", "batch_date"]].copy()
    scores_df["fraud_probability"] = proba.round(6)
    scores_df["predicted_fraud"] = (proba >= fraud_threshold).astype(int)
    scores_df["scored_at"] = pd.Timestamp.utcnow()

    _persist_scores(scores_df, db_url)

    n_flagged = scores_df["predicted_fraud"].sum()
    logger.info(
        "Scored %d transactions for %s — %d flagged as fraud (%.2f%%)",
        len(scores_df),
        batch_date,
        n_flagged,
        100 * n_flagged / len(scores_df),
    )
    return scores_df


def _persist_scores(scores_df: pd.DataFrame, db_url: str) -> None:
    """Write scored predictions to `scores.daily_predictions`, replacing any existing rows."""
    if scores_df.empty:
        return
    engine = create_engine(db_url)
    batch_date = scores_df["batch_date"].iloc[0]
    # to_sql must receive a Connection (not Engine) for pandas 2.x + SQLAlchemy 1.4.x.
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM scores.daily_predictions WHERE batch_date = :d"),
            {"d": str(batch_date)},
        )
        scores_df.to_sql(
            "daily_predictions",
            conn,
            schema="scores",
            if_exists="append",
            index=False,
            method="multi",
        )
    logger.info("Persisted %d score rows to scores.daily_predictions", len(scores_df))
