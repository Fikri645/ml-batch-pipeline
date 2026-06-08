"""
Model training script — run once to produce the model artifact.

Generates a synthetic historical dataset (90 days), engineers features,
trains a LightGBM classifier, and saves to models/lgbm_fraud.pkl.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --days 60 --output models/lgbm_fraud.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Make project root importable when running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb  # noqa: E402

from src.config import FEATURE_COLUMNS, MODEL_PATH  # noqa: E402
from src.data_generator import generate_batch  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Feature engineering (Python-side, mirrors dbt mart) ─────────────────────


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the same feature set as `marts.fct_transaction_features`.

    This is the Python equivalent of the dbt SQL; used only for training.
    In the live pipeline, dbt produces features before the scoring step.
    """
    df = raw_df.sort_values("transaction_time").copy()
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])

    # ── Temporal ─────────────────────────────────────────────────────────────
    df["hour_of_day"] = df["transaction_time"].dt.hour
    df["day_of_week"] = df["transaction_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ── Haversine distance ────────────────────────────────────────────────────
    lat1 = np.radians(df["lat"].values)
    lat2 = np.radians(df["merchant_lat"].values)
    lon1 = np.radians(df["long"].values)
    lon2 = np.radians(df["merchant_long"].values)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["distance_km"] = 6371.0 * 2 * np.arcsin(np.sqrt(a))

    # ── Velocity features (rolling, per card) ─────────────────────────────────
    df = df.sort_values(["card_number", "transaction_time"])
    result_parts = []

    for card_id, grp in df.groupby("card_number", sort=False):
        grp = grp.sort_values("transaction_time").copy()
        times = grp["transaction_time"]
        amounts = grp["amount"].values

        txn_count_1h = []
        txn_count_24h = []
        txn_count_7d = []
        amt_sum_1h = []
        amt_ratio = []

        for i, (ts, amt) in enumerate(zip(times, amounts)):
            window = grp[times <= ts]
            w1h = grp[(ts - times <= pd.Timedelta("1h")) & (times <= ts)]
            w24h = grp[(ts - times <= pd.Timedelta("24h")) & (times <= ts)]
            w7d = grp[(ts - times <= pd.Timedelta("7d")) & (times <= ts)]

            txn_count_1h.append(len(w1h))
            txn_count_24h.append(len(w24h))
            txn_count_7d.append(len(w7d))
            amt_sum_1h.append(w1h["amount"].sum())

            prev_30 = amounts[max(0, i - 30):i]
            ref_amt = prev_30.mean() if len(prev_30) > 0 else amt
            amt_ratio.append(amt / ref_amt if ref_amt > 0 else 1.0)

        grp["txn_count_1h"] = txn_count_1h
        grp["txn_count_24h"] = txn_count_24h
        grp["txn_count_7d"] = txn_count_7d
        grp["amt_sum_1h"] = amt_sum_1h
        grp["amt_ratio"] = amt_ratio
        result_parts.append(grp)

    return pd.concat(result_parts, ignore_index=True)


# ── Training ──────────────────────────────────────────────────────────────────


def generate_training_data(
    days: int = 90,
    n_per_day: int = 500,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Generate *days* of synthetic transaction history."""
    if end_date is None:
        end_date = datetime.utcnow()
    logger.info("Generating %d days of synthetic data (%d txns/day)…", days, n_per_day)
    batches = []
    for d in range(days):
        day = end_date - timedelta(days=days - d)
        batch = generate_batch(day, n_transactions=n_per_day)
        batches.append(batch)
    df = pd.concat(batches, ignore_index=True)
    logger.info("Total rows: %d | fraud rows: %d", len(df), df["is_fraud"].sum())
    return df


def train(
    days: int = 90,
    n_per_day: int = 500,
    output_path: Path = MODEL_PATH,
) -> None:
    """Train the LightGBM fraud classifier and save to *output_path*."""
    raw_df = generate_training_data(days=days, n_per_day=n_per_day)

    logger.info("Engineering features…")
    feat_df = engineer_features(raw_df)
    feat_df = feat_df.dropna(subset=FEATURE_COLUMNS)

    X = feat_df[FEATURE_COLUMNS]
    y = feat_df["is_fraud"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info("Training LightGBM on %d samples (val=%d)…", len(X_train), len(X_val))
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        class_weight="balanced",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )

    proba_val = model.predict_proba(X_val)[:, 1]
    roc = roc_auc_score(y_val, proba_val)
    pr_auc = average_precision_score(y_val, proba_val)
    logger.info("Validation  ROC-AUC=%.4f  PR-AUC=%.4f", roc, pr_auc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info("Model saved to %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument("--days", type=int, default=90, help="Days of history to generate")
    parser.add_argument("--n-per-day", type=int, default=500, help="Transactions per day")
    parser.add_argument("--output", type=Path, default=MODEL_PATH, help="Output path for model artifact")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(days=args.days, n_per_day=args.n_per_day, output_path=args.output)
