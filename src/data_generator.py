"""
Synthetic transaction generator for the batch scoring pipeline demo.

Produces Sparkov-like daily transaction batches without requiring the
Kaggle dataset download — the focus of this project is the pipeline, not the data.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

CATEGORIES = [
    "grocery_pos",
    "entertainment",
    "food_dining",
    "gas_transport",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel",
]

# Indonesia approximate bounding box
LAT_RANGE = (-8.5, 6.0)
LONG_RANGE = (95.0, 141.0)

CARD_POOL_SIZE = 300
MERCHANT_POOL_SIZE = 800


# ── Card pool ─────────────────────────────────────────────────────────────────


def get_card_pool(seed: int = 42) -> pd.DataFrame:
    """
    Generate a stable pool of card profiles (home location + typical spend).

    The pool is deterministic — same seed = same cards across all daily batches,
    which gives the velocity / behavioral features meaningful signal.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "card_number": [f"CARD_{i:04d}" for i in range(CARD_POOL_SIZE)],
            "home_lat": rng.uniform(*LAT_RANGE, CARD_POOL_SIZE),
            "home_long": rng.uniform(*LONG_RANGE, CARD_POOL_SIZE),
            # Typical spend per card — lognormal
            "typical_amount": rng.lognormal(mean=3.8, sigma=0.8, size=CARD_POOL_SIZE).clip(
                5, 800
            ),
        }
    )


# ── Batch generation ──────────────────────────────────────────────────────────


def generate_batch(
    date: datetime,
    n_transactions: int = 500,
    fraud_rate: float = 0.006,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic batch of transactions for *date*.

    Parameters
    ----------
    date : datetime
        Target date for the batch.
    n_transactions : int
        Total number of transactions.
    fraud_rate : float
        Proportion of fraudulent transactions (Sparkov baseline ~0.6%).
    seed : int | None
        Random seed; defaults to the Unix timestamp of *date* for
        reproducibility while keeping batches distinct per day.

    Returns
    -------
    pd.DataFrame
        Columns: transaction_id, card_number, merchant, category, amount,
                 lat, long, merchant_lat, merchant_long, transaction_time,
                 is_fraud, batch_date.
    """
    if seed is None:
        seed = int(date.replace(hour=0, minute=0, second=0).timestamp())
    rng = np.random.default_rng(seed)

    cards = get_card_pool()
    n_fraud = max(1, int(n_transactions * fraud_rate))
    # ── Timestamps ────────────────────────────────────────────────────────────
    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds = rng.uniform(0, 86400, n_transactions).astype(int)
    timestamps = sorted(day_start + timedelta(seconds=int(s)) for s in seconds)

    # ── Cards ─────────────────────────────────────────────────────────────────
    card_idx = rng.choice(len(cards), size=n_transactions)
    sel = cards.iloc[card_idx].reset_index(drop=True)

    # ── Fraud flags ───────────────────────────────────────────────────────────
    is_fraud = np.zeros(n_transactions, dtype=bool)
    fraud_idx = rng.choice(n_transactions, size=n_fraud, replace=False)
    is_fraud[fraud_idx] = True

    # ── Amounts ───────────────────────────────────────────────────────────────
    legit_amounts = rng.lognormal(
        mean=np.log(sel["typical_amount"].values + 1), sigma=0.7
    ).clip(1, 2000)
    fraud_amounts = rng.lognormal(mean=np.log(550), sigma=0.9).clip(50, 3000)
    amounts = np.where(is_fraud, fraud_amounts, legit_amounts).round(2)

    # ── Merchant locations ────────────────────────────────────────────────────
    # Legit: near home; fraud: anywhere in Indonesia
    noise_lat = rng.normal(0, 0.15, n_transactions)
    noise_long = rng.normal(0, 0.15, n_transactions)
    random_lat = rng.uniform(*LAT_RANGE, n_transactions)
    random_long = rng.uniform(*LONG_RANGE, n_transactions)

    merchant_lat = np.where(
        is_fraud,
        random_lat,
        (sel["home_lat"].values + noise_lat).clip(*LAT_RANGE),
    ).round(6)
    merchant_long = np.where(
        is_fraud,
        random_long,
        (sel["home_long"].values + noise_long).clip(*LONG_RANGE),
    ).round(6)

    # ── Categories ────────────────────────────────────────────────────────────
    # Fraud skews toward online shopping / travel
    cat_weights = np.ones(len(CATEGORIES))
    online_idx = [CATEGORIES.index(c) for c in ("misc_net", "shopping_net", "travel")]
    cat_weights[online_idx] *= 2.5
    cat_probs = cat_weights / cat_weights.sum()
    categories = rng.choice(CATEGORIES, size=n_transactions, p=cat_probs)

    # ── Merchants ─────────────────────────────────────────────────────────────
    merchants = [
        f"MERCH_{rng.integers(1, MERCHANT_POOL_SIZE + 1):04d}" for _ in range(n_transactions)
    ]

    # ── Transaction IDs ───────────────────────────────────────────────────────
    txn_ids = [
        hashlib.md5(
            f"{ts.isoformat()}_{card}_{amt:.2f}".encode(), usedforsecurity=False
        ).hexdigest()[:16]
        for ts, card, amt in zip(timestamps, sel["card_number"], amounts)
    ]

    return pd.DataFrame(
        {
            "transaction_id": txn_ids,
            "card_number": sel["card_number"].values,
            "merchant": merchants,
            "category": categories,
            "amount": amounts,
            "lat": sel["home_lat"].values.round(6),
            "long": sel["home_long"].values.round(6),
            "merchant_lat": merchant_lat,
            "merchant_long": merchant_long,
            "transaction_time": timestamps,
            "is_fraud": is_fraud,
            "batch_date": date.date(),
        }
    )
