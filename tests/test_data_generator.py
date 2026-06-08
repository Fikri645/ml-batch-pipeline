"""Unit tests for src/data_generator.py."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data_generator import generate_batch, get_card_pool


# ── Card pool ─────────────────────────────────────────────────────────────────

class TestGetCardPool:
    def test_returns_dataframe(self):
        pool = get_card_pool()
        assert isinstance(pool, pd.DataFrame)

    def test_default_size(self):
        pool = get_card_pool()
        assert len(pool) == 300

    def test_deterministic_with_same_seed(self):
        pool_a = get_card_pool(seed=42)
        pool_b = get_card_pool(seed=42)
        pd.testing.assert_frame_equal(pool_a, pool_b)

    def test_required_columns(self):
        pool = get_card_pool()
        for col in ("card_number", "home_lat", "home_long"):
            assert col in pool.columns, f"Missing column: {col}"

    def test_indonesia_coordinates(self):
        """All cards should have home coordinates within Indonesia's bounding box."""
        pool = get_card_pool()
        assert pool["home_lat"].between(-11.0, 6.0).all(), "Some latitudes outside Indonesia"
        assert pool["home_long"].between(95.0, 141.0).all(), "Some longitudes outside Indonesia"


# ── Batch generation ──────────────────────────────────────────────────────────

class TestGenerateBatch:
    def test_returns_dataframe(self, sample_batch):
        assert isinstance(sample_batch, pd.DataFrame)

    def test_row_count(self):
        batch = generate_batch(datetime(2026, 1, 1), n_transactions=200, seed=7)
        assert len(batch) == 200

    def test_required_columns(self, sample_batch):
        expected = {
            "transaction_id", "card_number", "merchant",
            "amount", "lat", "long",
            "merchant_lat", "merchant_long",
            "transaction_time", "is_fraud", "batch_date", "category",
        }
        missing = expected - set(sample_batch.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_fraud_rate_approximate(self):
        """Actual fraud rate should be close to the requested rate."""
        batch = generate_batch(datetime(2026, 1, 1), n_transactions=1000, fraud_rate=0.05, seed=1)
        actual_rate = batch["is_fraud"].mean()
        assert abs(actual_rate - 0.05) < 0.02, f"Fraud rate {actual_rate:.3f} too far from 0.05"

    def test_no_fraud_when_rate_zero(self):
        batch = generate_batch(datetime(2026, 1, 1), n_transactions=100, fraud_rate=0.0, seed=2)
        assert batch["is_fraud"].sum() == 0

    def test_amounts_positive(self, sample_batch):
        assert (sample_batch["amount"] > 0).all()

    def test_transaction_ids_unique(self, sample_batch):
        assert sample_batch["transaction_id"].nunique() == len(sample_batch)

    def test_batch_date_column(self):
        date = datetime(2026, 5, 20)
        batch = generate_batch(date, n_transactions=50, seed=3)
        assert (batch["batch_date"] == date.date()).all()

    def test_deterministic_with_seed(self):
        date = datetime(2026, 2, 10)
        batch_a = generate_batch(date, n_transactions=50, seed=77)
        batch_b = generate_batch(date, n_transactions=50, seed=77)
        pd.testing.assert_frame_equal(batch_a, batch_b)

    def test_different_seeds_differ(self):
        date = datetime(2026, 2, 10)
        batch_a = generate_batch(date, n_transactions=50, seed=1)
        batch_b = generate_batch(date, n_transactions=50, seed=2)
        # At least some amounts should differ
        assert not (batch_a["amount"].values == batch_b["amount"].values).all()

    def test_fraud_amounts_larger_on_average(self):
        """Fraud transactions should have larger mean amounts than legit."""
        batch = generate_batch(datetime(2026, 1, 1), n_transactions=2000, fraud_rate=0.1, seed=42)
        fraud_mean = batch.loc[batch["is_fraud"] == 1, "amount"].mean()
        legit_mean = batch.loc[batch["is_fraud"] == 0, "amount"].mean()
        assert fraud_mean > legit_mean, "Fraud amounts should be larger on average"
