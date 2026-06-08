"""Unit tests for src/score.py — batch scoring logic."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_feature_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Return a minimal features DataFrame that mirrors the dbt mart output."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:04d}" for i in range(n)],
            "batch_date": datetime(2026, 3, 15).date(),
            **{col: rng.uniform(0, 10, n) for col in FEATURE_COLUMNS},
        }
    )


def _make_mock_model(n: int, fraud_prob: float = 0.05):
    """Return a mock LightGBM-like model that returns constant probabilities."""
    model = MagicMock()
    proba = np.full((n, 2), [[1 - fraud_prob, fraud_prob]])
    model.predict_proba.return_value = proba
    return model


# ── load_model ────────────────────────────────────────────────────────────────

class TestLoadModel:
    @patch("src.score.joblib.load")
    def test_loads_from_path(self, mock_load):
        from src.score import load_model
        mock_load.return_value = MagicMock()
        model = load_model("/fake/path/model.pkl")
        mock_load.assert_called_once_with("/fake/path/model.pkl")
        assert model is mock_load.return_value

    @patch("src.score.joblib.load", side_effect=FileNotFoundError("no file"))
    def test_raises_on_missing_file(self, _):
        from src.score import load_model
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/model.pkl")


# ── score_batch ───────────────────────────────────────────────────────────────

class TestScoreBatch:
    """Integration-level tests with DB and model mocked out."""

    @patch("src.score.create_engine")
    @patch("src.score.joblib.load")
    def test_returns_row_count(self, mock_load, mock_engine):
        from src.score import score_batch

        n = 40
        feat_df = _make_feature_df(n=n, seed=1)
        mock_load.return_value = _make_mock_model(n)

        conn = MagicMock()
        conn.__enter__ = lambda s: conn
        conn.__exit__ = MagicMock(return_value=False)
        mock_engine.return_value.connect.return_value = conn

        with patch("src.score.pd.read_sql", return_value=feat_df):
            result = score_batch(
                batch_date=datetime(2026, 3, 15),
                model_path="/fake/model.pkl",
                db_url="postgresql://fake/fake",
                fraud_threshold=0.5,
            )

        assert result["n_scored"] == n

    @patch("src.score.create_engine")
    @patch("src.score.joblib.load")
    def test_fraud_flagged_count_matches_threshold(self, mock_load, mock_engine):
        from src.score import score_batch

        n = 100
        feat_df = _make_feature_df(n=n, seed=2)

        # All scores = 0.8 → all should be flagged at threshold=0.5
        model = MagicMock()
        model.predict_proba.return_value = np.full((n, 2), [[0.2, 0.8]])
        mock_load.return_value = model

        conn = MagicMock()
        conn.__enter__ = lambda s: conn
        conn.__exit__ = MagicMock(return_value=False)
        mock_engine.return_value.connect.return_value = conn

        with patch("src.score.pd.read_sql", return_value=feat_df):
            result = score_batch(
                batch_date=datetime(2026, 3, 15),
                model_path="/fake/model.pkl",
                db_url="postgresql://fake/fake",
                fraud_threshold=0.5,
            )

        assert result["n_fraud_flagged"] == n

    @patch("src.score.create_engine")
    @patch("src.score.joblib.load")
    def test_no_fraud_below_threshold(self, mock_load, mock_engine):
        from src.score import score_batch

        n = 60
        feat_df = _make_feature_df(n=n, seed=3)

        # All scores = 0.1 → none should be flagged at threshold=0.5
        model = MagicMock()
        model.predict_proba.return_value = np.full((n, 2), [[0.9, 0.1]])
        mock_load.return_value = model

        conn = MagicMock()
        conn.__enter__ = lambda s: conn
        conn.__exit__ = MagicMock(return_value=False)
        mock_engine.return_value.connect.return_value = conn

        with patch("src.score.pd.read_sql", return_value=feat_df):
            result = score_batch(
                batch_date=datetime(2026, 3, 15),
                model_path="/fake/model.pkl",
                db_url="postgresql://fake/fake",
                fraud_threshold=0.5,
            )

        assert result["n_fraud_flagged"] == 0

    @patch("src.score.create_engine")
    @patch("src.score.joblib.load")
    def test_empty_batch_returns_zero_counts(self, mock_load, mock_engine):
        from src.score import score_batch

        empty_df = _make_feature_df(n=0, seed=4)
        mock_load.return_value = _make_mock_model(0)

        conn = MagicMock()
        conn.__enter__ = lambda s: conn
        conn.__exit__ = MagicMock(return_value=False)
        mock_engine.return_value.connect.return_value = conn

        with patch("src.score.pd.read_sql", return_value=empty_df):
            result = score_batch(
                batch_date=datetime(2026, 3, 15),
                model_path="/fake/model.pkl",
                db_url="postgresql://fake/fake",
                fraud_threshold=0.5,
            )

        assert result["n_scored"] == 0
        assert result["n_fraud_flagged"] == 0

    @patch("src.score.create_engine")
    @patch("src.score.joblib.load")
    def test_result_contains_required_keys(self, mock_load, mock_engine):
        from src.score import score_batch

        n = 20
        feat_df = _make_feature_df(n=n, seed=5)
        mock_load.return_value = _make_mock_model(n)

        conn = MagicMock()
        conn.__enter__ = lambda s: conn
        conn.__exit__ = MagicMock(return_value=False)
        mock_engine.return_value.connect.return_value = conn

        with patch("src.score.pd.read_sql", return_value=feat_df):
            result = score_batch(
                batch_date=datetime(2026, 3, 15),
                model_path="/fake/model.pkl",
                db_url="postgresql://fake/fake",
            )

        for key in ("n_scored", "n_fraud_flagged", "fraud_rate", "batch_date"):
            assert key in result, f"Missing key in result: {key}"
