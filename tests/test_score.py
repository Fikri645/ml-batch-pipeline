"""Unit tests for src/score.py — batch scoring logic."""

from __future__ import annotations

from datetime import date
from pathlib import Path
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
            "card_number": [f"CARD_{i:04d}" for i in range(n)],
            "batch_date": date(2026, 3, 15),
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
    @patch.object(Path, "exists", return_value=True)
    def test_loads_from_path(self, mock_exists, mock_load):
        from src.score import load_model
        mock_load.return_value = MagicMock()
        model = load_model("/fake/path/model.pkl")
        mock_load.assert_called_once()
        assert model is mock_load.return_value

    def test_raises_on_missing_file(self):
        """A genuinely non-existent path raises FileNotFoundError."""
        from src.score import load_model
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/definitely/not/there/model.pkl")


# ── score_batch ───────────────────────────────────────────────────────────────

class TestScoreBatch:
    """
    Integration-level tests with DB and model mocked out.
    score_batch() returns a DataFrame with columns:
      transaction_id, card_number, batch_date, fraud_probability,
      predicted_fraud, scored_at.
    """

    def _run_score_batch(
        self,
        mock_load,
        mock_engine,
        feat_df,
        fraud_threshold=0.5,
    ):
        """Helper: run score_batch with mocked DB and model, patch persist."""
        conn = MagicMock()
        conn.__enter__ = lambda s: conn
        conn.__exit__ = MagicMock(return_value=False)
        mock_engine.return_value.connect.return_value = conn
        mock_engine.return_value.begin.return_value = conn

        with patch("src.score.pd.read_sql", return_value=feat_df), \
             patch("src.score._persist_scores"):
            from src.score import score_batch
            return score_batch(
                batch_date=date(2026, 3, 15),
                model_path="/fake/model.pkl",
                db_url="postgresql://fake/fake",
                fraud_threshold=fraud_threshold,
            )

    @patch("src.score.create_engine")
    @patch("src.score.joblib.load")
    @patch.object(Path, "exists", return_value=True)
    def test_returns_dataframe_with_correct_row_count(
        self, mock_exists, mock_load, mock_engine
    ):
        n = 40
        feat_df = _make_feature_df(n=n, seed=1)
        mock_load.return_value = _make_mock_model(n)
        result = self._run_score_batch(mock_load, mock_engine, feat_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == n

    @patch("src.score.create_engine")
    @patch("src.score.joblib.load")
    @patch.object(Path, "exists", return_value=True)
    def test_all_flagged_when_high_prob(self, mock_exists, mock_load, mock_engine):
        n = 100
        feat_df = _make_feature_df(n=n, seed=2)
        model = MagicMock()
        model.predict_proba.return_value = np.full((n, 2), [[0.2, 0.8]])
        mock_load.return_value = model
        result = self._run_score_batch(mock_load, mock_engine, feat_df, fraud_threshold=0.5)
        assert int(result["predicted_fraud"].sum()) == n

    @patch("src.score.create_engine")
    @patch("src.score.joblib.load")
    @patch.object(Path, "exists", return_value=True)
    def test_none_flagged_when_low_prob(self, mock_exists, mock_load, mock_engine):
        n = 60
        feat_df = _make_feature_df(n=n, seed=3)
        model = MagicMock()
        model.predict_proba.return_value = np.full((n, 2), [[0.9, 0.1]])
        mock_load.return_value = model
        result = self._run_score_batch(mock_load, mock_engine, feat_df, fraud_threshold=0.5)
        assert int(result["predicted_fraud"].sum()) == 0

    @patch("src.score.create_engine")
    @patch("src.score.joblib.load")
    @patch.object(Path, "exists", return_value=True)
    def test_empty_batch_returns_empty_dataframe(
        self, mock_exists, mock_load, mock_engine
    ):
        empty_df = _make_feature_df(n=0, seed=4)
        mock_load.return_value = _make_mock_model(0)
        result = self._run_score_batch(mock_load, mock_engine, empty_df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch("src.score.create_engine")
    @patch("src.score.joblib.load")
    @patch.object(Path, "exists", return_value=True)
    def test_result_contains_required_columns(
        self, mock_exists, mock_load, mock_engine
    ):
        n = 20
        feat_df = _make_feature_df(n=n, seed=5)
        mock_load.return_value = _make_mock_model(n)
        result = self._run_score_batch(mock_load, mock_engine, feat_df)
        for col in ("transaction_id", "card_number", "batch_date",
                    "fraud_probability", "predicted_fraud", "scored_at"):
            assert col in result.columns, f"Missing column: {col}"

    @patch("src.score.create_engine")
    @patch("src.score.joblib.load")
    @patch.object(Path, "exists", return_value=True)
    def test_fraud_probability_in_unit_interval(
        self, mock_exists, mock_load, mock_engine
    ):
        n = 50
        feat_df = _make_feature_df(n=n, seed=6)
        mock_load.return_value = _make_mock_model(n, fraud_prob=0.3)
        result = self._run_score_batch(mock_load, mock_engine, feat_df)
        assert result["fraud_probability"].between(0.0, 1.0).all()
