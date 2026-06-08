"""Shared fixtures for the ml-batch-pipeline test suite."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.data_generator import generate_batch


@pytest.fixture(scope="session")
def sample_date() -> datetime:
    """A fixed date for deterministic tests."""
    return datetime(2026, 3, 15)


@pytest.fixture(scope="session")
def sample_batch(sample_date) -> "pd.DataFrame":
    """A small, deterministic transaction batch for unit tests."""
    return generate_batch(date=sample_date, n_transactions=100, fraud_rate=0.05, seed=99)
