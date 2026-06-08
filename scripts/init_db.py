"""
Database initialisation script.

Creates all schemas and tables required by the pipeline.
Run once before the first pipeline execution:

    python scripts/init_db.py
"""

import logging
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

# Make project root importable when running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATABASE_URL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DDL = """
-- ─────────────────────────────────────────────
-- Schemas
-- ─────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS marts;
CREATE SCHEMA IF NOT EXISTS scores;
CREATE SCHEMA IF NOT EXISTS audit;

-- ─────────────────────────────────────────────
-- raw.transactions  — daily landing zone
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw.transactions (
    transaction_id   TEXT        NOT NULL,
    card_number      TEXT        NOT NULL,
    merchant         TEXT,
    category         TEXT,
    amount           NUMERIC(12, 2),
    lat              DOUBLE PRECISION,
    long             DOUBLE PRECISION,
    merchant_lat     DOUBLE PRECISION,
    merchant_long    DOUBLE PRECISION,
    transaction_time TIMESTAMPTZ,
    is_fraud         BOOLEAN,
    batch_date       DATE        NOT NULL,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (transaction_id, batch_date)
);

-- ─────────────────────────────────────────────
-- scores.daily_predictions  — model output
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scores.daily_predictions (
    transaction_id    TEXT,
    card_number       TEXT,
    batch_date        DATE,
    fraud_probability DOUBLE PRECISION,
    predicted_fraud   SMALLINT,
    scored_at         TIMESTAMPTZ,
    PRIMARY KEY (transaction_id, batch_date)
);

-- ─────────────────────────────────────────────
-- audit.pipeline_runs  — one row per daily run
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit.pipeline_runs (
    id               SERIAL PRIMARY KEY,
    batch_date       DATE,
    n_transactions   INTEGER,
    n_fraud_flagged  INTEGER,
    fraud_rate       DOUBLE PRECISION,
    drift_detected   BOOLEAN,
    drifted_features TEXT,
    max_psi          DOUBLE PRECISION,
    run_status       TEXT,
    run_at           TIMESTAMPTZ
);

-- ─────────────────────────────────────────────
-- audit.drift_reports  — PSI per feature per day
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit.drift_reports (
    id           SERIAL PRIMARY KEY,
    batch_date   DATE,
    feature_name TEXT,
    psi          DOUBLE PRECISION,
    is_drifted   BOOLEAN,
    checked_at   TIMESTAMPTZ
);
"""


def main():
    logger.info("Connecting to database...")
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        for stmt in DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    logger.info("Database initialised successfully.")


if __name__ == "__main__":
    main()
