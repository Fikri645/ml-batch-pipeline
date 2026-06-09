"""
Simulate a real-time transaction stream.

Runs in a continuous loop, inserting mini-batches of synthetic transactions
into raw.transactions every N seconds.  The Airflow DAG (running every 10 min)
will pick up all accumulated data, score it, and log any fraud detected.

Run this script in a second terminal alongside the Airflow stack:

    docker compose exec airflow-scheduler bash -c \\
        "cd /opt/airflow/project && python scripts/simulate_stream.py"

Or with custom options:

    docker compose exec airflow-scheduler bash -c \\
        "cd /opt/airflow/project && python scripts/simulate_stream.py \\
         --interval 60 --batch-size 30 --fraud-rate 0.02"

Options
-------
--interval    Seconds between each mini-batch insertion  (default: 120)
--batch-size  Transactions per mini-batch                (default: 50)
--fraud-rate  Fraction of fraudulent transactions        (default: 0.006 ≈ 0.6%)
--runs        Stop after this many batches (0 = run forever, default: 0)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text

from src.config import DATABASE_URL, FRAUD_RATE_BASELINE
from src.data_generator import generate_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Database helpers ──────────────────────────────────────────────────────────


def _insert_batch(engine, df) -> int:
    """
    Append a batch to raw.transactions.

    Uses ON CONFLICT DO NOTHING so the script is safe to re-run and so the
    Airflow DAG's own ingest task won't clash (different seeds = different IDs).
    Returns the number of rows actually inserted (after conflict filtering).
    """
    df_clean = df.copy()
    df_clean["is_fraud"] = df_clean["is_fraud"].astype(bool)

    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO raw.transactions
                (transaction_id, card_number, merchant, category, amount,
                 lat, long, merchant_lat, merchant_long,
                 transaction_time, is_fraud, batch_date)
                VALUES
                (:transaction_id, :card_number, :merchant, :category, :amount,
                 :lat, :long, :merchant_lat, :merchant_long,
                 :transaction_time, :is_fraud, :batch_date)
                ON CONFLICT (transaction_id, batch_date) DO NOTHING
            """),
            df_clean.to_dict("records"),
        )
    return result.rowcount


def _get_today_count(engine, batch_date) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT COUNT(*) FROM raw.transactions WHERE batch_date = :d"),
            {"d": str(batch_date)},
        ).scalar()


# ── Main loop ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a live transaction stream into raw.transactions."
    )
    parser.add_argument(
        "--interval", type=int, default=120,
        help="Seconds between batches (default: 120)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Transactions per batch (default: 50)",
    )
    parser.add_argument(
        "--fraud-rate", type=float, default=FRAUD_RATE_BASELINE,
        help=f"Fraud fraction per batch (default: {FRAUD_RATE_BASELINE})",
    )
    parser.add_argument(
        "--runs", type=int, default=0,
        help="Stop after N batches; 0 = run forever (default: 0)",
    )
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    batch_num = 0
    total_inserted = 0
    total_fraud = 0

    logger.info(
        "Stream simulator started — %d txns every %ds  "
        "(fraud rate %.1f%%)  |  Ctrl+C to stop",
        args.batch_size, args.interval, 100 * args.fraud_rate,
    )
    logger.info("Airflow UI → http://localhost:8080  (admin / admin)")
    logger.info("─" * 60)

    while True:
        batch_num += 1
        now = datetime.utcnow()
        batch_date = now.date()

        # Millisecond timestamp ensures a fresh random seed every batch
        seed = int(now.timestamp() * 1000) % (2 ** 31)

        df = generate_batch(
            date=datetime(now.year, now.month, now.day),
            n_transactions=args.batch_size,
            fraud_rate=args.fraud_rate,
            seed=seed,
        )

        inserted = _insert_batch(engine, df)
        today_total = _get_today_count(engine, batch_date)

        n_fraud_batch = int(df["is_fraud"].sum())
        total_inserted += inserted
        total_fraud += n_fraud_batch

        logger.info(
            "Batch #%-3d  +%d txns (%d fraud)  |  hari ini: %d txns  |  "
            "session total: %d txns / %d fraud  |  next in %ds",
            batch_num, inserted, n_fraud_batch,
            today_total,
            total_inserted, total_fraud,
            args.interval,
        )

        if args.runs > 0 and batch_num >= args.runs:
            logger.info("Reached --runs %d. Stopping.", args.runs)
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stream simulator stopped by user.")
        sys.exit(0)
