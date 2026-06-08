"""
Entry point for the Cloud Run scorer container.

Usage:
    python scorer_entrypoint.py                    # score yesterday
    python scorer_entrypoint.py --date 2026-06-07  # score a specific date
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Batch fraud scorer (Cloud Run Job)")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=1),
        help="Batch date to score (YYYY-MM-DD). Default: yesterday.",
    )
    args = parser.parse_args()

    log.info("Starting batch scoring for %s", args.date)

    from src.score import score_batch

    scores = score_batch(batch_date=args.date)

    if scores.empty:
        log.warning("No transactions scored for %s.", args.date)
        sys.exit(0)

    n_flagged = scores["predicted_fraud"].sum()
    log.info(
        "Done. Scored %d transactions — %d flagged as fraud (%.2f%%).",
        len(scores),
        n_flagged,
        100 * n_flagged / len(scores),
    )


if __name__ == "__main__":
    main()
