"""Central configuration — reads from environment variables / .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from src/)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=False)

# ── Database ────────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "pipeline_db")
DB_USER = os.getenv("DB_USER", "pipeline_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "pipeline_pass")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ── Model ───────────────────────────────────────────────────────────────────
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(_ROOT / "models" / "lgbm_fraud.pkl")))

FEATURE_COLUMNS = [
    "amount",
    "txn_count_1h",
    "txn_count_24h",
    "txn_count_7d",
    "amt_sum_1h",
    "amt_ratio",
    "distance_km",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

# ── Pipeline ─────────────────────────────────────────────────────────────────
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
FRAUD_RATE_BASELINE = float(os.getenv("FRAUD_RATE_BASELINE", "0.006"))

# ── Drift ────────────────────────────────────────────────────────────────────
# PSI thresholds: 0–0.1 negligible, 0.1–0.2 minor, >0.2 major drift
PSI_ALERT_THRESHOLD = float(os.getenv("PSI_ALERT_THRESHOLD", "0.2"))
REFERENCE_WINDOW_DAYS = int(os.getenv("REFERENCE_WINDOW_DAYS", "30"))

# ── GCP Cloud Run ─────────────────────────────────────────────────────────────
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_REGION = os.getenv("GCP_REGION", "asia-southeast1")
CLOUD_RUN_JOB_NAME = os.getenv("CLOUD_RUN_JOB_NAME", "fraud-batch-scorer")
