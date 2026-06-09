"""
fraud_scoring_pipeline — daily batch fraud scoring DAG.

Schedule: daily at 01:00 UTC (after the previous day's transactions are complete).

Task graph:
    ingest_transactions
        → run_dbt_staging
            → run_dbt_marts
                → test_dbt
                    → batch_score
                        → compute_psi
                            → check_drift_threshold (BranchPythonOperator)
                                ├── handle_drift_alert
                                └── log_clean_run
                                    └── log_run_summary  (both branches converge here)

Features demonstrated:
  • PythonOperator for Python callable tasks
  • BashOperator to invoke the dbt CLI
  • BranchPythonOperator for conditional execution (drift gate)
  • XCom to pass small payloads between tasks
  • TaskGroup to bundle related tasks
  • trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS for join after branch
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task, task_group
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

# Make the project src/ importable inside the Airflow container
# (mounted at /opt/airflow/project in docker-compose.yml)
_PROJECT_ROOT = Path("/opt/airflow/project")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

log = logging.getLogger(__name__)

# ── Default args ──────────────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner": "fikri",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

DBT_DIR = str(_PROJECT_ROOT / "dbt")
DBT_PROFILES_DIR = str(_PROJECT_ROOT / "dbt")
MODEL_PATH = str(_PROJECT_ROOT / "models" / "lgbm_fraud.pkl")

# PSI threshold — must match src/config.py
PSI_ALERT_THRESHOLD = float(os.getenv("PSI_ALERT_THRESHOLD", "0.2"))

# Retention policy for raw.transactions (simulation mode)
RETENTION_MAX_ROWS = 10_000   # eviction triggers when today's count exceeds this
RETENTION_EVICT_ROWS = 500    # number of oldest rows to delete per eviction pass


# ── DAG ───────────────────────────────────────────────────────────────────────


@dag(
    dag_id="fraud_scoring_pipeline",
    description="Fraud scoring pipeline — runs every 10 min, accumulates data for simulation.",
    schedule="*/10 * * * *",  # every 10 minutes (simulation / demo mode)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["fraud", "batch-scoring", "dbt", "drift-detection"],
    doc_md="""
## Fraud Batch Scoring Pipeline

Scores all transactions from the previous calendar day for fraud risk.

### Steps
1. **ingest** — generate a synthetic daily batch and insert into `raw.transactions`
2. **dbt staging** — type-cast and validate via `stg_transactions` view
3. **dbt marts** — compute velocity / geo / temporal features via SQL window functions
4. **dbt test** — run schema tests; fail fast if data quality breaks
5. **batch_score** — load LightGBM model artifact, predict fraud probability
6. **compute_psi** — PSI across all features vs 30-day reference window
7. **drift gate** — branch on whether any feature PSI > 0.20
8. **log_run_summary** — persist audit record to `audit.pipeline_runs`
""",
)
def fraud_scoring_pipeline():

    # ── 1. Ingest ─────────────────────────────────────────────────────────────

    @task(task_id="ingest_transactions")
    def ingest_transactions(**context) -> dict:
        """
        Generate a synthetic mini-batch and APPEND it into raw.transactions.

        Each 10-minute run uses execution_date as the random seed, so each run
        produces a statistically different set of transactions.  Data accumulates
        over the day (ON CONFLICT DO NOTHING prevents duplicates on retries).

        The simulate_stream.py script can run in parallel to add extra batches
        between DAG runs, simulating a live feed.
        """
        from sqlalchemy import create_engine, text

        from src.config import BATCH_SIZE, DATABASE_URL, FRAUD_RATE_BASELINE
        from src.data_generator import generate_batch

        execution_date: datetime = context["logical_date"]
        # Use today's date as batch_date so data accumulates throughout the day
        batch_date = execution_date.date()
        # Use the full execution timestamp as seed → different data every 10-min run
        seed = int(execution_date.timestamp()) % (2**31)

        log.info(
            "Generating batch for %s (seed=%d, %d transactions)…",
            batch_date, seed, BATCH_SIZE,
        )
        df = generate_batch(
            date=datetime.combine(batch_date, datetime.min.time()),
            n_transactions=BATCH_SIZE,
            fraud_rate=FRAUD_RATE_BASELINE,
            seed=seed,
        )
        df["is_fraud"] = df["is_fraud"].astype(bool)

        engine = create_engine(DATABASE_URL)
        # Append-only with ON CONFLICT DO NOTHING so:
        #   • retries are safe (same seed = same IDs → silently skipped)
        #   • multiple runs per day accumulate data (different seeds = different IDs)
        #   • simulate_stream.py can co-exist without data loss
        with engine.begin() as conn:
            conn.execute(
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
                df.to_dict("records"),
            )

        with engine.connect() as conn:
            total = conn.execute(
                text("SELECT COUNT(*) FROM raw.transactions WHERE batch_date = :d"),
                {"d": str(batch_date)},
            ).scalar()

        # ── Retention: drop oldest rows once the cap is hit ───────────────────
        if total > RETENTION_MAX_ROWS:
            with engine.begin() as conn:
                deleted = conn.execute(
                    text("""
                        DELETE FROM raw.transactions
                        WHERE ctid IN (
                            SELECT ctid FROM raw.transactions
                            WHERE batch_date = :d
                            ORDER BY ingested_at ASC
                            LIMIT :n
                        )
                    """),
                    {"d": str(batch_date), "n": RETENTION_EVICT_ROWS},
                ).rowcount
            total -= deleted
            log.info(
                "Retention: deleted %d oldest rows (cap=%d).  Total now: %d.",
                deleted, RETENTION_MAX_ROWS, total,
            )

        log.info(
            "Added %d txns for %s.  Total accumulated today: %d rows.",
            len(df), batch_date, total,
        )
        return {"batch_date": str(batch_date), "n_rows": total}

    # ── 2–4. dbt tasks ────────────────────────────────────────────────────────

    @task_group(group_id="dbt_transform")
    def dbt_transform_group(ingest_result: dict):

        dbt_run_staging = BashOperator(
            task_id="run_dbt_staging",
            bash_command=(
                f"cd {DBT_DIR} && "
                f"dbt run --select staging --profiles-dir {DBT_PROFILES_DIR} --no-version-check"
            ),
            env={
                **os.environ,
                "DBT_TARGET": "dev",
            },
        )

        dbt_run_marts = BashOperator(
            task_id="run_dbt_marts",
            bash_command=(
                f"cd {DBT_DIR} && "
                f"dbt run --select marts --profiles-dir {DBT_PROFILES_DIR} --no-version-check"
            ),
        )

        dbt_test = BashOperator(
            task_id="test_dbt",
            bash_command=(
                f"cd {DBT_DIR} && "
                f"dbt test --profiles-dir {DBT_PROFILES_DIR} --no-version-check"
            ),
        )

        dbt_run_staging >> dbt_run_marts >> dbt_test
        return dbt_test

    # ── 5. Batch scoring ──────────────────────────────────────────────────────

    @task(task_id="batch_score")
    def batch_score(ingest_result: dict) -> dict:
        """Load model and score the day's features from the dbt mart."""
        from pathlib import Path

        from src.score import score_batch

        batch_date = date.fromisoformat(ingest_result["batch_date"])
        scores_df = score_batch(batch_date=batch_date, model_path=Path(MODEL_PATH))

        if scores_df.empty:
            return {"batch_date": str(batch_date), "n_scored": 0, "n_fraud_flagged": 0}

        n_flagged = int(scores_df["predicted_fraud"].sum())
        return {
            "batch_date": str(batch_date),
            "n_scored": len(scores_df),
            "n_fraud_flagged": n_flagged,
            "fraud_rate": round(n_flagged / len(scores_df), 6),
        }

    # ── 6. PSI drift detection ────────────────────────────────────────────────

    @task(task_id="compute_psi")
    def compute_psi(score_result: dict) -> dict:
        """Compute PSI for all features vs the 30-day reference window."""
        from src.drift import compute_drift

        batch_date = date.fromisoformat(score_result["batch_date"])
        report = compute_drift(batch_date=batch_date)

        return {
            "batch_date": str(batch_date),
            "drift_detected": report.overall_drift,
            "drifted_features": report.drifted_features,
            "max_psi": report.max_psi,
            "psi_by_feature": report.psi_by_feature,
        }

    # ── 7. Branch gate ────────────────────────────────────────────────────────

    def _branch_on_drift(**context) -> str:
        """Return task_id of the next branch based on drift detection result."""
        # XCom value is fetched from the upstream task
        ti = context["ti"]
        result = ti.xcom_pull(task_ids="compute_psi")
        if result and result.get("drift_detected"):
            return "handle_drift_alert"
        return "log_clean_run"

    branch = BranchPythonOperator(
        task_id="check_drift_threshold",
        python_callable=_branch_on_drift,
        provide_context=True,
    )

    @task(task_id="handle_drift_alert")
    def handle_drift_alert(**context) -> None:
        """Send drift alert and log warning."""
        from src.alerts import send_drift_alert
        from src.drift import DriftReport

        result = context["ti"].xcom_pull(task_ids="compute_psi")
        batch_date = date.fromisoformat(result["batch_date"])
        report = DriftReport(
            batch_date=batch_date,
            psi_by_feature=result["psi_by_feature"],
            drifted_features=result["drifted_features"],
            overall_drift=True,
            max_psi=result["max_psi"],
        )
        log.warning("Drift detected for %s: %s", batch_date, report.summary())
        send_drift_alert(report)

    @task(task_id="log_clean_run")
    def log_clean_run(**context) -> None:
        """Log a successful clean run."""
        result = context["ti"].xcom_pull(task_ids="compute_psi")
        log.info("Clean run for %s — no drift detected.", result.get("batch_date"))

    # ── 8. Audit log ──────────────────────────────────────────────────────────

    @task(
        task_id="log_run_summary",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    def log_run_summary(**context) -> None:
        """Persist a RunSummary to audit.pipeline_runs regardless of branch taken."""
        from datetime import date as date_type

        from src.alerts import RunSummary, log_run

        ti = context["ti"]
        ingest_r = ti.xcom_pull(task_ids="ingest_transactions") or {}
        score_r = ti.xcom_pull(task_ids="batch_score") or {}
        drift_r = ti.xcom_pull(task_ids="compute_psi") or {}

        batch_date = date_type.fromisoformat(
            ingest_r.get("batch_date", str(date_type.today()))
        )
        drift_detected = drift_r.get("drift_detected", False)

        summary = RunSummary(
            batch_date=batch_date,
            n_transactions=score_r.get("n_scored", 0),
            n_fraud_flagged=score_r.get("n_fraud_flagged", 0),
            fraud_rate=score_r.get("fraud_rate", 0.0),
            drift_detected=drift_detected,
            drifted_features=drift_r.get("drifted_features", []),
            max_psi=drift_r.get("max_psi", 0.0),
            run_status="drift_alert" if drift_detected else "success",
        )
        log_run(summary)
        log.info("Run summary logged for %s.", batch_date)

    # ── DAG wiring ────────────────────────────────────────────────────────────

    ingest_result = ingest_transactions()
    dbt_done = dbt_transform_group(ingest_result)
    score_result = batch_score(ingest_result)
    drift_result = compute_psi(score_result)

    ingest_result >> dbt_done >> score_result >> drift_result >> branch

    alert_task = handle_drift_alert()
    clean_task = log_clean_run()
    summary_task = log_run_summary()

    branch >> [alert_task, clean_task]
    [alert_task, clean_task] >> summary_task


# Instantiate the DAG
fraud_scoring_pipeline_dag = fraud_scoring_pipeline()
