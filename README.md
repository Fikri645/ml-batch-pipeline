# ml-batch-pipeline

[![CI](https://github.com/Fikri645/ml-batch-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Fikri645/ml-batch-pipeline/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Airflow 2.9](https://img.shields.io/badge/Airflow-2.9-017CEE?logo=apacheairflow)](https://airflow.apache.org/)
[![dbt 1.8](https://img.shields.io/badge/dbt-1.8-FF694B?logo=dbt)](https://docs.getdbt.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green)](https://lightgbm.readthedocs.io/)

**Production-grade fraud-detection batch pipeline** orchestrated with Apache Airflow, feature-engineered with dbt, and deployed as a Cloud Run Job on GCP. Scores 500+ synthetic Indonesian transactions daily with PSI-based feature drift monitoring and Slack alerting.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Docker Compose (Local)                           │
│                                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐  │
│  │  PostgreSQL  │    │              Apache Airflow 2.9                  │  │
│  │     :5432    │    │                                                  │  │
│  │              │    │  fraud_scoring_pipeline  (daily @ 01:00 UTC)    │  │
│  │  airflow_meta│    │                                                  │  │
│  │  pipeline_db │    │  ingest_transactions                            │  │
│  │    schemas:  │    │        │                                         │  │
│  │    - raw     │◄───┤  ┌─────▼───────────────────────────────────┐   │  │
│  │    - staging │    │  │  dbt_transform_group (TaskGroup)         │   │  │
│  │    - marts   │    │  │    run_dbt_staging → run_dbt_marts       │   │  │
│  │    - scores  │    │  │             └──→ test_dbt                │   │  │
│  │    - audit   │    │  └─────┬───────────────────────────────────┘   │  │
│  └──────────────┘    │        │                                         │  │
│                       │  batch_score                                    │  │
│                       │        │                                         │  │
│                       │  compute_psi                                    │  │
│                       │        │                                         │  │
│                       │  check_drift_threshold (BranchPython)          │  │
│                       │     ├── handle_drift_alert                     │  │
│                       │     └── log_clean_run                          │  │
│                       │              └──────────────┐                  │  │
│                       │                   log_run_summary              │  │
│                       └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

                                      │
                          ┌───────────▼────────────┐
                          │    GCP Cloud Run Job    │
                          │    (fraud-scorer)       │
                          │                         │
                          │  Triggered by:          │
                          │  • Airflow (via API)    │
                          │  • Cloud Scheduler      │
                          │  • Manual gcloud CLI    │
                          └─────────────────────────┘
```

### Data Flow

```
Synthetic generator
  → raw.transactions (PostgreSQL)
  → dbt staging view  (stg_transactions)
  → dbt mart table    (fct_transaction_features)  ← window fn velocity + Haversine
  → LightGBM scorer   (scores.daily_predictions)
  → PSI drift audit   (audit.drift_reports)
  → Slack alert       (if PSI > 0.20)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | Apache Airflow 2.9 — LocalExecutor, TaskFlow API |
| Feature engineering | dbt-postgres 1.8 — staging views + mart tables |
| ML model | LightGBM 4.3 — gradient boosting fraud classifier |
| Drift detection | Population Stability Index (PSI) — 10-bin histogram |
| Storage | PostgreSQL 15 — 5 schemas (raw / staging / marts / scores / audit) |
| Containerization | Docker Compose (local), Cloud Run Jobs (GCP) |
| CI | GitHub Actions — lint, unit tests, DAG import, dbt compile |

---

## Project Structure

```
ml-batch-pipeline/
├── src/
│   ├── config.py          # Central config: feature columns, thresholds
│   ├── data_generator.py  # Synthetic Sparkov-like transaction generator
│   ├── score.py           # Batch scorer: load model → predict → persist
│   ├── drift.py           # PSI drift detection + DriftReport dataclass
│   └── alerts.py          # RunSummary dataclass, Slack webhook alerts
├── scripts/
│   ├── init_db.py         # Create all schemas and tables
│   └── train_model.py     # Train LightGBM on 90 days of synthetic data
├── dbt/
│   ├── dbt_project.yml
│   └── models/
│       ├── staging/
│       │   ├── sources.yml          # raw.transactions source definition
│       │   └── stg_transactions.sql # Type casts + null filters
│       └── marts/
│           ├── schema.yml                   # not_null / unique tests
│           ├── fct_transaction_features.sql # Velocity + Haversine features
│           └── fct_daily_summary.sql        # Daily aggregate stats
├── dags/
│   └── fraud_scoring_pipeline.py   # Main Airflow DAG
├── docker/
│   ├── Dockerfile.airflow      # Airflow image with project deps
│   ├── Dockerfile.scorer       # Minimal Cloud Run scorer image
│   ├── scorer_entrypoint.py    # CLI entrypoint for Cloud Run Job
│   ├── airflow-init.sh         # DB migrate + user create
│   └── init-multiple-dbs.sh    # Creates airflow_meta + pipeline_db
├── tests/
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_data_generator.py  # Generator unit tests
│   ├── test_drift.py           # PSI calculation unit tests
│   └── test_score.py           # Scorer unit tests (mocked DB)
├── models/                     # Model artifacts (gitignored)
├── plugins/                    # Airflow plugins (empty)
├── .github/workflows/ci.yml    # CI pipeline
├── docker-compose.yml
├── requirements.txt
├── requirements-scorer.txt     # Minimal deps for Cloud Run image
└── .env.example
```

---

## Quick Start (Local Docker)

### Prerequisites

- Docker Desktop ≥ 24.0
- Docker Compose ≥ 2.20
- 4 GB RAM available to Docker

### 1 — Clone and configure

```bash
git clone https://github.com/Fikri645/ml-batch-pipeline.git
cd ml-batch-pipeline

cp .env.example .env
# Edit .env:
#   - Set AIRFLOW__CORE__FERNET_KEY  (generate below)
#   - Set AIRFLOW__WEBSERVER__SECRET_KEY
```

Generate the Fernet key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 2 — Start the stack

```bash
# Build images and bring up Postgres + Airflow
docker compose up --build -d

# Wait ~30 seconds for Airflow to initialise, then check health:
docker compose ps
```

### 3 — Initialise the pipeline database

```bash
# Create schemas (raw, staging, marts, scores, audit) and tables
docker compose exec airflow-scheduler bash -c \
  "cd /opt/airflow/project && python scripts/init_db.py"
```

### 4 — Train the model

```bash
# Generate 90 days of synthetic data and train LightGBM (~2 minutes)
docker compose exec airflow-scheduler bash -c \
  "cd /opt/airflow/project && python scripts/train_model.py"
```

### 5 — Trigger the DAG

Open [http://localhost:8080](http://localhost:8080) → login `admin / admin`

Enable **fraud_scoring_pipeline** and trigger a manual run, or wait for the daily schedule (01:00 UTC).

---

## Running Tests Locally

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run all unit tests with coverage
pytest tests/ --cov=src --cov-report=term-missing -v

# Lint
flake8 src/ scripts/ dags/ tests/
```

---

## dbt Models

### Staging — `stg_transactions` (view)

Cleans `raw.transactions`: type casts, filters null IDs and zero/negative amounts.

### Mart — `fct_transaction_features` (table)

Core feature mart used by the scorer. Key features computed with SQL window functions:

| Feature | SQL Technique |
|---------|---------------|
| `txn_count_1h` | `COUNT(*) OVER … RANGE BETWEEN INTERVAL '1 hour' PRECEDING` |
| `txn_count_24h` | Same, 24-hour window |
| `txn_count_7d` | Same, 7-day window |
| `amt_sum_1h` | `SUM(amount) OVER … RANGE 1h` |
| `amt_ratio` | `amount / NULLIF(amt_mean_prev_30, 0)` |
| `distance_km` | Haversine formula in SQL |
| `hour_of_day` | `EXTRACT(HOUR FROM transaction_time)` |
| `is_weekend` | `EXTRACT(DOW FROM …) IN (0, 6)` |

### Mart — `fct_daily_summary` (table)

Aggregate stats per `batch_date` — used for monitoring dashboards.

---

## Drift Detection

PSI (Population Stability Index) is computed daily for each feature, comparing the current batch against a 30-day rolling reference window.

| PSI Range | Interpretation | Action |
|-----------|---------------|--------|
| < 0.10 | Negligible drift | None |
| 0.10 – 0.20 | Minor drift | Log warning |
| > 0.20 | Major drift | Slack alert + audit record |

Drift results are persisted to `audit.drift_reports` and surfaced in the Airflow task log via XCom.

---

## GCP Deployment (Cloud Run Job)

### Prerequisites

- GCP project with billing enabled
- `gcloud` CLI authenticated: `gcloud auth login`
- Artifact Registry API enabled

### 1 — Create Artifact Registry repository

```bash
gcloud artifacts repositories create ml-batch-pipeline \
  --repository-format=docker \
  --location=asia-southeast1 \
  --description="ml-batch-pipeline images"
```

### 2 — Build and push the scorer image

```bash
export GCP_PROJECT_ID=your-gcp-project-id
export GCP_REGION=asia-southeast1
export IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/ml-batch-pipeline/fraud-scorer:latest"

# Authenticate Docker to Artifact Registry
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

# Build the scorer image
docker build \
  -f docker/Dockerfile.scorer \
  -t "${IMAGE_URI}" \
  .

# Push to Artifact Registry
docker push "${IMAGE_URI}"
```

### 3 — Create the Cloud Run Job

```bash
gcloud run jobs create fraud-scorer \
  --image="${IMAGE_URI}" \
  --region="${GCP_REGION}" \
  --task-timeout=30m \
  --max-retries=1 \
  --parallelism=1 \
  --set-env-vars="DATABASE_URL=postgresql://user:pass@HOST:5432/pipeline_db" \
  --set-env-vars="MODEL_PATH=/app/models/lgbm_fraud.pkl"
```

> **Note:** For production, use Secret Manager for `DATABASE_URL`:
> ```bash
> --set-secrets="DATABASE_URL=pipeline-db-url:latest"
> ```

### 4 — Execute the job manually

```bash
# Score yesterday's batch
gcloud run jobs execute fraud-scorer --region=${GCP_REGION}

# Score a specific date
gcloud run jobs execute fraud-scorer \
  --region=${GCP_REGION} \
  --args="--date,2026-03-15"
```

### 5 — Schedule with Cloud Scheduler (optional)

```bash
# Trigger daily at 01:30 UTC (30 min after Airflow DAG completes transform)
gcloud scheduler jobs create http fraud-scorer-daily \
  --schedule="30 1 * * *" \
  --uri="https://${GCP_REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${GCP_PROJECT_ID}/jobs/fraud-scorer:run" \
  --http-method=POST \
  --oauth-service-account-email="your-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --location="${GCP_REGION}"
```

### Required IAM roles

| Service Account | Role |
|----------------|------|
| Cloud Run Job SA | `roles/cloudsql.client` |
| Cloud Scheduler SA | `roles/run.invoker` |

---

## Slack Alerts (Optional)

Set `SLACK_WEBHOOK_URL` in `.env` to receive notifications:

- **Drift alert** — triggered when any feature PSI > 0.20 (configurable via `PSI_ALERT_THRESHOLD`)
- **Clean run** — daily confirmation with fraud rate and volume stats

Create an Incoming Webhook at [api.slack.com/apps](https://api.slack.com/apps).

---

## Key Design Decisions

**Why dbt for feature engineering?**
SQL window functions with `RANGE BETWEEN INTERVAL` express velocity features more naturally than Python loops, and the dbt DAG documents every transformation. The Python `engineer_features()` in `train_model.py` mirrors the SQL logic exactly, ensuring training/serving consistency.

**Why LocalExecutor?**
For a portfolio demo on a single machine, LocalExecutor keeps the stack simple (no Redis/Celery). The code is Executor-agnostic — switch to `CeleryExecutor` or `KubernetesExecutor` by changing one env var.

**Why Cloud Run Jobs (not Cloud Run Services)?**
The scorer is a run-to-completion workload, not a long-running service. Cloud Run Jobs provide auto-scaling, retries, and pay-per-execution billing — ideal for batch scoring.

**Why synthetic data?**
No Kaggle account or dataset download required. The generator produces statistically realistic Indonesian transaction patterns (home location clustering, fraud geo-distance, category biases) and is fully deterministic with a fixed seed.

---

## License

MIT
