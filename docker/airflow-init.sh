#!/usr/bin/env bash
# Airflow one-time init: DB migrate + admin user + optional pipeline DB setup
set -e

case "$1" in
    "init-db")
        echo "[init] Creating pipeline schemas and tables..."
        python /opt/airflow/project/scripts/init_db.py
        ;;
    "train")
        echo "[init] Training model artifact..."
        python /opt/airflow/project/scripts/train_model.py
        ;;
    *)
        echo "[init] Running Airflow DB migrate..."
        airflow db migrate
        if [ "${_AIRFLOW_WWW_USER_CREATE:-false}" = "true" ]; then
            airflow users create \
                --username "${_AIRFLOW_WWW_USER_USERNAME:-admin}" \
                --password "${_AIRFLOW_WWW_USER_PASSWORD:-admin}" \
                --firstname Admin \
                --lastname User \
                --role Admin \
                --email admin@example.com
        fi
        ;;
esac
