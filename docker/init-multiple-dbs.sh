#!/usr/bin/env bash
# Creates both the Airflow metadata DB and the pipeline data warehouse DB
# on PostgreSQL first start.
set -e

function create_db() {
    local db=$1
    echo "Creating database '$db'..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        SELECT 'CREATE DATABASE ${db}'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${db}')\gexec
EOSQL
}

IFS=',' read -ra DBS <<< "$POSTGRES_MULTIPLE_DATABASES"
for db in "${DBS[@]}"; do
    create_db "$db"
done
echo "Databases ready."
