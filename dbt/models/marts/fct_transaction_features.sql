{{
    config(
        materialized='table',
        schema='marts',
    )
}}

/*
Feature engineering mart — mirrors the Python logic in scripts/train_model.py.

Features computed here:
  Velocity
    txn_count_1h     # transactions for this card in the last 1 hour
    txn_count_24h    # transactions for this card in the last 24 hours
    txn_count_7d     # transactions for this card in the last 7 days
    amt_sum_1h       # total amount for this card in the last 1 hour

  Behavioural
    amt_mean_prev_30  # rolling mean of this card's last 30 transactions (excludes current)
    amt_ratio         # amount / amt_mean_prev_30 — anomalous spending ratio

  Geographic
    distance_km       # haversine distance between card home location and merchant

  Temporal
    hour_of_day
    day_of_week       # 0=Sunday … 6=Saturday
    is_weekend

NOTE: PostgreSQL window functions with RANGE BETWEEN INTERVAL are used for the
time-based velocity windows. This requires transaction_time to be TIMESTAMPTZ.
*/

WITH base AS (
    SELECT * FROM {{ ref('stg_transactions') }}
),

velocity AS (
    SELECT
        transaction_id,
        card_number,
        transaction_time,
        amount,

        -- 1-hour velocity
        COUNT(*) OVER (
            PARTITION BY card_number
            ORDER BY transaction_time
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
        ) AS txn_count_1h,

        SUM(amount) OVER (
            PARTITION BY card_number
            ORDER BY transaction_time
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
        ) AS amt_sum_1h,

        -- 24-hour velocity
        COUNT(*) OVER (
            PARTITION BY card_number
            ORDER BY transaction_time
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        ) AS txn_count_24h,

        -- 7-day velocity
        COUNT(*) OVER (
            PARTITION BY card_number
            ORDER BY transaction_time
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
        ) AS txn_count_7d,

        -- Rolling mean of prior 30 transactions (excludes self)
        AVG(amount) OVER (
            PARTITION BY card_number
            ORDER BY transaction_time
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS amt_mean_prev_30

    FROM base
),

geo AS (
    SELECT
        transaction_id,
        /*
         Haversine approximation in SQL.
         R = 6371 km. Result is in kilometres.
         Accuracy is sufficient for our purposes (< 0.5% error for distances < 500 km).
        */
        6371.0 * 2.0 * ASIN(
            SQRT(
                POWER(SIN((RADIANS(merchant_lat) - RADIANS(card_lat)) / 2.0), 2)
                + COS(RADIANS(card_lat))
                  * COS(RADIANS(merchant_lat))
                  * POWER(SIN((RADIANS(merchant_long) - RADIANS(card_long)) / 2.0), 2)
            )
        ) AS distance_km
    FROM base
)

SELECT
    b.transaction_id,
    b.card_number,
    b.merchant,
    b.category,
    b.amount,
    b.card_lat,
    b.card_long,
    b.merchant_lat,
    b.merchant_long,
    b.transaction_time,
    b.is_fraud,
    b.batch_date,

    -- Velocity
    v.txn_count_1h,
    v.txn_count_24h,
    v.txn_count_7d,
    v.amt_sum_1h,

    -- Behavioural
    COALESCE(v.amt_mean_prev_30, b.amount)                    AS amt_mean_prev_30,
    b.amount / NULLIF(COALESCE(v.amt_mean_prev_30, b.amount), 0) AS amt_ratio,

    -- Geographic
    g.distance_km,

    -- Temporal
    EXTRACT(HOUR FROM b.transaction_time)::INTEGER             AS hour_of_day,
    EXTRACT(DOW  FROM b.transaction_time)::INTEGER             AS day_of_week,
    CASE WHEN EXTRACT(DOW FROM b.transaction_time) IN (0, 6)
         THEN 1 ELSE 0
    END                                                        AS is_weekend

FROM base     b
JOIN velocity v USING (transaction_id)
JOIN geo      g USING (transaction_id)
