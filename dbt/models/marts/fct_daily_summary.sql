{{
    config(
        materialized='table',
        schema='marts',
    )
}}

/*
Daily summary mart — aggregate statistics per batch_date.

Used by the drift detection step to track distribution statistics over time
and by the pipeline audit log.
*/

SELECT
    batch_date,

    -- Volume
    COUNT(*)                                        AS n_transactions,
    COUNT(DISTINCT card_number)                     AS n_unique_cards,

    -- Amount distribution
    AVG(amount)                                     AS avg_amount,
    PERCENTILE_CONT(0.50) WITHIN GROUP
        (ORDER BY amount)                           AS median_amount,
    PERCENTILE_CONT(0.95) WITHIN GROUP
        (ORDER BY amount)                           AS p95_amount,
    STDDEV(amount)                                  AS std_amount,

    -- Fraud (ground-truth labels — for evaluation only)
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)       AS n_fraud,
    AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END)   AS fraud_rate,

    -- Velocity stats
    AVG(txn_count_1h)                               AS avg_txn_count_1h,
    AVG(txn_count_24h)                              AS avg_txn_count_24h,
    AVG(amt_ratio)                                  AS avg_amt_ratio,

    -- Distance stats
    AVG(distance_km)                                AS avg_distance_km,
    PERCENTILE_CONT(0.95) WITHIN GROUP
        (ORDER BY distance_km)                      AS p95_distance_km

FROM {{ ref('fct_transaction_features') }}

GROUP BY batch_date
ORDER BY batch_date DESC
