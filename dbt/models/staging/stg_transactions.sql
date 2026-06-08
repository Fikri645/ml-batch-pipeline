{{
    config(
        materialized='view',
        schema='staging',
    )
}}

/*
Staging layer — cleans and casts raw transaction data.

- Casts to correct types
- Filters obviously invalid rows (null IDs, zero/negative amounts)
- Adds a cleaned_at metadata column
*/

SELECT
    transaction_id::TEXT                                    AS transaction_id,
    card_number::TEXT                                       AS card_number,
    merchant::TEXT                                          AS merchant,
    category::TEXT                                          AS category,
    amount::NUMERIC(12, 2)                                  AS amount,
    lat::DOUBLE PRECISION                                   AS card_lat,
    long::DOUBLE PRECISION                                  AS card_long,
    merchant_lat::DOUBLE PRECISION                          AS merchant_lat,
    merchant_long::DOUBLE PRECISION                         AS merchant_long,
    transaction_time::TIMESTAMPTZ                           AS transaction_time,
    is_fraud::BOOLEAN                                       AS is_fraud,
    batch_date::DATE                                        AS batch_date,
    NOW()                                                   AS cleaned_at

FROM {{ source('raw', 'transactions') }}

WHERE
    transaction_id IS NOT NULL
    AND amount > 0
    AND batch_date IS NOT NULL
