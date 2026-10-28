-- Cohort retention template
-- Groups clients by the quarter they signed, then shows what share
-- still have an active contract in each later quarter.
--
-- Expected tables (generic schema, no company data):
--   accounts(account_id, signed_date)
--   contracts(account_id, start_date, end_date)

WITH cohorts AS (
    SELECT
        account_id,
        DATE_TRUNC('quarter', signed_date) AS cohort_quarter
    FROM accounts
),

quarters AS (
    SELECT DISTINCT DATE_TRUNC('quarter', start_date) AS quarter
    FROM contracts
),

active AS (
    SELECT DISTINCT
        c.account_id,
        q.quarter
    FROM contracts c
    JOIN quarters q
      ON q.quarter BETWEEN DATE_TRUNC('quarter', c.start_date)
                       AND DATE_TRUNC('quarter', c.end_date)
),

cohort_sizes AS (
    SELECT cohort_quarter, COUNT(*) AS cohort_size
    FROM cohorts
    GROUP BY cohort_quarter
)

SELECT
    co.cohort_quarter,
    a.quarter                                        AS activity_quarter,
    (EXTRACT(YEAR FROM a.quarter) - EXTRACT(YEAR FROM co.cohort_quarter)) * 4
      + (EXTRACT(QUARTER FROM a.quarter) - EXTRACT(QUARTER FROM co.cohort_quarter))
                                                     AS quarters_since_signing,
    COUNT(DISTINCT a.account_id)                     AS active_accounts,
    cs.cohort_size,
    ROUND(100.0 * COUNT(DISTINCT a.account_id) / cs.cohort_size, 1) AS retention_pct
FROM cohorts co
JOIN active a        ON a.account_id = co.account_id
                    AND a.quarter >= co.cohort_quarter
JOIN cohort_sizes cs ON cs.cohort_quarter = co.cohort_quarter
GROUP BY co.cohort_quarter, a.quarter, cs.cohort_size
ORDER BY co.cohort_quarter, a.quarter;
