WITH AssetTiers AS (
    SELECT 
        "bank name",
        "state",
        "total assets",
        "total deposits",
        CAST("quarter date" AS DATE) AS quarter_date,
        CASE 
            WHEN "total assets" < 500000 THEN 'Below $500M'
            WHEN "total assets" BETWEEN 500000 AND 1000000 THEN '$500M - $1B'
            WHEN "total assets" > 1000000 THEN 'Above $1B'
        END AS asset_tier
    FROM "s3_crawler_database"."processed_data"
),
DepositChange AS (
    SELECT 
        current."bank name",
        current."state",
        current."total assets",
        current."total deposits",
        current.quarter_date,
        current.asset_tier, -- Ensure asset_tier is included here
        previous."total deposits" AS prev_total_deposits,
        CASE 
            WHEN previous."total deposits" IS NOT NULL AND 
                 (current."total deposits" - previous."total deposits") / previous."total deposits" <= -0.05 THEN 'Yes'
            ELSE 'No'
        END AS deposit_decline_flag
    FROM AssetTiers current
    LEFT JOIN AssetTiers previous
    ON current."bank name" = previous."bank name"
       AND DATE_ADD('month', -3, current.quarter_date) = previous.quarter_date
)
SELECT 
    "bank name",
    "state",
    "total assets",
    "total deposits",
    quarter_date,
    asset_tier, -- Include this in the final SELECT
    deposit_decline_flag
FROM DepositChange
ORDER BY "bank name", quarter_date;
