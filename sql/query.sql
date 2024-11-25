CREATE TABLE "s3_crawler_database"."final_view2" WITH (format = 'parquet') AS WITH AssetTiers AS (
  SELECT "bank name",
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
FirstSecondQuarter AS (
  SELECT "bank name",
    "state",
    MAX(
      CASE
        WHEN EXTRACT(
          QUARTER
          FROM quarter_date
        ) = 1 THEN "total deposits" ELSE NULL
      END
    ) AS first_quarter_deposits,
    MAX(
      CASE
        WHEN EXTRACT(
          QUARTER
          FROM quarter_date
        ) = 2 THEN "total deposits" ELSE NULL
      END
    ) AS second_quarter_deposits,
    MAX(
      CASE
        WHEN EXTRACT(
          QUARTER
          FROM quarter_date
        ) = 1 THEN "total assets" ELSE NULL
      END
    ) AS first_quarter_assets,
    MAX(
      CASE
        WHEN EXTRACT(
          QUARTER
          FROM quarter_date
        ) = 2 THEN "total assets" ELSE NULL
      END
    ) AS second_quarter_assets,
    MAX(
      CASE
        WHEN EXTRACT(
          QUARTER
          FROM quarter_date
        ) = 1 THEN asset_tier ELSE NULL
      END
    ) AS first_quarter_asset_tier,
    MAX(
      CASE
        WHEN EXTRACT(
          QUARTER
          FROM quarter_date
        ) = 2 THEN asset_tier ELSE NULL
      END
    ) AS second_quarter_asset_tier
  FROM AssetTiers
  WHERE EXTRACT(
      QUARTER
      FROM quarter_date
    ) IN (1, 2) -- Focus on Q1 and Q2
  GROUP BY "bank name",
    "state"
),
DepositChangeFlag AS (
  SELECT "bank name",
    "state",
    first_quarter_deposits,
    second_quarter_deposits,
    first_quarter_assets,
    second_quarter_assets,
    first_quarter_asset_tier,
    second_quarter_asset_tier,
    CASE
      WHEN second_quarter_deposits < first_quarter_deposits THEN 'Yes' ELSE 'No'
    END AS deposit_decline_flag,
    CASE
      WHEN first_quarter_assets IS NOT NULL
      AND second_quarter_assets IS NOT NULL THEN ROUND(
        (
          (second_quarter_assets - first_quarter_assets) * 100.0
        ) / first_quarter_assets,
        2
      ) ELSE NULL
    END AS asset_decline_percent
  FROM FirstSecondQuarter
)
SELECT "bank name",
  "state",
  first_quarter_assets,
  second_quarter_assets,
  first_quarter_asset_tier AS asset_tier,
  first_quarter_deposits,
  second_quarter_deposits,
  deposit_decline_flag,
  asset_decline_percent
FROM DepositChangeFlag
ORDER BY "bank name";