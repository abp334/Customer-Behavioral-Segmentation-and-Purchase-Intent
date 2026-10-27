"""
src/etl.py
==========
Modular Extract-Load-Transform (ELT) Pipeline for REES46 eCommerce Data.
Builds the Medallion Architecture:
  - Bronze: Raw sampled events (from Phase 1)
  - Silver: Cleaned, deduplicated, enriched events with parsed hierarchies
  - Gold: Star Schema dimensional warehouse (DuckDB) & Customer Feature Mart
"""

import os
import time
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

from src.config import (
    PROJECT_ROOT, BRONZE_DIR, SILVER_DIR, GOLD_DIR,
    DUCKDB_PATH, FEATURE_WINDOW_START, FEATURE_WINDOW_END,
    LABEL_WINDOW_START, LABEL_WINDOW_END
)


def get_duckdb_connection(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection configured for optimal local performance."""
    path_str = str(db_path) if db_path else str(DUCKDB_PATH)
    con = duckdb.connect(path_str)
    # Enable multi-threading (up to 8 threads on Apple M1)
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='4GB';")
    return con


def build_silver(bronze_dir: Path = BRONZE_DIR, silver_dir: Path = SILVER_DIR) -> dict:
    """
    Transform Bronze Parquet into Silver Parquet:
      1. Filter out invalid/non-positive prices (price > 0)
      2. Deduplicate exact duplicate rows
      3. Disambiguate session key: concat(user_id, '_', user_session)
      4. Parse category_code hierarchy into category_level_1 and category_level_2
      5. Impute missing brands and categories with 'unknown'
      6. Extract temporal attributes (hour_of_day, day_of_week, is_weekend)
      7. Save as Snappy-compressed Parquet partitioned by event_date
    """
    t0 = time.perf_counter()
    silver_events_path = silver_dir / "events_oct"
    silver_events_path.parent.mkdir(parents=True, exist_ok=True)
    
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='4GB';")

    bronze_glob = str(bronze_dir / "events_oct" / "**" / "*.parquet")
    
    print("\n" + "=" * 60)
    print("EXECUTING SILVER LAYER TRANSFORMATION")
    print("=" * 60)
    
    # Check input bronze row count
    bronze_count = con.execute(f"SELECT count(*) FROM '{bronze_glob}'").fetchone()[0]
    print(f"Bronze input rows    : {bronze_count:>12,}")

    silver_query = f"""
    CREATE OR REPLACE TEMP TABLE silver_clean AS
    SELECT DISTINCT
        user_id,
        event_time,
        event_date,
        event_type,
        product_id,
        category_id,
        COALESCE(category_code, 'unknown') AS category_code,
        CASE 
            WHEN category_code IS NOT NULL AND contains(category_code, '.') THEN split_part(category_code, '.', 1)
            WHEN category_code IS NOT NULL AND category_code != '' THEN category_code
            ELSE 'unknown'
        END AS category_level_1,
        CASE 
            WHEN category_code IS NOT NULL AND array_length(string_split(category_code, '.')) > 1 THEN split_part(category_code, '.', 2)
            ELSE 'unknown'
        END AS category_level_2,
        COALESCE(brand, 'unknown') AS brand,
        price,
        COALESCE(user_session, 'unknown_session') AS user_session,
        concat(CAST(user_id AS VARCHAR), '_', COALESCE(user_session, 'unknown_session')) AS session_id,
        date_part('hour', event_time) AS hour_of_day,
        dayofweek(event_date) AS day_of_week,
        CASE WHEN dayofweek(event_date) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend
    FROM '{bronze_glob}'
    WHERE price > 0;
    """
    con.execute(silver_query)

    silver_count = con.execute("SELECT count(*) FROM silver_clean").fetchone()[0]
    dropped_count = bronze_count - silver_count
    distinct_users = con.execute("SELECT count(distinct user_id) FROM silver_clean").fetchone()[0]
    distinct_sessions = con.execute("SELECT count(distinct session_id) FROM silver_clean").fetchone()[0]
    
    print(f"Silver cleaned rows  : {silver_count:>12,}  (Dropped {dropped_count:,} invalid/dup rows)")
    print(f"Distinct users       : {distinct_users:>12,}")
    print(f"Distinct sessions    : {distinct_sessions:>12,}")

    # Export to Silver Parquet partitioned by event_date
    export_path = str(silver_events_path)
    con.execute(f"""
    COPY silver_clean TO '{export_path}' (
        FORMAT PARQUET,
        PARTITION_BY (event_date),
        OVERWRITE_OR_IGNORE 1,
        COMPRESSION SNAPPY
    );
    """)
    elapsed = time.perf_counter() - t0
    print(f"✅ Silver Parquet saved to: {export_path} (took {elapsed:.2f}s)")
    con.close()

    return {
        "bronze_count": bronze_count,
        "silver_count": silver_count,
        "dropped_count": dropped_count,
        "distinct_users": distinct_users,
        "distinct_sessions": distinct_sessions,
        "elapsed_seconds": round(elapsed, 2)
    }


def build_star_schema(silver_dir: Path = SILVER_DIR, duckdb_path: Path = DUCKDB_PATH) -> dict:
    """
    Build Star Schema in DuckDB warehouse:
      - dim_date (Date dimension, 31 days)
      - dim_product (Product dimension: category, brand, median price, view/cart/buy counts)
      - dim_user (User dimension: customer tenure, total sessions, lifetime spend)
      - fact_events (Fact table: event details with foreign keys to all dimensions)
    """
    t0 = time.perf_counter()
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    con = get_duckdb_connection(duckdb_path)

    silver_glob = str(silver_dir / "events_oct" / "**" / "*.parquet")

    print("\n" + "=" * 60)
    print("BUILDING DUCKDB STAR SCHEMA")
    print("=" * 60)

    # 1. Dimension: dim_date
    con.execute("""
    CREATE OR REPLACE TABLE dim_date AS
    WITH dates AS (
        SELECT DISTINCT event_date AS full_date
        FROM read_parquet(?)
    )
    SELECT
        CAST(strftime(full_date, '%Y%m%d') AS INTEGER) AS date_key,
        full_date,
        day(full_date) AS day_of_month,
        strftime(full_date, '%A') AS day_name,
        dayofweek(full_date) AS day_of_week,
        CASE WHEN dayofweek(full_date) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
        week(full_date) AS week_of_year
    FROM dates
    ORDER BY full_date;
    """, [silver_glob])
    dim_date_count = con.execute("SELECT count(*) FROM dim_date").fetchone()[0]
    print(f"dim_date built        : {dim_date_count:>10,} rows")

    # 2. Dimension: dim_product
    con.execute("""
    CREATE OR REPLACE TABLE dim_product AS
    SELECT
        product_id,
        first(category_id) AS category_id,
        first(category_code) AS category_code,
        first(category_level_1) AS category_level_1,
        first(category_level_2) AS category_level_2,
        first(brand) AS brand,
        round(avg(price), 2) AS avg_price,
        round(median(price), 2) AS median_price,
        round(min(price), 2) AS min_price,
        round(max(price), 2) AS max_price,
        count(*) FILTER (WHERE event_type = 'view') AS total_views,
        count(*) FILTER (WHERE event_type = 'cart') AS total_carts,
        count(*) FILTER (WHERE event_type = 'purchase') AS total_purchases
    FROM read_parquet(?)
    GROUP BY product_id;
    """, [silver_glob])
    dim_product_count = con.execute("SELECT count(*) FROM dim_product").fetchone()[0]
    print(f"dim_product built     : {dim_product_count:>10,} rows")

    # 3. Dimension: dim_user
    con.execute("""
    CREATE OR REPLACE TABLE dim_user AS
    SELECT
        user_id,
        min(event_time) AS first_event_time,
        max(event_time) AS last_event_time,
        count(distinct event_date) AS active_days,
        count(distinct session_id) AS total_sessions,
        count(*) AS total_events,
        count(*) FILTER (WHERE event_type = 'purchase') AS total_purchases,
        round(coalesce(sum(price) FILTER (WHERE event_type = 'purchase'), 0.0), 2) AS lifetime_spend
    FROM read_parquet(?)
    GROUP BY user_id;
    """, [silver_glob])
    dim_user_count = con.execute("SELECT count(*) FROM dim_user").fetchone()[0]
    print(f"dim_user built        : {dim_user_count:>10,} rows")

    # 4. Fact Table: fact_events
    con.execute("""
    CREATE OR REPLACE TABLE fact_events AS
    SELECT
        row_number() OVER () AS event_id,
        event_time,
        CAST(strftime(event_date, '%Y%m%d') AS INTEGER) AS date_key,
        event_date,
        user_id,
        product_id,
        session_id,
        event_type,
        price,
        hour_of_day,
        is_weekend
    FROM read_parquet(?);
    """, [silver_glob])
    fact_count = con.execute("SELECT count(*) FROM fact_events").fetchone()[0]
    print(f"fact_events built     : {fact_count:>10,} rows")

    elapsed = time.perf_counter() - t0
    print(f"✅ Star Schema created in DuckDB warehouse: {duckdb_path} ({elapsed:.2f}s)")
    con.close()

    return {
        "dim_date_count": dim_date_count,
        "dim_product_count": dim_product_count,
        "dim_user_count": dim_user_count,
        "fact_events_count": fact_count,
        "elapsed_seconds": round(elapsed, 2)
    }


def build_gold_features(silver_dir: Path = SILVER_DIR,
                        gold_dir: Path = GOLD_DIR,
                        duckdb_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    """
    Construct the Customer Behavioral Feature Mart (Gold Layer):
      - Strictly uses FEATURE WINDOW (Oct 01 to Oct 20) for all input signals
      - Derives TARGET LABEL from LABEL WINDOW (Oct 21 to Oct 31)
      - Computes RFM, funnel rates, breadth of exploration, session density
      - Saves as Parquet, CSV sample, and DuckDB table 'gold_user_features'
    """
    t0 = time.perf_counter()
    gold_dir.mkdir(parents=True, exist_ok=True)
    con = get_duckdb_connection(duckdb_path)
    silver_glob = str(silver_dir / "events_oct" / "**" / "*.parquet")

    print("\n" + "=" * 60)
    print("CONSTRUCTING GOLD CUSTOMER FEATURE MART")
    print("=" * 60)
    print(f"Feature window : {FEATURE_WINDOW_START} → {FEATURE_WINDOW_END} (Signals)")
    print(f"Label window   : {LABEL_WINDOW_START} → {LABEL_WINDOW_END} (Target)")

    feature_query = f"""
    CREATE OR REPLACE TABLE gold_user_features AS
    WITH silver AS (
        SELECT * FROM read_parquet('{silver_glob}')
    ),
    feat_window AS (
        SELECT
            user_id,
            -- Recency & Tenure (relative to cutoff date 2019-10-20)
            datediff('day', max(event_date), DATE '{FEATURE_WINDOW_END}') AS recency_days,
            datediff('day', min(event_date), DATE '{FEATURE_WINDOW_END}') AS tenure_days,
            count(distinct event_date) AS active_days,
            count(distinct session_id) AS total_sessions,
            count(*) AS total_events,
            
            -- Funnel event volume
            count(*) FILTER (WHERE event_type = 'view') AS view_count,
            count(*) FILTER (WHERE event_type = 'cart') AS cart_count,
            count(*) FILTER (WHERE event_type = 'purchase') AS purchase_count,
            
            -- Monetary metrics
            round(coalesce(sum(price) FILTER (WHERE event_type = 'purchase'), 0.0), 2) AS total_spend,
            round(coalesce(avg(price) FILTER (WHERE event_type = 'purchase'), 0.0), 2) AS avg_purchase_price,
            round(coalesce(max(price) FILTER (WHERE event_type = 'purchase'), 0.0), 2) AS max_purchase_price,
            round(max(price), 2) AS max_price_interacted,
            round(avg(price), 2) AS avg_price_interacted,
            
            -- Breadth of exploration
            count(distinct product_id) AS n_unique_products,
            count(distinct category_level_1) AS n_unique_cat1,
            count(distinct brand) AS n_unique_brands,
            
            -- Behavioral ratios & temporal patterns
            round(count(*) FILTER (WHERE is_weekend = 1) * 1.0 / count(*), 4) AS weekend_event_ratio,
            round(coalesce(count(*) FILTER (WHERE event_type = 'cart') * 1.0 / nullif(count(*) FILTER (WHERE event_type = 'view'), 0), 0.0), 4) AS cart_to_view_ratio,
            round(coalesce(count(*) FILTER (WHERE event_type = 'purchase') * 1.0 / nullif(count(*) FILTER (WHERE event_type = 'cart'), 0), 0.0), 4) AS purchase_to_cart_ratio,
            round(coalesce(count(*) FILTER (WHERE event_type = 'purchase') * 1.0 / nullif(count(*) FILTER (WHERE event_type = 'view'), 0), 0.0), 4) AS purchase_to_view_ratio,
            round(count(*) * 1.0 / count(distinct session_id), 2) AS events_per_session
        FROM silver
        WHERE event_date <= DATE '{FEATURE_WINDOW_END}'
        GROUP BY user_id
    ),
    label_window AS (
        SELECT
            user_id,
            count(*) AS future_purchases,
            round(coalesce(sum(price), 0.0), 2) AS future_spend
        FROM silver
        WHERE event_date >= DATE '{LABEL_WINDOW_START}' AND event_type = 'purchase'
        GROUP BY user_id
    )
    SELECT
        f.*,
        CASE WHEN coalesce(l.future_purchases, 0) > 0 THEN 1 ELSE 0 END AS will_purchase_next_10d,
        coalesce(l.future_purchases, 0) AS future_purchases,
        coalesce(l.future_spend, 0.0) AS future_spend
    FROM feat_window f
    LEFT JOIN label_window l ON f.user_id = l.user_id;
    """
    con.execute(feature_query)

    df_features = con.execute("SELECT * FROM gold_user_features").df()
    n_users = len(df_features)
    n_target_pos = (df_features["will_purchase_next_10d"] == 1).sum()
    pos_rate = df_features["will_purchase_next_10d"].mean() * 100

    print(f"Customer feature rows: {n_users:>10,}")
    print(f"Feature columns      : {df_features.shape[1]:>10}")
    print(f"Future buyers (target): {n_target_pos:>10,} ({pos_rate:.2f}% positive class rate)")

    # Save to Parquet
    parquet_out = gold_dir / "user_features.parquet"
    df_features.to_parquet(parquet_out, index=False, compression="snappy")
    print(f"✅ User features saved to Parquet: {parquet_out}")

    # Save small CSV sample (1,000 rows) for easy spreadsheet & Tableau preview
    csv_sample_out = gold_dir / "user_features_sample.csv"
    df_features.head(1000).to_csv(csv_sample_out, index=False)
    print(f"📄 User features sample CSV saved: {csv_sample_out}")

    con.close()
    elapsed = time.perf_counter() - t0
    print(f"⏱  Feature Mart generation completed in {elapsed:.2f}s")

    return df_features


def run_phase2_pipeline():
    """Run full Phase 2 pipeline sequentially."""
    print("\n" + "#" * 60)
    print("STARTING PHASE 2 ELT PIPELINE")
    print("#" * 60)
    start_total = time.perf_counter()

    silver_stats = build_silver()
    star_stats = build_star_schema()
    df_features = build_gold_features()

    total_time = time.perf_counter() - start_total
    print("\n" + "=" * 60)
    print(f"PHASE 2 COMPLETED SUCCESSFULLY IN {total_time:.2f}s")
    print("=" * 60)
    return silver_stats, star_stats, df_features


if __name__ == "__main__":
    run_phase2_pipeline()
