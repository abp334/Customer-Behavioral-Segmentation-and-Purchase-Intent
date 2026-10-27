"""
============================================================
NOTEBOOK 02 – ELT PIPELINE, STAR SCHEMA & FEATURE MART
============================================================
Phase 2  |  Syllabus Units: 1 (Lifecycle), 2 (Big Data & Architecture),
                           6 (Data Transformation & Consistency)
============================================================

OBJECTIVE
---------
1. Cleanse Bronze data into Silver Parquet (handling nulls, zero prices, duplicates).
2. Deconstruct dot-separated hierarchy in `category_code` into distinct levels.
3. Establish a Dimensional Model (Star Schema) inside DuckDB:
   - Fact: `fact_events`
   - Dimensions: `dim_date`, `dim_product`, `dim_user`
4. Engineer Customer Behavioral Features (Gold Layer) with strict temporal windowing:
   - Feature Window : Oct 01 → Oct 20 (Customer signals)
   - Label Window   : Oct 21 → Oct 31 (Target: will_purchase_next_10d)
5. Generate reconciliation audit tables and visual profiles.

WHY THIS MATTERS (viva talking points)
---------------------------------------
• ETL vs ELT: We use ELT. Raw data is loaded into storage first (Bronze), then
  transformed in-engine using DuckDB vectorized execution. This is scalable,
  reproducible, and avoids intermediate serialization bottlenecks.
• Star Schema vs 3NF / Snowflake:
  - 3NF minimizes redundancy via heavy normalization, requiring 5-10 joins per query.
  - Star Schema denormalizes dimensions around a central fact table, optimizing for
    fast aggregations, BI tools (Tableau/Power BI), and OLAP workloads.
• Temporal Windowing (Preventing Data Leakage):
  In predictive modeling, using future data to predict the future causes catastrophic
  test leakage. We freeze feature extraction on Oct 20 at 23:59:59. Only purchases in
  Oct 21–31 define the positive target label.
"""

# ============================================================
# CELL 0 – Imports & Environment Setup
# ============================================================
import sys, os, time, warnings
warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
if os.path.basename(os.getcwd()) == "notebooks":
    sys.path.insert(0, os.path.dirname(os.getcwd()))
else:
    sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    PROJECT_ROOT, BRONZE_DIR, SILVER_DIR, GOLD_DIR,
    REPORTS_DIR, DUCKDB_PATH, FEATURE_WINDOW_START,
    FEATURE_WINDOW_END, LABEL_WINDOW_START, LABEL_WINDOW_END
)
from src.etl import build_silver, build_star_schema, build_gold_features

print("=" * 60)
print("PHASE 2 ENVIRONMENT INITIALIZED")
print("=" * 60)
print(f"Project Root : {PROJECT_ROOT}")
print(f"DuckDB Path  : {DUCKDB_PATH}")
print(f"DuckDB Ver   : {duckdb.__version__}")

# ============================================================
# CELL 1 – Inspect Bronze Layer Input
# ============================================================
con = duckdb.connect()
bronze_glob = str(BRONZE_DIR / "events_oct" / "**" / "*.parquet")

bronze_count = con.execute(f"SELECT count(*) FROM '{bronze_glob}'").fetchone()[0]
print("\n" + "=" * 60)
print("BRONZE LAYER INPUT AUDIT")
print("=" * 60)
print(f"Bronze total events : {bronze_count:,}")

# Preview raw sample
df_preview = con.execute(f"SELECT * FROM '{bronze_glob}' LIMIT 5").df()
print("\nSample Bronze Records:")
print(df_preview[["event_time", "event_type", "product_id", "category_code", "brand", "price", "user_id"]])

# ============================================================
# CELL 2 – Execute Silver Layer Transformations
# ============================================================
"""
TRANSFORMATION RULES (Unit 6: Data Consistency & Cleaning):
1. Remove non-positive prices: `price > 0`.
2. Deduplicate exact duplicate event records.
3. Unpack `category_code`:
   - "electronics.smartphone" → category_level_1 = "electronics", category_level_2 = "smartphone"
   - Fallback for nulls: "unknown"
4. Impute missing brands: COALESCE(brand, 'unknown')
5. Composite session key: `concat(user_id, '_', user_session)` guarantees session isolation.
"""
silver_stats = build_silver(BRONZE_DIR, SILVER_DIR)

# Audit Silver Parquet on disk
silver_glob = str(SILVER_DIR / "events_oct" / "**" / "*.parquet")
silver_audit = con.execute(f"""
SELECT 
    count(*) AS total_rows,
    count(distinct user_id) AS n_users,
    count(distinct product_id) AS n_products,
    count(distinct session_id) AS n_sessions,
    count(*) FILTER (WHERE brand = 'unknown') AS unknown_brands,
    count(*) FILTER (WHERE category_level_1 = 'unknown') AS unknown_cat1,
    min(price) AS min_price,
    max(price) AS max_price
FROM '{silver_glob}';
""").df()

print("\nSilver Data Hygiene Audit:")
print(silver_audit.to_string(index=False))

# ============================================================
# CELL 3 – Construct Dimensional Model (Star Schema) in DuckDB
# ============================================================
"""
DIMENSIONAL MODELING DESIGN (Star Schema):
- Central Fact: `fact_events` (Granularity: 1 row per clickstream action)
  - Foreign keys: date_key, user_id, product_id
  - Degenerate dimension: session_id, event_type
  - Measures: price
- Dimension 1: `dim_date` (Role-playing date dimension with calendar attributes)
- Dimension 2: `dim_product` (Product master with hierarchical category rollup & aggregated counts)
- Dimension 3: `dim_user` (Customer master with tenure, lifetime sessions, and lifetime spend)
"""
star_stats = build_star_schema(SILVER_DIR, DUCKDB_PATH)

# Verify Star Schema inside DuckDB
con_warehouse = duckdb.connect(str(DUCKDB_PATH))
tables = con_warehouse.execute("SHOW TABLES;").df()
print("\nWarehouse Tables:")
print(tables.to_string(index=False))

# Query facts joined to dimensions (validating referential integrity)
test_star_query = """
SELECT 
    d.day_name,
    p.category_level_1,
    f.event_type,
    count(*) AS event_count,
    round(sum(f.price), 2) AS total_gmv
FROM fact_events f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_product p ON f.product_id = p.product_id
WHERE f.event_type = 'purchase'
GROUP BY d.day_name, p.category_level_1, f.event_type
ORDER BY total_gmv DESC
LIMIT 5;
"""
print("\nTop 5 Purchase Categories by Day (Testing Star Schema Joins):")
print(con_warehouse.execute(test_star_query).df().to_string(index=False))

# ============================================================
# CELL 4 – Engineer Gold Customer Feature Mart
# ============================================================
"""
TEMPORAL TRAIN/TEST SPLIT (Preventing Data Leakage):
- Feature Observation Window : Oct 01 → Oct 20 (Input features X)
- Future Outcome Window      : Oct 21 → Oct 31 (Target label y)

FEATURES ENGINEERED:
1. RFM:
   - Recency  : Days since last interaction relative to Oct 20
   - Frequency: Total sessions, total events, view count, cart count, purchase count
   - Monetary : Total historical spend, avg purchase price, max item price
2. Behavioral Ratios:
   - cart_to_view_ratio: Intent intensity
   - purchase_to_cart_ratio: Checkout completion rate
   - events_per_session: Session depth
3. Exploration Breadth:
   - n_unique_products, n_unique_cat1, n_unique_brands
4. Target Label:
   - `will_purchase_next_10d`: Binary indicator (1 if purchase occurred in Oct 21-31, else 0)
"""
df_features = build_gold_features(SILVER_DIR, GOLD_DIR, DUCKDB_PATH)

# ============================================================
# CELL 5 – Reconciliation & Audit Reporting
# ============================================================
counts_log = [
    {"stage": "01_raw_csv", "count": 42448764, "desc": "Raw October clickstream log"},
    {"stage": "02_bronze_sample", "count": silver_stats["bronze_count"], "desc": "Deterministic 5% user sample"},
    {"stage": "03_silver_clean", "count": silver_stats["silver_count"], "desc": "Deduplicated, price > 0, imputed"},
    {"stage": "04_gold_users", "count": len(df_features), "desc": "Feature-window active customer profiles"}
]
recon_df = pd.DataFrame(counts_log)
recon_df["pct_of_sample"] = (recon_df["count"] / silver_stats["bronze_count"] * 100).round(2)
recon_df.to_csv(REPORTS_DIR / "reconciliation_phase2.csv", index=False)

print("\n" + "=" * 60)
print("PHASE 2 RECONCILIATION TABLE")
print("=" * 60)
print(recon_df.to_string(index=False))

# Save Star Schema Summary
schema_summary = pd.DataFrame([
    {"table": "dim_date", "rows": star_stats["dim_date_count"], "type": "Dimension"},
    {"table": "dim_product", "rows": star_stats["dim_product_count"], "type": "Dimension"},
    {"table": "dim_user", "rows": star_stats["dim_user_count"], "type": "Dimension"},
    {"table": "fact_events", "rows": star_stats["fact_events_count"], "type": "Fact"},
    {"table": "gold_user_features", "rows": len(df_features), "type": "Feature Mart"}
])
schema_summary.to_csv(REPORTS_DIR / "phase2_star_schema_summary.csv", index=False)

# ============================================================
# CELL 6 – Feature Distribution Visualizations
# ============================================================
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Phase 2 — Gold Feature Mart & Target Label Profile", fontsize=16, fontweight="bold")

# 6a. Target Label Distribution (Severe Class Imbalance)
ax = axes[0, 0]
target_counts = df_features["will_purchase_next_10d"].value_counts()
target_pct = df_features["will_purchase_next_10d"].value_counts(normalize=True) * 100
bars = ax.bar(["No Purchase (0)", "Purchased (1)"], target_counts.values, color=["#3498db", "#e74c3c"], width=0.5)
ax.set_title(f"Target Label: Next 10-Day Purchase\n(Positive Rate: {target_pct.iloc[1]:.2f}%)", fontweight="bold")
ax.set_ylabel("Customer Count")
for bar, count, pct in zip(bars, target_counts.values, target_pct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500, f"{count:,}\n({pct:.1f}%)",
            ha="center", fontsize=10, fontweight="bold")

# 6b. Recency Distribution
ax = axes[0, 1]
ax.hist(df_features["recency_days"], bins=20, color="#2ecc71", edgecolor="white", alpha=0.85)
ax.set_title("Customer Recency Distribution (Days to Oct 20)", fontweight="bold")
ax.set_xlabel("Days Since Last Event")
ax.set_ylabel("Customer Frequency")

# 6c. Total Events per User (Log Scale)
ax = axes[1, 0]
ax.hist(df_features["total_events"], bins=np.logspace(0, 3.5, 30), color="#9b59b6", edgecolor="white", alpha=0.85)
ax.set_xscale("log")
ax.set_title("Total Events per Customer (Log Scale)", fontweight="bold")
ax.set_xlabel("Event Count")
ax.set_ylabel("Customer Frequency")

# 6d. Historical Spend by Target Class (Boxplot)
ax = axes[1, 1]
sns.boxplot(data=df_features, x="will_purchase_next_10d", y="total_spend", ax=ax,
            palette=["#3498db", "#e74c3c"], showfliers=False)
ax.set_title("Historical Spend in Feature Window by Future Purchase Status", fontweight="bold")
ax.set_xticklabels(["Did Not Purchase", "Purchased"])
ax.set_xlabel("Future Purchase Label (Next 10 Days)")
ax.set_ylabel("Total Spend ($)")

plt.tight_layout()
plot_path = REPORTS_DIR / "phase2_features_overview.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n📊 Feature profile plot saved: {plot_path}")

# ============================================================
# CELL 7 – Phase 2 Summary
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2 COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"""
Summary Deliverables:
1. Silver Parquet Layer  : {SILVER_DIR / 'events_oct'} ({silver_stats['silver_count']:,} clean events)
2. Star Schema Warehouse : {DUCKDB_PATH}
   - dim_date            : {star_stats['dim_date_count']} records
   - dim_product         : {star_stats['dim_product_count']:,} records
   - dim_user            : {star_stats['dim_user_count']:,} records
   - fact_events         : {star_stats['fact_events_count']:,} records
3. Gold Customer Mart    : {GOLD_DIR / 'user_features.parquet'} ({len(df_features):,} customers, 24 features)
4. Target Imbalance      : {target_pct.iloc[1]:.2f}% positive class rate ({target_counts.iloc[1]:,} buyers)
5. Audit & Visuals       : {REPORTS_DIR / 'reconciliation_phase2.csv'} & {plot_path}
""")
con_warehouse.close()
con.close()
