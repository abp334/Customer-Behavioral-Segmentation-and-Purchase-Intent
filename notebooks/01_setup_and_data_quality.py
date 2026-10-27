"""
============================================================
NOTEBOOK 01 – ENVIRONMENT SETUP & DATA QUALITY CHECKS
============================================================
Phase 1  |  Syllabus Units: 1 (DS Lifecycle), 2 (Big Data)
============================================================

OBJECTIVE
---------
1. Verify the environment (Python, Java, PySpark).
2. Load the raw REES46 October CSV with PySpark (memory-safe).
3. Sample 5% of users by deterministic hash (full histories intact).
4. Save the sample as Parquet (Bronze layer).
5. Produce a data dictionary and initial data quality report.
6. Demonstrate CSV vs Parquet speed difference (Unit 2 evidence).

WHY THIS MATTERS (viva talking points)
---------------------------------------
• We use PySpark even locally because the full file is ~5.5 GB / 42M rows.
  pandas would need ~12 GB of RAM to hold that; our laptop has 8 GB.
  PySpark reads lazily and only materializes the rows we ask for.
• We sample by user_id (not random rows) so each sampled user's full
  clickstream stays intact — essential for session analysis and the
  temporal feature/label window split.
• We convert to Parquet immediately: columnar format, Snappy compression,
  column pruning. This cuts disk from ~5.5 GB CSV to ~250 MB Parquet and
  speeds up every downstream read by 5-10×.

RUN INSTRUCTIONS
----------------
1. Activate the virtual environment:
      source .venv/bin/activate
2. Set JAVA_HOME (add to your ~/.zshrc for persistence):
      export JAVA_HOME=$(/usr/libexec/java_home -v 17)
3. Launch Jupyter or run cell-by-cell in VS Code.
"""

# ============================================================
# CELL 0 – Imports & environment verification
# ============================================================
import sys, os, time, warnings
warnings.filterwarnings("ignore")

# Make sure we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), ""))
# If running from notebooks/ folder, adjust:
if os.path.basename(os.getcwd()) == "notebooks":
    sys.path.insert(0, os.path.dirname(os.getcwd()))
else:
    sys.path.insert(0, os.getcwd())

from src.config import *
from src.utils import (
    deterministic_user_sample, log_row_count,
    reconciliation_table, quality_report,
    check_negative_prices, check_duplicate_rows,
    check_session_user_integrity, timer, memory_usage_mb
)

import numpy as np
import pandas as pd

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)
print(f"Python       : {sys.version}")
print(f"NumPy        : {np.__version__}")
print(f"pandas       : {pd.__version__}")
print(f"Project root : {PROJECT_ROOT}")
print(f"Raw data dir : {RAW_DIR}")

# Check Java
java_home = os.environ.get("JAVA_HOME", "NOT SET")
print(f"JAVA_HOME    : {java_home}")

# ============================================================
# CELL 1 – Initialize PySpark (local mode, tuned for 8 GB M1)
# ============================================================
"""
WHY LOCAL MODE?
  PySpark in local[*] uses all CPU cores as executor threads.
  On M1 (8 performance + efficiency cores), this gives us
  parallel reads and transforms without a cluster.
  
  We cap driver memory at 4g to leave room for the OS and
  other apps. Shuffle partitions = 8 to match core count
  (default 200 is wasteful for local mode).

JAVA 17 NOTE:
  PySpark 3.5 needs Java 17.  Java 22 breaks because Spark
  uses JVM internal APIs that Java 22 blocks by default.
  We installed openjdk@17 and set JAVA_HOME accordingly.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# JAVA_HOME and PYSPARK_PYTHON are set automatically by src/config.py
# (Java 17 for Spark compatibility, venv Python 3.10 for worker parity)
print(f"JAVA_HOME        : {os.environ.get('JAVA_HOME', 'NOT SET')}")
print(f"PYSPARK_PYTHON   : {os.environ.get('PYSPARK_PYTHON', 'NOT SET')}")

spark = (
    SparkSession.builder
    .master("local[*]")                             # use all cores
    .appName("DSC_GP_Phase1")
    .config("spark.driver.memory", SPARK_DRIVER_MEMORY)    # 4g
    .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTS)  # 8
    .config("spark.local.dir", SPARK_LOCAL_DIR)
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print(f"\n✅ Spark {spark.version} running in local mode")
print(f"   Driver memory : {SPARK_DRIVER_MEMORY}")
print(f"   Shuffle parts : {SPARK_SHUFFLE_PARTS}")

# ============================================================
# CELL 2 – Load raw CSV with PySpark
# ============================================================
"""
DATA DICTIONARY (REES46 eCommerce Behavior Data)
──────────────────────────────────────────────────
Column          Type      Description
──────────────  ────────  ────────────────────────────────────────────
event_time      string    UTC timestamp, e.g. "2019-10-01 00:00:00 UTC"
event_type      string    One of: view, cart, remove_from_cart, purchase
product_id      long      Unique product identifier
category_id     long      Category identifier (internal)
category_code   string    Human-readable category (e.g. "electronics.smartphone")
                          *** HIGH NULL RATE expected (~50%) ***
brand           string    Product brand; NULL if unbranded / unknown
price           double    Product price in USD (should be > 0)
user_id         long      Unique customer identifier
user_session    string    UUID grouping events in one browsing session

DATA TYPES (structured / semi-structured / unstructured)
────────────────────────────────────────────────────────
• Structured  : price (numeric), product_id, user_id, category_id
• Semi-struct : category_code (dot-separated hierarchy: "electronics.smartphone")
• Temporal    : event_time (needs parsing from string)
• Categorical : event_type, brand
"""

print("\n" + "=" * 60)
print("LOADING RAW CSV WITH PYSPARK")
print("=" * 60)

# Check if the raw file exists
if not RAW_OCT_CSV.exists():
    print(f"\n⚠️  Raw file not found at: {RAW_OCT_CSV}")
    print("""
╔══════════════════════════════════════════════════════════════╗
║  DOWNLOAD INSTRUCTIONS                                      ║
║                                                              ║
║  Option A – Kaggle CLI (recommended):                        ║
║    pip install kaggle                                        ║
║    kaggle datasets download -d mkechinov/                    ║
║      ecommerce-behavior-data-from-multi-category-store       ║
║      -f 2019-Oct.csv.gz  -p data/raw/                       ║
║    gunzip data/raw/2019-Oct.csv.gz                           ║
║                                                              ║
║  Option B – Browser:                                         ║
║    1. Go to https://www.kaggle.com/datasets/mkechinov/       ║
║       ecommerce-behavior-data-from-multi-category-store      ║
║    2. Download "2019-Oct.csv" (5.5 GB)                       ║
║    3. Move it to: data/raw/2019-Oct.csv                      ║
║                                                              ║
║  Option C – Use a smaller sample (if bandwidth limited):     ║
║    kaggle datasets download -d mkechinov/                    ║
║      ecommerce-behavior-data-from-multi-category-store       ║
║      -f 2019-Oct.csv.gz  -p data/raw/                       ║
║    # Then run the sampling cell below                        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    # For now, stop here — re-run this cell after downloading
    raise FileNotFoundError(f"Place 2019-Oct.csv in {RAW_DIR}")

# Define schema explicitly (faster than inferSchema, and type-safe)
schema = T.StructType([
    T.StructField("event_time",    T.StringType(),  True),
    T.StructField("event_type",    T.StringType(),  True),
    T.StructField("product_id",    T.LongType(),    True),
    T.StructField("category_id",   T.LongType(),    True),
    T.StructField("category_code", T.StringType(),  True),
    T.StructField("brand",         T.StringType(),  True),
    T.StructField("price",         T.DoubleType(),  True),
    T.StructField("user_id",       T.LongType(),    True),
    T.StructField("user_session",  T.StringType(),  True),
])

t0 = time.perf_counter()
raw_sdf = (
    spark.read
    .option("header", "true")
    .schema(schema)                   # explicit schema = no extra scan
    .csv(str(RAW_OCT_CSV))
)
csv_load_time = time.perf_counter() - t0
print(f"⏱  CSV schema applied in {csv_load_time:.2f}s (lazy — no data read yet)")

# ============================================================
# CELL 3 – Row count & basic shape (forces one full scan)
# ============================================================
counts_log = []    # audit trail for reconciliation table

t0 = time.perf_counter()
raw_count = raw_sdf.count()
csv_count_time = time.perf_counter() - t0

print(f"\nRaw CSV rows     : {raw_count:>12,}")
print(f"Columns          : {len(raw_sdf.columns):>12}")
print(f"⏱  Full count took {csv_count_time:.1f}s")
counts_log.append({"stage": "01_raw_csv", "row_count": raw_count})

# ============================================================
# CELL 4 – Deterministic 5% user sample
# ============================================================
"""
WHY SAMPLE BY USER, NOT BY ROW?
  Random row sampling would break user histories. Imagine user
  A has 100 events; random 5% row sampling might keep 5 random
  events from A's timeline — destroying session structure, funnel
  analysis, and the temporal feature/label window.

  Instead we:
  1. Collect all distinct user_ids.
  2. Hash each with MD5 → take mod 100 → keep if < 5.
  3. Filter the full DataFrame to keep only these users.

  This keeps ~5% of users with ALL their events intact.

WHY MD5 HASH?
  user_id % 100 would be biased if IDs are sequential (e.g. all
  ending in 00-04 are structurally different from others).
  MD5 scatters any ID pattern uniformly across buckets.
"""

print("\n" + "=" * 60)
print("DETERMINISTIC USER SAMPLING (5%)")
print("=" * 60)

# Step 1: Get all distinct user_ids (this is a Spark action)
t0 = time.perf_counter()
all_user_ids = (
    raw_sdf
    .select("user_id")
    .distinct()
    .rdd.flatMap(lambda x: x)
    .collect()
)
n_total_users = len(all_user_ids)
print(f"Total distinct users : {n_total_users:,}")
print(f"⏱  Distinct users collected in {time.perf_counter()-t0:.1f}s")

# Step 2: Apply deterministic hash filter
sampled_ids = [uid for uid in all_user_ids
               if deterministic_user_sample(uid, USER_HASH_MOD,
                                            int(USER_SAMPLE_FRACTION * USER_HASH_MOD))]
n_sampled_users = len(sampled_ids)
print(f"Sampled users        : {n_sampled_users:,}  "
      f"({n_sampled_users/n_total_users*100:.1f}%)")

# Step 3: Broadcast the sampled IDs and filter using native Broadcast Hash Join
# (broadcast is efficient when the table is small: ~150k user IDs is ~1.5 MB in memory)
# Using a native join avoids Python UDF row-by-row serialization over 42M rows.
sampled_users_df = spark.createDataFrame(
    [(int(uid),) for uid in sampled_ids],
    schema=T.StructType([T.StructField("user_id", T.LongType(), False)])
)

sampled_sdf = raw_sdf.join(F.broadcast(sampled_users_df), on="user_id", how="inner")

sample_count = sampled_sdf.count()
print(f"Sampled rows         : {sample_count:,}  "
      f"({sample_count/raw_count*100:.1f}% of raw rows)")
counts_log.append({"stage": "02_user_sampled", "row_count": sample_count})

# ============================================================
# CELL 5 – Parse event_time and save as Bronze Parquet
# ============================================================
"""
BRONZE LAYER (Medallion Architecture)
  Bronze = raw data, minimally parsed, in an efficient format.
  We parse event_time from string to timestamp and save as
  Snappy-compressed Parquet, partitioned by date.

  Partitioning by date means:
  - Each date gets its own folder of Parquet files
  - Downstream queries that filter by date skip irrelevant folders
  - This is how data lakes work at scale (S3, ADLS, GCS)
"""

print("\n" + "=" * 60)
print("SAVING BRONZE PARQUET (partitioned by date)")
print("=" * 60)

# Parse the timestamp: "2019-10-01 00:00:00 UTC" → proper TimestampType
bronze_sdf = (
    sampled_sdf
    .withColumn("event_time",
                F.to_timestamp(F.col("event_time"), "yyyy-MM-dd HH:mm:ss 'UTC'"))
    .withColumn("event_date", F.to_date(F.col("event_time")))
)

# Write partitioned Parquet
bronze_path = str(BRONZE_DIR / "events_oct")

t0 = time.perf_counter()
(
    bronze_sdf
    .repartition(8)              # 8 partitions ≈ 8 cores
    .write
    .mode("overwrite")
    .partitionBy("event_date")   # one folder per day
    .parquet(bronze_path)
)
parquet_write_time = time.perf_counter() - t0
print(f"✅ Bronze Parquet written to: {bronze_path}")
print(f"⏱  Write took {parquet_write_time:.1f}s")

# ============================================================
# CELL 6 – CSV vs Parquet speed comparison (Unit 2 evidence)
# ============================================================
"""
UNIT 2 EVIDENCE: Big Data storage formats
  We compare reading the same data from CSV vs Parquet.
  Parquet should be 5-10× faster because:
  1. Columnar: only reads the columns you need
  2. Compressed: Snappy compression → less disk I/O
  3. Metadata: stores min/max per row group → predicate pushdown
"""

print("\n" + "=" * 60)
print("CSV vs PARQUET SPEED COMPARISON")
print("=" * 60)

# Read from Parquet
t0 = time.perf_counter()
bronze_read = spark.read.parquet(bronze_path)
parquet_count = bronze_read.count()
parquet_read_time = time.perf_counter() - t0

print(f"Parquet read + count : {parquet_count:,} rows in {parquet_read_time:.2f}s")
print(f"CSV full count       : {raw_count:,} rows in {csv_count_time:.1f}s")
print(f"Speedup factor       : {csv_count_time / parquet_read_time:.1f}×")

# Column-pruned read (only 3 columns)
t0 = time.perf_counter()
_ = spark.read.parquet(bronze_path).select("user_id", "event_type", "price").count()
pruned_time = time.perf_counter() - t0
print(f"Parquet (3 cols only) : {pruned_time:.2f}s  "
      f"(column pruning skips unused data)")

# Disk size comparison
import subprocess
csv_size = os.path.getsize(RAW_OCT_CSV) / 1024**2
parquet_result = subprocess.run(
    ["du", "-sm", bronze_path], capture_output=True, text=True
)
parquet_size = float(parquet_result.stdout.split()[0])
print(f"\nCSV on disk          : {csv_size:,.0f} MB")
print(f"Parquet on disk      : {parquet_size:,.0f} MB  "
      f"({parquet_size/csv_size*100:.0f}% of CSV)")

counts_log.append({"stage": "03_bronze_parquet", "row_count": parquet_count})

# ============================================================
# CELL 7 – Load Bronze into pandas for quality checks
# ============================================================
"""
Now we switch from PySpark to pandas for detailed quality analysis.
The sampled data (~2M rows) fits comfortably in 8 GB RAM.
We stop Spark to free its 4 GB driver memory.
"""

print("\n" + "=" * 60)
print("LOADING BRONZE INTO PANDAS")
print("=" * 60)

# Read Parquet with pandas (via PyArrow — fast and memory-efficient)
pdf = pd.read_parquet(bronze_path)
pdf["event_date"] = pd.to_datetime(pdf["event_date"]).dt.date
log_row_count(pdf, "04_pandas_bronze", counts_log)
memory_usage_mb(pdf)

# Stop Spark to free memory
spark.stop()
print("🔴 Spark stopped (memory freed for pandas work)")

# ============================================================
# CELL 8 – Data Dictionary (print for the report)
# ============================================================
print("\n" + "=" * 60)
print("DATA DICTIONARY")
print("=" * 60)

data_dict = pd.DataFrame({
    "Column": ["event_time", "event_type", "product_id", "category_id",
               "category_code", "brand", "price", "user_id",
               "user_session", "event_date"],
    "Type": ["timestamp", "string (categorical)", "int64",
             "int64", "string (hierarchical)", "string (categorical)",
             "float64", "int64", "string (UUID)", "date"],
    "Description": [
        "UTC timestamp of the event",
        "Type of behavior: view, cart, remove_from_cart, purchase",
        "Unique product identifier",
        "Internal category ID (numeric)",
        "Human-readable category hierarchy (e.g. electronics.smartphone). ~50% NULL.",
        "Product brand name. NULL if unknown/unbranded.",
        "Product price in USD. Should be > 0.",
        "Unique customer identifier (our primary key for segmentation)",
        "UUID grouping events into a single browsing session",
        "Date extracted from event_time (added by us for partitioning)"
    ],
    "Structured?": [
        "Structured (temporal)", "Structured (categorical)", "Structured (numeric)",
        "Structured (numeric)", "Semi-structured (dot hierarchy)", "Structured (categorical)",
        "Structured (numeric)", "Structured (numeric)", "Structured (identifier)",
        "Structured (temporal)"
    ]
})
print(data_dict.to_string(index=False))

# ============================================================
# CELL 9 – Data Quality Report
# ============================================================
print("\n" + "=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)

qr = quality_report(pdf)
# Display key columns (top_values can be wide)
display_cols = ["column", "dtype", "non_null", "null_count", "null_pct",
                "n_unique", "min", "max", "mean", "median"]
print(qr[display_cols].to_string(index=False))

# Save for the report
qr.to_csv(REPORTS_DIR / "data_quality_report_raw.csv", index=False)
print(f"\n📄 Quality report saved to: {REPORTS_DIR / 'data_quality_report_raw.csv'}")

# ============================================================
# CELL 10 – Specific quality checks
# ============================================================
print("\n" + "=" * 60)
print("SPECIFIC QUALITY CHECKS")
print("=" * 60)

# 10a. Negative / zero prices
print("\n--- Price Check ---")
bad_prices = check_negative_prices(pdf)
if len(bad_prices) > 0:
    print("Sample of bad prices:")
    print(bad_prices.head())

# 10b. Exact duplicates
print("\n--- Duplicate Check ---")
n_dup = check_duplicate_rows(pdf)

# 10c. Event type distribution
print("\n--- Event Type Distribution ---")
event_dist = pdf["event_type"].value_counts()
event_pct  = pdf["event_type"].value_counts(normalize=True) * 100
event_summary = pd.DataFrame({"count": event_dist, "pct": event_pct.round(2)})
print(event_summary)

# 10d. Session-user integrity
print("\n--- Session-User Integrity ---")
bad_sessions = check_session_user_integrity(pdf)

# 10e. Date range
print("\n--- Date Range ---")
print(f"Min event_time : {pdf['event_time'].min()}")
print(f"Max event_time : {pdf['event_time'].max()}")
print(f"Unique dates   : {pdf['event_date'].nunique()}")
print(f"Date range     : {pdf['event_date'].min()} → {pdf['event_date'].max()}")

# 10f. Events per day (quick sanity check)
print("\n--- Events per Day ---")
daily_counts = pdf.groupby("event_date").size().sort_index()
print(daily_counts.to_string())

# 10g. Users with suspiciously high event counts
print("\n--- Top 10 Users by Event Count ---")
user_events = pdf.groupby("user_id").size().sort_values(ascending=False)
print(user_events.head(10))

# ============================================================
# CELL 11 – Reconciliation Table (audit trail)
# ============================================================
print("\n" + "=" * 60)
print("RECONCILIATION TABLE (row count audit)")
print("=" * 60)

recon = reconciliation_table(counts_log)
print(recon.to_string(index=False))

# Save for the report
recon.to_csv(REPORTS_DIR / "reconciliation_phase1.csv", index=False)
print(f"\n📄 Reconciliation table saved to: "
      f"{REPORTS_DIR / 'reconciliation_phase1.csv'}")

# ============================================================
# CELL 12 – Quick visualizations
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Phase 1 – Data Quality Overview", fontsize=16, fontweight="bold")

# 12a. Event type distribution
ax = axes[0, 0]
event_dist.plot(kind="bar", ax=ax, color=["#2196F3", "#FF9800", "#4CAF50", "#F44336"])
ax.set_title("Event Type Distribution")
ax.set_ylabel("Count")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=45)

# 12b. Price distribution (log scale)
ax = axes[0, 1]
valid_prices = pdf.loc[pdf["price"] > 0, "price"]
ax.hist(valid_prices, bins=100, color="#2196F3", alpha=0.7, edgecolor="white")
ax.set_xscale("log")
ax.set_title("Price Distribution (log scale)")
ax.set_xlabel("Price (USD)")
ax.set_ylabel("Frequency")

# 12c. Events per day
ax = axes[1, 0]
daily_counts.plot(ax=ax, marker="o", markersize=3, color="#4CAF50", linewidth=1)
ax.set_title("Events per Day (October)")
ax.set_xlabel("Date")
ax.set_ylabel("Event Count")
ax.tick_params(axis='x', rotation=45)

# 12d. Null rates by column
ax = axes[1, 1]
null_pcts = pdf.isnull().mean() * 100
null_pcts = null_pcts[null_pcts > 0].sort_values(ascending=True)
null_pcts.plot(kind="barh", ax=ax, color="#F44336")
ax.set_title("Null Rate by Column (%)")
ax.set_xlabel("Null %")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "phase1_quality_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n📊 Plot saved to: {REPORTS_DIR / 'phase1_quality_overview.png'}")

# ============================================================
# CELL 13 – Summary for the report
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1 COMPLETE – SUMMARY")
print("=" * 60)
print(f"""
Dataset       : REES46 eCommerce Behavior Data (October 2019)
Raw rows      : {raw_count:,}
Sampled users : {n_sampled_users:,} / {n_total_users:,}  ({n_sampled_users/n_total_users*100:.1f}%)
Sampled rows  : {sample_count:,}  ({sample_count/raw_count*100:.1f}%)
Bronze format : Parquet (Snappy), partitioned by event_date
Columns       : {len(pdf.columns)}
Date range    : {pdf['event_date'].min()} → {pdf['event_date'].max()}
Null hotspots : category_code (~50%), brand (~15-20%)
Price issues  : {len(bad_prices)} rows with price ≤ 0
Duplicates    : {n_dup} exact duplicate rows
Session issues: {len(bad_sessions)} sessions span >1 user

Next: Phase 2 will clean these issues (Silver layer) and build
the Gold feature tables + DuckDB star schema.
""")

# ============================================================
# END OF NOTEBOOK 01
# ============================================================
