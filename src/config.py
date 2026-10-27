"""
src/config.py
=============
Central configuration for paths, constants, and random seeds.
Every notebook imports from here so nothing is hard-coded twice.
"""

import os
from pathlib import Path

# ── Project root (two levels up from src/) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data layer paths (Medallion architecture) ───────────────────
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
BRONZE_DIR   = PROJECT_ROOT / "data" / "bronze"
SILVER_DIR   = PROJECT_ROOT / "data" / "silver"
GOLD_DIR     = PROJECT_ROOT / "data" / "gold"

# ── Output paths ────────────────────────────────────────────────
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
DASHBOARD_DIR= PROJECT_ROOT / "dashboard"
APP_DIR      = PROJECT_ROOT / "app"

# ── Ensure all directories exist ────────────────────────────────
for d in [RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR,
          MODELS_DIR, REPORTS_DIR, DASHBOARD_DIR, APP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────
RANDOM_SEED = 42

# ── Sampling ────────────────────────────────────────────────────
# We keep 5 % of users (by deterministic hash) so full user
# histories stay intact.  Adjust if your machine struggles.
USER_SAMPLE_FRACTION = 0.05        # 5 % of distinct user_ids
USER_HASH_MOD        = 100         # hash(user_id) % 100 < 5

# ── Temporal windows (October) ──────────────────────────────────
FEATURE_WINDOW_START = "2019-10-01"
FEATURE_WINDOW_END   = "2019-10-20"   # inclusive
LABEL_WINDOW_START   = "2019-10-21"
LABEL_WINDOW_END     = "2019-10-31"   # inclusive

# ── Dataset files ───────────────────────────────────────────────
RAW_OCT_CSV  = RAW_DIR / "2019-Oct.csv"
RAW_NOV_CSV  = RAW_DIR / "2019-Nov.csv"

# ── DuckDB warehouse ───────────────────────────────────────────
DUCKDB_PATH  = GOLD_DIR / "warehouse.duckdb"

# ── Column schema (expected in the raw REES46 CSVs) ────────────
RAW_COLUMNS = [
    "event_time",       # UTC timestamp  (e.g. 2019-10-01 00:00:00 UTC)
    "event_type",       # view | cart | remove_from_cart | purchase
    "product_id",       # int64
    "category_id",      # int64
    "category_code",    # dot-separated hierarchy, often NULL
    "brand",            # string, often NULL
    "price",            # float (USD)
    "user_id",          # int64
    "user_session",     # UUID string
]

# ── PySpark tuning for 8 GB M1 laptop ──────────────────────────
SPARK_DRIVER_MEMORY  = "4g"         # leave ~4 GB for OS + other apps
SPARK_SHUFFLE_PARTS  = 8            # M1 has 8 cores
SPARK_LOCAL_DIR      = str(PROJECT_ROOT / "data" / ".spark_tmp")

# ── PySpark Python version fix ──────────────────────────────────
# The system default python is 3.14 but our venv is 3.10.
# PySpark workers inherit the system python unless we tell them
# explicitly.  Without this, you get:
#   PYTHON_VERSION_MISMATCH: worker 3.14 != driver 3.10
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
os.environ.setdefault("PYSPARK_PYTHON",        VENV_PYTHON)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON",  VENV_PYTHON)

# ── Java 17 for PySpark (Java 22 breaks Spark reflection) ──────
JAVA17_HOME = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
if os.path.exists(JAVA17_HOME):
    os.environ["JAVA_HOME"] = JAVA17_HOME
