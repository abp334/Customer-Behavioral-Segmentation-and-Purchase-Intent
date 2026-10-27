"""
src/utils.py
============
Reusable helper functions used across all notebooks.
Keeps notebooks clean; logic tested once, used everywhere.
"""

import hashlib
import time
import functools
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────
# 1. DETERMINISTIC USER SAMPLING
# ─────────────────────────────────────────────────────────────────
def deterministic_user_sample(user_id: int, mod: int = 100, threshold: int = 5) -> bool:
    """
    Return True if user_id should be kept in the sample.

    Uses MD5 hash so the sample is:
      - deterministic  (same user_id → same True/False every run)
      - uniform         (MD5 distributes evenly across buckets)
      - history-safe    (we keep ALL events for sampled users)

    Why MD5 and not user_id % mod?
      Because user_id values are often sequential or clustered,
      which makes modulo biased.  MD5 scatters them uniformly.

    Parameters
    ----------
    user_id   : the raw user_id integer
    mod       : bucket count  (100 → 1% granularity)
    threshold : keep if hash_bucket < threshold  (5 → 5%)
    """
    hash_hex = hashlib.md5(str(user_id).encode()).hexdigest()
    bucket   = int(hash_hex, 16) % mod
    return bucket < threshold


def get_sampled_user_ids(all_user_ids, mod=100, threshold=5):
    """
    Given an array-like of user_ids, return the set of ids that
    pass the deterministic hash filter.
    """
    return {uid for uid in all_user_ids
            if deterministic_user_sample(uid, mod, threshold)}


# ─────────────────────────────────────────────────────────────────
# 2. LOGGING / AUDITING HELPERS
# ─────────────────────────────────────────────────────────────────
def log_row_count(df, stage_name: str, counts_log: list | None = None):
    """
    Print and optionally record the row count of a DataFrame.
    Works with both pandas DataFrames and PySpark DataFrames.
    """
    if hasattr(df, "count") and callable(df.count) and not isinstance(df, pd.DataFrame):
        # PySpark DataFrame
        n = df.count()
    else:
        n = len(df)
    msg = f"[{stage_name}]  rows = {n:>12,}"
    print(msg)
    if counts_log is not None:
        counts_log.append({"stage": stage_name, "row_count": n})
    return n


def reconciliation_table(counts_log: list) -> pd.DataFrame:
    """
    Build a tidy reconciliation table from a list of
    {"stage": ..., "row_count": ...} dicts.
    Adds a '% of raw' column for quick auditing.
    """
    df = pd.DataFrame(counts_log)
    raw_count = df["row_count"].iloc[0] if len(df) > 0 else 1
    df["pct_of_raw"] = (df["row_count"] / raw_count * 100).round(2)
    return df


# ─────────────────────────────────────────────────────────────────
# 3. DATA QUALITY HELPERS
# ─────────────────────────────────────────────────────────────────
def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-shot data quality summary for every column:
      - dtype, non-null count, null count, null %,
        unique count, top 3 values, min, max (for numerics).
    """
    records = []
    for col in df.columns:
        series = df[col]
        rec = {
            "column":     col,
            "dtype":      str(series.dtype),
            "non_null":   series.notna().sum(),
            "null_count": series.isna().sum(),
            "null_pct":   round(series.isna().mean() * 100, 2),
            "n_unique":   series.nunique(),
        }
        # Top 3 most frequent values
        top3 = series.value_counts().head(3)
        rec["top_values"] = dict(top3)

        if pd.api.types.is_numeric_dtype(series):
            rec["min"] = series.min()
            rec["max"] = series.max()
            rec["mean"] = round(series.mean(), 4)
            rec["median"] = round(series.median(), 4)
        else:
            rec["min"] = rec["max"] = rec["mean"] = rec["median"] = None

        records.append(rec)
    return pd.DataFrame(records)


def check_negative_prices(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """Return rows where price <= 0 (should not exist in valid data)."""
    mask = df[price_col] <= 0
    bad = df[mask]
    print(f"Rows with {price_col} <= 0 : {len(bad):,}  "
          f"({len(bad)/len(df)*100:.3f}%)")
    return bad


def check_duplicate_rows(df: pd.DataFrame) -> int:
    """Count exact duplicate rows."""
    n_dup = df.duplicated().sum()
    print(f"Exact duplicate rows : {n_dup:,}  ({n_dup/len(df)*100:.3f}%)")
    return n_dup


def check_session_user_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag sessions that span multiple user_ids.
    A valid session belongs to exactly one user.
    """
    session_users = df.groupby("user_session")["user_id"].nunique()
    bad_sessions = session_users[session_users > 1]
    print(f"Sessions with >1 user_id : {len(bad_sessions):,}")
    return bad_sessions


# ─────────────────────────────────────────────────────────────────
# 4. TIMING DECORATOR (for CSV vs Parquet comparison)
# ─────────────────────────────────────────────────────────────────
def timer(func):
    """Decorator that prints elapsed wall-clock time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"⏱  {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


# ─────────────────────────────────────────────────────────────────
# 5. MEMORY USAGE
# ─────────────────────────────────────────────────────────────────
def memory_usage_mb(df: pd.DataFrame) -> float:
    """Return DataFrame memory usage in MB."""
    mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage: {mb:.1f} MB")
    return mb
