"""Shared paths and lazy-loaded warehouse for API routes."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DATA_DIR = PROJECT_ROOT / "data" / "app_data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


@lru_cache(maxsize=1)
def get_warehouse() -> pd.DataFrame:
    return pd.read_parquet(APP_DATA_DIR / "scored_customers_warehouse.parquet")
