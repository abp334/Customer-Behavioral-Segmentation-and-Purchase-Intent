"""Batch CSV scoring."""

import io
from typing import List

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.deps import APP_DATA_DIR
from src.scoring_engine import NUMERIC_FEATURES, score_batch_dataframe

router = APIRouter(prefix="/api", tags=["batch"])

SIMPLE_COLUMNS = [
    "customer_id",
    "recency_days",
    "total_spend",
    "cart_count",
    "total_sessions",
    "total_events",
]

PERSONA_SHORT = {
    0: "Dormant",
    1: "Window Shoppers",
    2: "Champions",
    3: "Active Explorers",
}


def _expand_simple_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Accept simple CRM-ish columns and expand to model features when needed."""
    out = df.copy()
    for col in NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = 0.0

    if "total_sessions" in df.columns:
        sessions = pd.to_numeric(df["total_sessions"], errors="coerce").fillna(1).clip(lower=1)
    else:
        sessions = pd.Series([5.0] * len(df))

    if "total_events" in df.columns:
        events = pd.to_numeric(df["total_events"], errors="coerce").fillna(sessions)
    else:
        events = sessions * 3

    if "cart_count" in df.columns:
        carts = pd.to_numeric(df["cart_count"], errors="coerce").fillna(0).clip(lower=0)
    else:
        carts = pd.Series([0.0] * len(df))

    if "total_spend" in df.columns:
        spend = pd.to_numeric(df["total_spend"], errors="coerce").fillna(0).clip(lower=0)
    else:
        spend = pd.Series([0.0] * len(df))

    if "recency_days" in df.columns:
        recency = pd.to_numeric(df["recency_days"], errors="coerce").fillna(7).clip(lower=0)
    else:
        recency = pd.Series([7.0] * len(df))

    # Only fill gaps when the advanced template columns were missing / zeroed
    missing_advanced = not all(c in df.columns for c in NUMERIC_FEATURES)
    if missing_advanced:
        views = (events - carts).clip(lower=1)
        purchases = (spend > 0).astype(float)
        avg_price = spend.where(purchases > 0, 0) / purchases.replace(0, 1)
        out["recency_days"] = recency
        out["tenure_days"] = recency.clip(lower=14)
        out["active_days"] = (sessions / 2).clip(lower=1, upper=14)
        out["total_sessions"] = sessions
        out["total_events"] = events
        out["view_count"] = views
        out["cart_count"] = carts
        out["purchase_count"] = purchases
        out["total_spend"] = spend
        out["avg_purchase_price"] = avg_price
        out["max_price_interacted"] = pd.concat([avg_price, spend], axis=1).max(axis=1).clip(lower=50)
        out["n_unique_products"] = (views / 3).clip(lower=1, upper=50)
        out["n_unique_cat1"] = (views / 10).clip(lower=1, upper=8)
        out["n_unique_brands"] = (views / 5).clip(lower=1, upper=20)
        out["weekend_event_ratio"] = 0.3
        out["cart_to_view_ratio"] = (carts / views).fillna(0)
        out["events_per_session"] = events / sessions

    return out


def _product_rows(scored: pd.DataFrame, original: pd.DataFrame) -> List[dict]:
    id_col = None
    for candidate in ("customer_id", "customer_code", "user_id"):
        if candidate in original.columns:
            id_col = candidate
            break

    rows = []
    for i, row in scored.iterrows():
        cid = original.iloc[i][id_col] if id_col is not None else f"ROW-{i + 1}"
        cluster = int(row["predicted_cluster"])
        rows.append(
            {
                "customer_id": str(cid),
                "purchase_likelihood": round(float(row["calibrated_purchase_prob"]) * 100, 1),
                "purchase_probability": float(row["calibrated_purchase_prob"]),
                "segment": PERSONA_SHORT.get(cluster, row["persona_name"]),
                "segment_full": row["persona_name"],
                "intent_rank": int(row["intent_rank"]),
                "total_spend": round(float(row.get("total_spend", 0) or 0), 2),
                "cart_count": float(row.get("cart_count", 0) or 0),
                "recency_days": float(row.get("recency_days", 0) or 0),
            }
        )
    rows.sort(key=lambda r: r["intent_rank"])
    return rows


@router.post("/batch")
async def score_batch(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows.")

    expanded = _expand_simple_csv(df)
    try:
        scored = score_batch_dataframe(expanded)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Align original index with scored (score_batch sorts)
    scored_reset = scored.reset_index(drop=True)
    # Rebuild original order map via intent — use scored order for product rows
    original_aligned = expanded.loc[scored.index].reset_index(drop=True)
    rows = _product_rows(scored_reset, original_aligned)

    expected_buyers = sum(r["purchase_probability"] for r in rows)
    return {
        "row_count": len(rows),
        "expected_buyers": round(expected_buyers, 1),
        "mean_likelihood": round(sum(r["purchase_likelihood"] for r in rows) / len(rows), 1),
        "high_intent_count": sum(1 for r in rows if r["purchase_likelihood"] >= 10),
        "rows": rows,
    }


@router.get("/batch/demo")
def score_demo_batch():
    path = APP_DATA_DIR / "demo_batch_customers.csv"
    df = pd.read_csv(path)
    expanded = _expand_simple_csv(df)
    scored = score_batch_dataframe(expanded)
    scored_reset = scored.reset_index(drop=True)
    original_aligned = expanded.loc[scored.index].reset_index(drop=True)
    rows = _product_rows(scored_reset, original_aligned)
    expected_buyers = sum(r["purchase_probability"] for r in rows)
    return {
        "row_count": len(rows),
        "expected_buyers": round(expected_buyers, 1),
        "mean_likelihood": round(sum(r["purchase_likelihood"] for r in rows) / len(rows), 1),
        "high_intent_count": sum(1 for r in rows if r["purchase_likelihood"] >= 10),
        "rows": rows,
        "source": "demo",
    }


@router.get("/batch/template")
def download_simple_template():
    csv = (
        "customer_id,recency_days,total_spend,cart_count,total_sessions,total_events\n"
        "CUST-001,2,190.00,4,7,17\n"
        "CUST-002,0,0,0,42,265\n"
        "CUST-003,10,382.81,1,10,180\n"
    )
    return StreamingResponse(
        io.BytesIO(csv.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=nexus_batch_template.csv"},
    )
