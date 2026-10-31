"""Customer browse + single-profile scoring (no memorized IDs required)."""

from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.deps import get_warehouse
from backend.schemas import SimpleProfileRequest
from src.scoring_engine import NUMERIC_FEATURES, score_customer_vector

router = APIRouter(prefix="/api", tags=["customers"])

PERSONA_SHORT = {
    0: "Dormant",
    1: "Window Shoppers",
    2: "Champions",
    3: "Active Explorers",
}

SEGMENT_ALIASES = {
    "dormant": 0,
    "window": 1,
    "window shoppers": 1,
    "champions": 2,
    "champion": 2,
    "explorers": 3,
    "active explorers": 3,
    "active": 3,
}


def _productize_score(raw: Dict[str, Any], features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cluster = int(raw["cluster_id"])
    return {
        "purchase_likelihood": round(float(raw["calibrated_prob"]) * 100, 1),
        "purchase_probability": float(raw["calibrated_prob"]),
        "intent_band": raw["intent_decile"],
        "segment": PERSONA_SHORT.get(cluster, raw["persona_name"]),
        "segment_full": raw["persona_name"],
        "cluster_id": cluster,
        "why": raw["reasons"],
        "recommended_action": raw["recommended_action"],
        "features": {
            "recency_days": features.get("recency_days") if features else None,
            "total_spend": features.get("total_spend") if features else None,
            "cart_count": features.get("cart_count") if features else None,
            "total_sessions": features.get("total_sessions") if features else None,
            "total_events": features.get("total_events") if features else None,
        }
        if features
        else None,
        "science": raw.get("the_science"),
    }


def _row_to_features(row: pd.Series) -> Dict[str, Any]:
    return {col: float(row.get(col, 0) or 0) for col in NUMERIC_FEATURES}


def _simple_to_features(body: SimpleProfileRequest) -> Dict[str, Any]:
    sessions = max(1.0, body.total_sessions)
    events = max(body.total_events, sessions)
    views = max(1.0, events - body.cart_count)
    purchases = 1.0 if body.total_spend > 0 else 0.0
    avg_price = (body.total_spend / purchases) if purchases else 0.0
    return {
        "recency_days": body.recency_days,
        "tenure_days": max(body.recency_days, 14.0),
        "active_days": min(14.0, max(1.0, sessions / 2)),
        "total_sessions": sessions,
        "total_events": events,
        "view_count": views,
        "cart_count": body.cart_count,
        "purchase_count": purchases,
        "total_spend": body.total_spend,
        "avg_purchase_price": avg_price,
        "max_price_interacted": max(avg_price, body.total_spend, 50.0),
        "n_unique_products": min(50.0, max(1.0, views / 3)),
        "n_unique_cat1": min(8.0, max(1.0, views / 10)),
        "n_unique_brands": min(20.0, max(1.0, views / 5)),
        "weekend_event_ratio": 0.3,
        "cart_to_view_ratio": (body.cart_count / views) if views else 0.0,
        "events_per_session": events / sessions,
    }


def _row_summary(r: pd.Series) -> Dict[str, Any]:
    cid = int(r["cluster"])
    return {
        "customer_id": int(r["user_id"]),
        "display_name": f"Shopper {str(int(r['user_id']))[-4:]}",
        "segment": PERSONA_SHORT.get(cid, r["persona_name"]),
        "cluster_id": cid,
        "purchase_likelihood": round(float(r["calibrated_prob"]) * 100, 1),
        "total_spend": round(float(r["total_spend"]), 2),
        "recency_days": float(r["recency_days"]),
        "cart_count": float(r["cart_count"]),
        "total_sessions": float(r["total_sessions"]),
    }


@router.get("/customers/browse")
def browse_customers(
    segment: Optional[str] = Query(None, description="dormant | window | champions | explorers | all"),
    min_likelihood: float = Query(0, ge=0, le=100),
    max_recency_days: Optional[float] = Query(None, ge=0),
    min_spend: float = Query(0, ge=0),
    has_cart: Optional[bool] = Query(None),
    sort: str = Query("likelihood", description="likelihood | spend | recency"),
    limit: int = Query(40, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Browse warehouse customers with filters — click a row in the UI to score."""
    df = get_warehouse().copy()

    if segment and segment.lower() not in ("all", ""):
        key = segment.lower().strip()
        if key.isdigit():
            df = df[df["cluster"] == int(key)]
        elif key in SEGMENT_ALIASES:
            df = df[df["cluster"] == SEGMENT_ALIASES[key]]
        else:
            raise HTTPException(status_code=400, detail="Unknown segment filter.")

    df = df[df["calibrated_prob"] * 100 >= min_likelihood]
    if max_recency_days is not None:
        df = df[df["recency_days"] <= max_recency_days]
    if min_spend > 0:
        df = df[df["total_spend"] >= min_spend]
    if has_cart is True:
        df = df[df["cart_count"] > 0]
    elif has_cart is False:
        df = df[df["cart_count"] <= 0]

    if sort == "spend":
        df = df.sort_values("total_spend", ascending=False)
    elif sort == "recency":
        df = df.sort_values("recency_days", ascending=True)
    else:
        df = df.sort_values("calibrated_prob", ascending=False)

    total = len(df)
    page = df.iloc[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "customers": [_row_summary(r) for _, r in page.iterrows()],
        "filters": {
            "segment": segment or "all",
            "min_likelihood": min_likelihood,
            "max_recency_days": max_recency_days,
            "min_spend": min_spend,
            "has_cart": has_cart,
            "sort": sort,
        },
    }


@router.get("/customers/{user_id}")
def get_customer(user_id: int):
    df = get_warehouse()
    match = df[df["user_id"] == user_id]
    if match.empty:
        raise HTTPException(
            status_code=404,
            detail="Customer not found in the warehouse. Pick someone from the browse list.",
        )
    row = match.iloc[0]
    features = _row_to_features(row)
    scored = score_customer_vector(features, cluster_id=int(row["cluster"]))
    result = _productize_score(scored, features)
    result["customer_id"] = int(user_id)
    result["display_name"] = f"Shopper {str(user_id)[-4:]}"
    result["source"] = "warehouse"
    result["snapshot"] = _row_summary(row)
    return result


@router.post("/score")
def score_simple_profile(body: SimpleProfileRequest):
    features = _simple_to_features(body)
    scored = score_customer_vector(features)
    result = _productize_score(scored, features)
    result["customer_id"] = None
    result["display_name"] = "Custom profile"
    result["source"] = "profile_builder"
    return result
