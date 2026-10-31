"""Persona profiles and segment map points."""

import json
import re

from fastapi import APIRouter

from backend.deps import APP_DATA_DIR, get_warehouse

router = APIRouter(prefix="/api", tags=["personas"])

PERSONA_SHORT = {
    0: "Dormant",
    1: "Window Shoppers",
    2: "Champions",
    3: "Active Explorers",
}

PERSONA_ACTIONS = {
    0: "Low-cost email reactivation, or exclude from expensive channels.",
    1: "Product recommendation emails based on recently viewed items.",
    2: "VIP early access and dedicated support for high-value spenders.",
    3: "Time-boxed discount on recently viewed categories.",
}


def _strip_emoji(text: str) -> str:
    return re.sub(r"[\U00010000-\U0010ffff\u2600-\u27BF]", "", text).strip()


@router.get("/personas")
def list_personas():
    with open(APP_DATA_DIR / "persona_profiles.json") as f:
        profiles = json.load(f)

    personas = []
    for key in ["0", "1", "2", "3"]:
        p = profiles[key]
        cid = int(key)
        personas.append(
            {
                "cluster_id": cid,
                "name": PERSONA_SHORT[cid],
                "full_name": p["persona_name"],
                "customer_count": p["customer_count"],
                "pct_of_total": p["pct_of_total"],
                "conversion_rate_pct": p["conversion_rate_pct"],
                "mean_spend": p["mean_spend"],
                "median_recency_days": p["median_recency_days"],
                "median_sessions": p["median_sessions"],
                "recommended_action": PERSONA_ACTIONS[cid],
                # Science fields kept for methodology deep-link, not product cards
                "wilson_ci_lower_pct": p.get("wilson_ci_lower_pct"),
                "wilson_ci_upper_pct": p.get("wilson_ci_upper_pct"),
            }
        )
    return {"personas": personas, "population": 110956}


@router.get("/segments/map")
def segment_map(limit: int = 2000):
    df = get_warehouse()
    sample = df.sample(n=min(limit, len(df)), random_state=7)
    points = []
    for _, r in sample.iterrows():
        cid = int(r["cluster"])
        points.append(
            {
                "x": round(float(r["pca_x"]), 3),
                "y": round(float(r["pca_y"]), 3),
                "cluster_id": cid,
                "segment": PERSONA_SHORT[cid],
                "purchase_likelihood": round(float(r["calibrated_prob"]) * 100, 1),
            }
        )
    return {"points": points, "note": "2D view of behavioral similarity among a sample of customers."}
