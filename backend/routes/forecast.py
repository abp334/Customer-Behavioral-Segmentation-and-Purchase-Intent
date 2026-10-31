"""7-day traffic outlook — product-facing copy, honest series alignment."""

import json
import re

from fastapi import APIRouter

from backend.deps import APP_DATA_DIR

router = APIRouter(prefix="/api", tags=["forecast"])


def _clean_action(text: str) -> str:
    cleaned = re.sub(r"[\U00010000-\U0010ffff\u2600-\u27BF]", "", text or "").strip()
    return cleaned or "Standard operations"


@router.get("/forecast")
def get_forecast():
    with open(APP_DATA_DIR / "forecast_7d.json") as f:
        data = json.load(f)

    # Historical series uses total_events (~55–80k) while forecast predicted_traffic
    # aligns with active_users scale (~9–16k). Present them as separate series so
    # the chart does not invent a cliff.
    history = []
    for h in data["historical_series"][-14:]:
        history.append(
            {
                "date": h["full_date"],
                "day_name": h["day_name"],
                "active_users": h["active_users"],
                "total_events": h["total_events"],
                "total_purchases": h["total_purchases"],
            }
        )

    forecast = []
    for frow in data["forecast"]:
        forecast.append(
            {
                "date": frow["forecast_date"],
                "day_name": frow["day_of_week"],
                "predicted_visitors": frow["predicted_traffic"],
                "range_low": frow["pi_80_lower"],
                "range_high": frow["pi_80_upper"],
                "action": _clean_action(frow["recommended_action"]),
            }
        )

    # Best days = highest predicted visitor days
    ranked = sorted(forecast, key=lambda x: x["predicted_visitors"], reverse=True)
    best_days = [d["day_name"] for d in ranked[:2]]

    return {
        "headline": f"Prioritize campaigns on {best_days[0]} and {best_days[1]}.",
        "subtitle": (
            "Outlook for the next 7 days based on October 2019 traffic patterns. "
            "History shows daily active users; forecast predicts visitor volume on the same scale."
        ),
        "best_days": best_days,
        "history": history,
        "forecast": forecast,
        "methodology_hint": "Model order, MAPE, and stationarity tests live under Methodology.",
        "metrics_for_methodology": {
            "model_order": data.get("sarima_order"),
            "holdout_mape_pct": data.get("holdout_mape_pct"),
            "holdout_rmse": data.get("holdout_rmse"),
        },
    }
