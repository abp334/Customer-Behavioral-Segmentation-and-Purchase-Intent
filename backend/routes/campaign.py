"""Campaign planning: EV targeting + PDF export. Lagrange study only via methodology."""

import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from backend.schemas import CampaignRequest
from src.campaign_optimizer import run_campaign_optimization
from src.pdf_generator import generate_campaign_pdf

router = APIRouter(prefix="/api", tags=["campaign"])

VALID_GOALS = {
    "maximize_profit",
    "win_back_lapsed",
    "convert_cart_abandoners",
    "reward_loyal_champions",
}

GOAL_LABELS = {
    "maximize_profit": "Maximize overall commercial return",
    "win_back_lapsed": "Win back lapsed customers",
    "convert_cart_abandoners": "Convert cart abandoners",
    "reward_loyal_champions": "Reward loyal champions",
}


@router.post("/campaign")
def plan_campaign(body: CampaignRequest):
    if body.goal not in VALID_GOALS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid goal. Choose one of: {', '.join(sorted(VALID_GOALS))}",
        )
    result = run_campaign_optimization(
        goal=body.goal,
        total_budget=body.total_budget,
        cost_per_contact=body.cost_per_contact,
    )
    # Product payload — no Lagrange on the face
    full = result["full_targeted_df"]
    targets = full.head(50).to_dict(orient="records")
    return {
        "goal": body.goal,
        "goal_title": result["goal_title"],
        "story": (
            "Under your budget and cost per contact, we contact the top customers "
            "ranked by predicted expected value (purchase likelihood × typical segment spend)."
        ),
        "total_budget": result["total_budget"],
        "cost_per_contact": result["cost_per_contact"],
        "eligible_audience_size": result["eligible_audience_size"],
        "target_count": result["target_count"],
        "actual_spent": result["actual_spent"],
        "predicted_buyers": result["expected_buyers"],
        "predicted_conversion_pct": result["conversion_rate_targeted_pct"],
        "predicted_revenue": result["expected_revenue"],
        "predicted_net": result["expected_net_profit"],
        "predicted_roi_pct": result["roi_pct"],
        "disclaimer": (
            "Figures are predicted expected values from historical behavior — "
            "not measured causal lift from contacting these customers."
        ),
        "targets": targets,
    }


@router.post("/campaign/pdf")
def campaign_pdf(body: CampaignRequest):
    if body.goal not in VALID_GOALS:
        raise HTTPException(status_code=400, detail="Invalid goal.")
    result = run_campaign_optimization(
        goal=body.goal,
        total_budget=body.total_budget,
        cost_per_contact=body.cost_per_contact,
    )
    pdf_bytes = generate_campaign_pdf(result)
    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=nexus_campaign_brief.pdf"},
    )


@router.post("/campaign/csv")
def campaign_csv(body: CampaignRequest):
    if body.goal not in VALID_GOALS:
        raise HTTPException(status_code=400, detail="Invalid goal.")
    result = run_campaign_optimization(
        goal=body.goal,
        total_budget=body.total_budget,
        cost_per_contact=body.cost_per_contact,
    )
    csv_buf = io.StringIO()
    result["full_targeted_df"].to_csv(csv_buf, index=False)
    return StreamingResponse(
        io.BytesIO(csv_buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=nexus_campaign_targets.csv"},
    )
