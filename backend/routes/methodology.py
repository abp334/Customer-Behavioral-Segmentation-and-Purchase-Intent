"""Methodology payloads for graders — models, stats, Lagrange, pipeline."""

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.deps import APP_DATA_DIR, REPORTS_DIR
from src.campaign_optimizer import solve_lagrange_allocation

router = APIRouter(prefix="/api/methodology", tags=["methodology"])


def _read_csv(name: str):
    path = REPORTS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing report: {name}")
    df = pd.read_csv(path)
    return df.fillna("").to_dict(orient="records")


@router.get("/overview")
def methodology_overview():
    return {
        "title": "Methodology",
        "audience": "Instructors and technical reviewers",
        "sections": [
            {"id": "models", "title": "Intent model competition"},
            {"id": "calibration", "title": "Calibration and decision threshold"},
            {"id": "segments", "title": "Segmentation statistics"},
            {"id": "optimization", "title": "Lagrange budget study"},
            {"id": "forecast", "title": "Traffic forecast method"},
            {"id": "pipeline", "title": "Data pipeline"},
        ],
        "dataset": {
            "name": "REES46 multi-category eCommerce",
            "period": "October 2019",
            "events": 42448764,
            "feature_mart_customers": 110956,
            "app_warehouse_sample": 10000,
        },
    }


@router.get("/models")
def methodology_models():
    return {
        "comparison": _read_csv("intent_model_comparison.csv"),
        "coefficients": json.loads((APP_DATA_DIR / "model_coefficients.json").read_text()),
        "notes": [
            "Champion logistic regression with Platt scaling for calibrated purchase probabilities.",
            "Features: 17 behavioral numerics + 4 cluster indicators.",
            "Temporal split guards against leakage from the label window.",
        ],
    }


@router.get("/calibration")
def methodology_calibration():
    return {
        "threshold_analysis": _read_csv("optimal_threshold_analysis.csv"),
        "convergence": _read_csv("optimization_convergence.csv"),
        "notes": [
            "Cost-sensitive threshold τ* balances false positives vs missed buyers.",
            "Convergence CSV supports gradient-descent training diagnostics.",
        ],
    }


@router.get("/segments")
def methodology_segments():
    with open(APP_DATA_DIR / "persona_profiles.json") as f:
        profiles = json.load(f)
    return {
        "profiles": profiles,
        "clustering_metrics": _read_csv("clustering_metrics.csv"),
        "hypothesis_tests": _read_csv("hypothesis_test_results.csv"),
        "pairwise": _read_csv("segment_pairwise_comparisons.csv"),
        "notes": [
            "K-Means (k=4) on PCA-reduced standardized features.",
            "Wilson intervals and chi-square tests quantify segment conversion differences.",
        ],
    }


@router.get("/optimization")
def methodology_optimization(budget: float = 5000.0):
    allocations, lambda_star = solve_lagrange_allocation(budget)
    return {
        "budget": budget,
        "lambda_star": round(lambda_star, 4),
        "allocations": allocations,
        "report_table": _read_csv("lagrange_budget_allocation.csv"),
        "notes": [
            "This concave revenue study is separate from the product Campaign Plan.",
            "The product contacts top-N customers by predicted expected value under budget/CPA capacity.",
            "Lagrange equalizes marginal returns across segments: R_i(b_i) = α_i ln(1 + β_i b_i).",
        ],
    }


@router.get("/forecast")
def methodology_forecast():
    with open(APP_DATA_DIR / "forecast_7d.json") as f:
        forecast = json.load(f)
    return {
        "model_order": forecast.get("sarima_order"),
        "holdout_mape_pct": forecast.get("holdout_mape_pct"),
        "holdout_rmse": forecast.get("holdout_rmse"),
        "holdout_mae": forecast.get("holdout_mae"),
        "evaluation": _read_csv("timeseries_forecast_evaluation.csv"),
        "stationarity": _read_csv("timeseries_stationarity_tests.csv"),
        "notes": [
            "Seasonal SARIMA captures weekly shopping cycles.",
            "Product Traffic page uses active-user scale for history vs visitor forecast.",
        ],
    }


@router.get("/pipeline")
def methodology_pipeline():
    return {
        "medallion": [
            {
                "layer": "Bronze",
                "role": "Raw clickstream ingest and quality checks",
                "artifact": "data_quality_report_raw.csv",
            },
            {
                "layer": "Silver",
                "role": "Cleaned events and star-schema dimensions",
                "artifact": "phase2_star_schema_summary.csv",
            },
            {
                "layer": "Gold",
                "role": "User feature mart, scores, and campaign inputs",
                "artifact": "scored_customers_warehouse.parquet",
            },
        ],
        "quality": _read_csv("data_quality_report_raw.csv"),
        "star_schema": _read_csv("phase2_star_schema_summary.csv"),
        "notes": [
            "Offline notebooks (01–07) reproduce each phase.",
            "The production app serves precomputed Gold artifacts plus live model scoring.",
        ],
    }
