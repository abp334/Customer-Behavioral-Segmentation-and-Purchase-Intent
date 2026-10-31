"""
src/scoring_engine.py
=====================
Core scoring, probability calibration, and explainability engine.
Computes both business recommendations (Result Tab) and exact applied mathematics
(The Science Tab) with worked numbers for any single customer or batch upload.
"""

import os, json, warnings
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
APP_DATA_DIR = PROJECT_ROOT / "data" / "app_data"

# Load persistent models and data contracts
_CALIBRATOR = joblib.load(MODELS_DIR / "calibrated_intent_model.joblib")
_CHAMPION_MODEL = joblib.load(MODELS_DIR / "intent_model_champion.joblib")
_SCALER = joblib.load(MODELS_DIR / "scaler.joblib")
_SCALER_17NUM = joblib.load(MODELS_DIR / "scaler_17num.joblib")
_PCA_5 = joblib.load(MODELS_DIR / "pca_5comp.joblib")
_KMEANS = joblib.load(MODELS_DIR / "kmeans_4clusters.joblib")

with open(APP_DATA_DIR / "persona_profiles.json") as f:
    _PERSONA_PROFILES = json.load(f)

with open(APP_DATA_DIR / "model_coefficients.json") as f:
    _COEFFICIENTS = json.load(f)

NUMERIC_FEATURES = [
    "recency_days", "tenure_days", "active_days", "total_sessions",
    "total_events", "view_count", "cart_count", "purchase_count",
    "total_spend", "avg_purchase_price", "max_price_interacted",
    "n_unique_products", "n_unique_cat1", "n_unique_brands",
    "weekend_event_ratio", "cart_to_view_ratio", "events_per_session"
]

SKEWED_FEATURES = [
    "total_spend", "total_events", "view_count", "cart_count", "purchase_count",
    "total_sessions", "avg_purchase_price", "max_price_interacted", "n_unique_products"
]

PERSONA_NAMES = {
    0: "Dormant Bouncers (Cold / Inactive)",
    1: "Recent Window Shoppers (Warm Bouncers)",
    2: "Champions (High-Value Spenders)",
    3: "Active Explorers (High Engagement)"
}


def score_customer_vector(feature_dict: Dict[str, Any], cluster_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Score a single customer feature dictionary.
    Returns full Result payload and exact 'The Science' mathematical decomposition.
    """
    # 1. Clean and log-transform skewed features
    raw_vals = {}
    transformed_vals = {}
    for col in NUMERIC_FEATURES:
        val = float(feature_dict.get(col, 0.0) or 0.0)
        raw_vals[col] = val
        if col in SKEWED_FEATURES:
            transformed_vals[col] = float(np.log1p(max(0.0, val)))
        else:
            transformed_vals[col] = val

    # 2. Standardize 17 numeric features for PCA and Centroid Distance
    x_num_arr = np.array([[transformed_vals[c] for c in NUMERIC_FEATURES]])
    z_num = _SCALER_17NUM.transform(x_num_arr)[0]
    
    # 3. Determine cluster if not provided (via PCA projection + KMeans)
    pca_coords = _PCA_5.transform(z_num.reshape(1, -1))[0]
    if cluster_id is None:
        cluster_id = int(_KMEANS.predict(pca_coords.reshape(1, -1))[0])
    
    # Calculate Euclidean distance to each persona centroid in 17D space
    distances = {}
    for k in range(4):
        c_vec = np.array(_PERSONA_PROFILES[str(k)]["centroid_17d"])
        d = float(np.linalg.norm(z_num - c_vec))
        distances[PERSONA_NAMES[k]] = round(d, 3)

    # 4. Prepare 21-feature design matrix for Supervised Champion Model
    dummies = {"cluster_0": 0.0, "cluster_1": 0.0, "cluster_2": 0.0, "cluster_3": 0.0}
    dummies[f"cluster_{cluster_id}"] = 1.0

    full_row = {}
    for col in NUMERIC_FEATURES:
        full_row[col] = transformed_vals[col]
    for c in ["cluster_0", "cluster_1", "cluster_2", "cluster_3"]:
        full_row[c] = dummies[c]

    feat_names = _COEFFICIENTS["feature_names"]
    x_full_arr = np.array([[full_row[f] for f in feat_names]])
    x_scaled = _SCALER.transform(x_full_arr)[0]

    # 5. Compute probabilities (Raw & Calibrated)
    raw_prob = float(_CHAMPION_MODEL.predict_proba(x_scaled.reshape(1, -1))[0, 1])
    cal_prob = float(_CALIBRATOR.predict_proba(x_scaled.reshape(1, -1))[0, 1])

    # 6. Exact Log-Odds Decomposition for The Science Tab
    intercept = float(_COEFFICIENTS["intercept"])
    coefs = _COEFFICIENTS["coefficients"]
    log_odds_contributions = []
    
    for i, f_name in enumerate(feat_names):
        z_i = float(x_scaled[i])
        w_i = float(coefs[i])
        contrib = w_i * z_i
        log_odds_contributions.append({
            "feature": f_name,
            "raw_value": round(raw_vals.get(f_name, dummies.get(f_name, 0.0)), 2),
            "z_score": round(z_i, 3),
            "coefficient": round(w_i, 4),
            "contribution": round(contrib, 4),
            "odds_ratio": round(float(_COEFFICIENTS["odds_ratios"][i]), 3)
        })

    # Sort contributions by absolute impact
    sorted_contribs = sorted(log_odds_contributions, key=lambda x: abs(x["contribution"]), reverse=True)
    calculated_logit = intercept + sum(item["contribution"] for item in log_odds_contributions)
    sigmoid_calc = 1.0 / (1.0 + np.exp(-calculated_logit))

    # 7. Generate Plain-Language Reasons (Result Tab)
    reasons = []
    recency = raw_vals.get("recency_days", 99)
    spend = raw_vals.get("total_spend", 0.0)
    cart = raw_vals.get("cart_count", 0)
    events = raw_vals.get("total_events", 0)
    unique_prods = raw_vals.get("n_unique_products", 0)

    if spend > 150:
        reasons.append(f"Significant customer investment with ${spend:,.2f} historical spend.")
    elif cart > 0:
        reasons.append(f"High immediate purchase intent with {int(cart)} items added to cart.")
    if recency <= 3:
        reasons.append(f"Recent engagement window (visited within last {int(recency)} days).")
    elif recency >= 10:
        reasons.append(f"Dormant risk: inactive for {int(recency)} days.")
    if unique_prods >= 5:
        reasons.append(f"Broad catalog exploration across {int(unique_prods)} distinct products.")
    if not reasons:
        reasons.append("Single-session evaluation with limited prior interaction.")

    # Tailored business action
    if cluster_id == 2:
        action = "VIP Loyalty Concierge: Provide exclusive early access to premium arrivals and dedicated customer support."
    elif cluster_id == 3:
        action = "Intent Trigger: Deliver an automated 10% discount on recently viewed categories with 48h expiration."
    elif cluster_id == 1:
        action = "Remarketing Nudge: Trigger relevant product recommendation emails based on top viewed items."
    else:
        action = "Re-engagement Campaign: Low-cost email reactivation or exclude from high-cost direct promotion."

    # Intent Decile compared to population
    if cal_prob >= 0.20:
        decile = "Top 10% (High Intent)"
    elif cal_prob >= 0.08:
        decile = "Top 20% (Elevated Intent)"
    elif cal_prob >= 0.0387:
        decile = "Above Average Intent"
    else:
        decile = "Lower 50% (Low Intent)"

    return {
        "persona_name": PERSONA_NAMES[cluster_id],
        "cluster_id": cluster_id,
        "calibrated_prob": round(cal_prob, 4),
        "raw_prob": round(raw_prob, 4),
        "intent_decile": decile,
        "reasons": reasons[:3],
        "recommended_action": action,
        "the_science": {
            "intercept_beta_0": round(intercept, 4),
            "calculated_logit": round(calculated_logit, 4),
            "raw_sigmoid_prob": round(sigmoid_calc, 4),
            "calibrated_platt_prob": round(cal_prob, 4),
            "distances_to_centroids": distances,
            "nearest_persona": min(distances, key=distances.get),
            "pca_coords_2d": [round(float(pca_coords[0]), 3), round(float(pca_coords[1]), 3)],
            "feature_contributions": sorted_contribs
        }
    }


def score_batch_dataframe(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Score an uploaded DataFrame containing the required customer feature columns.
    Returns an enriched DataFrame ready for review and CSV export.
    """
    if df_input is None or len(df_input) == 0:
        raise ValueError("Uploaded file contains zero customer records. Please provide a CSV with at least one row.")

    df = df_input.copy()
    # Check and impute missing required columns
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    # 1. Transform skewed features
    df_trans = df[NUMERIC_FEATURES].copy()
    for col in SKEWED_FEATURES:
        df_trans[col] = np.log1p(np.maximum(0.0, df_trans[col]))

    # 2. Project onto PCA and predict Cluster
    X_num_scaled = _SCALER_17NUM.transform(df_trans.values)
    Z_pca = _PCA_5.transform(X_num_scaled)
    clusters = _KMEANS.predict(Z_pca)
    
    # 3. Create dummy variables
    dummies = pd.get_dummies(clusters, prefix="cluster", drop_first=False).astype(float)
    for c in ["cluster_0", "cluster_1", "cluster_2", "cluster_3"]:
        if c not in dummies.columns:
            dummies[c] = 0.0
    dummies = dummies[["cluster_0", "cluster_1", "cluster_2", "cluster_3"]]

    # 4. Supervised champion model scoring
    feat_names = _COEFFICIENTS["feature_names"]
    X_full = pd.concat([df_trans.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)[feat_names]
    X_scaled = _SCALER.transform(X_full.values)

    cal_probs = _CALIBRATOR.predict_proba(X_scaled)[:, 1]
    raw_probs = _CHAMPION_MODEL.predict_proba(X_scaled)[:, 1]

    # Enrich input DataFrame
    df["predicted_cluster"] = clusters
    df["persona_name"] = [PERSONA_NAMES[k] for k in clusters]
    df["calibrated_purchase_prob"] = cal_probs.round(4)
    df["raw_purchase_prob"] = raw_probs.round(4)
    df["pca_component_1"] = Z_pca[:, 0].round(3)
    df["pca_component_2"] = Z_pca[:, 1].round(3)

    # Priority Rank: higher calibrated probability = rank 1
    df["intent_rank"] = df["calibrated_purchase_prob"].rank(ascending=False, method="min").astype(int)
    
    # Sort by intent rank
    df.sort_values("intent_rank", inplace=True)
    return df
