"""
app/main.py — NEXUS | Customer Behavioral Intelligence Platform
================================================================
Enterprise-grade Data Science Capstone Application uniting:
  F1: Customer Intelligence & Real-Time Platt-Calibrated Scorer
  F2: Marketing Campaign ROI & Lagrange Multiplier Budget Allocator
  F3: Batch Customer CSV Scoring Engine
  F4: Unsupervised Behavioral Segmentation & Interactive 2D PCA Map
  F5: 7-Day Traffic Outlook & Seasonal SARIMA Forecasting Engine
  F6: Model Audit, Lakehouse Architecture & Optimization Replay

Grounded in 42,448,764 REES46 eCommerce clickstream events.
Crystal-clear Dark Slate styling matching SkillCompass India.
"""

import sys
import json
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.scoring_engine import (
    score_customer_vector,
    score_batch_dataframe,
    NUMERIC_FEATURES,
    PERSONA_NAMES
)
from src.campaign_optimizer import run_campaign_optimization
from src.pdf_generator import generate_campaign_pdf

APP_DATA_DIR = PROJECT_ROOT / "data" / "app_data"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ─────────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & GLOBAL DARK THEME
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS | Customer Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Dark Theme Global Canvas */
.stApp {
    background-color: #0F172A !important;
    color: #F8FAFC !important;
}

/* Headers with Gradient Accent */
.main-header {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818CF8 0%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}
.sub-header {
    font-size: 0.98rem;
    color: #94A3B8 !important;
    margin-bottom: 1.2rem;
    font-weight: 400;
}

/* Metrics High Contrast */
[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    font-size: 1.65rem !important;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* Pill Badges */
.pill-champion {
    display: inline-block;
    background-color: #064E3B !important;
    color: #6EE7B7 !important;
    border: 1px solid #059669 !important;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.82rem;
    font-weight: 700;
}
.pill-explorer {
    display: inline-block;
    background-color: #1E1B4B !important;
    color: #C7D2FE !important;
    border: 1px solid #4F46E5 !important;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.82rem;
    font-weight: 700;
}
.pill-shopper {
    display: inline-block;
    background-color: #451A03 !important;
    color: #FDE68A !important;
    border: 1px solid #B45309 !important;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.82rem;
    font-weight: 700;
}
.pill-bouncer {
    display: inline-block;
    background-color: #450A0A !important;
    color: #FCA5A5 !important;
    border: 1px solid #DC2626 !important;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.82rem;
    font-weight: 700;
}

/* Custom Cards */
.custom-card {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
}
.custom-card-highlight {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-left: 5px solid #6366F1 !important;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 14px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
}
.takeaway-card {
    background-color: #064E3B !important;
    color: #ECFDF5 !important;
    border: 1px solid #059669 !important;
    border-left: 5px solid #10B981 !important;
    padding: 14px 16px;
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 0.92rem;
    line-height: 1.5;
}

/* Streamlit Tabs Styling */
button[data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
}
button[aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom-color: #38BDF8 !important;
}

/* Plotly Canvas Container */
.stPlotlyChart {
    min-height: 380px !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# 2. CACHED DATA RESOURCES
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_app_resources():
    warehouse = pd.read_parquet(APP_DATA_DIR / "scored_customers_warehouse.parquet")
    with open(APP_DATA_DIR / "persona_profiles.json") as f:
        persona_profiles = json.load(f)
    with open(APP_DATA_DIR / "forecast_7d.json") as f:
        forecast_payload = json.load(f)
    with open(APP_DATA_DIR / "model_coefficients.json") as f:
        model_coefs = json.load(f)
    with open(APP_DATA_DIR / "sample_upload_template.csv", "rb") as f:
        template_bytes = f.read()

    demo_batch_df = pd.read_csv(APP_DATA_DIR / "demo_batch_customers.csv")
    model_benchmarks = pd.read_csv(REPORTS_DIR / "intent_model_comparison.csv")
    threshold_analysis = pd.read_csv(REPORTS_DIR / "optimal_threshold_analysis.csv")
    gd_convergence = pd.read_csv(REPORTS_DIR / "optimization_convergence.csv")
    forecast_eval = pd.read_csv(REPORTS_DIR / "timeseries_forecast_evaluation.csv")
    stationarity_df = pd.read_csv(REPORTS_DIR / "timeseries_stationarity_tests.csv")

    return (
        warehouse, persona_profiles, forecast_payload, model_coefs,
        template_bytes, demo_batch_df, model_benchmarks, threshold_analysis,
        gd_convergence, forecast_eval, stationarity_df
    )

(
    df_warehouse, persona_profiles, forecast_payload, model_coefs,
    sample_template_bytes, df_demo_batch, df_model_benchmarks,
    df_threshold_analysis, df_gd_convergence, df_forecast_eval, df_stationarity
) = load_app_resources()

# ─────────────────────────────────────────────────────────────────
# 3. TOP BRAND HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("<div class='main-header'>🧠 NEXUS | Customer Behavioral Intelligence Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Enterprise Behavioral Segmentation, Purchase Intent Prediction & Constrained Marketing Optimization • Grounded in 42.4M REES46 Events</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# 4. SIDEBAR: INTERACTIVE SHOPPER BUILDER & CONTROLS
# ─────────────────────────────────────────────────────────────────
st.sidebar.header("🛠️ Interactive Shopper Builder")
st.sidebar.caption("Construct any shopper behavioral profile live or choose a preset commercial archetype.")

# Scenario Presets
preset_choice = st.sidebar.selectbox(
    "Load Scenario Preset:",
    [
        "Custom Shopper (Adjust Below)",
        "👑 VIP Champion ($650 Spend, 2d Recency)",
        "🛒 High-Intent Cart Abandoner (2 Cart Items, 1d)",
        "🛍️ Window Shopper (12 Views, 0 Cart)",
        "❄️ Lapsed Inactive (16d Dormant)"
    ]
)

if "VIP Champion" in preset_choice:
    def_rec, def_spend, def_cart, def_events, def_prods, def_sess = 2, 650.0, 2, 35, 12, 5
elif "Cart Abandoner" in preset_choice:
    def_rec, def_spend, def_cart, def_events, def_prods, def_sess = 1, 0.0, 3, 22, 8, 3
elif "Window Shopper" in preset_choice:
    def_rec, def_spend, def_cart, def_events, def_prods, def_sess = 3, 0.0, 0, 15, 6, 1
elif "Lapsed Inactive" in preset_choice:
    def_rec, def_spend, def_cart, def_events, def_prods, def_sess = 16, 25.0, 0, 2, 1, 1
else:
    def_rec, def_spend, def_cart, def_events, def_prods, def_sess = 2, 320.0, 1, 20, 5, 3

# Tactile Sliders for Live Profile Building
recency_val = st.sidebar.slider("📅 Days Inactive (Recency):", min_value=0, max_value=20, value=def_rec, help="0 = Active today, 20 = 20 days dormant")
spend_val = st.sidebar.slider("💰 Historical Lifetime Spend ($):", min_value=0.0, max_value=1200.0, value=float(def_spend), step=25.0)
cart_val = st.sidebar.slider("🛒 Active Items in Cart:", min_value=0, max_value=6, value=def_cart)
events_val = st.sidebar.slider("👀 Browsing Activity Volume (Events):", min_value=1, max_value=50, value=def_events)

# Optional Expander for Database ID Lookup (For Examiners)
with st.sidebar.expander("🔍 Or Search Database by 9-Digit ID (10k Users)"):
    manual_id = st.number_input("Customer ID:", min_value=100000000, max_value=999999999, value=556431564, step=1)
    use_manual_id = st.checkbox("Load exact record from database", value=False)

if use_manual_id:
    cust_match = df_warehouse[df_warehouse["user_id"] == manual_id]
    if len(cust_match) > 0:
        cust_row = cust_match.iloc[0].to_dict()
    else:
        cust_row = df_warehouse.iloc[0].to_dict()
        st.sidebar.warning(f"⚠️ ID {manual_id} not found in database. Showing default profile.")
    display_user_label = f"ID: {cust_row['user_id']}"
else:
    # Build feature vector dynamically from the tactile controls
    p_count = 1 if spend_val > 50 else (2 if spend_val > 300 else 0)
    avg_p_price = spend_val / max(1, p_count) if spend_val > 0 else 0.0
    v_count = max(0, events_val - cart_val - p_count)
    cust_row = {
        "user_id": "Custom Profile",
        "recency_days": recency_val,
        "tenure_days": max(recency_val + 2, 14),
        "active_days": max(1, def_sess),
        "total_sessions": def_sess,
        "total_events": events_val,
        "view_count": v_count,
        "cart_count": cart_val,
        "purchase_count": p_count,
        "total_spend": spend_val,
        "avg_purchase_price": avg_p_price,
        "max_price_interacted": max(25.0, avg_p_price * 1.5),
        "n_unique_products": def_prods,
        "n_unique_cat1": min(4, max(1, def_prods // 3)),
        "n_unique_brands": min(5, max(1, def_prods // 2)),
        "weekend_event_ratio": 0.28,
        "cart_to_view_ratio": cart_val / max(1, v_count),
        "events_per_session": events_val / max(1, def_sess)
    }
    display_user_label = f"{preset_choice.split('(')[0].strip()}"

# Real-time scoring for active customer
feat_dict = {col: cust_row[col] for col in NUMERIC_FEATURES if col in cust_row}
single_score = score_customer_vector(feat_dict, cluster_id=int(cust_row.get("cluster", 0)) if use_manual_id else None)

# Active profile sidebar card
st.sidebar.markdown(f"""
<div class='custom-card'>
    <div style='font-size:0.75rem; color:#94A3B8; font-weight:700;'>LIVE EVALUATION STATUS</div>
    <div style='font-size:1.1rem; font-weight:800; color:#F8FAFC;'>{display_user_label}</div>
    <div style='margin-top:6px;'>
        <span class='{"pill-champion" if single_score["cluster_id"]==2 else ("pill-explorer" if single_score["cluster_id"]==3 else ("pill-shopper" if single_score["cluster_id"]==1 else "pill-bouncer"))}'>
            {single_score["persona_name"]}
        </span>
    </div>
    <div style='margin-top:8px; font-size:0.83rem; color:#CBD5E1; line-height:1.6;'>
        <b>Calibrated Intent:</b> <span style='color:#38BDF8; font-weight:700;'>{single_score["calibrated_prob"]*100:.1f}%</span><br/>
        <b>Historical Spend:</b> ${cust_row["total_spend"]:,.2f}<br/>
        <b>Cart Items:</b> {int(cust_row["cart_count"])} items<br/>
        <b>Recency:</b> {int(cust_row["recency_days"])} days ago
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# 5. MAIN APPLICATION TABS (CLEAN, NATIVE, FOCUSED)
# ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎯 F1: Customer Diagnostic",
    "💼 F2: Campaign ROI Optimizer",
    "📁 F3: Batch CSV Scorer",
    "📊 F4: Persona Explorer & 2D Map",
    "📈 F5: 7-Day Traffic Outlook",
    "🔬 F6: Technical Audit & Lakehouse"
])

# =================================================================
# TAB 1: F1 - CUSTOMER DIAGNOSTIC & NEXT-BEST-ACTION
# =================================================================
with tabs[0]:
    st.subheader(f"Customer Diagnostic & Next-Best-Action ({display_user_label})")
    
    st.markdown(f"""
    <div class='takeaway-card'>
        <b>Executive Takeaway:</b> Shopper profile <b>{display_user_label}</b> is classified into 
        <b>{single_score['persona_name']}</b> with a <b>{single_score['calibrated_prob']*100:.1f}% calibrated purchase likelihood</b> ({single_score['intent_decile']}).<br/>
        <b>Prescribed Strategic Action:</b> {single_score['recommended_action']}
    </div>
    """, unsafe_allow_html=True)

    # 4 High-level Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Calibrated Purchase Intent", f"{single_score['calibrated_prob']*100:.1f}%", f"{single_score['calibrated_prob']*100 - 3.87:+.1f}% vs Store Base")
    with m2:
        st.metric("Assigned Persona", single_score['persona_name'].split('(')[0].strip())
    with m3:
        st.metric("Historical Spend", f"${cust_row['total_spend']:,.2f}")
    with m4:
        st.metric("Inactivity Recency", f"{int(cust_row['recency_days'])} days ago")

    st.markdown("---")

    col_footprint, col_gauge = st.columns([1.2, 1])

    with col_footprint:
        st.markdown("##### 📋 Behavioral Footprint (20-Day Window)")
        
        # Native Streamlit metrics grid (clean, reliable, responsive)
        g1, g2, g3 = st.columns(3)
        with g1:
            st.metric("Total Sessions", int(cust_row["total_sessions"]))
            st.metric("Unique Products", int(cust_row["n_unique_products"]))
        with g2:
            st.metric("Total Events", int(cust_row["total_events"]))
            st.metric("Cart Items", int(cust_row["cart_count"]))
        with g3:
            st.metric("Avg Item Price", f"${cust_row['avg_purchase_price']:,.2f}")
            st.metric("Weekend Activity", f"{cust_row.get('weekend_event_ratio', 0.2)*100:.0f}%")

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("##### 🔍 Key Intent Drivers Identified by Model")
        for reason in single_score["reasons"]:
            st.markdown(f"- <span style='color:#38BDF8; font-weight:600;'>{reason}</span>", unsafe_allow_html=True)

    with col_gauge:
        st.markdown("##### ⚡ Calibrated Purchase Probability Gauge")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=single_score["calibrated_prob"] * 100,
            number={"suffix": "%", "font": {"size": 38, "color": "#F8FAFC"}},
            title={"text": "Calibrated 10-Day Purchase Intent", "font": {"size": 13, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, 50], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": "#6366F1", "thickness": 0.3},
                "bgcolor": "#1E293B",
                "steps": [
                    {"range": [0, 3.87], "color": "#334155"},
                    {"range": [3.87, 10], "color": "#1E1B4B"},
                    {"range": [10, 25], "color": "#312E81"},
                    {"range": [25, 50], "color": "#4338CA"}
                ],
                "threshold": {
                    "line": {"color": "#10B981", "width": 4},
                    "thickness": 0.75,
                    "value": 3.87
                }
            }
        ))
        fig_gauge.update_layout(
            height=280,
            autosize=True,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#1E293B",
            plot_bgcolor="#1E293B",
            font=dict(color="#F8FAFC")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption("Green tick indicates 3.87% baseline store conversion rate.")

    # Expandable Science Drawer (Keeps main page clean!)
    with st.expander("🔬 How Was This Calculated? (Applied Mathematics & Worked Numbers)"):
        st.markdown("##### 1. 17D Euclidean Distance to Cluster Centroids")
        st.latex(r"d(x, c_k) = \|z - \mu_k\|_2 = \sqrt{\sum_{j=1}^{17} \left(\frac{x_j - \mu_j}{\sigma_j} - c_{k,j}\right)^2}")
        
        dist_records = [
            {"Persona Centroid": k, "Euclidean Distance": v, "Status": "🌟 NEAREST ASSIGNED" if k == single_score["the_science"]["nearest_persona"] else "Alternative"}
            for k, v in single_score["the_science"]["distances_to_centroids"].items()
        ]
        st.dataframe(pd.DataFrame(dist_records), use_container_width=True, hide_index=True)

        st.markdown("##### 2. Logistic Sum & Platt Sigmoid Calibration")
        st.latex(r"\text{logit}(x) = \beta_0 + \sum_{j=1}^{21} \beta_j z_j \implies P_{\text{calibrated}} = \frac{1}{1 + \exp(A \cdot \text{logit} + B)}")
        
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Raw Intercept (β₀)", f"{single_score['the_science']['intercept_beta_0']:.4f}")
        with s2:
            st.metric("Computed Logit", f"{single_score['the_science']['calculated_logit']:.4f}")
        with s3:
            st.metric("Raw Model Prob", f"{single_score['the_science']['raw_sigmoid_prob']*100:.1f}%")
        with s4:
            st.metric("Platt Calibrated Prob", f"{single_score['the_science']['calibrated_platt_prob']*100:.2f}%")

        st.markdown("##### 3. Top Feature Log-Odds Contributions")
        contrib_df = pd.DataFrame(single_score["the_science"]["feature_contributions"]).head(8)
        st.dataframe(
            contrib_df[["feature", "raw_value", "z_score", "coefficient", "contribution", "odds_ratio"]].rename(columns={
                "feature": "Feature", "raw_value": "Raw Value", "z_score": "Standardized (z)",
                "coefficient": "Weight (β)", "contribution": "Log-Odds Contribution", "odds_ratio": "Odds Ratio"
            }),
            use_container_width=True,
            hide_index=True
        )

# =================================================================
# TAB 2: F2 - MARKETING CAMPAIGN ROI & LAGRANGE ALLOCATOR
# =================================================================
with tabs[1]:
    st.subheader("💼 Marketing Campaign Builder & Lagrange Multiplier Allocator")
    st.caption("Analytically maximizes commercial return under budget constraints using Karush-Kuhn-Tucker (KKT) optimality conditions.")

    c_g, c_b, c_c = st.columns([1.5, 1, 1])
    with c_g:
        camp_goal = st.selectbox(
            "Select Campaign Objective:",
            options=[
                ("win_back_lapsed", "Win-Back Lapsed Customers (Recency >= 7d)"),
                ("convert_cart_abandoners", "Convert Cart Abandoners (Cart Items >= 1)"),
                ("reward_loyal_champions", "Reward Loyal Champions (High Historical Spend)"),
                ("maximize_profit", "Maximize Overall Commercial Return (Unrestricted)")
            ],
            format_func=lambda x: x[1]
        )[0]

    with c_b:
        budget_input = st.number_input("Total Marketing Budget ($):", min_value=10.0, max_value=100000.0, value=5000.0, step=500.0)
    with c_c:
        cost_input = st.number_input("Cost per Targeted Contact ($):", min_value=0.10, max_value=50.0, value=1.00, step=0.25)

    if budget_input < cost_input:
        st.warning(f"⚠️ Total budget (${budget_input:,.2f}) is less than the cost per contact (${cost_input:,.2f}). Please increase the budget or reduce contact cost to fund at least 1 customer.")

    camp_res = run_campaign_optimization(camp_goal, total_budget=max(cost_input, budget_input), cost_per_contact=cost_input)

    st.markdown(f"""
    <div class='takeaway-card'>
        <b>Executive Takeaway:</b> Allocating <b>${camp_res['total_budget']:,.2f}</b> at ${camp_res['cost_per_contact']:.2f}/user funds 
        <b>{camp_res['target_count']:,} targeted customers</b>. Projected to yield <b>{camp_res['expected_buyers']} buyers</b> and 
        <b>${camp_res['expected_revenue']:,.2f} GMV</b> (Net Profit: <b>+${camp_res['expected_net_profit']:,.2f}</b> | Commercial ROI: <b>+{camp_res['roi_pct']}%</b>).
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Targeted Audience", f"{camp_res['target_count']:,}", f"of {camp_res['eligible_audience_size']:,} eligible")
    with k2:
        st.metric("Expected Buyers", f"{camp_res['expected_buyers']}", f"{camp_res['conversion_rate_targeted_pct']}% conversion")
    with k3:
        st.metric("Projected GMV", f"${camp_res['expected_revenue']:,.2f}", f"Net: +${camp_res['expected_net_profit']:,.2f}")
    with k4:
        st.metric("Commercial ROI", f"+{camp_res['roi_pct']}%", f"λ* = {camp_res['the_science']['lambda_star']:.2f}")

    st.markdown("---")

    col_targets, col_alloc = st.columns([1.3, 1])

    with col_targets:
        st.markdown("##### 🏆 Priority Scored Target List (Top Scored Customers)")
        st.dataframe(camp_res["full_targeted_df"].head(12), use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        with d1:
            csv_bytes = camp_res["full_targeted_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download Target List (CSV)",
                data=csv_bytes,
                file_name=f"targets_{camp_goal}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d2:
            pdf_bytes = generate_campaign_pdf(camp_res)
            st.download_button(
                "📄 Download Executive Brief (PDF)",
                data=pdf_bytes,
                file_name=f"executive_brief_{camp_goal}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

    with col_alloc:
        st.markdown("##### ⚖️ Lagrange Multiplier Optimal Allocation")
        alloc_list = list(camp_res["lagrange_allocations"].values())
        
        fig_pie = px.pie(
            alloc_list,
            names="persona_name",
            values="budget_allocated",
            hole=0.45,
            color="persona_name",
            color_discrete_map={
                "Champions": "#10B981",
                "Active Explorers": "#6366F1",
                "Recent Window Shoppers": "#F59E0B",
                "Dormant Bouncers": "#64748B"
            },
            template="plotly_dark"
        )
        fig_pie.update_layout(
            height=280,
            autosize=True,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#1E293B",
            plot_bgcolor="#1E293B",
            font=dict(color="#F8FAFC")
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        alloc_df = pd.DataFrame(alloc_list)[["persona_name", "budget_allocated", "budget_share_pct", "marginal_return_per_dollar"]]
        st.dataframe(
            alloc_df.rename(columns={
                "persona_name": "Segment", "budget_allocated": "Budget ($)",
                "budget_share_pct": "Share (%)", "marginal_return_per_dollar": "Marginal Return"
            }),
            use_container_width=True,
            hide_index=True
        )

    with st.expander("🔬 The Science: Analytical Lagrange Multipliers & Diminishing Returns"):
        st.markdown("##### Constrained Optimization Formulation (Lagrangian)")
        st.latex(r"\max_{b_1, \dots, b_k} \sum_{i=1}^k \alpha_i \ln(1 + \beta_i b_i) \quad \text{s.t.} \quad \sum_{i=1}^k b_i = B, \quad b_i \ge 0")
        st.latex(r"\mathcal{L}(b, \lambda) = \sum_{i=1}^k \alpha_i \ln(1 + \beta_i b_i) - \lambda \left(\sum_{i=1}^k b_i - B\right)")
        st.latex(r"\frac{\partial \mathcal{L}}{\partial b_i} = \frac{\alpha_i \beta_i}{1 + \beta_i b_i} - \lambda^* = 0 \implies b_i^*(\lambda^*) = \max\left(0, \frac{\alpha_i}{\lambda^*} - \frac{1}{\beta_i}\right)")
        st.info(f"💡 **Equal Marginal Return:** At optimal budget allocation, the marginal revenue of the last dollar spent across all funded personas is exactly equalized to **λ* = {camp_res['the_science']['lambda_star']:.4f}**.")
        for asm in camp_res["the_science"]["assumptions"]:
            st.markdown(f"- ⚠️ *{asm}*")

# =================================================================
# TAB 3: F3 - BATCH CUSTOMER CSV SCORING ENGINE
# =================================================================
with tabs[2]:
    st.subheader("📁 Batch Customer CSV Scoring Engine")
    st.caption("Upload a spreadsheet of customers to execute real-time feature standardization, PCA projection, persona classification, and Platt probability calibration.")

    c_b_up, c_b_demo = st.columns([2, 1])

    with c_b_demo:
        st.download_button(
            "📥 Download 5-Row CSV Template",
            data=sample_template_bytes,
            file_name="customer_batch_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        load_demo = st.button("💡 Load Demo 20-Customer Batch", use_container_width=True)

    with c_b_up:
        uploaded_csv = st.file_uploader("Upload Customer CSV File:", type=["csv"])

    target_batch_df = None
    if uploaded_csv is not None:
        try:
            raw_df = pd.read_csv(uploaded_csv)
            if len(raw_df) == 0:
                st.error("⚠️ The uploaded CSV contains zero data rows. Please upload a file with customer records.")
            else:
                target_batch_df = raw_df
                st.success(f"Uploaded {len(target_batch_df):,} customers from `{uploaded_csv.name}`!")
        except Exception as e:
            st.error(f"⚠️ Error reading CSV file: {e}. Please ensure the file is a valid comma-separated spreadsheet.")
    elif load_demo:
        target_batch_df = df_demo_batch.copy()
        st.success("Loaded 20 real pre-configured demo test customer profiles!")
    else:
        target_batch_df = df_demo_batch.copy()

    if target_batch_df is not None and len(target_batch_df) > 0:
        try:
            with st.spinner("Executing calibrated ML scoring pipeline..."):
                scored_batch_df = score_batch_dataframe(target_batch_df)
        except Exception as e:
            st.error(f"⚠️ Scoring pipeline error: {e}")
            scored_batch_df = None

        if scored_batch_df is not None:
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Total Customers Scored", f"{len(scored_batch_df):,}")
            with b2:
                st.metric("Expected Buyers", f"{int(scored_batch_df['calibrated_purchase_prob'].sum()):,}")
            with b3:
                st.metric("Mean Purchase Intent", f"{scored_batch_df['calibrated_purchase_prob'].mean()*100:.2f}%")
            with b4:
                high_intent_count = (scored_batch_df['calibrated_purchase_prob'] >= 0.10).sum()
                st.metric("High-Intent Customers (≥10%)", f"{high_intent_count:,}")

            st.markdown("---")
            st.markdown("##### 📋 Batch Scored Customer Table")
        
        display_batch_cols = [
            "intent_rank", "customer_code" if "customer_code" in scored_batch_df else "predicted_cluster",
            "persona_name", "calibrated_purchase_prob", "total_spend", "cart_count", "recency_days"
        ]
        available_cols = [c for c in display_batch_cols if c in scored_batch_df.columns]
        
        st.dataframe(
            scored_batch_df[available_cols].rename(columns={
                "intent_rank": "Priority Rank", "customer_code": "Customer ID",
                "predicted_cluster": "Cluster", "persona_name": "Assigned Persona",
                "calibrated_purchase_prob": "Purchase Probability", "total_spend": "Spend ($)",
                "cart_count": "Cart Items", "recency_days": "Recency (d)"
            }).head(25),
            use_container_width=True,
            hide_index=True
        )

        batch_csv_export = scored_batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download Fully Scored Batch Results (CSV)",
            data=batch_csv_export,
            file_name="nexus_scored_batch_results.csv",
            mime="text/csv",
            type="primary"
        )

# =================================================================
# TAB 4: F4 - BEHAVIORAL PERSONA EXPLORER & 2D PCA MAP
# =================================================================
with tabs[3]:
    st.subheader("📊 Behavioral Customer Personas & Interactive 2D SVD/PCA Map")
    st.caption("K-Means clustering ($k=4$) on 5 orthogonal PCA dimensions, capturing 80.3% of total behavioral variance.")

    c_p1, c_p2 = st.columns(2)
    with c_p1:
        p_primary = st.selectbox("Select Primary Focus Persona:", list(persona_profiles.keys()), format_func=lambda k: persona_profiles[k]["persona_name"])
    with c_p2:
        p_secondary = st.selectbox("Select Comparison Persona:", [k for k in persona_profiles.keys() if k != p_primary], format_func=lambda k: persona_profiles[k]["persona_name"])

    info_1 = persona_profiles[p_primary]
    info_2 = persona_profiles[p_secondary]

    cp1, cp2 = st.columns(2)
    for col, data, is_prim in zip([cp1, cp2], [info_1, info_2], [True, False]):
        with col:
            st.markdown(f"""
            <div class='custom-card-highlight' style='border-left-color: {"#6366F1" if is_prim else "#94A3B8"} !important;'>
                <div style='font-size:0.75rem; color:#94A3B8; font-weight:700;'>{"PRIMARY PERSONA" if is_prim else "COMPARISON SEGMENT"}</div>
                <div style='font-size:1.25rem; font-weight:800; color:#F8FAFC;'>{data['persona_name']}</div>
                <div style='margin-top:8px; font-size:0.85rem; color:#CBD5E1; line-height:1.6;'>
                    <b>Customer Count:</b> {data['customer_count']:,} ({data['pct_of_total']}%)<br/>
                    <b>10-Day Conversion:</b> <span style='color:#34D399; font-weight:700;'>{data['conversion_rate_pct']}%</span><br/>
                    <b>95% Wilson CI:</b> [{data['wilson_ci_lower_pct']}%, {data['wilson_ci_upper_pct']}%]<br/>
                    <b>Historical Spend:</b> ${data['mean_spend']:,.2f}<br/>
                    <b>Prescribed Action:</b> {data['recommended_action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🗺️ Interactive 2D Customer Map (PCA / SVD Latent Space)")
    st.write("2,000 customers sampled from warehouse projected onto PC1 (Engagement Volume) and PC2 (Monetary Intensity). Star markers indicate cluster centroids.")

    sample_pts = df_warehouse.sample(n=min(2000, len(df_warehouse)), random_state=42).copy()

    fig_pca = px.scatter(
        sample_pts,
        x="pca_x",
        y="pca_y",
        color="persona_name",
        color_discrete_map={
            "Champions (High-Value Spenders)": "#10B981",
            "Active Explorers (High Engagement)": "#6366F1",
            "Recent Window Shoppers (Warm Bouncers)": "#F59E0B",
            "Dormant Bouncers (Cold / Inactive)": "#64748B"
        },
        hover_data={
            "user_id": True,
            "persona_name": True,
            "total_spend": ":$.2f",
            "recency_days": True,
            "calibrated_prob": ":.2%",
            "pca_x": False,
            "pca_y": False
        },
        labels={
            "pca_x": "Principal Component 1 (Engagement Volume)",
            "pca_y": "Principal Component 2 (Monetary Intensity)",
            "persona_name": "Cluster Persona"
        },
        template="plotly_dark",
        opacity=0.7
    )

    for k in range(4):
        c_pt = persona_profiles[str(k)]["pca_center_2d"]
        fig_pca.add_trace(go.Scatter(
            x=[c_pt[0]],
            y=[c_pt[1]],
            mode="markers+text",
            marker=dict(symbol="star", size=16, color="#F8FAFC", line=dict(color="#0F172A", width=1.5)),
            name=f"Centroid C{k}",
            text=[f"C{k}"],
            textposition="top center",
            showlegend=False
        ))

    fig_pca.update_layout(
        height=480,
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC")
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    with st.expander("🔬 The Science: Chi-Square Independence Test, Wilson CIs & SVD"):
        st.markdown("##### 1. Chi-Square Test of Independence for Conversion Rates")
        st.latex(r"\chi^2 = 2841.6, \quad df = 3, \quad p < 10^{-300}, \quad \text{Cramér's } V = 0.160")
        st.caption("Indicates a statistically indisputable difference in 10-day conversion across personas.")

        st.markdown("##### 2. Exact 95% Wilson Score Confidence Interval")
        st.latex(r"\text{CI}_{95\%} = \frac{p + \frac{z^2}{2n} \pm z \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}")
        st.markdown(f"For {info_1['persona_name']}: $n={info_1['customer_count']:,}, p={info_1['conversion_rate_pct']}\\% \\implies$ **[{info_1['wilson_ci_lower_pct']}%, {info_1['wilson_ci_upper_pct']}%]**")

        st.markdown("##### 3. SVD Singular Values & PCA Variance Explained")
        st.latex(r"X_{\text{scaled}} = U \Sigma V^T \implies \lambda_i(C) = \frac{\sigma_i^2}{n - 1}")
        pca_var_df = pd.DataFrame([
            {"Component": "PC 1", "Variance Explained": "40.43%", "Cumulative": "40.43%", "Dominant Factor": "Total events, views, sessions (Engagement)"},
            {"Component": "PC 2", "Variance Explained": "17.05%", "Cumulative": "57.48%", "Dominant Factor": "Total spend, purchase count (Monetary Intensity)"},
            {"Component": "PC 3", "Variance Explained": "10.18%", "Cumulative": "67.66%", "Dominant Factor": "Tenure vs Recency (Lifecycle Maturity)"},
            {"Component": "PC 4", "Variance Explained": "7.03%", "Cumulative": "74.69%", "Dominant Factor": "Unique products & categories (Catalog Exploration)"},
            {"Component": "PC 5", "Variance Explained": "5.63%", "Cumulative": "80.33%", "Dominant Factor": "Cart-to-view ratio (Immediate Intent)"}
        ])
        st.dataframe(pca_var_df, use_container_width=True, hide_index=True)

# =================================================================
# TAB 5: F5 - 7-DAY TRAFFIC OUTLOOK (SARIMA)
# =================================================================
with tabs[4]:
    st.subheader("📈 7-Day Traffic Outlook & Seasonal SARIMA Projection")
    st.caption("Captures weekly Day-of-Week seasonality ($s=7$) to optimize marketing touchpoint timing.")

    st.markdown("""
    <div class='takeaway-card'>
        <b>Operational Action Line:</b> Schedule primary campaigns for <b>Friday and Saturday</b>, 
        when traffic surges by <b>+15.2% over baseline</b> (~15,900 daily events). Mid-week days (Tuesday-Wednesday) represent the weekly engagement trough.
    </div>
    """, unsafe_allow_html=True)

    tf1, tf2, tf3, tf4 = st.columns(4)
    with tf1:
        st.metric("Holdout MAPE", f"{forecast_payload['holdout_mape_pct']}%", "Outperforms all baselines")
    with tf2:
        st.metric("Holdout RMSE", f"{int(forecast_payload['holdout_rmse']):,} events")
    with tf3:
        st.metric("Peak Engagement Window", "Friday & Saturday", "+15% surge")
    with tf4:
        st.metric("Forecast Horizon", "7 Days Ahead", "Nov 1 to Nov 7")

    st.markdown("---")
    st.markdown("##### 📊 31-Day History & 7-Day Ahead Projection with 95% Uncertainty Bands")

    hist_df = pd.DataFrame(forecast_payload["historical_series"])
    fc_df = pd.DataFrame(forecast_payload["forecast"])

    fig_ts = go.Figure()

    fig_ts.add_trace(go.Scatter(
        x=hist_df["full_date"],
        y=hist_df["total_events"],
        mode="lines+markers",
        name="Historical Events",
        line=dict(color="#F8FAFC", width=2.5),
        marker=dict(size=4)
    ))

    fig_ts.add_trace(go.Scatter(
        x=fc_df["forecast_date"].tolist() + fc_df["forecast_date"].tolist()[::-1],
        y=fc_df["pi_95_upper"].tolist() + fc_df["pi_95_lower"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="95% Prediction Interval"
    ))

    fig_ts.add_trace(go.Scatter(
        x=fc_df["forecast_date"],
        y=fc_df["predicted_traffic"],
        mode="lines+markers",
        name="SARIMA Forecast",
        line=dict(color="#38BDF8", width=3, dash="dash"),
        marker=dict(size=6, color="#38BDF8")
    ))

    fig_ts.update_layout(
        height=400,
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Total Events Count",
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        legend=dict(orientation="h", y=-0.18)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("##### 📅 Day-by-Day Tactical Operational Schedule")
    st.dataframe(
        fc_df[["forecast_date", "day_of_week", "predicted_traffic", "pi_95_lower", "pi_95_upper", "recommended_action"]].rename(columns={
            "forecast_date": "Date", "day_of_week": "Day", "predicted_traffic": "Predicted Events",
            "pi_95_lower": "95% Lower Band", "pi_95_upper": "95% Upper Band", "recommended_action": "Tactical Action"
        }),
        use_container_width=True,
        hide_index=True
    )

    with st.expander("🔬 The Science: SARIMA Model Specification & Diagnostics"):
        st.markdown("##### SARIMA Specification")
        st.latex(r"\text{SARIMA}(1,1,1) \times (1,0,0)_7: \quad (1 - \phi_1 B)(1 - \Phi_1 B^7)(1 - B) y_t = (1 + \theta_1 B) \epsilon_t")
        st.dataframe(df_forecast_eval, use_container_width=True, hide_index=True)
        st.dataframe(df_stationarity, use_container_width=True, hide_index=True)
        st.warning("⚠️ **History Duration Notice:** The model was estimated on 31 days of continuous telemetry. It captures weekly cycles ($s=7$), but cannot capture multi-month or annual trends.")

# =================================================================
# TAB 6: F6 - TECHNICAL AUDIT & LAKEHOUSE ARCHITECTURE
# =================================================================
with tabs[5]:
    st.subheader("🔬 Supervised Model Audit & Lakehouse Architecture")
    st.caption("Production engineering documentation, model performance audit, and data lakehouse diagnostics.")

    col_bench, col_thresh = st.columns([1.1, 1])

    with col_bench:
        st.markdown("##### 🏆 Supervised Model Competition")
        st.dataframe(
            df_model_benchmarks.rename(columns={
                "model": "Model Candidate", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
                "brier_score": "Brier Score", "f1_score_tau50": "F1 (τ=0.50)", "fit_time_seconds": "Train Time (s)"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.info("🏆 **Champion Model: Logistic Regression** achieved highest PR-AUC (0.2450) with instant inference (<1ms) and 100% mathematical interpretability.")

    with col_thresh:
        st.markdown("##### 💰 Cost-Sensitive Threshold Optimization (τ*)")
        st.markdown("""
        - **Revenue per Converted Buyer ($R$):** `$50.00`
        - **Cost per Targeted Offer ($C$):** `$2.00`
        - **Commercial Profit Function:** $\\text{Profit}(\\tau) = \\text{TP}(\\tau) \\cdot R - (\\text{TP} + \\text{FP}) \\cdot C$
        """)
        
        opt_row = df_threshold_analysis.loc[df_threshold_analysis["expected_net_profit"].idxmax()]
        st.markdown(f"""
        <div class='custom-card-highlight' style='border-left-color: #10B981 !important;'>
            <b>Optimal Decision Threshold:</b> <span style='color:#34D399; font-weight:700;'>τ* = {opt_row['threshold']:.2f}</span><br/>
            <b>Max Expected Net Profit:</b> <span style='color:#F8FAFC; font-weight:700;'>${opt_row['expected_net_profit']:,.2f}</span><br/>
            <b>Captured Buyers (Recall):</b> {opt_row['recall']*100:.1f}% &middot; <b>Precision:</b> {opt_row['precision']*100:.1f}%
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🛡️ Leakage-Safe Temporal Split Architecture")
    st.markdown("""
    ```
    October 1, 2019                                 October 20, 2019 (23:59:59)              October 31, 2019
    ├───────────────────────────────────────────────────────┤───────────────────────────────────────┤
                 FEATURE WINDOW (Days 1–20)                           TARGET EVALUATION (Days 21–31)
    - Compute 17 customer-level behavioral features          - Target label: will_purchase_next_10d
      (Recency, Spend, Sessions, Unique Products)            - ZERO forward events leak into features!
    ```
    """)

    st.markdown("---")
    st.markdown("##### 🏗️ Medallion Data Engineering Pipeline & The 5 Vs")
    c_b1, c_b2, c_b3 = st.columns(3)
    with c_b1:
        st.markdown("""
        <div class='custom-card' style='border-left: 4px solid #F59E0B;'>
            <div style='font-size:0.75rem; color:#94A3B8; font-weight:700;'>BRONZE LAYER</div>
            <div style='font-size:1.15rem; font-weight:800; color:#F8FAFC;'>42,448,764 Events</div>
            <div style='font-size:0.82rem; color:#CBD5E1; margin-top:4px;'>
                Partitioned Snappy Parquet &middot; 5.66 GB raw CSV &middot; Date-partitioned validation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_b2:
        st.markdown("""
        <div class='custom-card' style='border-left: 4px solid #6366F1;'>
            <div style='font-size:0.75rem; color:#94A3B8; font-weight:700;'>SILVER LAYER</div>
            <div style='font-size:1.15rem; font-weight:800; color:#F8FAFC;'>DuckDB Star Schema</div>
            <div style='font-size:0.82rem; color:#CBD5E1; margin-top:4px;'>
                Cleaned fact_events &middot; Dimension tables &middot; Outliers pruned &middot; High-speed OLAP.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_b3:
        st.markdown("""
        <div class='custom-card' style='border-left: 4px solid #10B981;'>
            <div style='font-size:0.75rem; color:#94A3B8; font-weight:700;'>GOLD LAYER</div>
            <div style='font-size:1.15rem; font-weight:800; color:#F8FAFC;'>110,956 User Mart</div>
            <div style='font-size:0.82rem; color:#CBD5E1; margin-top:4px;'>
                17 behavioral features &middot; 4 K-Means personas &middot; Platt-calibrated champion model.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### ⚡ Interactive Gradient Descent Loss Replay")
    
    max_ep = int(df_gd_convergence["epoch"].max())
    ep_slider = st.slider("Scrub Training Epoch:", min_value=1, max_value=max_ep, value=max_ep, step=1)
    sub_gd = df_gd_convergence[df_gd_convergence["epoch"] <= ep_slider]

    fig_gd = go.Figure()
    fig_gd.add_trace(go.Scatter(x=sub_gd["epoch"], y=sub_gd["batch_gd_loss"], mode="lines+markers", name="Batch GD", line=dict(color="#818CF8", width=2.5)))
    fig_gd.add_trace(go.Scatter(x=sub_gd["epoch"], y=sub_gd["sgd_loss"], mode="lines+markers", name="Stochastic GD", line=dict(color="#F43F5E", width=1.5, dash="dot")))
    fig_gd.add_trace(go.Scatter(x=sub_gd["epoch"], y=sub_gd["mini_batch_gd_loss"], mode="lines+markers", name="Mini-Batch GD (b=256)", line=dict(color="#10B981", width=2.5)))
    fig_gd.update_layout(
        height=360,
        autosize=True,
        xaxis_title="Epoch",
        yaxis_title="Binary Cross-Entropy Loss",
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        legend=dict(orientation="h", y=-0.18)
    )
    st.plotly_chart(fig_gd, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# 6. CAPSTONE ENGINEERING GROUP FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; color:#94A3B8; padding: 10px 0;'>
    <div>
        <b style='color:#F8FAFC;'>NEXUS Intelligence Platform</b> &middot; Final Year BSc Data Science Group Capstone Project<br/>
        42.4M REES46 Events &middot; Zero Lookahead Leakage &middot; Platt-Calibrated Champion Engine
    </div>
    <div style='text-align:right;'>
        <b>Engineering Division (4 Roles):</b><br/>
        1. Data Engineering & ELT &middot; 2. PCA & Unsupervised Learning &middot; 3. Supervised Modeling & Calibration &middot; 4. Lagrange Optimization & Platform UI
    </div>
</div>
""", unsafe_allow_html=True)
