"""
app/main.py
===========
Customer Behavioral Segmentation & Purchase Intent Prediction
Production Streamlit Application — Capstone Project
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import GOLD_DIR, MODELS_DIR, REPORTS_DIR, DUCKDB_PATH
from src.budget_optimization import solve_lagrange_budget, revenue_function

# Streamlit Page Configuration
st.set_page_config(
    page_title="eCommerce Customer Analytics & Intent AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1e3a8a; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #475569; margin-bottom: 1.5rem; }
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 4px; padding: 8px 16px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_app_data():
    """Load cached dataset summaries and profiles."""
    try:
        df_seg = pd.read_csv(REPORTS_DIR / "customer_segments_profile.csv")
    except Exception:
        df_seg = pd.DataFrame()

    try:
        df_daily = pd.read_csv(REPORTS_DIR / "daily_timeseries_data.csv")
        df_daily["full_date"] = pd.to_datetime(df_daily["full_date"])
    except Exception:
        df_daily = pd.DataFrame()

    try:
        df_forecast = pd.read_csv(REPORTS_DIR / "timeseries_forecast_evaluation.csv")
    except Exception:
        df_forecast = pd.DataFrame()

    return df_seg, df_daily, df_forecast


@st.cache_resource
def load_ml_pipeline():
    """Load serialized champion model and scaler."""
    try:
        model = joblib.load(MODELS_DIR / "intent_model_champion.joblib")
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        features = joblib.load(MODELS_DIR / "feature_names.joblib")
        return model, scaler, features
    except Exception as e:
        return None, None, None


df_seg, df_daily, df_forecast = load_app_data()
model, scaler, feature_names = load_ml_pipeline()

# ── Sidebar Navigation ──────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart-loaded.png", width=64)
    st.markdown("### **Capstone Intelligence**")
    st.caption("**Dataset**: REES46 eCommerce Clickstream (October 2019)")
    st.caption("**Hardware**: Apple Silicon M1 (8 GB)")
    st.divider()

    st.markdown("#### **Business Questions**")
    st.markdown("""
    - **Q1**: What behavioral segments exist?
    - **Q2**: Who will purchase in next 10d?
    - **Q3**: Do segments differ in conversion?
    - **Q4**: What will traffic look like?
    - **Q5**: How should budget be allocated?
    """)
    st.divider()
    st.caption("Data Science Capstone — End-to-End Pipeline")

# Main Header
st.markdown('<div class="main-header">🛒 Customer Behavioral Segmentation & Purchase Intent AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Medallion Data Lakehouse · Linear Algebra · Supervised ML · Lagrange Budget Optimization · Time Series</div>', unsafe_allow_html=True)

# ── Tabs Navigation ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏢 Executive Overview",
    "👥 Behavioral Personas (Q1 & Q3)",
    "🎯 Live Intent Predictor (Q2)",
    "💰 Budget Optimizer (Q5)",
    "⏱️ Traffic Forecasting (Q4)"
])

# ================================================================
# TAB 1: Executive Overview
# ================================================================
with tab1:
    st.markdown("### **Platform Scale & Medallion Pipeline Overview**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Raw Events Scanned", "42.45M", "5.67 GB CSV")
    with col2:
        st.metric("Bronze Sampled", "2.10M", "100 MB Parquet (98% drop)")
    with col3:
        st.metric("Active Customers", "110,956", "Oct 01 - Oct 20")
    with col4:
        st.metric("October GMV", "$10.84M", "36,675 Purchases")
    with col5:
        st.metric("10-Day Intent Rate", "3.87%", "4,292 Buyers")

    st.markdown("---")
    
    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.markdown("#### **Medallion Lakehouse Architecture**")
        st.markdown("""
        - **Raw Layer (5.67 GB CSV)**: 42,448,764 clickstream events loaded with PySpark in local multi-core mode.
        - **Bronze Layer (100 MB Parquet)**: Deterministic 5% user-level sample (151,177 users) preserving complete behavioral histories, partitioned across 31 dates.
        - **Silver Layer (98 MB Parquet)**: Deduplicated events with cleaned prices ($> 0$), parsed category hierarchies, and composite session keys.
        - **Gold Layer (DuckDB Star Schema)**: `dim_date`, `dim_product`, `dim_user`, and `fact_events` in embedded `warehouse.duckdb`.
        - **Feature Mart (Parquet)**: 110,956 user profiles with strict temporal split (Oct 01–20 features vs Oct 21–31 target) guaranteeing **zero target leakage**.
        """)

    with c_right:
        st.markdown("#### **Full Behavioral Conversion Funnel**")
        funnel_data = pd.DataFrame({
            "Stage": ["1. Product Views", "2. Added to Cart", "3. Purchases"],
            "Event Count": [2017907, 46256, 36675],
            "Percentage": ["96.05%", "2.20%", "1.75%"],
            "Drop-off / Conversion": ["100% (Baseline)", "2.29% (View to Cart)", "79.28% (Cart to Purchase)"]
        })
        st.dataframe(funnel_data, width="stretch", hide_index=True)
        st.info("💡 **Key Finding**: Cart-to-purchase completion is high (~79%), identifying that the primary commercial leak is in the View-to-Cart discovery transition (~2.3%).")

# ================================================================
# TAB 2: Customer Behavioral Personas (Q1 & Q3)
# ================================================================
with tab2:
    st.markdown("### **Customer Behavioral Personas (Question Q1 & Q3)**")
    st.markdown("Unsupervised clustering ($k=4$) on top 5 Principal Components derived via Singular Value Decomposition (SVD).")

    if not df_seg.empty:
        c1, c2, c3, c4 = st.columns(4)
        cols = [c1, c2, c3, c4]
        colors = ["#7f8c8d", "#3498db", "#27ae60", "#2980b9"]

        for idx, row in df_seg.iterrows():
            with cols[idx]:
                st.markdown(f"#### **{row['persona_name'].split('(')[0]}**")
                st.markdown(f"**Cluster {row['cluster']}** — *{row['pct_of_total']}% of user base*")
                st.metric("10-Day Conversion", f"{row['next_10d_conversion_rate']}%")
                st.write(f"- **Customer Count**: {int(row['customer_count']):,}")
                st.write(f"- **Median Recency**: {row['recency_days']:.0f} days")
                st.write(f"- **Median Events**: {row['total_events']:.0f} actions")
                st.write(f"- **Mean Spend**: ${row['total_spend']:.2f}")

        st.markdown("---")
        st.markdown("#### **Statistical Proof of Segment Differences (Question Q3)**")
        st.success("""
        - **Omnibus $\\chi^2$ Test**: $\\chi^2 = 6,295.96$ ($df=3, p < 10^{-300}$, Cramér's $V = 0.2382$). Conversion differences across personas are verified.
        - **Post-Hoc Pairwise $Z$-Tests**: All 6 persona pairings remain statistically distinct after Bonferroni correction ($p_{\\text{adj}} < 10^{-14}$).
        - **Champions (16.93%)** convert **13.2× higher** than Dormant Bouncers (1.28%) and **3.5× higher** than Active Explorers (4.90%).
        """)

# ================================================================
# TAB 3: Live Purchase Intent Predictor (Q2)
# ================================================================
with tab3:
    st.markdown("### **Real-Time Purchase Intent Predictor (Question Q2)**")
    st.markdown("Live inference using the champion **Balanced Logistic Regression / LightGBM** pipeline trained in Phase 5.")

    if model is not None and scaler is not None:
        p_col1, p_col2, p_col3 = st.columns(3)

        with p_col1:
            st.markdown("##### **Recency & Tenure Signals**")
            in_recency = st.slider("Recency (Days since last event)", 0, 20, 3)
            in_tenure = st.slider("Customer Tenure (Days since first event)", 0, 20, 12)
            in_active_days = st.slider("Distinct Active Days", 1, 15, 3)
            in_cluster = st.selectbox("Behavioral Persona Cluster", [
                "Cluster 2: Champions", "Cluster 3: Active Explorers",
                "Cluster 1: Recent Window Shoppers", "Cluster 0: Dormant Bouncers"
            ])
            cluster_id = int(in_cluster.split(":")[0].replace("Cluster", "").strip())

        with p_col2:
            st.markdown("##### **Browsing Engagement Signals**")
            in_sessions = st.slider("Total Sessions", 1, 30, 4)
            in_events = st.slider("Total Clickstream Events", 1, 150, 25)
            in_views = st.slider("Product Views", 1, 120, 20)
            in_unique_prods = st.slider("Unique Products Viewed", 1, 50, 8)
            in_weekend_ratio = st.slider("Weekend Activity Ratio", 0.0, 1.0, 0.35)

        with p_col3:
            st.markdown("##### **Cart & Monetary Signals**")
            in_carts = st.slider("Cart Actions", 0, 15, 3)
            in_purchases = st.slider("Past Purchases", 0, 10, 1)
            in_spend = st.number_input("Total Past Spend ($)", min_value=0.0, max_value=5000.0, value=120.0, step=10.0)
            in_max_price = st.number_input("Max Item Price Interacted ($)", min_value=0.0, max_value=3000.0, value=350.0, step=25.0)

        # Derived metrics
        cart_to_view = in_carts / max(1, in_views)
        events_per_sess = in_events / max(1, in_sessions)
        avg_purch_price = in_spend / max(1, in_purchases) if in_purchases > 0 else 0.0

        if st.button("🚀 Score Customer Purchase Intent", type="primary"):
            # Construct feature vector matching trained model schema
            raw_input = {
                "recency_days": in_recency,
                "tenure_days": in_tenure,
                "active_days": in_active_days,
                "total_sessions": np.log1p(in_sessions),
                "total_events": np.log1p(in_events),
                "view_count": np.log1p(in_views),
                "cart_count": np.log1p(in_carts),
                "purchase_count": np.log1p(in_purchases),
                "total_spend": np.log1p(in_spend),
                "avg_purchase_price": np.log1p(avg_purch_price),
                "max_price_interacted": np.log1p(in_max_price),
                "n_unique_products": np.log1p(in_unique_prods),
                "n_unique_cat1": 2,
                "n_unique_brands": 3,
                "weekend_event_ratio": in_weekend_ratio,
                "cart_to_view_ratio": cart_to_view,
                "events_per_session": events_per_sess,
                "cluster_0": 1.0 if cluster_id == 0 else 0.0,
                "cluster_1": 1.0 if cluster_id == 1 else 0.0,
                "cluster_2": 1.0 if cluster_id == 2 else 0.0,
                "cluster_3": 1.0 if cluster_id == 3 else 0.0
            }

            input_df = pd.DataFrame([raw_input])[feature_names]
            input_scaled = scaler.transform(input_df)
            prob = float(model.predict_proba(input_scaled)[0, 1])
            is_buyer = prob >= 0.50

            st.markdown("---")
            res_c1, res_c2, res_c3 = st.columns([1, 1, 1.2])

            with res_c1:
                st.metric("Predicted 10-Day Purchase Propensity", f"{prob*100:.1f}%")
                st.progress(min(1.0, prob))

            with res_c2:
                if is_buyer:
                    st.success("🎯 **HIGH INTENT BUYER**\n(Prob $\ge$ 50% Threshold)")
                else:
                    st.warning("⏳ **LOW / MODERATE INTENT**\n(Prob < 50% Threshold)")

            with res_c3:
                st.markdown("##### **Recommended Marketing Action**")
                if prob >= 0.60:
                    st.markdown("⭐ **VIP Immediate Action**: High-value closing window. Deploy personalized offer or checkout discount within 24 hours.")
                elif prob >= 0.40:
                    st.markdown("🛒 **Cart Nurture**: Automated abandoned-cart email sequence and recommended matching accessories.")
                else:
                    st.markdown("👀 **Organic Retargeting**: Retain on organic browse / low-cost standard newsletters. Do not spend paid CPC budget.")
    else:
        st.error("ML model artifacts not found. Please verify Phase 5 execution.")

# ================================================================
# TAB 4: Lagrange Budget Optimizer (Q5)
# ================================================================
with tab4:
    st.markdown("### **Constrained Marketing Budget Allocator (Question Q5)**")
    st.markdown("Microeconomic optimization using **Lagrange Multipliers** on concave diminishing-returns revenue functions:")
    st.latex(r"\max_{b_0, \dots, b_3} \sum_{i=0}^3 \alpha_i \ln(1 + \beta_i b_i) \quad \text{subject to} \quad \sum_{i=0}^3 b_i = B_{\text{total}}, \quad b_i \ge 0")

    budget_slider = st.slider("Select Total Campaign Marketing Budget ($)", min_value=1000, max_value=50000, value=10000, step=1000)

    alphas = np.array([4000.0, 5000.0, 25000.0, 12000.0])
    betas  = np.array([0.0005, 0.0010, 0.0080, 0.0025])
    seg_names = ["Dormant Bouncers", "Recent Window Shoppers", "Champions", "Active Explorers"]
    seg_counts = [39911, 36245, 11266, 23534]

    b_opt, lam_star = solve_lagrange_budget(alphas, betas, float(budget_slider))
    r_opt = revenue_function(b_opt, alphas, betas)
    tot_rev = np.sum(r_opt)
    net_profit = tot_rev - budget_slider
    roi = (net_profit / budget_slider) * 100.0

    # Uniform comparison
    b_uni = np.full(4, budget_slider / 4.0)
    r_uni = revenue_function(b_uni, alphas, betas)
    tot_rev_uni = np.sum(r_uni)
    profit_uni = tot_rev_uni - budget_slider

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        st.metric("Total Campaign Budget", f"${budget_slider:,.0f}")
    with b_col2:
        st.metric("Optimal Shadow Price (λ*)", f"{lam_star:.3f}", "Marginal $/dollar")
    with b_col3:
        st.metric("Expected Gross Revenue", f"${tot_rev:,.2f}")
    with b_col4:
        st.metric("Expected Net Profit", f"${net_profit:,.2f}", f"+${net_profit - profit_uni:,.2f} vs Equal")

    st.markdown("---")
    alloc_c1, alloc_c2 = st.columns([1.2, 1])

    with alloc_c1:
        st.markdown("#### **Optimal Budget Allocations Across Personas**")
        alloc_table = pd.DataFrame({
            "Persona": seg_names,
            "Optimal Budget ($)": [f"${x:,.2f}" for x in b_opt],
            "Share (%)": [f"{x/budget_slider*100:.1f}%" for x in b_opt],
            "Expected Revenue ($)": [f"${x:,.2f}" for x in r_opt]
        })
        st.dataframe(alloc_table, width="stretch", hide_index=True)

    with alloc_c2:
        st.markdown("#### **Lagrange Multiplier Allocation Breakdown**")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
        # Filter out 0 budgets for pie
        non_zero = b_opt > 0
        ax_pie.pie(b_opt[non_zero], labels=[seg_names[i] for i in range(4) if non_zero[i]],
                   autopct="%1.1f%%", colors=["#3498db", "#27ae60", "#2980b9"], startangle=140)
        st.pyplot(fig_pie)
        plt.close(fig_pie)

# ================================================================
# TAB 5: Time Series Traffic & Sales Forecaster (Q4)
# ================================================================
with tab5:
    st.markdown("### **Time Series Analysis & Traffic Forecaster (Question Q4)**")
    st.markdown("Out-of-sample holdout benchmarking using **Seasonal SARIMA(1, 1, 1) $\times$ (1, 0, 0)$_7$**.")

    if not df_daily.empty:
        ts_c1, ts_c2 = st.columns([1.5, 1])

        with ts_c1:
            st.markdown("#### **Daily Platform Events Trajectory (October 2019)**")
            fig_ts, ax_ts = plt.subplots(figsize=(10, 4.5))
            ax_ts.plot(df_daily["full_date"], df_daily["total_events"], "b-o", linewidth=2, markersize=4, label="Daily Events")
            ax_ts.set_ylabel("Total Clickstream Events")
            ax_ts.grid(True, linestyle="--", alpha=0.5)
            # Shade holdout test week
            ax_ts.axvspan(pd.to_datetime("2019-10-25"), pd.to_datetime("2019-10-31"), color="yellow", alpha=0.2, label="7-Day Holdout Forecast Window")
            ax_ts.legend(loc="lower left")
            st.pyplot(fig_ts)
            plt.close(fig_ts)

        with ts_c2:
            st.markdown("#### **Forecast Benchmarking & Day-of-Week Rhythm**")
            st.metric("Champion SARIMA MAPE", "3.99%", "96.01% Out-of-Sample Accuracy")
            st.write("##### **Weekly Seasonality Surge**")
            st.write("- **Peak Traffic Days**: Sundays (~72k events) and Fridays (~71k events)")
            st.write("- **Trough Days**: Thursdays (~63k events)")
            st.info("💡 **Operational Insight**: Pre-scale web application pods on Saturday night ahead of Sunday conversion surge, and schedule database batch maintenance on Thursdays.")

st.markdown("---")
st.caption("Capstone Project — Data Science · REES46 eCommerce Behavior Dataset · Complete Phase 1-8 Implementation")
