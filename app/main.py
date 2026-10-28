"""
app/main.py — Educational Interactive Intelligence Platform & Viva Guide
========================================================================
Customer Behavioral Segmentation & Purchase Intent Prediction
A self-explanatory, interactive dashboard designed for faculty evaluation,
student presentations, and business exploration.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import GOLD_DIR, MODELS_DIR, REPORTS_DIR
from src.budget_optimization import solve_lagrange_budget, revenue_function

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG & MODERN DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Intelligence Platform — Capstone Project",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Responsive, Glassmorphism, Semantic Cards)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

    /* Hero Header */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .hero h1 { font-size: 2.3rem; font-weight: 800; margin-bottom: 0.4rem; color: #ffffff; letter-spacing: -0.02em; }
    .hero p { font-size: 1.05rem; color: #c7d2fe; margin-top: 0; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.5; }

    /* Section Titles */
    .section-header {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.65rem;
        font-weight: 800;
        margin: 1.5rem 0 0.5rem 0;
        letter-spacing: -0.01em;
    }

    /* Educational Narrative Cards & Semantic Paragraph Styling */
    .explain-card, .insight-box, .attention-box, .viva-card {
        padding: 1.15rem 1.35rem;
        border-radius: 0 12px 12px 0;
        margin: 0.9rem 0;
        line-height: 1.65;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .explain-card p, .insight-box p, .attention-box p, .viva-card p {
        margin: 0.45rem 0;
        line-height: 1.65;
    }
    .explain-card p:first-child, .insight-box p:first-child, .attention-box p:first-child, .viva-card p:first-child {
        margin-top: 0;
    }
    .explain-card p:last-child, .insight-box p:last-child, .attention-box p:last-child, .viva-card p:last-child {
        margin-bottom: 0;
    }
    .explain-card ul, .insight-box ul, .attention-box ul, .viva-card ul {
        margin: 0.4rem 0 0.6rem 0;
        padding-left: 1.25rem;
    }
    .explain-card li, .insight-box li, .attention-box li, .viva-card li {
        margin-bottom: 0.35rem;
        line-height: 1.6;
    }

    .explain-card {
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        font-size: 0.96rem;
        color: #1e293b;
    }
    .explain-card strong { color: #4338ca; font-weight: 700; }

    .insight-box {
        background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
        border-left: 4px solid #10b981;
        font-size: 0.95rem;
        color: #064e3b;
    }
    .insight-box strong { color: #047857; font-weight: 700; }

    .attention-box {
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
        border-left: 4px solid #f59e0b;
        font-size: 0.95rem;
        color: #78350f;
    }
    .attention-box strong { color: #b45309; font-weight: 700; }

    .viva-card {
        background: #faf5ff;
        border-left: 4px solid #9333ea;
        color: #3b0764;
        box-shadow: 0 2px 5px rgba(147, 51, 234, 0.08);
    }
    .viva-card h4 { margin-top: 0; color: #6b21a8; font-weight: 700; }

    /* Summary Metric Cards */
    .metric-row { display: flex; gap: 1rem; margin: 1.2rem 0; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 160px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    .metric-card .value { font-size: 1.85rem; font-weight: 800; color: #0f172a; line-height: 1.2; }
    .metric-card .label { font-size: 0.82rem; font-weight: 600; color: #64748b; margin-top: 0.35rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-card .delta { font-size: 0.8rem; color: #10b981; font-weight: 600; margin-top: 0.25rem; }

    /* Persona Profile Cards */
    .persona-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.04);
        transition: transform 0.25s, box-shadow 0.25s;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .persona-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 24px rgba(0,0,0,0.08);
    }
    .persona-card .emoji { font-size: 2.4rem; margin-bottom: 0.4rem; }
    .persona-card .name { font-weight: 800; font-size: 1.15rem; color: #0f172a; }
    .persona-card .badge {
        display: inline-block; padding: 0.2rem 0.55rem;
        border-radius: 9999px; font-size: 0.72rem; font-weight: 700;
        margin-top: 0.3rem; margin-bottom: 0.5rem; width: fit-content;
    }
    .persona-card .pct { color: #64748b; font-size: 0.85rem; }
    .persona-card .conv { font-size: 1.55rem; font-weight: 800; margin: 0.4rem 0; }
    .persona-card .desc { font-size: 0.88rem; color: #475569; line-height: 1.55; flex-grow: 1; }

    /* Streamlit overrides */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING (Cached & Robust)
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    d = {}
    try:
        d['segments'] = pd.read_csv(REPORTS_DIR / "customer_segments_profile.csv")
    except:
        d['segments'] = pd.DataFrame()

    try:
        daily_path = REPORTS_DIR / "daily_timeseries_data.csv"
        df_d = pd.read_csv(daily_path)
        if 'full_date' in df_d.columns:
            df_d['full_date'] = pd.to_datetime(df_d['full_date'])
        elif 'date' in df_d.columns:
            df_d['full_date'] = pd.to_datetime(df_d['date'])
        d['daily'] = df_d
    except Exception as e:
        d['daily'] = pd.DataFrame()

    try:
        d['forecast'] = pd.read_csv(REPORTS_DIR / "timeseries_forecast_evaluation.csv")
    except:
        d['forecast'] = pd.DataFrame()

    try:
        d['models'] = pd.read_csv(REPORTS_DIR / "intent_model_comparison.csv")
    except:
        d['models'] = pd.DataFrame()

    try:
        funnel_p = PROJECT_ROOT / "dashboard" / "tableau_category_funnel.csv"
        d['funnel'] = pd.read_csv(funnel_p)
    except:
        d['funnel'] = pd.DataFrame()

    try:
        d['hypothesis'] = pd.read_csv(REPORTS_DIR / "hypothesis_test_results.csv")
    except:
        d['hypothesis'] = pd.DataFrame()

    try:
        d['pairwise'] = pd.read_csv(REPORTS_DIR / "segment_pairwise_comparisons.csv")
    except:
        d['pairwise'] = pd.DataFrame()

    try:
        d['tukey'] = pd.read_csv(REPORTS_DIR / "tukey_spend_differences.csv")
    except:
        d['tukey'] = pd.DataFrame()

    try:
        d['threshold'] = pd.read_csv(REPORTS_DIR / "optimal_threshold_analysis.csv")
    except:
        d['threshold'] = pd.DataFrame()

    try:
        d['lagrange'] = pd.read_csv(REPORTS_DIR / "lagrange_budget_allocation.csv")
    except:
        d['lagrange'] = pd.DataFrame()

    try:
        d['scenarios'] = pd.read_csv(PROJECT_ROOT / "dashboard" / "tableau_lagrange_scenarios.csv")
    except:
        d['scenarios'] = pd.DataFrame()

    return d


@st.cache_resource
def load_ml():
    try:
        model = joblib.load(MODELS_DIR / "intent_model_champion.joblib")
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        features = joblib.load(MODELS_DIR / "feature_names.joblib")
        return model, scaler, features
    except Exception as e:
        return None, None, None


data = load_all_data()
model, scaler, feature_names = load_ml()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Guided Walkthrough")
    page = st.radio(
        "Choose a section:",
        [
            "🏠 Project Overview",
            "🔍 Understanding the Data",
            "👥 Customer Segments (Q1)",
            "🎯 Purchase Predictor (Q2)",
            "📊 Are Segments Real? (Q3)",
            "📈 Traffic Forecasting (Q4)",
            "💰 Budget Optimizer (Q5)",
            "🎓 Viva Defense Cheat Sheet",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.83rem; color: #475569; line-height: 1.6; background: #f8fafc; padding: 0.9rem; border-radius: 10px; border: 1px solid #e2e8f0;'>
        <div style="font-weight: 700; color: #1e293b; margin-bottom: 0.4rem;">📌 Project Specifications:</div>
        <div>• <strong>Dataset:</strong> REES46 eCommerce</div>
        <div>• <strong>Timespan:</strong> October 2019</div>
        <div>• <strong>Raw Events:</strong> 42.4M (5.67 GB)</div>
        <div>• <strong>Customer Cohort:</strong> 110,956 users</div>
        <div>• <strong>Conversion Base:</strong> 3.87%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.caption("Developed for Academic Viva & Practical Business Decision Making")


# ═════════════════════════════════════════════════════════════════
# PAGE 1: PROJECT OVERVIEW
# ═════════════════════════════════════════════════════════════════
if page == "🏠 Project Overview":
    st.markdown("""
    <div class="hero">
        <h1>Customer Intelligence & Decision Platform</h1>
        <p>A complete data science pipeline answering five foundational business questions from 42.4 million eCommerce customer actions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-card">
        <p><strong>The Business Challenge:</strong></p>
        <p>Imagine you operate an online eCommerce superstore with millions of monthly visitors. Most visitors browse casually and leave within 30 seconds. A minority add items to their carts. An even smaller fraction actually completes a transaction.</p>
        <p>To compete profitably, you cannot treat all customers equally. You need to know: <strong>Who are they? Who will buy next? Are these behavioral differences real? What will tomorrow's traffic be? And where should every marketing dollar go?</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">The Five Questions We Answer</div>', unsafe_allow_html=True)

    q1, q2 = st.columns(2)
    with q1:
        st.markdown("""
        #### 👥 Q1. Behavioral Segmentation (Unsupervised Learning)
        *Who are our customers?*

        Instead of guessing demographic buckets, we apply **Principal Component Analysis (PCA)** and **K-Means Clustering** across 25 behavioral dimensions to automatically discover 4 distinct customer personas.

        #### 🎯 Q2. Purchase Intent Prediction (Supervised Learning)
        *Who will buy in the next 10 days?*

        Using historical browsing signals (Oct 1–20), we train models to identify prospective buyers for the subsequent period (Oct 21–31) with zero data leakage, beating random guessing by **6.3× PR-AUC**.

        #### 📊 Q3. Hypothesis Testing (Inferential Statistics)
        *Are customer segment differences real or just random noise?*

        We run rigorous **Chi-Square Tests of Independence, Bonferroni-corrected 2-Sample Z-Tests, and Tukey HSD** to mathematically prove that observed differences ($p < 10^{-14}$) are genuine.
        """)
    with q2:
        st.markdown("""
        #### 📈 Q4. Time Series Traffic Forecasting (SARIMA)
        *How many visitors will arrive next week?*

        We uncover a 7-day cyclical rhythm (weekend surges vs weekday lulls) and build an out-of-sample **SARIMA model** delivering **96.01% forecast accuracy** (3.99% holdout MAPE).

        #### 💰 Q5. Optimal Marketing Budget Allocation (Lagrange Multipliers)
        *How should marketing dollars be distributed to maximize revenue?*

        Accounting for diminishing returns across segments, we derive the exact analytical solution via **KKT constrained optimization**, unlocking **+$1,800 to +$3,500 in net profit lift** over equal budget splits.
        """)

    st.markdown("---")
    st.markdown('<div class="section-header">Data Pipeline Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-card">
        <p><strong>The 4-Layer Medallion Pipeline:</strong></p>
        <ul>
            <li><strong>1. Raw Ingestion Layer:</strong> 42,448,764 clickstream events (5.67 GB CSV). Too memory-intensive for direct desktop exploration.</li>
            <li><strong>2. Bronze Layer (User-Hash Sampling):</strong> We select a deterministic 5% sample of <em>users</em> (<code>user_id % 100 &lt; 5</code>). This preserves 100% of each sampled user's journey (2.1M events, 100 MB).</li>
            <li><strong>3. Silver Layer (Data Hygiene):</strong> Removes negative prices, cleans corrupted product taxonomies, handles missing values, and builds an event-to-session star schema.</li>
            <li><strong>4. Gold Layer (Feature Engineering & Target Formulation):</strong> Constructs a 25-feature customer profile table partitioned into a historical feature window (Oct 1–20) and a future ground-truth target window (Oct 21–31).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-row">
        <div class="metric-card">
            <div class="value">42.4M</div>
            <div class="label">Raw Clickstream Events</div>
            <div class="delta">5.67 GB Raw Storage</div>
        </div>
        <div class="metric-card">
            <div class="value">2.1M</div>
            <div class="label">Sampled Events (100% History)</div>
            <div class="delta">Deterministic 5% Hash</div>
        </div>
        <div class="metric-card">
            <div class="value">110,956</div>
            <div class="label">Unique Customer Cohort</div>
            <div class="delta">25 Engineered Features</div>
        </div>
        <div class="metric-card">
            <div class="value">3.87%</div>
            <div class="label">Baseline Conversion Rate</div>
            <div class="delta">4,292 Buyers in 10-day Window</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🎓 Faculty Check: Why sample 5% of users instead of 5% of rows?"):
        st.markdown("""
        **Crucial viva answer:**

        If we randomly sampled 5% of *individual rows*, a customer who visited 20 times would only retain 1 single action in our dataset! Their browsing sessions would be shredded, cart-to-view ratios would be destroyed, and temporal sequence would vanish.

        By sampling 5% of *user IDs* via modular hashing (`user_id % 100 < 5`), we retain **100% of every click, cart addition, and purchase** for all 110,956 sampled users. The behavioral integrity remains mathematically identical to the full 42.4M event population.
        """)

    with st.expander("🎓 Faculty Check: How do we prevent Data Leakage?"):
        st.markdown("""
        **Crucial viva answer:**

        We enforced a strict **temporal split** rather than a randomized train/test split:

        - **Feature Window (Oct 1 – Oct 20):** Features (sessions, carts, spend, recency) are computed exclusively from events before midnight on Oct 20.
        - **Target Window (Oct 21 – Oct 31):** The binary target (`will_purchase_next_10d`) indicates whether the customer purchased during Oct 21–31.

        The model never sees a single event from the future during feature generation. This guarantees zero future-data leakage.
        """)


# ═════════════════════════════════════════════════════════════════
# PAGE 2: UNDERSTANDING THE DATA
# ═════════════════════════════════════════════════════════════════
elif page == "🔍 Understanding the Data":
    st.markdown('<div class="section-header">🔍 Understanding the Data</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-card">
        <p><strong>The Clickstream Footprint:</strong></p>
        <p>Every interaction an eCommerce user takes generates an atomic telemetry event: a timestamp, event type (<code>view</code>, <code>cart</code>, <code>purchase</code>), product identifier, category breadcrumb, price, and visitor ID.</p>
    </div>
    """, unsafe_allow_html=True)

    # Clean Synthetic-Style Sample
    st.markdown("#### 📋 Atomic Event Record Example")
    sample_df = pd.DataFrame({
        'event_time': ['2019-10-01 09:15:22', '2019-10-01 09:15:45', '2019-10-01 09:16:10', '2019-10-01 09:18:33', '2019-10-01 09:20:01'],
        'event_type': ['view', 'view', 'cart', 'view', 'purchase'],
        'product_name': ['Samsung Galaxy S10', 'Phone Protective Case', 'Samsung Galaxy S10', 'Fast Charger Cable', 'Samsung Galaxy S10'],
        'category': ['electronics.smartphone', 'accessories.phone', 'electronics.smartphone', 'accessories.cable', 'electronics.smartphone'],
        'price_usd': [899.99, 24.99, 899.99, 14.50, 899.99],
        'user_id': [51234890, 51234890, 51234890, 51234890, 51234890]
    })
    st.dataframe(sample_df, hide_index=True, use_container_width=True)
    st.caption("A customer's shopping journey: browsing products → placing Galaxy S10 in cart → checking an accessory → completing checkout.")

    st.markdown("---")
    st.markdown("#### 🔽 The Macro eCommerce Conversion Funnel")

    st.markdown("""
    <div class="explain-card">
        <p><strong>Where does the drop-off occur?</strong></p>
        <p>In eCommerce, understanding the funnel stage with the steepest attrition determines where product and engineering resources should be deployed.</p>
    </div>
    """, unsafe_allow_html=True)

    funnel_stages = ['👁️ Product Views', '🛒 Added to Cart', '💳 Completed Purchases']
    funnel_counts = [2017907, 46256, 36675]

    fig_funnel = go.Figure(go.Funnel(
        y=funnel_stages,
        x=funnel_counts,
        textinfo="value+percent initial",
        marker=dict(color=['#6366f1', '#f59e0b', '#10b981']),
        connector=dict(line=dict(color="#cbd5e1", width=2))
    ))
    fig_funnel.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family="Inter", size=13)
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <p><strong>💡 Critical Discovery:</strong></p>
        <ul>
            <li><strong>Step 1 (View → Cart):</strong> 97.71% of viewers drop off before adding anything to cart! Only <strong>2.29%</strong> ever add to cart.</li>
            <li><strong>Step 2 (Cart → Purchase):</strong> Once an item enters the cart, <strong>79.28%</strong> proceed to complete the transaction!</li>
        </ul>
        <p><strong>Strategic Takeaway:</strong> The problem is NOT checkout abandonment; it is product discovery and purchase motivation. Optimization efforts must focus on search relevance, item recommendations, and browse engagement.</p>
    </div>
    """, unsafe_allow_html=True)

    # Category Revenue Breakdown
    if not data.get('funnel', pd.DataFrame()).empty:
        st.markdown("---")
        st.markdown("#### 🏷️ Top Product Categories by Gross Merchandise Value (GMV)")
        df_f = data['funnel'].sort_values('gmv', ascending=True).tail(8)

        fig_cat = go.Figure(go.Bar(
            y=df_f['category_level_1'],
            x=df_f['gmv'],
            orientation='h',
            marker=dict(
                color=df_f['gmv'],
                colorscale='Purples'
            ),
            text=[f"${v:,.0f}" for v in df_f['gmv']],
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Gross Revenue: $%{x:,.0f}<extra></extra>"
        ))
        fig_cat.update_layout(
            height=380,
            margin=dict(l=20, r=90, t=30, b=30),
            xaxis_title="Gross Merchandise Value ($ USD)",
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # Daily Traffic Pattern
    if not data.get('daily', pd.DataFrame()).empty:
        st.markdown("---")
        st.markdown("#### 📅 Daily Traffic Trajectory (October 2019)")
        df_d = data['daily']

        fig_daily = go.Figure()
        colors = ['#f59e0b' if w == 1 else '#6366f1' for w in df_d['is_weekend']]
        fig_daily.add_trace(go.Bar(
            x=df_d['full_date'],
            y=df_d['total_events'],
            marker_color=colors,
            hovertemplate="<b>%{x|%A, %b %d}</b><br>Traffic: %{y:,} events<extra></extra>"
        ))
        fig_daily.update_layout(
            title="Daily Website Interactions (Orange = Weekend Days)",
            xaxis_title="Date",
            yaxis_title="Total Events",
            height=350,
            margin=dict(l=20, r=20, t=50, b=30),
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_daily, use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PAGE 3: CUSTOMER SEGMENTS (Q1)
# ═════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segments (Q1)":
    st.markdown('<div class="section-header">👥 Customer Behavioral Segments (Q1)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-card">
        <p><strong>The Question:</strong> Who are our customers, and how do their behaviors naturally cluster without human bias?</p>
        <p><strong>The Technique:</strong> We engineered 25 behavioral dimensions per customer, applied <strong>Principal Component Analysis (PCA)</strong> to distill 80%+ variance into 5 orthogonal dimensions, and executed <strong>K-Means Clustering</strong> with $k=4$.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🎓 Faculty Check: Why apply PCA before K-Means?"):
        st.markdown("""
        **Crucial viva answer:**

        1. **Curse of Dimensionality:** In 25 dimensions, Euclidean distances between points converge, making clusters diffuse and meaningless.
        2. **Multicollinearity Removal:** Behavioral features like `view_count`, `total_events`, and `total_sessions` have high pairwise correlations. PCA projects them onto orthogonal principal axes, preventing redundant features from dominating distance calculations.
        3. **Computational Efficiency:** Clustering on 5 principal components is orders of magnitude faster while retaining 80.2% of the total dataset variance.
        """)

    if not data.get('segments', pd.DataFrame()).empty:
        df_seg = data['segments']

        # Accurate Persona Mapping by Cluster ID
        persona_info = {
            2: {
                "emoji": "🏆", "name": "Champions", "color": "#10b981", "badge": "High Value Spenders",
                "desc": "Elite 10.15% of users. High frequency, extensive browsing, and high average spend ($627). They achieve a massive 16.93% conversion rate."
            },
            3: {
                "emoji": "🔍", "name": "Active Explorers", "color": "#3b82f6", "badge": "High Engagement",
                "desc": "21.21% of users. Highly active browsing (~20 events across 13 unique products) with moderate spend and 4.90% conversion. Primed for conversion incentives."
            },
            1: {
                "emoji": "👀", "name": "Recent Window Shoppers", "color": "#f59e0b", "badge": "Warm Browsers",
                "desc": "32.67% of users. Visited recently (recency 3 days) with brief, light exploration. Low spend, 2.00% conversion. Responsive to cart nudges."
            },
            0: {
                "emoji": "💤", "name": "Dormant Bouncers", "color": "#ef4444", "badge": "Cold / Inactive",
                "desc": "Largest cohort (35.97%). High recency (13 days absent), bounced after 1–2 views, zero carts, and 1.28% conversion. Unprofitable for paid retargeting."
            }
        }

        st.markdown("### The Four Discovered Customer Personas")
        card_cols = st.columns(4)

        for col, cid in zip(card_cols, [2, 3, 1, 0]):
            seg_row = df_seg[df_seg['cluster'] == cid].iloc[0]
            info = persona_info[cid]

            with col:
                st.markdown(f"""
                <div class="persona-card" style="border-top: 5px solid {info['color']};">
                    <div class="emoji">{info['emoji']}</div>
                    <div class="name">{info['name']}</div>
                    <span class="badge" style="background: {info['color']}22; color: {info['color']};">{info['badge']}</span>
                    <div class="pct">{seg_row['pct_of_total']}% of Cohort ({int(seg_row['customer_count']):,} Users)</div>
                    <div class="conv" style="color: {info['color']};">{seg_row['next_10d_conversion_rate']}% Convert</div>
                    <div class="desc">{info['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("---")
        st.markdown("### 🔬 Interactive Segment Deep Dive")

        compare_metric = st.selectbox(
            "Select metric to compare across segments:",
            ["10-Day Conversion Rate (%)", "Average Spend ($ USD)", "Median Total Events", "Recency (Days Since Last Seen)", "Active Days Tenure"]
        )

        metric_map = {
            "10-Day Conversion Rate (%)": ("next_10d_conversion_rate", "Conversion Rate (%)"),
            "Average Spend ($ USD)": ("total_spend", "Average Spend ($ USD)"),
            "Median Total Events": ("total_events", "Events per Customer"),
            "Recency (Days Since Last Seen)": ("recency_days", "Recency (Days)"),
            "Active Days Tenure": ("tenure_days", "Tenure (Days)")
        }
        col_name, axis_label = metric_map[compare_metric]

        # Sorted by order: Champions, Active Explorers, Window Shoppers, Dormant Bouncers
        ordered_cids = [2, 3, 1, 0]
        plot_df = pd.DataFrame([
            {
                "Cluster Name": persona_info[cid]["name"],
                "Value": df_seg[df_seg['cluster'] == cid].iloc[0][col_name],
                "Color": persona_info[cid]["color"]
            }
            for cid in ordered_cids
        ])

        fig_compare = go.Figure(go.Bar(
            x=plot_df["Cluster Name"],
            y=plot_df["Value"],
            marker_color=plot_df["Color"],
            text=[f"{v:,.2f}" if v < 100 else f"{v:,.0f}" for v in plot_df["Value"]],
            textposition='outside'
        ))
        fig_compare.update_layout(
            yaxis_title=axis_label,
            height=380,
            margin=dict(l=20, r=20, t=30, b=30),
            font=dict(family="Inter", size=13)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # Behavioral Radar Chart
        st.markdown("### 🕸️ Multi-Dimensional Behavioral Fingerprint")
        st.markdown("Normalized 0–1 behavioral geometry illustrating cluster divergence.")

        radar_metrics = ['Recency Score', 'Events Volume', 'Spend Intensity', 'Cart Additions', 'Purchase Frequency', 'Product Breadth']
        fig_radar = go.Figure()

        for cid in ordered_cids:
            row = df_seg[df_seg['cluster'] == cid].iloc[0]
            vals = [
                1.0 - (row['recency_days'] / max(df_seg['recency_days'].max(), 1)), # inverted: higher is more recent
                row['total_events'] / max(df_seg['total_events'].max(), 1),
                row['total_spend'] / max(df_seg['total_spend'].max(), 1),
                row['cart_count'] / max(df_seg['cart_count'].max(), 1),
                row['purchase_count'] / max(df_seg['purchase_count'].max(), 1),
                row['n_unique_products'] / max(df_seg['n_unique_products'].max(), 1)
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_metrics + [radar_metrics[0]],
                fill='toself',
                name=persona_info[cid]["name"],
                line_color=persona_info[cid]["color"],
                opacity=0.65
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            height=450,
            font=dict(family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PAGE 4: PURCHASE PREDICTOR (Q2)
# ═════════════════════════════════════════════════════════════════
elif page == "🎯 Purchase Predictor (Q2)":
    st.markdown('<div class="section-header">🎯 Purchase Intent Predictor (Q2)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-card">
        <p><strong>The Question:</strong> Given a user's browsing activities over the last 20 days, what is the probability they will make a purchase in the subsequent 10 days?</p>
        <p><strong>Business Value:</strong> Rather than blasting generic discounts to everyone, the business identifies high-intent customers for precision closing and suppresses ads for low-intent bouncers.</p>
    </div>
    """, unsafe_allow_html=True)

    # Model Evaluation Comparison
    if not data.get('models', pd.DataFrame()).empty:
        st.markdown("### 🏅 Model Benchmark Comparison")
        st.markdown("Evaluated on out-of-sample holdout test partition (22,192 customers).")

        df_m = data['models']
        c_m1, c_m2 = st.columns([1.2, 1])

        with c_m1:
            fig_models = go.Figure(go.Bar(
                x=df_m['model'],
                y=df_m['pr_auc'],
                marker_color=['#10b981', '#3b82f6', '#f59e0b', '#94a3b8'],
                text=[f"{v:.4f}" for v in df_m['pr_auc']],
                textposition='outside'
            ))
            fig_models.update_layout(
                yaxis_title="PR-AUC Score (Higher is Better)",
                height=350,
                margin=dict(l=20, r=20, t=30, b=30),
                font=dict(family="Inter")
            )
            st.plotly_chart(fig_models, use_container_width=True)

        with c_m2:
            st.markdown("""
            <div class="viva-card">
                <h4>🎓 Why PR-AUC Over ROC-AUC / Accuracy?</h4>
                <p><strong>1. The Accuracy Paradox:</strong> Baseline buyers are only 3.87%. A trivial model predicting "Nobody Buys" achieves <strong>96.13% accuracy</strong>, but is completely useless!</p>
                <p><strong>2. ROC-AUC is Deluded:</strong> ROC-AUC includes False Positives divided by the massive True Negative class (~96%), making even mediocre models look artificially great (~0.80).</p>
                <p><strong>3. PR-AUC is Honest:</strong> Evaluates Precision and Recall exclusively on the positive minority class. Our Champion achieves <strong>0.2450 PR-AUC</strong> (6.3× higher than 0.0387 baseline).</p>
            </div>
            """, unsafe_allow_html=True)

    # Threshold Optimization Curve
    if not data.get('threshold', pd.DataFrame()).empty:
        st.markdown("---")
        st.markdown("### 📈 Decision Threshold & Business Profit Optimization")
        st.markdown("Machine learning outputs probabilities $\hat{p} \in [0, 1]$. But what cut-off threshold $\\tau$ maximizes commercial profitability?")

        df_t = data['threshold']
        best_row = df_t.loc[df_t['expected_net_profit'].idxmax()]

        fig_thresh = go.Figure()
        fig_thresh.add_trace(go.Scatter(
            x=df_t['threshold'], y=df_t['expected_net_profit'],
            mode='lines', line=dict(color='#10b981', width=3),
            name="Expected Net Profit ($ USD)"
        ))
        fig_thresh.add_vline(
            x=best_row['threshold'], line=dict(color='#ef4444', dash='dash', width=2),
            annotation_text=f"Optimal τ* = {best_row['threshold']:.2f} (+${best_row['expected_net_profit']:,.0f})",
            annotation_position="top left"
        )
        fig_thresh.update_layout(
            title="Expected Net Profit vs Classification Threshold (Campaign Cost: $1/target, Value: $25/converted buyer)",
            xaxis_title="Classification Probability Threshold (τ)",
            yaxis_title="Expected Net Profit ($ USD)",
            height=360,
            margin=dict(l=20, r=20, t=50, b=30),
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_thresh, use_container_width=True)

        st.markdown(f"""
        <div class="insight-box">
            <p><strong>💡 Business Profit Optimum:</strong> Setting the operational threshold to <strong>τ* = {best_row['threshold']:.2f}</strong> targets {int(best_row['targeted_users']):,} prospective buyers, yielding <strong>+${best_row['expected_net_profit']:,.0f} in net profit</strong>.</p>
            <p>Setting τ too low (&lt;0.15) loses money due to excessive targeting costs, while setting τ too high (&gt;0.75) starves conversions.</p>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Scoring Simulator
    st.markdown("---")
    st.markdown("### 🕹️ Live Customer Intent Scoring Simulator")
    st.markdown("Select a persona archetype or adjust behavioral sliders to score a prospective visitor:")

    if model is not None:
        preset_choice = st.radio(
            "Select Preset Archetype:",
            ["🏆 Champion Customer", "🔍 Active Explorer", "👀 Window Shopper", "💤 Dormant Bouncer", "✏️ Custom Sliders"],
            horizontal=True
        )

        preset_params = {
            "🏆 Champion Customer":  dict(rec=1, ten=18, act=10, sess=15, evts=80, views=60, carts=8, purch=5, spend=800, maxp=1200, uprods=25, wknd=0.30, clust=2),
            "🔍 Active Explorer":    dict(rec=3, ten=12, act=5, sess=8, evts=35, views=30, carts=3, purch=1, spend=120, maxp=400, uprods=15, wknd=0.40, clust=3),
            "👀 Window Shopper":     dict(rec=2, ten=4, act=2, sess=2, evts=8, views=7, carts=1, purch=0, spend=0, maxp=200, uprods=5, wknd=0.50, clust=1),
            "💤 Dormant Bouncer":    dict(rec=15, ten=15, act=1, sess=1, evts=2, views=2, carts=0, purch=0, spend=0, maxp=50, uprods=1, wknd=0.0, clust=0),
        }

        p = preset_params.get(preset_choice, preset_params["👀 Window Shopper"])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📅 Temporal Activity**")
            in_rec = st.slider("Recency (Days since last visit)", 0, 20, p['rec'])
            in_ten = st.slider("Tenure (Days since first visit)", 0, 20, p['ten'])
            in_act = st.slider("Active Days", 1, 15, p['act'])
        with c2:
            st.markdown("**🖱️ Browse Behavior**")
            in_sess = st.slider("Total Sessions", 1, 25, p['sess'])
            in_evts = st.slider("Total Clickstream Events", 1, 150, p['evts'])
            in_views = st.slider("Product Views", 1, 120, p['views'])
            in_uprods = st.slider("Unique Products Viewed", 1, 40, p['uprods'])
        with c3:
            st.markdown("**💳 Purchase & Monetary Signals**")
            in_carts = st.slider("Cart Additions", 0, 15, p['carts'])
            in_purch = st.slider("Historical Purchases", 0, 10, p['purch'])
            in_spend = st.number_input("Historical Spend ($ USD)", 0.0, 5000.0, float(p['spend']), step=50.0)
            in_maxp = st.number_input("Max Product Price Viewed ($ USD)", 0.0, 3000.0, float(p['maxp']), step=50.0)

        cart_to_view = in_carts / max(1, in_views)
        events_per_sess = in_evts / max(1, in_sess)
        avg_purch = in_spend / max(1, in_purch) if in_purch > 0 else 0.0
        cluster_id = p['clust']

        # Live Inference Execution
        raw_feat = {
            "recency_days": in_rec, "tenure_days": in_ten, "active_days": in_act,
            "total_sessions": np.log1p(in_sess), "total_events": np.log1p(in_evts),
            "view_count": np.log1p(in_views), "cart_count": np.log1p(in_carts),
            "purchase_count": np.log1p(in_purch), "total_spend": np.log1p(in_spend),
            "avg_purchase_price": np.log1p(avg_purch),
            "max_price_interacted": np.log1p(in_maxp),
            "n_unique_products": np.log1p(in_uprods),
            "n_unique_cat1": 2, "n_unique_brands": 3,
            "weekend_event_ratio": p.get('wknd', 0.35),
            "cart_to_view_ratio": cart_to_view,
            "events_per_session": events_per_sess,
            "cluster_0": 1.0 if cluster_id == 0 else 0.0,
            "cluster_1": 1.0 if cluster_id == 1 else 0.0,
            "cluster_2": 1.0 if cluster_id == 2 else 0.0,
            "cluster_3": 1.0 if cluster_id == 3 else 0.0,
        }

        input_df = pd.DataFrame([raw_feat])[feature_names]
        input_scaled = scaler.transform(input_df)
        prob = float(model.predict_proba(input_scaled)[0, 1])

        st.markdown("")
        res_col1, res_col2 = st.columns([1, 1.4])

        with res_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%', 'font': {'size': 44, 'family': 'Inter'}},
                title={'text': "Predicted 10-Day Purchase Probability", 'font': {'size': 15}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': '#10b981' if prob >= 0.70 else '#f59e0b' if prob >= 0.35 else '#ef4444'},
                    'steps': [
                        {'range': [0, 35], 'color': '#fee2e2'},
                        {'range': [35, 70], 'color': '#fef3c7'},
                        {'range': [70, 100], 'color': '#d1fae5'},
                    ],
                    'threshold': {'line': {'color': '#0f172a', 'width': 3}, 'thickness': 0.8, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), font=dict(family="Inter"))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            if prob >= 0.70:
                st.markdown(f"""
                <div class="insight-box" style="margin-top: 1.5rem;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #047857; margin-bottom: 0.5rem;">🎯 HIGH PURCHASE INTENT ({prob*100:.1f}%)</div>
                    <p><strong>Segment Classification:</strong> Highly engaged prospective buyer.</p>
                    <p><strong>Prescribed Action:</strong> Deploy an immediate time-limited closing offer or VIP customer support assistance within 24 hours. They have already qualified themselves.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prob >= 0.35:
                st.markdown(f"""
                <div class="attention-box" style="margin-top: 1.5rem;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #b45309; margin-bottom: 0.5rem;">🛒 MODERATE PURCHASE INTENT ({prob*100:.1f}%)</div>
                    <p><strong>Segment Classification:</strong> Active consideration / Cart browser.</p>
                    <p><strong>Prescribed Action:</strong> Trigger an automated abandoned-cart reminder with dynamic product recommendations or social proof reviews to overcome hesitation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 1.15rem 1.35rem; border-radius: 0 12px 12px 0; margin-top: 1.5rem; color: #991b1b; line-height: 1.6;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #b91c1c; margin-bottom: 0.5rem;">💤 LOW PURCHASE INTENT ({prob*100:.1f}%)</div>
                    <p style="margin: 0.45rem 0;"><strong>Segment Classification:</strong> Cold / Inactive browser.</p>
                    <p style="margin: 0.45rem 0;"><strong>Prescribed Action:</strong> Suppress paid retargeting ad expenditure. Maintain low-cost engagement exclusively through weekly batch email newsletters.</p>
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# PAGE 5: ARE SEGMENTS REAL? (Q3)
# ═════════════════════════════════════════════════════════════════
elif page == "📊 Are Segments Real? (Q3)":
    st.markdown('<div class="section-header">📊 Statistical Significance & Hypothesis Testing (Q3)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-card">
        <p><strong>The Scientific Question:</strong> Are the observed conversion gaps between segments statistically significant, or could they have arisen by random sampling noise?</p>
        <p><strong>Null Hypothesis ($H_0$):</strong> All four segments share an identical population conversion probability ($p_0 = p_1 = p_2 = p_3 = 3.87\%$).</p>
        <p><strong>Alternative Hypothesis ($H_1$):</strong> At least one segment exhibits a statistically distinct conversion rate.</p>
    </div>
    """, unsafe_allow_html=True)

    # Core Test Statistics Cards
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card">
            <div class="value" style="color: #10b981;">χ² = 6,296</div>
            <div class="label">Chi-Square Statistic</div>
            <div class="delta">Degrees of Freedom: 3</div>
        </div>
        <div class="metric-card">
            <div class="value" style="color: #10b981;">p < 10⁻³⁰⁰</div>
            <div class="label">Asymptotic P-Value</div>
            <div class="delta">Far Below α = 0.05</div>
        </div>
        <div class="metric-card">
            <div class="value" style="color: #10b981;">V = 0.238</div>
            <div class="label">Cramér's V (Effect Size)</div>
            <div class="delta">Medium-Strong Effect</div>
        </div>
        <div class="metric-card">
            <div class="value" style="color: #10b981;">6 / 6 Pairs</div>
            <div class="label">Bonferroni Pairs Reject H₀</div>
            <div class="delta">Zero Spurious Findings</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <p><strong>✅ Statistical Verdict: Null Hypothesis Overwhelmingly Rejected.</strong></p>
        <p>The Chi-Square test yielding $\chi^2 = 6,296.2$ with $p \approx 0$ proves the behavioral divergence is mathematically impossible to produce by random fluctuation.</p>
        <p>Furthermore, <strong>Cramér's V = 0.238</strong> verifies this is not merely a tiny statistical artifact driven by a large sample size ($N = 110,956$), but a substantial real-world commercial effect.</p>
    </div>
    """, unsafe_allow_html=True)

    # Pairwise Comparisons Table
    if not data.get('pairwise', pd.DataFrame()).empty:
        st.markdown("---")
        st.markdown("### 🔬 Post-Hoc Pairwise 2-Sample Z-Tests (Multiple Comparison Corrections)")
        st.markdown("When conducting 6 simultaneous pairwise tests, the chance of a false positive inflates to $\\alpha_{FWER} = 1 - (1-0.05)^6 = 26.5\\%$. We applied **Bonferroni ($\alpha/6 = 0.0083$)** and **Benjamini-Hochberg FDR** corrections.")

        df_p = data['pairwise']
        display_pairs = []
        for _, r in df_p.iterrows():
            display_pairs.append({
                "Pairwise Comparison": r['pair'],
                "Group 1 Rate": f"{r['p1_rate_pct']:.2f}%",
                "Group 2 Rate": f"{r['p2_rate_pct']:.2f}%",
                "Delta (p2 - p1)": f"{r['diff_pct']:+.2f}%",
                "Z-Statistic": f"{r['z_statistic']:.2f}",
                "Adjusted p-val (Bonf)": "< 10⁻¹⁴" if r['p_val_bonferroni'] == 0 else f"{r['p_val_bonferroni']:.2e}",
                "Reject H₀?": "✅ Significant" if r['reject_bonferroni'] else "❌ Insignificant"
            })
        st.dataframe(pd.DataFrame(display_pairs), hide_index=True, use_container_width=True)

    # Tukey Spend Table
    if not data.get('tukey', pd.DataFrame()).empty:
        st.markdown("---")
        st.markdown("### 💰 Tukey HSD Post-Hoc Test on Customer Spend ($\log(1 + Spend)$)")
        st.markdown("ANOVA ($F = 72,139$, $p \\approx 0$) followed by Tukey Honest Significant Differences (HSD) test across all segment pairs:")

        df_t = data['tukey']
        tukey_disp = []
        for _, r in df_t.iterrows():
            tukey_disp.append({
                "Group 1": r['group1_name'],
                "Group 2": r['group2_name'],
                "Mean Diff (Log Spend)": f"{r['meandiff']:+.4f}",
                "Adjusted p-value": f"{r['p-adj']:.4f}",
                "95% CI Lower": f"{r['lower']:+.4f}",
                "95% CI Upper": f"{r['upper']:+.4f}",
                "Statistically Different?": "✅ Significant" if r['reject'] else "➖ Negligible"
            })
        st.dataframe(pd.DataFrame(tukey_disp), hide_index=True, use_container_width=True)
        st.caption("Key insight: Champions spend drastically more than all three other cohorts ($p < 0.001$). The other three segments show negligible spending differences between each other.")


# ═════════════════════════════════════════════════════════════════
# PAGE 6: TRAFFIC FORECASTING (Q4)
# ═════════════════════════════════════════════════════════════════
elif page == "📈 Traffic Forecasting (Q4)":
    st.markdown('<div class="section-header">📈 Time Series Traffic Forecasting (Q4)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-card">
        <p><strong>The Question:</strong> What will platform traffic look like over the next 7 days, and can engineering proactively plan server auto-scaling?</p>
        <p><strong>The Finding:</strong> eCommerce clickstream events follow a predictable 7-day cyclical heartbeat, surging during weekends and troughing mid-week.</p>
    </div>
    """, unsafe_allow_html=True)

    if not data.get('daily', pd.DataFrame()).empty:
        df_d = data['daily']

        # Weekly Cyclical Rhythm Bar Chart
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_avg = df_d.groupby('day_name')['total_events'].mean().reindex(dow_order)

        fig_dow = go.Figure(go.Bar(
            x=dow_order,
            y=dow_avg.values,
            marker_color=['#64748b', '#64748b', '#64748b', '#64748b', '#3b82f6', '#f59e0b', '#10b981'],
            text=[f"{v:,.0f}" for v in dow_avg.values],
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Average Traffic: %{y:,.0f} events<extra></extra>"
        ))
        fig_dow.update_layout(
            title="Weekly Traffic Heartbeat (Average Events by Day of Week)",
            yaxis_title="Average Daily Events",
            height=360,
            margin=dict(l=20, r=20, t=50, b=30),
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_dow, use_container_width=True)

        # 7-Day Holdout Forecast Chart
        st.markdown("---")
        st.markdown("### 🏆 Out-of-Sample Holdout Forecast (Oct 25 – Oct 31)")
        st.markdown("Trained on Days 1–24 (Oct 1–24), evaluated on unseen Days 25–31 (Oct 25–31).")

        train_df = df_d.iloc[:24]
        test_df = df_d.iloc[24:]

        # SARIMA holdout predictions from Phase 7 benchmark
        sarima_preds = [65778, 65337, 65515, 64574, 65004, 64448, 63676]

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=train_df['full_date'], y=train_df['total_events'],
            mode='lines+markers', name="Historical Training (Days 1–24)",
            line=dict(color='#94a3b8', width=2)
        ))
        fig_ts.add_trace(go.Scatter(
            x=test_df['full_date'], y=test_df['total_events'],
            mode='lines+markers', name="Actual Test Ground Truth (Days 25–31)",
            line=dict(color='#0f172a', width=3),
            marker=dict(size=7)
        ))
        fig_ts.add_trace(go.Scatter(
            x=test_df['full_date'], y=sarima_preds,
            mode='lines+markers', name="SARIMA Champion Forecast (MAPE: 3.99%)",
            line=dict(color='#10b981', width=3, dash='dash'),
            marker=dict(symbol='square', size=7)
        ))
        fig_ts.add_vline(
            x=pd.to_datetime('2019-10-24 12:00:00').timestamp() * 1000,
            line=dict(color='gray', dash='dot', width=2),
            annotation_text="Train / Test Split",
            annotation_position="top left"
        )
        fig_ts.update_layout(
            title="Daily Traffic: Actual vs SARIMA(1,1,1)×(1,0,0)₇ Predictions",
            xaxis_title="Date (October 2019)",
            yaxis_title="Total Clickstream Interactions",
            height=420,
            margin=dict(l=20, r=20, t=50, b=30),
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # Forecast Models Benchmark Table
    if not data.get('forecast', pd.DataFrame()).empty:
        st.markdown("### 📊 Forecasting Model Evaluation Benchmark")
        df_fc = data['forecast']

        fc_disp = []
        for _, r in df_fc.iterrows():
            fc_disp.append({
                "Forecasting Architecture": r['model'],
                "Holdout MAPE (%)": f"{r['mape_pct']:.2f}%",
                "Accuracy (%)": f"{100 - r['mape_pct']:.2f}%",
                "Mean Absolute Error (MAE)": f"±{r['mae']:,.0f} events",
                "Root Mean Squared Error (RMSE)": f"{r['rmse']:,.0f}"
            })
        st.dataframe(pd.DataFrame(fc_disp), hide_index=True, use_container_width=True)

    with st.expander("🎓 Faculty Check: Why did SARIMA outperform ARIMA and Holt-Winters?"):
        st.markdown("""
        **Crucial viva answer:**

        1. **Weekly Seasonality Period $s = 7$:** Standard ARIMA only tracks consecutive day lags ($t-1, t-2$). It cannot capture the recurring weekly spike that occurs every 7 days without massive order inflation.
        2. **Multiplicative SARIMA formulation:** $\\text{SARIMA}(1,1,1) \\times (1,0,0)_7$ combines non-seasonal AR/MA momentum with a 7-day seasonal autoregressive term ($P=1, s=7$).
        3. **Stationarity:** Augmented Dickey-Fuller (ADF) test on raw traffic yielded $p = 0.21$ (non-stationary). Applying first differencing ($d=1$) achieved strict stationarity ($p < 0.01$), allowing the seasonal filters to capture true variance.
        """)


# ═════════════════════════════════════════════════════════════════
# PAGE 7: BUDGET OPTIMIZER (Q5)
# ═════════════════════════════════════════════════════════════════
elif page == "💰 Budget Optimizer (Q5)":
    st.markdown('<div class="section-header">💰 Marketing Budget Optimization (Q5)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-card">
        <p><strong>The Problem:</strong> Given a fixed marketing budget $B$, how should capital be allocated across the four customer segments to maximize total commercial revenue?</p>
        <p><strong>The Economic Reality:</strong> Each segment exhibits diminishing marginal returns modeled by concave response curves:</p>
        <p style="text-align: center; margin: 0.6rem 0;">$$R_i(b_i) = \\alpha_i \\cdot (1 - e^{-\\beta_i b_i})$$</p>
        <p>where $\\alpha_i$ is maximum saturation revenue and $\\beta_i$ is customer acquisition efficiency.</p>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Budget Simulator
    st.markdown("### 🎛️ Interactive Budget Allocation Engine")
    budget = st.slider(
        "Total Marketing Campaign Budget ($ USD):",
        min_value=1000, max_value=50000, value=10000, step=1000
    )

    alphas = np.array([4000.0, 5000.0, 25000.0, 12000.0])
    betas = np.array([0.0005, 0.0010, 0.0080, 0.0025])
    seg_names = ["💤 Dormant Bouncers", "👀 Window Shoppers", "🏆 Champions", "🔍 Active Explorers"]
    seg_colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6']

    b_opt, lam_star = solve_lagrange_budget(alphas, betas, float(budget))
    r_opt = revenue_function(b_opt, alphas, betas)
    net_profit = float(np.sum(r_opt) - budget)
    roi = (net_profit / budget) * 100

    # Benchmark: Equal Split Strategy
    b_eq = np.full(4, budget / 4.0)
    r_eq = revenue_function(b_eq, alphas, betas)
    profit_eq = float(np.sum(r_eq) - budget)
    lift = net_profit - profit_eq

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="value">${budget:,}</div>
            <div class="label">Campaign Budget</div>
        </div>
        <div class="metric-card">
            <div class="value" style="color: #10b981;">${net_profit:,.0f}</div>
            <div class="label">Optimized Net Profit</div>
            <div class="delta">ROI: {roi:,.0f}%</div>
        </div>
        <div class="metric-card">
            <div class="value" style="color: #64748b;">${profit_eq:,.0f}</div>
            <div class="label">Equal Split Profit</div>
        </div>
        <div class="metric-card">
            <div class="value" style="color: #10b981;">+${lift:,.0f}</div>
            <div class="label">Optimization Lift</div>
            <div class="delta">Extra Profit vs Naive Split</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    alloc_c1, alloc_c2 = st.columns([1.2, 1])

    with alloc_c1:
        alloc_df = pd.DataFrame({
            'Segment': seg_names,
            'Budget': b_opt,
            'Revenue': r_opt,
            'Color': seg_colors
        })

        fig_alloc = go.Figure(go.Bar(
            x=alloc_df['Segment'],
            y=alloc_df['Budget'],
            marker_color=alloc_df['Color'],
            text=[f"${v:,.0f}" for v in alloc_df['Budget']],
            textposition='outside'
        ))
        fig_alloc.update_layout(
            title="Optimal Budget Distribution ($ USD)",
            yaxis_title="Allocated Capital ($ USD)",
            height=370,
            margin=dict(l=20, r=20, t=50, b=30),
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    with alloc_c2:
        non_zero = b_opt > 0.01
        fig_pie = go.Figure(go.Pie(
            labels=[seg_names[i] for i in range(4) if non_zero[i]],
            values=b_opt[non_zero],
            marker_colors=[seg_colors[i] for i in range(4) if non_zero[i]],
            hole=0.45,
            textinfo='label+percent'
        ))
        fig_pie.update_layout(
            title="Capital Share (%)",
            height=370,
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(family="Inter"),
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(f"""
    <div class="attention-box">
        <p><strong>⚠️ Why do Dormant Bouncers receive ${b_opt[0]:.0f}?</strong></p>
        <p>This is a fundamental triumph of mathematical optimization over human intuition:</p>
        <p>Under the <strong>Karush-Kuhn-Tucker (KKT) stationarity condition</strong>, capital must only be allocated where the marginal return exceeds the shadow price $\lambda^* = {lam_star:.2f}$.</p>
        <ul>
            <li>Champions produce <strong>$200 in marginal return</strong> on their first dollar.</li>
            <li>Dormant Bouncers produce only <strong>$2.00 in marginal return</strong> on their first dollar.</li>
        </ul>
        <p>Because Dormant Bouncers' maximum marginal return ($2.00) is strictly lower than the shadow price $\lambda^* = {lam_star:.2f}$, <strong>every single dollar redirected away from Bouncers into Champions or Explorers generates superior net revenue</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    # Budget Scenarios Curve
    if not data.get('scenarios', pd.DataFrame()).empty:
        st.markdown("---")
        st.markdown("### 📊 Budget Allocation Trajectory Across Scale ($5K → $100K)")
        df_sc = data['scenarios']

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=df_sc['campaign_budget'], y=df_sc['alloc_champions'], mode='lines+markers', name="Champions", line=dict(color='#10b981', width=3)))
        fig_sc.add_trace(go.Scatter(x=df_sc['campaign_budget'], y=df_sc['alloc_active_explorers'], mode='lines+markers', name="Active Explorers", line=dict(color='#3b82f6', width=3)))
        fig_sc.add_trace(go.Scatter(x=df_sc['campaign_budget'], y=df_sc['alloc_window_shoppers'], mode='lines+markers', name="Window Shoppers", line=dict(color='#f59e0b', width=2)))
        fig_sc.add_trace(go.Scatter(x=df_sc['campaign_budget'], y=df_sc['alloc_dormant_bouncers'], mode='lines+markers', name="Dormant Bouncers", line=dict(color='#ef4444', width=2)))

        fig_sc.update_layout(
            title="How Capital Flow Transitions Across Scaling Budget Scenarios",
            xaxis_title="Total Campaign Budget ($ USD)",
            yaxis_title="Allocated Budget ($ USD)",
            height=380,
            margin=dict(l=20, r=20, t=50, b=30),
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("Notice how Dormant Bouncers and Window Shoppers receive $0 at lower budgets, and only receive residual capital when Champions approach diminishing return saturation.")


# ═════════════════════════════════════════════════════════════════
# PAGE 8: VIVA DEFENSE & PRESENTATION CHEAT SHEET
# ═════════════════════════════════════════════════════════════════
elif page == "🎓 Viva Defense Cheat Sheet":
    st.markdown("""
    <div class="hero">
        <h1>🎓 Faculty Viva Defense & Presentation Cheat Sheet</h1>
        <p>Master the project narrative, anticipate rigorous professor questions, and deliver crisp, confident explanations.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🗣️ 60-Second Elevator Pitch & 5-Min Walkthrough", "❓ Top 8 Viva Questions & Prepared Answers", "📖 Formula & Jargon Decoder"])

    with tab1:
        st.markdown("### 🎙️ The 60-Second Executive Summary")
        st.markdown("""
        <div class="viva-card">
            <p><em>"Good morning professors. This project addresses the central business challenge of modern eCommerce: turning massive, unstructured clickstream data into profitable marketing decisions."</em></p>
            <p><em>"Using 42.4 million real events from October 2019, we engineered a leak-free 4-layer medallion pipeline that profiled 110,956 unique customers. We then answered five core questions:"</em></p>
            <ul>
                <li><strong>1. Behavioral Segmentation:</strong> Unsupervised PCA + K-Means discovered 4 distinct behavioral segments, from high-converting Champions (16.9%) to cold Bouncers (1.3%).</li>
                <li><strong>2. Purchase Intent:</strong> Supervised classification predicted 10-day forward purchase intent, outperforming random guessing by 6.3× PR-AUC.</li>
                <li><strong>3. Inferential Testing:</strong> Statistical hypothesis testing (Chi-Square and Bonferroni-corrected Z-tests) proved these segment differences are overwhelmingly genuine ($p &lt; 10^{-14}$).</li>
                <li><strong>4. Traffic Forecasting:</strong> Time series SARIMA modeling accurately forecasted 7-day platform traffic with 96% holdout accuracy (3.99% MAPE).</li>
                <li><strong>5. Capital Optimization:</strong> And Lagrange multiplier optimization mathematically solved the marketing budget allocation, unlocking thousands in profit lift over naive equal splits.</li>
            </ul>
            <p><em>"Every phase connects seamlessly into an end-to-end decision intelligence platform."</em></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⏱️ The 5-Minute Live Viva Walkthrough Script")
        st.markdown("""
        Follow this exact screen-by-screen presentation flow:

        - **Minute 1 — Project Overview:** Open on Page 1. Emphasize the **42.4M event scale**, the **5% user-hash sampling** (keeping customer journeys intact), and the **leak-free temporal split** (Oct 1–20 features vs Oct 21–31 labels).
        - **Minute 2 — Customer Segments (Q1):** Navigate to Page 3. Point out the **Persona Cards**. Highlight that Champions represent only 10% of users but convert at 16.93%, while Bouncers make up 36% but convert at only 1.28%. Show the radar chart to prove multi-dimensional separation.
        - **Minute 3 — Purchase Predictor (Q2):** Go to Page 4. Show the **PR-AUC benchmark** and explain why accuracy is meaningless on 3.87% imbalanced data. Slide the preset to 'Champion' and show the 96%+ score, then switch to 'Bouncer' and show the low score. Show the profit optimization curve peaking at $\\tau^* = 0.50$.
        - **Minute 4 — Are Segments Real? (Q3) & Traffic (Q4):** Glance at Page 5 to highlight the $\chi^2 = 6,296$ and Cramér's V = 0.24 (proving non-randomness). Flip to Page 6 to show the **7-day weekly traffic heartbeat** and the SARIMA holdout forecast curve (3.99% error).
        - **Minute 5 — Budget Optimizer (Q5):** Finish on Page 7. Move the budget slider. Point out why the mathematical solution gives **$0 to Dormant Bouncers** ("Don't water dead plants when your roses are thirsty"). Show the **+$1,800 to +$3,500 profit lift** over equal split.
        """)

    with tab2:
        st.markdown("### ❓ Top 8 Tough Viva Questions & Prepared Answers")

        questions = [
            ("Q1. Why did you sample 5% of users instead of 5% of rows?",
             "If you sample rows, you fragment individual shopping sessions. A user with 20 actions would only have 1 random action left, destroying session depth and cart-to-view ratios. Hashing user_id (user_id % 100 < 5) preserves 100% of the sequential journey for every sampled customer."),

            ("Q2. Why did you enforce a temporal split rather than random train_test_split?",
             "Random splitting in time series causes severe data leakage. If we use future purchases to predict past browsing, the model memorizes the timeline. A temporal split (Oct 1–20 for features, Oct 21–31 for labels) mimics real-world deployment where you must forecast future actions using only past history."),

            ("Q3. Why is PR-AUC better than Accuracy or ROC-AUC for this project?",
             "Only 3.87% of visitors convert. A naive model predicting 'no purchase' for everyone achieves 96.13% accuracy while catching zero buyers. ROC-AUC is flattered by the huge True Negative count. PR-AUC evaluates Precision and Recall exclusively on the positive minority class of interest."),

            ("Q4. How did you determine K=4 for customer clustering?",
             "We ran K-Means for k=2 through k=8 on the 5 PCA components. The Elbow curve showed an inflection point at k=4, and the Davies-Bouldin index dropped significantly. Furthermore, k=4 produced four clean, actionable commercial personas (Champions, Explorers, Shoppers, Bouncers) without splintering into redundant sub-clusters."),

            ("Q5. Why did the budget optimizer give $0 to Dormant Bouncers?",
             "Under Karush-Kuhn-Tucker (KKT) optimality, capital is only allocated where marginal revenue exceeds the shadow price λ*. Champions generate ~$200 in marginal revenue on their first dollar, while Bouncers generate only $2.00. Because Bouncers never surpass the system shadow price, spending on them yields an opportunity loss."),

            ("Q6. What is the difference between statistical significance (p-value) and practical significance (effect size)?",
             "With large sample sizes (N = 110,956), even trivial differences produce tiny p-values. Cramér's V measures effect magnitude independent of sample size. Our V = 0.238 confirms a medium-strong commercial effect, proving the segments truly exhibit distinct behavior."),

            ("Q7. Why did you apply Bonferroni and FDR corrections in pairwise tests?",
             "Testing 6 segment pairs simultaneously inflates the Family-Wise Error Rate (FWER) to 26.5% at α=0.05. Bonferroni divides α by 6 (α_adj = 0.0083). All 6 pairs still rejected H0 with p < 10⁻¹⁴, proving zero false positive discoveries."),

            ("Q8. Why does SARIMA beat standard ARIMA and Holt-Winters for traffic forecasting?",
             "Platform traffic has a prominent 7-day cyclical seasonality (weekend surges). Standard ARIMA only captures consecutive lag momentum (t-1, t-2). SARIMA incorporates a seasonal autoregressive filter (s=7) that explicitly models Sunday-to-Sunday and Friday-to-Friday patterns, yielding 3.99% MAPE.")
        ]

        for q, a in questions:
            with st.expander(f"📌 {q}"):
                st.markdown("**How to answer in 30 seconds:**")
                st.markdown(a)

    with tab3:
        st.markdown("### 📖 Plain-English Decoder for Formulas & Terms")

        st.markdown("""
        | Mathematical Term | Plain English Meaning | Why It Matters In Viva |
        | :--- | :--- | :--- |
        | **$\lambda^*$ (Shadow Price)** | The extra revenue earned if budget increases by $1 | At optimum, every funded segment has this exact marginal return. |
        | **PR-AUC** | Area under Precision-Recall Curve | The gold standard evaluation for imbalanced classification (3.87% base). |
        | **Cramér's V** | Effect size metric for categorical associations | Proves segment conversion differences are substantial (V = 0.24), not just p-hacked. |
        | **Bonferroni $\\alpha / m$** | Dividing significance threshold by number of tests | Prevents false discoveries when doing multiple hypothesis tests (6 pairs). |
        | **$\text{SARIMA}(1,1,1)(1,0,0)_7$** | AR(1) + differencing + MA(1) with 7-day seasonal lag | Captures both day-to-day momentum and weekly weekend peaks. |
        | **Temporal Split** | Training on past (Oct 1–20), evaluating on future (Oct 21–31) | Guarantees zero data leakage; mimics live production deployment. |
        | **PCA (5 Components)** | Rotating 25 features to find the 5 strongest variance axes | Overcomes the curse of dimensionality and multicollinearity in K-Means. |
        """)


# ─────────────────────────────────────────────────────────────────
# PERSISTENT FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.82rem; padding: 1.2rem 0; line-height: 1.6;">
    <div>Customer Behavioral Segmentation & Purchase Intent Prediction — Data Science Capstone Project</div>
    <div style="color: #cbd5e1; font-size: 0.78rem;">REES46 eCommerce Behavior Dataset · October 2019 · Python, Streamlit, Scikit-Learn, Statsmodels</div>
</div>
""", unsafe_allow_html=True)
