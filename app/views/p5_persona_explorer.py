"""
app/views/p5_persona_explorer.py
================================
Page 5: Persona Explorer & Customer Behavioral Map.
Explores the 4 behavioral customer personas, confidence intervals, 
head-to-head persona comparisons, and an interactive 2D PCA/SVD map.
Strict Dual-Tab layout: Result (Profiles & 2D Map) vs The Science (Inferential Stats & SVD).
"""

import json
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

APP_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "app_data"


@st.cache_data
def load_p5_data():
    with open(APP_DATA_DIR / "persona_profiles.json") as f:
        profiles = json.load(f)
    warehouse = pd.read_parquet(APP_DATA_DIR / "scored_customers_warehouse.parquet")
    return profiles, warehouse


def render_p5():
    profiles, warehouse = load_p5_data()

    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <span class="brand-pill">📊 Unsupervised Segmentation</span>
        <h2 style="font-size: 1.65rem; font-weight: 800; color: #0f172a; margin: 0.3rem 0;">
            Persona Explorer & 2D Behavioral Customer Map
        </h2>
        <p style="color: #64748b; font-size: 0.92rem; margin: 0;">
            Discover the 4 distinct customer personas uncovered by K-Means clustering ($k=4$) on 5 orthogonal PCA dimensions, capturing 80.3% of behavioral variance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_result, tab_science = st.tabs(["🎯 Result: Personas & 2D Map", "🔬 The Science: Chi-Square, Wilson CIs & SVD"])

    # Persona Selector
    persona_options = [
        ("2", "Champions (High-Value Spenders)"),
        ("3", "Active Explorers (High Engagement)"),
        ("1", "Recent Window Shoppers (Warm Bouncers)"),
        ("0", "Dormant Bouncers (Cold / Inactive)")
    ]

    with tab_result:
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            p_id = st.selectbox(
                "Select Primary Persona to Inspect:",
                options=persona_options,
                format_func=lambda x: x[1]
            )[0]
        with c_sel2:
            p_compare_id = st.selectbox(
                "Compare Head-to-Head Against:",
                options=[opt for opt in persona_options if opt[0] != p_id],
                format_func=lambda x: x[1]
            )[0]

        p_info = profiles[p_id]
        p_comp = profiles[p_compare_id]

        # Takeaway Banner
        st.markdown(f"""
        <div class="takeaway-banner">
            <strong>Key Insight:</strong> <b>{p_info['persona_name']}</b> represents 
            <b>{p_info['pct_of_total']}% of store visitors</b> with a 10-day conversion rate of 
            <b>{p_info['conversion_rate_pct']}%</b> (95% CI: [{p_info['wilson_ci_lower_pct']}%, {p_info['wilson_ci_upper_pct']}%]).<br>
            <strong>Prescribed Action:</strong> {p_info['recommended_action']}
        </div>
        """, unsafe_allow_html=True)

        # Head-to-Head Comparison Grid
        st.markdown("### Head-to-Head Persona Comparison")
        col_c1, col_c2 = st.columns(2)

        def render_persona_summary_box(p_data, is_primary=True):
            border_color = "#4f46e5" if is_primary else "#94a3b8"
            title_prefix = "PRIMARY FOCUS" if is_primary else "COMPARISON SEGMENT"
            st.markdown(f"""
            <div class="nexus-card" style="border-top: 4px solid {border_color};">
                <span style="font-size: 0.72rem; font-weight: 700; color: #64748b; letter-spacing: 0.05em;">{title_prefix}</span>
                <h3 style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin: 0.2rem 0 0.8rem 0;">{p_data['persona_name']}</h3>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-bottom: 0.8rem;">
                    <div style="background: #f8fafc; padding: 0.6rem; border-radius: 6px;">
                        <span style="font-size: 0.72rem; color: #64748b;">CUSTOMER BASE</span>
                        <div style="font-size: 1.1rem; font-weight: 800;">{p_data['customer_count']:,} ({p_data['pct_of_total']}%)</div>
                    </div>
                    <div style="background: #f8fafc; padding: 0.6rem; border-radius: 6px;">
                        <span style="font-size: 0.72rem; color: #64748b;">10-DAY CONVERSION</span>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #10b981;">{p_data['conversion_rate_pct']}%</div>
                    </div>
                    <div style="background: #f8fafc; padding: 0.6rem; border-radius: 6px;">
                        <span style="font-size: 0.72rem; color: #64748b;">HISTORICAL SPEND</span>
                        <div style="font-size: 1.1rem; font-weight: 800;">${p_data['mean_spend']:,.2f}</div>
                    </div>
                    <div style="background: #f8fafc; padding: 0.6rem; border-radius: 6px;">
                        <span style="font-size: 0.72rem; color: #64748b;">95% WILSON CI</span>
                        <div style="font-size: 0.95rem; font-weight: 700;">[{p_data['wilson_ci_lower_pct']}%, {p_data['wilson_ci_upper_pct']}%]</div>
                    </div>
                </div>
                <div style="font-size: 0.82rem; color: #475569;">
                    <b>Median Activity:</b> {int(p_data['median_events'])} events &middot; {int(p_data['median_sessions'])} sessions &middot; {int(p_data['median_recency_days'])} days recency
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_c1:
            render_persona_summary_box(p_info, is_primary=True)
        with col_c2:
            render_persona_summary_box(p_comp, is_primary=False)

        st.markdown("---")

        # 2D PCA Interactive Scatter Map
        st.markdown("### Interactive 2D Customer Map (PCA / SVD Coordinates)")
        st.write("2,000 customers sampled across all personas projected onto the top 2 orthogonal principal components (57.5% variance). Centroid stars mark cluster centers.")

        sample_pts = warehouse.sample(n=min(2000, len(warehouse)), random_state=42).copy()

        fig_map = px.scatter(
            sample_pts,
            x="pca_x",
            y="pca_y",
            color="persona_name",
            color_discrete_map={
                "Champions (High-Value Spenders)": "#10b981",
                "Active Explorers (High Engagement)": "#6366f1",
                "Recent Window Shoppers (Warm Bouncers)": "#f59e0b",
                "Dormant Bouncers (Cold / Inactive)": "#94a3b8"
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
            opacity=0.65
        )

        # Add centroid markers
        for k in range(4):
            c_center = profiles[str(k)]["pca_center_2d"]
            fig_map.add_trace(go.Scatter(
                x=[c_center[0]],
                y=[c_center[1]],
                mode="markers+text",
                marker=dict(symbol="star", size=15, color="#0f172a", line=dict(color="white", width=1.5)),
                name=f"Centroid: {profiles[str(k)]['persona_name'].split('(')[0]}",
                text=[f"C{k}"],
                textposition="top center",
                showlegend=False
            ))

        fig_map.update_layout(height=480, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_map, use_container_width=True)

    with tab_science:
        st.markdown(f"""
        <div class="science-box">
            <div class="science-title">🔬 Statistical Rigor & Linear Algebra (Units 3, 4, 5)</div>
            <p style="font-size: 0.88rem; color: #475569;">
                Inferential hypothesis tests, confidence intervals, and dimensionality reduction underpinning behavioral segmentation.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("#### 1. Chi-Square Test of Independence")
            st.latex(r"\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}} = 2841.6, \quad df = 3, \quad p < 0.0001")
            st.markdown("""
            - **Hypothesis**: $H_0$ (Conversion rate is independent of persona) vs. $H_1$ (Conversion differs by persona).
            - **Decision**: Reject $H_0$ ($p = 2.4 \times 10^{-616}$).
            - **Effect Size (Cramér's V)**:
            """)
            st.latex(r"V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, c-1)}} = \sqrt{\frac{2841.6}{110956 \cdot 1}} = 0.160")
            st.info("💡 **Interpretation:** A Cramér's V of 0.160 indicates a moderate-to-strong practical association in behavioral clickstream data.")

        with col_s2:
            st.markdown(f"#### 2. Wilson Score 95% Confidence Interval ({p_info['persona_name'].split('(')[0]})")
            st.latex(r"\text{CI}_{95\%} = \frac{p + \frac{z^2}{2n} \pm z \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}")
            st.markdown(f"""
            - **Sample Size ($n$):** `{p_info['customer_count']:,}`
            - **Observed Conversion ($p$):** `{p_info['conversion_rate_pct']}%`
            - **Critical Value ($z$):** `1.96`
            - **Computed 95% Interval:** `[{p_info['wilson_ci_lower_pct']}%, {p_info['wilson_ci_upper_pct']}%]`
            """)
            st.caption("Wilson score interval is strictly preferred over naive Wald approximation for low base-rate proportions.")

        st.markdown("---")
        st.markdown("#### 3. SVD / PCA Decomposition Verifications")
        st.latex(r"X_{\text{scaled}} = U \Sigma V^T \quad \text{where} \quad C = \frac{1}{n-1} X_{\text{scaled}}^T X_{\text{scaled}} = V \left(\frac{\Sigma^2}{n-1}\right) V^T")
        
        la_df = pd.DataFrame([
            {"Component": "PC 1", "Interpretation": "General Engagement Volume (views, events, sessions)", "Explained Variance": "40.43%", "Cumulative": "40.43%"},
            {"Component": "PC 2", "Interpretation": "Monetary & Purchase Intensity (spend, cart count)", "Explained Variance": "17.05%", "Cumulative": "57.48%"},
            {"Component": "PC 3", "Interpretation": "Lifecycle Recency vs. Tenure", "Explained Variance": "10.18%", "Cumulative": "67.66%"},
            {"Component": "PC 4", "Interpretation": "Catalog Breadth vs. Specialization", "Explained Variance": "7.03%", "Cumulative": "74.69%"},
            {"Component": "PC 5", "Interpretation": "Price Sensitivity & Cart Conversion", "Explained Variance": "5.63%", "Cumulative": "80.33%"}
        ])
        st.dataframe(la_df, use_container_width=True, hide_index=True)

