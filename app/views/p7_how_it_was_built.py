"""
app/views/p7_how_it_was_built.py
================================
Page 7: How It Was Built (The Architecture & Rigor Suite).
Covers:
1. Data Engineering Pipeline (Bronze -> Silver -> Gold with verified row counts)
2. The 5 Vs of Big Data applied to REES46 eCommerce clickstream
3. Leakage-safe temporal split design
4. Interactive Gradient Descent replay (Batch vs Mini-batch vs Stochastic GD)
"""

from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


@st.cache_data
def load_gd_data():
    return pd.read_csv(REPORTS_DIR / "optimization_convergence.csv")


def render_p7():
    gd_df = load_gd_data()

    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <span class="brand-pill">🏗️ Engineering & Architecture Suite</span>
        <h2 style="font-size: 1.65rem; font-weight: 800; color: #0f172a; margin: 0.3rem 0;">
            How It Was Built: System Architecture & Rigor Audit
        </h2>
        <p style="color: #64748b; font-size: 0.92rem; margin: 0;">
            Complete engineering documentation: ELT data lakehouse, temporal leakage barriers, and gradient descent optimization from scratch.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_arch, tab_gd = st.tabs([
        "🏗️ Lakehouse Pipeline & 5 Vs",
        "⚡ Gradient Descent Optimization Replay"
    ])

    with tab_arch:
        st.markdown("### 1. Medallion Data Engineering Pipeline")
        st.write("The platform processes raw clickstream logs into high-performance analytical feature marts using an ELT Medallion architecture.")

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("""
            <div class="nexus-card" style="border-left: 4px solid #b45309;">
                <span class="badge badge-shoppers">BRONZE LAYER</span>
                <h4 style="margin: 0.4rem 0; font-size: 1.05rem;">Raw Clickstream Ingestion</h4>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 0.4rem;">42,448,764 Events</div>
                <div style="font-size: 0.82rem; color: #64748b;">
                    <b>Format:</b> Snappy Parquet (partitioned by date)<br>
                    <b>Raw Size:</b> 5.66 GB CSV (REES46 Oct 2019)<br>
                    <b>Integrity:</b> Partition-level audit & date validation.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with p2:
            st.markdown("""
            <div class="nexus-card" style="border-left: 4px solid #6366f1;">
                <span class="badge badge-explorers">SILVER LAYER</span>
                <h4 style="margin: 0.4rem 0; font-size: 1.05rem;">Cleansed Star Schema</h4>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 0.4rem;">DuckDB OLAP Engine</div>
                <div style="font-size: 0.82rem; color: #64748b;">
                    <b>Schema:</b> Fact events + Dim users/products/time<br>
                    <b>Cleansing:</b> Sessionization, price outlier pruning<br>
                    <b>Performance:</b> Columnar compression & indexation.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with p3:
            st.markdown("""
            <div class="nexus-card" style="border-left: 4px solid #10b981;">
                <span class="badge badge-champions">GOLD LAYER</span>
                <h4 style="margin: 0.4rem 0; font-size: 1.05rem;">Feature Store & Personas</h4>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 0.4rem;">110,956 User Vectors</div>
                <div style="font-size: 0.82rem; color: #64748b;">
                    <b>Features:</b> 17 behavioral features (Days 1-20)<br>
                    <b>Labels:</b> Future 10-day conversion (Days 21-31)<br>
                    <b>Leakage Barrier:</b> Strict temporal segregation.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 2. The 5 Vs of Big Data Applied to REES46")
        
        v_table = pd.DataFrame({
            "The 5 Vs": ["Volume", "Velocity", "Variety", "Veracity", "Value"],
            "Empirical Evidence in REES46 Dataset": [
                "42.4M clickstream event rows (5.66 GB uncompressed CSV)",
                "1.37M events/day across peak hours; asynchronous user session logging",
                "Semi-structured clickstream (view, cart, purchase, price, multi-level category taxonomy)",
                "Null categories (~31%), non-standard session IDs, price anomalies (cleansed via DuckDB)",
                "Actionable behavioral segmentation and commercial conversion uplift ($2.00 cost / offer)"
            ],
            "Architectural Handling": [
                "Snappy-compressed partitioned Parquet + columnar DuckDB storage",
                "Sessionized windowing with 30-minute inactivity thresholds",
                "Snowflake dimensional star schema modeling (FactEvents, DimProduct, DimTime)",
                "Schema validation check, IQR monetary bounds, missing brand/category imputation",
                "Constrained Lagrange multiplier budget allocation and precision-calibrated targeting"
            ]
        })
        st.dataframe(v_table, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 3. Leakage-Safe Temporal Train/Test Architecture")
        st.markdown("""
        <div class="takeaway-banner">
            <strong>Critical Machine Learning Hygiene:</strong> 
            Random $k$-fold cross-validation on time-series clickstream data causes <b>catastrophic lookahead leakage</b>, 
            where features computed using future purchases predict past behavior. 
            Our platform strictly enforces a <b>Point-in-Time Temporal Split</b>:
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ```
        October 1, 2019                                 October 20, 2019 (23:59:59)              October 31, 2019
        ├───────────────────────────────────────────────────────┤───────────────────────────────────────┤
                     FEATURE WINDOW (20 Days)                             TARGET WINDOW (11 Days)
        - Compute 17 behavioral features                         - Evaluate true purchase label:
          (Recency, Frequency, Monetary spend,                     will_purchase_next_10d in {0, 1}
           Session counts, Category exploration)                 - Zero future events enter feature mart!
        ```
        """)

    with tab_gd:
        st.markdown("### Interactive Gradient Descent Convergence Replay")
        st.write("Compare the three fundamental first-order optimization paradigms implemented from scratch on the binary cross-entropy loss function.")

        max_epoch = int(gd_df["epoch"].max())
        selected_epoch = st.slider("Scrub Convergence Epoch:", min_value=1, max_value=max_epoch, value=max_epoch, step=1)

        subset_df = gd_df[gd_df["epoch"] <= selected_epoch]

        fig_gd = go.Figure()
        fig_gd.add_trace(go.Scatter(
            x=subset_df["epoch"], y=subset_df["batch_gd_loss"],
            mode="lines+markers", name="Batch GD (BGD)",
            line=dict(color="#4f46e5", width=2.5)
        ))
        fig_gd.add_trace(go.Scatter(
            x=subset_df["epoch"], y=subset_df["sgd_loss"],
            mode="lines+markers", name="Stochastic GD (SGD)",
            line=dict(color="#f43f5e", width=1.5, dash="dot")
        ))
        fig_gd.add_trace(go.Scatter(
            x=subset_df["epoch"], y=subset_df["mini_batch_gd_loss"],
            mode="lines+markers", name="Mini-Batch GD (MBGD, batch=256)",
            line=dict(color="#10b981", width=2.5)
        ))

        fig_gd.update_layout(
            height=420,
            xaxis_title="Epoch Number",
            yaxis_title="Binary Cross-Entropy Loss (Log-Loss)",
            xaxis=dict(range=[1, max_epoch]),
            yaxis=dict(range=[0.10, 0.70]),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=-0.18)
        )
        st.plotly_chart(fig_gd, use_container_width=True)

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.markdown("""
            **Batch GD (BGD)**
            - $\\nabla L(\\theta) = \\frac{1}{N} \\sum_{i=1}^N \\nabla l_i(\\theta)$
            - Deterministic, smooth descent.
            - Computationally heavy on large $N$.
            """)
        with col_g2:
            st.markdown("""
            **Stochastic GD (SGD)**
            - $\\nabla L(\\theta) = \\nabla l_i(\\theta)$
            - Noisy, high variance trajectory.
            - Fast iterations, escapes shallow saddle points.
            """)
        with col_g3:
            st.markdown("""
            **Mini-Batch GD (MBGD)**
            - $\\nabla L(\\theta) = \\frac{1}{B} \\sum_{j=1}^B \\nabla l_j(\\theta)$
            - Best of both worlds: vectorization speed + stable convergence.
            """)


