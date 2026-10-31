"""
app/views/p3_customer_lookup.py
===============================
Page 3: Customer Lookup & Batch Scoring Engine.
Enables store managers to evaluate individual customers or score batch CSV uploads.
Strict Dual-Tab layout: Result (Profile Card / Downloadable CSV) vs The Science (Applied Math).
"""

import json
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.scoring_engine import score_customer_vector, score_batch_dataframe, NUMERIC_FEATURES

APP_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "app_data"


@st.cache_data
def load_warehouse():
    return pd.read_parquet(APP_DATA_DIR / "scored_customers_warehouse.parquet")


@st.cache_data
def load_template_bytes():
    with open(APP_DATA_DIR / "sample_upload_template.csv", "rb") as f:
        return f.read()


def render_p3():
    warehouse = load_warehouse()
    template_bytes = load_template_bytes()

    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <span class="brand-pill">🔍 Store Operator Tool</span>
        <h2 style="font-size: 1.65rem; font-weight: 800; color: #0f172a; margin: 0.3rem 0;">
            Customer Intent Lookup & Batch Scoring Engine
        </h2>
        <p style="color: #64748b; font-size: 0.92rem; margin: 0;">
            Search any customer from the 10,000-user database or upload an external CSV for batch scoring. Runs the real calibrated model pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_result, tab_science = st.tabs(["🎯 Result: Customer Intelligence", "🔬 The Science: Worked Math & Attribution"])

    # Mode Selector: Single Lookup vs Batch CSV
    mode = st.radio(
        "Select Scoring Mode:",
        ["👤 Single Customer Lookup", "📁 Batch CSV Scoring Engine"],
        horizontal=True,
        label_visibility="collapsed"
    )

    with tab_result:
        if "Single Customer" in mode:
            st.markdown("---")
            c_input, c_sample = st.columns([2.5, 1.5])
            
            with c_input:
                user_id_input = st.number_input(
                    "Enter 9-digit Customer ID:",
                    min_value=100000000,
                    max_value=999999999,
                    value=int(warehouse.iloc[0]["user_id"]),
                    step=1,
                    key="p3_user_id_input"
                )

            with c_sample:
                st.write("")
                st.write("")
                sample_choice = st.selectbox(
                    "Or Try an Archetype Example:",
                    [
                        "Example 1: High-Value Champion (User 556431564)",
                        "Example 2: Active Cart Explorer (User 514274479)",
                        "Example 3: Recent Window Shopper (User 561841327)",
                        "Example 4: Cold Dormant Bouncer (User 554980851)"
                    ]
                )
                sample_id = int(sample_choice.split("User ")[-1].replace(")", ""))
                if st.button("Load Archetype Example", use_container_width=True):
                    user_id_input = sample_id

            # Find or fallback
            match = warehouse[warehouse["user_id"] == user_id_input]
            if len(match) == 0:
                match = warehouse.iloc[:1]
                st.warning(f"Customer ID {user_id_input} not found in 10,000-sample database. Displaying closest indexed customer.")
            
            cust_row = match.iloc[0]

            # Re-score on the fly to get exact science breakdown
            feat_dict = {col: cust_row[col] for col in NUMERIC_FEATURES if col in cust_row}
            score_res = score_customer_vector(feat_dict, cluster_id=int(cust_row["cluster"]))

            # Store in session state for Science Tab
            st.session_state["p3_current_score_res"] = score_res
            st.session_state["p3_current_row"] = cust_row

            # Takeaway Banner
            st.markdown(f"""
            <div class="takeaway-banner">
                <strong>Executive Takeaway:</strong> Customer <code>{cust_row['user_id']}</code> is classified as 
                <b>{score_res['persona_name']}</b> with a <b>{score_res['calibrated_prob']*100:.1f}% calibrated purchase likelihood</b> ({score_res['intent_decile']}).<br>
                <strong>Recommended Action:</strong> {score_res['recommended_action']}
            </div>
            """, unsafe_allow_html=True)

            # Profile Card
            c_card1, c_card2 = st.columns([1.2, 1])
            with c_card1:
                badge_style = (
                    "badge-champions" if score_res["cluster_id"] == 2 else
                    ("badge-explorers" if score_res["cluster_id"] == 3 else
                    ("badge-shoppers" if score_res["cluster_id"] == 1 else "badge-bouncers"))
                )
                st.markdown(f"""
                <div class="nexus-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <span style="font-size: 1.15rem; font-weight: 800; color: #0f172a;">Customer Intelligence Profile</span>
                        <span class="badge {badge_style}">{score_res['persona_name']}</span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin-bottom: 1.2rem;">
                        <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                            <span style="font-size: 0.72rem; color: #64748b; font-weight: 700;">HISTORICAL SPEND</span>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a;">${cust_row['total_spend']:,.2f}</div>
                        </div>
                        <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                            <span style="font-size: 0.72rem; color: #64748b; font-weight: 700;">CART ITEMS</span>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a;">{int(cust_row['cart_count'])}</div>
                        </div>
                        <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                            <span style="font-size: 0.72rem; color: #64748b; font-weight: 700;">RECENCY</span>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a;">{int(cust_row['recency_days'])}d ago</div>
                        </div>
                        <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                            <span style="font-size: 0.72rem; color: #64748b; font-weight: 700;">TOTAL SESSIONS</span>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a;">{int(cust_row['total_sessions'])}</div>
                        </div>
                        <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                            <span style="font-size: 0.72rem; color: #64748b; font-weight: 700;">TOTAL EVENTS</span>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a;">{int(cust_row['total_events'])}</div>
                        </div>
                        <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                            <span style="font-size: 0.72rem; color: #64748b; font-weight: 700;">PRODUCTS EXPLORED</span>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a;">{int(cust_row['n_unique_products'])}</div>
                        </div>
                    </div>

                    <div style="border-top: 1px solid #e2e8f0; padding-top: 0.8rem;">
                        <span style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase;">Top Decision Drivers:</span>
                        <ul style="margin: 0.4rem 0 0 0; padding-left: 1.2rem; font-size: 0.88rem; color: #1e293b;">
                            {''.join(f'<li>{r}</li>' for r in score_res['reasons'])}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c_card2:
                # Probability Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score_res["calibrated_prob"] * 100,
                    number={"suffix": "%", "font": {"size": 36, "color": "#0f172a"}},
                    title={"text": "Calibrated 10-Day Purchase Intent", "font": {"size": 14, "color": "#475569"}},
                    gauge={
                        "axis": {"range": [0, 50], "tickwidth": 1, "tickcolor": "#94a3b8"},
                        "bar": {"color": "#4f46e5", "thickness": 0.3},
                        "steps": [
                            {"range": [0, 3.87], "color": "#f1f5f9"},
                            {"range": [3.87, 10], "color": "#e0e7ff"},
                            {"range": [10, 25], "color": "#c7d2fe"},
                            {"range": [25, 50], "color": "#a5b4fc"}
                        ],
                        "threshold": {
                            "line": {"color": "#10b981", "width": 4},
                            "thickness": 0.75,
                            "value": 3.87
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.caption("Green tick indicates the 3.87% baseline store conversion rate.")

        else:
            # Batch CSV Mode
            st.markdown("---")
            st.subheader("Batch Customer Scoring Pipeline")
            st.write("Upload any CSV file conforming to the 17-feature behavioral schema. The pipeline will compute standardized PCA coordinates, assign personas, and calculate calibrated purchase probabilities.")

            c_up, c_temp = st.columns([2.5, 1])
            with c_temp:
                st.download_button(
                    label="📥 Download Sample CSV Template",
                    data=template_bytes,
                    file_name="sample_customer_upload_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with c_up:
                uploaded_file = st.file_uploader("Upload Customer CSV:", type=["csv"])

            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    st.success(f"Uploaded {len(df_upload):,} rows successfully!")
                    
                    with st.spinner("Executing calibrated ML scoring pipeline..."):
                        scored_df = score_batch_dataframe(df_upload)

                    st.markdown(f"""
                    <div class="takeaway-banner">
                        <strong>Batch Scoring Complete:</strong> Evaluated {len(scored_df):,} customers. 
                        <b>{int(scored_df['calibrated_purchase_prob'].sum()):,} expected buyers</b> identified. 
                        <b>{(scored_df['calibrated_purchase_prob'] >= 0.10).sum():,} customers</b> exceed the high-intent 10% threshold.
                    </div>
                    """, unsafe_allow_html=True)

                    c_d1, c_d2, c_d3 = st.columns(3)
                    with c_d1:
                        st.metric("Total Uploaded", f"{len(scored_df):,}")
                    with c_d2:
                        st.metric("Mean Purchase Intent", f"{scored_df['calibrated_purchase_prob'].mean()*100:.2f}%")
                    with c_d3:
                        st.metric("Top Persona Identified", scored_df["persona_name"].mode()[0])

                    st.dataframe(
                        scored_df[[
                            "intent_rank", "customer_code" if "customer_code" in scored_df else "predicted_cluster",
                            "persona_name", "calibrated_purchase_prob", "total_spend", "cart_count", "recency_days"
                        ]].head(25),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Export Button
                    csv_export = scored_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download Fully Scored Batch CSV",
                        data=csv_export,
                        file_name="nexus_batch_scored_customers.csv",
                        mime="text/csv",
                        type="primary"
                    )

                except Exception as e:
                    st.error(f"Error parsing uploaded file: {str(e)}. Please ensure the file matches the provided template format.")

    with tab_science:
        score_res = st.session_state.get("p3_current_score_res")
        if not score_res:
            st.info("Look up a customer in the Result tab to see their exact applied mathematics.")
        else:
            st.markdown(f"""
            <div class="science-box">
                <div class="science-title">🔬 Applied Math: Exact Inference Breakdown</div>
                <p style="font-size: 0.88rem; color: #475569;">
                    Step-by-step mathematical transformation for this exact customer:
                </p>
            </div>
            """, unsafe_allow_html=True)

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("#### Step 1: 17D Euclidean Distance to Centroids")
                st.latex(r"d(x, c_k) = \sqrt{\sum_{j=1}^{17} \left(\frac{x_j - \mu_j}{\sigma_j} - c_{k,j}\right)^2}")
                st.write(f"Assigned Cluster: **{score_res['persona_name']}** (Cluster {score_res['cluster_id']})")
                
                dist_records = [
                    {"Persona Centroid": k, "Euclidean Distance": v, "Status": "🌟 ASSIGNED MINIMUM" if k == score_res["the_science"]["nearest_persona"] else "Alternative"}
                    for k, v in score_res["the_science"]["distances_to_centroids"].items()
                ]
                st.dataframe(pd.DataFrame(dist_records), use_container_width=True, hide_index=True)

            with col_s2:
                st.markdown("#### Step 2: Logistic Sum & Platt Sigmoid Step")
                st.latex(r"\text{logit} = \beta_0 + \sum_{j=1}^{21} \beta_j z_j")
                st.latex(r"P_{\text{calibrated}} = \frac{1}{1 + \exp\left(A \cdot \text{logit} + B\right)}")
                st.markdown(f"""
                - **Raw Intercept ($\\beta_0$)**: `{score_res['the_science']['intercept_beta_0']}`
                - **Computed Log-Odds ($\\text{{logit}}$)**: `{score_res['the_science']['calculated_logit']}`
                - **Raw Sigmoid Output**: `{score_res['the_science']['raw_sigmoid_prob']*100:.2f}%`
                - **Platt Calibrated Probability**: `{score_res['the_science']['calibrated_platt_prob']*100:.2f}%`
                """)

            st.markdown("#### Step 3: Feature Log-Odds Contributions ($z_j \\cdot \\beta_j$)")
            st.caption("Shows how each standardized feature increases or decreases the log-odds of purchase.")
            c_df = pd.DataFrame(score_res["the_science"]["feature_contributions"])
            st.dataframe(c_df[["feature", "raw_value", "z_score", "coefficient", "contribution", "odds_ratio"]], use_container_width=True, hide_index=True)

