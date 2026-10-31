"""
app/views/p4_campaign_builder.py
================================
Page 4: Campaign Builder & Constrained Optimization Engine.
Allows store managers to set goals, budget, and contact costs to generate:
- Ranked customer target list
- Lagrange Multiplier optimal budget split
- Downloadable target CSV
- Downloadable 1-page PDF Executive Campaign Brief
Strict Dual-Tab layout: Result (ROI & Action) vs The Science (Lagrange KKT & Assumptions).
"""

from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.campaign_optimizer import run_campaign_optimization
from src.pdf_generator import generate_campaign_pdf


def render_p4():
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <span class="brand-pill">💼 Marketing ROI Optimizer</span>
        <h2 style="font-size: 1.65rem; font-weight: 800; color: #0f172a; margin: 0.3rem 0;">
            Campaign Builder & Lagrange Budget Allocator
        </h2>
        <p style="color: #64748b; font-size: 0.92rem; margin: 0;">
            Maximize commercial return under budget constraints. Leverages KKT optimality conditions to equalize marginal returns across customer segments.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_result, tab_science = st.tabs(["🎯 Result: Target Audience & PDF Brief", "🔬 The Science: Lagrange Optimality & Assumptions"])

    with st.container():
        # Campaign Parameter Inputs
        c_goal, c_budget, c_cost = st.columns([1.5, 1, 1])
        with c_goal:
            goal_key = st.selectbox(
                "Select Strategic Campaign Objective:",
                options=[
                    ("win_back_lapsed", "Win-Back Lapsed Customers (Recency >= 7d)"),
                    ("convert_cart_abandoners", "Convert Cart Abandoners (Cart Items >= 1)"),
                    ("reward_loyal_champions", "Reward & Retain Loyal Champions (High Spend)"),
                    ("maximize_profit", "Maximize Overall Commercial Return (Unrestricted)")
                ],
                format_func=lambda x: x[1]
            )[0]

        with c_budget:
            budget_val = st.number_input(
                "Total Campaign Budget ($):",
                min_value=200.0,
                max_value=100000.0,
                value=5000.0,
                step=500.0
            )

        with c_cost:
            cost_val = st.number_input(
                "Cost per Contact ($):",
                min_value=0.10,
                max_value=25.0,
                value=1.00,
                step=0.25
            )

    # Execute optimization pipeline
    res = run_campaign_optimization(goal_key, total_budget=budget_val, cost_per_contact=cost_val)

    with tab_result:
        # Takeaway Banner
        st.markdown(f"""
        <div class="takeaway-banner">
            <strong>Executive Takeaway:</strong> Allocating <b>${res['total_budget']:,.2f}</b> at ${res['cost_per_contact']:.2f}/contact funds 
            <b>{res['target_count']:,} targeted customers</b>. Expected to capture <b>{res['expected_buyers']} buyers</b> yielding 
            <b>${res['expected_revenue']:,.2f} projected GMV</b> (Expected Net Profit: <b>+${res['expected_net_profit']:,.2f}</b> | ROI: <b>+{res['roi_pct']}%</b>).
        </div>
        """, unsafe_allow_html=True)

        # KPI Summary Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Targeted Customers", f"{res['target_count']:,}", f"of {res['eligible_audience_size']:,} eligible")
        with k2:
            st.metric("Expected Buyers", f"{res['expected_buyers']}", f"{res['conversion_rate_targeted_pct']}% conversion")
        with k3:
            st.metric("Projected GMV", f"${res['expected_revenue']:,.2f}", f"Net: +${res['expected_net_profit']:,.2f}")
        with k4:
            st.metric("Expected Commercial ROI", f"+{res['roi_pct']}%", f"Shadow Price: {res['the_science']['lambda_star']:.2f}")

        st.markdown("---")

        c_table, c_alloc = st.columns([1.3, 1])

        with c_table:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                <h4 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: #0f172a;">
                    Priority Target List (Top Scored Customers)
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                res["full_targeted_df"].head(15),
                use_container_width=True,
                hide_index=True
            )

            # Download Buttons
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                csv_data = res["full_targeted_df"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Target List (CSV)",
                    data=csv_data,
                    file_name=f"campaign_targets_{goal_key}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with c_d2:
                # PDF Generation
                pdf_bytes = generate_campaign_pdf(res)
                st.download_button(
                    label="📄 Download Executive Brief (PDF)",
                    data=pdf_bytes,
                    file_name=f"campaign_executive_brief_{goal_key}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )

        with c_alloc:
            st.markdown("""
            <h4 style="margin: 0 0 0.6rem 0; font-size: 1.05rem; font-weight: 700; color: #0f172a;">
                Lagrange Multiplier Budget Allocation
            </h4>
            """, unsafe_allow_html=True)

            alloc_data = list(res["lagrange_allocations"].values())
            fig_alloc = px.pie(
                alloc_data,
                names="persona_name",
                values="budget_allocated",
                hole=0.45,
                color="persona_name",
                color_discrete_map={
                    "Champions": "#10b981",
                    "Active Explorers": "#6366f1",
                    "Recent Window Shoppers": "#f59e0b",
                    "Dormant Bouncers": "#94a3b8"
                }
            )
            fig_alloc.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_alloc, use_container_width=True)

            # Allocation Table
            alloc_df = pd.DataFrame(alloc_data)[["persona_name", "budget_allocated", "budget_share_pct", "marginal_return_per_dollar"]]
            alloc_df.rename(columns={
                "persona_name": "Segment",
                "budget_allocated": "Budget ($)",
                "budget_share_pct": "Share (%)",
                "marginal_return_per_dollar": "Marginal Return"
            }, inplace=True)
            st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    with tab_science:
        st.markdown(f"""
        <div class="science-box">
            <div class="science-title">🔬 Constrained Optimization & Lagrange Multipliers (Lagrangian Formulation)</div>
            <p style="font-size: 0.88rem; color: #475569;">
                The budget allocation solves the classical Karush-Kuhn-Tucker (KKT) constrained optimization problem 
                under concave diminishing-returns revenue functions:
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.markdown("#### The Lagrangian Formulation")
            st.latex(r"\max_{b_1, \dots, b_k} \sum_{i=1}^k \alpha_i \ln(1 + \beta_i b_i) \quad \text{s.t.} \quad \sum_{i=1}^k b_i = B, \quad b_i \ge 0")
            st.latex(r"\mathcal{L}(b, \lambda) = \sum_{i=1}^k \alpha_i \ln(1 + \beta_i b_i) - \lambda \left(\sum_{i=1}^k b_i - B\right)")
            
            st.markdown("#### KKT Stationarity Condition")
            st.latex(r"\frac{\partial \mathcal{L}}{\partial b_i} = \frac{\alpha_i \beta_i}{1 + \beta_i b_i} - \lambda^* = 0 \implies b_i^*(\lambda^*) = \max\left(0, \frac{\alpha_i}{\lambda^*} - \frac{1}{\beta_i}\right)")
            st.info(f"💡 **Equal Marginal Return:** At optimal budget allocation, the marginal revenue of the last dollar spent across all funded personas is exactly equalized to **$\\lambda^* = {res['the_science']['lambda_star']:.4f}$**.")

        with col_l2:
            st.markdown("#### Optimal Solution Check")
            st.write(f"- **Shadow Price ($\\lambda^*$):** `{res['the_science']['lambda_star']}` (marginal return per $1 budget increase)")
            st.write(f"- **Total Budget Constraint:** ${budget_val:,.2f}")
            st.write(f"- **Sum of Allocations ($\\sum b_i^*$):** ${res['the_science']['total_budget_check']:,.2f} *(Exact equality holds)*")

            st.markdown("#### Explicit Assumptions & Scientific Disclosures")
            for asm in res["the_science"]["assumptions"]:
                st.markdown(f"- ⚠️ *{asm}*")

