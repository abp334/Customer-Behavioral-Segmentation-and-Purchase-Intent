"""
app/views/p1_home.py
====================
Page 1: Executive Landing & System Overview.
Communicates the 20-second value proposition with a live interactive customer example,
features two big primary entry buttons (Gamified Benchmark vs Store Workflows),
and presents the 4-person Capstone Group architecture.
"""

from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from src.scoring_engine import score_customer_vector


def render_p1(set_page_callback):
    # Hero Section
    st.markdown("""
    <div class="brand-topbar" style="margin-bottom: 1.5rem;">
        <div>
            <span class="brand-pill">University Capstone Data Science Platform</span>
            <h1 class="brand-title" style="margin-top: 0.5rem; font-size: 2.2rem;">
                NEXUS | E-Commerce Intelligence & Decision Engine
            </h1>
            <p class="brand-subtitle" style="font-size: 1.05rem; max-width: 850px; line-height: 1.5;">
                An end-to-end analytical product transforming <b>42.4M clickstream events</b> into 
                unsupervised behavioral segmentation, Platt-calibrated purchase prediction, and Lagrange-optimized marketing allocations.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 20-Second Live Example Card
    st.markdown("""
    <div style="margin-bottom: 0.6rem;">
        <span style="font-size: 0.78rem; font-weight: 700; color: #64748b; letter-spacing: 0.05em; text-transform: uppercase;">
            ⚡ 20-Second Live Pipeline Preview
        </span>
    </div>
    """, unsafe_allow_html=True)

    live_example_col1, live_example_col2 = st.columns([1.4, 1])

    with live_example_col1:
        st.markdown("""
        <div class="nexus-card" style="border-left: 4.5px solid #4f46e5;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                <span style="font-weight: 800; font-size: 1.15rem; color: #0f172a;">Sample Profile: <code>CUST-7429</code></span>
                <span class="badge badge-champions">Champions (High-Value)</span>
            </div>
            <p style="font-size: 0.88rem; color: #475569; margin-bottom: 0.8rem;">
                Live evaluation of real clickstream signals through our calibrated logistic pipeline:
            </p>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin-bottom: 0.8rem;">
                <div style="background: #f8fafc; padding: 0.6rem; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <span style="font-size: 0.7rem; color: #64748b; font-weight: 600;">HISTORICAL SPEND</span>
                    <div style="font-weight: 800; font-size: 1rem; color: #0f172a;">$620.00</div>
                </div>
                <div style="background: #f8fafc; padding: 0.6rem; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <span style="font-size: 0.7rem; color: #64748b; font-weight: 600;">ACTIVE CART</span>
                    <div style="font-weight: 800; font-size: 1rem; color: #0f172a;">3 items</div>
                </div>
                <div style="background: #f8fafc; padding: 0.6rem; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <span style="font-size: 0.7rem; color: #64748b; font-weight: 600;">RECENCY</span>
                    <div style="font-weight: 800; font-size: 1rem; color: #0f172a;">2 days ago</div>
                </div>
            </div>
            <div style="font-size: 0.82rem; color: #15803d; background: #f0fdf4; padding: 0.6rem 0.8rem; border-radius: 6px;">
                <b>Model Takeaway:</b> 26.6% calibrated purchase intent (6.8x baseline). Prescribed action: Exclusive VIP Concierge voucher.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with live_example_col2:
        st.markdown("""
        <div class="nexus-card-dark">
            <span style="font-size: 0.75rem; font-weight: 700; color: #a5b4fc; letter-spacing: 0.05em; text-transform: uppercase;">
                Verified Empirical Benchmarks
            </span>
            <div style="margin-top: 0.6rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="font-size: 0.85rem; color: #cbd5e1;">Evaluated Clickstream:</span>
                    <span style="font-weight: 700; color: #ffffff;">42,448,764 Events</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="font-size: 0.85rem; color: #cbd5e1;">Test Set PR-AUC:</span>
                    <span style="font-weight: 700; color: #10b981;">0.2450 (6.3x Baseline)</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="font-size: 0.85rem; color: #cbd5e1;">Calibrated Brier Score:</span>
                    <span style="font-weight: 700; color: #38bdf8;">0.0331 (Platt Scaling)</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 0.85rem; color: #cbd5e1;">Traffic Forecast MAPE:</span>
                    <span style="font-weight: 700; color: #fbbf24;">3.99% (SARIMA)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two Big Entry Buttons
    st.markdown("### Choose Your Entry Workflow")
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        st.markdown("""
        <div class="nexus-card" style="text-align: center; border: 2px solid #6366f1; height: 100%;">
            <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">🎮</div>
            <h3 style="font-size: 1.3rem; font-weight: 800; color: #0f172a; margin-bottom: 0.3rem;">
                Play: Can You Beat the Model?
            </h3>
            <p style="font-size: 0.88rem; color: #64748b; margin-bottom: 1.2rem;">
                Test your commercial intuition against the champion ML model on 5 real, held-out test customers. Guess persona and conversion.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start Benchmark Game", key="btn_go_p2", type="primary", use_container_width=True):
            set_page_callback("p2")

    with btn_col2:
        st.markdown("""
        <div class="nexus-card" style="text-align: center; border: 2px solid #10b981; height: 100%;">
            <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">🏢</div>
            <h3 style="font-size: 1.3rem; font-weight: 800; color: #0f172a; margin-bottom: 0.3rem;">
                I Run a Store (Operator Tools)
            </h3>
            <p style="font-size: 0.88rem; color: #64748b; margin-bottom: 1.2rem;">
                Look up customer profiles, score batch CSVs, build profit-maximizing marketing campaigns, and download PDF briefs.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("💼 Enter Store Workflows", key="btn_go_p3", use_container_width=True):
            set_page_callback("p3")

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # Platform Capabilities Grid
    st.markdown("### Core Decision Engines")
    g1, g2, g3, g4 = st.columns(4)

    engines = [
        ("👤 Customer Lookup", "Instant single-user scoring and batch CSV pipeline with calibrated probabilities and decile rankings.", "p3"),
        ("💼 Campaign Builder", "Lagrange Multiplier budget optimizer equalizing marginal return per dollar across segments.", "p4"),
        ("📊 Persona Explorer", "4 behavioral personas with 95% Wilson CIs, Chi-square independence tests, and 2D PCA map.", "p5"),
        ("📈 Traffic Outlook", "Seasonal SARIMA(1,1,1)x(1,0,0)7 projection with day-of-week marketing scheduling.", "p6")
    ]

    for col, (title, desc, target_p) in zip([g1, g2, g3, g4], engines):
        with col:
            st.markdown(f"""
            <div class="nexus-card" style="height: 190px;">
                <h4 style="font-size: 0.98rem; font-weight: 800; color: #0f172a; margin: 0 0 0.4rem 0;">{title}</h4>
                <p style="font-size: 0.8rem; color: #64748b; line-height: 1.4; margin-bottom: 0.8rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {title.split()[1]}", key=f"quick_{target_p}", use_container_width=True):
                set_page_callback(target_p)

