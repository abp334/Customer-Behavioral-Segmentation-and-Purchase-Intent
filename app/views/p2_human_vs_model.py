"""
app/views/p2_human_vs_model.py
==============================
Page 2: "Meet the Shoppers" (Human vs Model Gamified Benchmark).
Tests human intuition against the champion calibrated machine learning model
on 5 real, held-out test customers extracted from October 2019 clickstream.
Features strict Dual-Tab layout: Result (Game) vs The Science (Applied Math).
"""

import json
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.scoring_engine import score_customer_vector

APP_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "app_data"


def load_test_timelines():
    with open(APP_DATA_DIR / "test_user_timelines.json") as f:
        return json.load(f)


def init_p2_state(timelines):
    if "p2_initialized" not in st.session_state:
        # Sample 5 cases: 2 Easy, 2 Medium, 1 Hard
        easy = [c for c in timelines if c["difficulty"] == "Easy"]
        med = [c for c in timelines if c["difficulty"] == "Medium"]
        hard = [c for c in timelines if c["difficulty"] == "Hard"]
        
        selected = []
        selected.extend(np.random.choice(easy, size=min(2, len(easy)), replace=False).tolist())
        selected.extend(np.random.choice(med, size=min(2, len(med)), replace=False).tolist())
        selected.extend(np.random.choice(hard, size=min(1, len(hard)), replace=False).tolist())
        np.random.shuffle(selected)

        st.session_state["p2_cases"] = selected
        st.session_state["p2_round"] = 0
        st.session_state["p2_human_score"] = 0
        st.session_state["p2_model_score"] = 0
        st.session_state["p2_revealed"] = False
        st.session_state["p2_history"] = []
        st.session_state["p2_initialized"] = True


def render_p2():
    timelines = load_test_timelines()
    init_p2_state(timelines)

    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <span class="brand-pill">🎮 Benchmark Challenge</span>
        <h2 style="font-size: 1.65rem; font-weight: 800; color: #0f172a; margin: 0.3rem 0;">
            Meet the Shoppers: Human Intuition vs. Machine Learning
        </h2>
        <p style="color: #64748b; font-size: 0.92rem; margin: 0;">
            Evaluate 5 real, held-out test customers from their 20-day clickstream histories. Guess their persona and whether they purchased in the subsequent 10 days.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_result, tab_science = st.tabs(["🎯 Result: Interactive Benchmark", "🔬 The Science: Real Math & Attribution"])

    curr_round = st.session_state["p2_round"]
    cases = st.session_state["p2_cases"]
    game_over = curr_round >= len(cases)

    with tab_result:
        # Top Scoreboard
        c_score1, c_score2, c_score3, c_reset = st.columns([1.2, 1.2, 1.6, 1])
        with c_score1:
            st.metric("Your Score (Human)", f"{st.session_state['p2_human_score']} / {curr_round if not game_over else len(cases)}")
        with c_score2:
            st.metric("Model Score (AI)", f"{st.session_state['p2_model_score']} / {curr_round if not game_over else len(cases)}")
        with c_score3:
            st.metric("Round Progress", f"Round {min(curr_round + 1, len(cases))} of {len(cases)}")
        with c_reset:
            st.write("")
            if st.button("🔄 Restart Game", key="p2_restart_btn"):
                st.session_state.pop("p2_initialized", None)
                st.rerun()

        st.markdown("---")

        if not game_over:
            current_case = cases[curr_round]

            col_left, col_right = st.columns([1.1, 1])

            with col_left:
                st.markdown(f"""
                <div class="nexus-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <span style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">Customer Profile: <code>{current_case['customer_code']}</code></span>
                        <span class="badge badge-explorers">{current_case['difficulty']} Round</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem; background: #f8fafc; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #e2e8f0;">
                        <div>
                            <span style="font-size: 0.75rem; color: #64748b; font-weight: 600;">HISTORICAL SPEND</span>
                            <div style="font-weight: 700; font-size: 1rem;">${current_case['total_spend']:,.2f}</div>
                        </div>
                        <div>
                            <span style="font-size: 0.75rem; color: #64748b; font-weight: 600;">TOTAL EVENTS</span>
                            <div style="font-weight: 700; font-size: 1rem;">{current_case['total_events']}</div>
                        </div>
                        <div>
                            <span style="font-size: 0.75rem; color: #64748b; font-weight: 600;">RECENCY</span>
                            <div style="font-weight: 700; font-size: 1rem;">{current_case['recency_days']}d ago</div>
                        </div>
                    </div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #334155; margin-bottom: 0.6rem;">
                        📅 Chronological Event Timeline (Days 1 to 20):
                    </div>
                """, unsafe_allow_html=True)

                # Render timeline nodes
                timeline_html = '<div style="max-height: 280px; overflow-y: auto; padding-right: 0.5rem;">'
                for ev in current_case["timeline"]:
                    dot_class = "purchase" if ev["action"] == "purchase" else ("cart" if ev["action"] == "cart" else "")
                    action_badge = f"<span style='font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: {'#10b981' if ev['action']=='purchase' else ('#f59e0b' if ev['action']=='cart' else '#6366f1')}'>[{ev['action']}]</span>"
                    timeline_html += f"""
                    <div class="timeline-step">
                        <div class="timeline-dot {dot_class}"></div>
                        <div class="timeline-date">Day {ev['day']} ({ev['date']}) {action_badge}</div>
                        <div class="timeline-desc">{ev['item']}</div>
                    </div>
                    """
                timeline_html += '</div></div>'
                st.markdown(timeline_html, unsafe_allow_html=True)

            with col_right:
                st.markdown("""
                <div class="nexus-card">
                    <h3 style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 0;">
                        Make Your Assessment
                    </h3>
                """, unsafe_allow_html=True)

                if not st.session_state["p2_revealed"]:
                    guess_persona = st.selectbox(
                        "1. Which persona does this customer belong to?",
                        options=[
                            "Dormant Bouncers (Cold / Inactive)",
                            "Recent Window Shoppers (Warm Bouncers)",
                            "Champions (High-Value Spenders)",
                            "Active Explorers (High Engagement)"
                        ],
                        key=f"guess_persona_{curr_round}"
                    )

                    guess_purchase = st.radio(
                        "2. Will this customer make a purchase in the next 10 days?",
                        options=["No (Will Not Purchase)", "Yes (Will Purchase)"],
                        key=f"guess_purchase_{curr_round}"
                    )

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🎯 Lock In & Reveal Model Truth", type="primary", use_container_width=True):
                        st.session_state["p2_revealed"] = True
                        st.session_state["p2_user_guess_purchase"] = 1 if "Yes" in guess_purchase else 0
                        st.session_state["p2_user_guess_persona"] = guess_persona

                        # Evaluate
                        true_p = current_case["true_next_10d_purchase"]
                        true_persona = current_case["persona_name"]
                        user_correct = (st.session_state["p2_user_guess_purchase"] == true_p)
                        
                        # Model decision threshold tau* = 0.08
                        model_pred = 1 if current_case["calibrated_prob"] >= 0.08 else 0
                        model_correct = (model_pred == true_p)

                        if user_correct:
                            st.session_state["p2_human_score"] += 1
                        if model_correct:
                            st.session_state["p2_model_score"] += 1

                        st.session_state["p2_history"].append({
                            "round": curr_round + 1,
                            "customer": current_case["customer_code"],
                            "true_outcome": "Purchased" if true_p == 1 else "No Purchase",
                            "user_guess": "Purchased" if st.session_state["p2_user_guess_purchase"] == 1 else "No Purchase",
                            "user_won": user_correct,
                            "model_prob": f"{current_case['calibrated_prob']*100:.1f}%",
                            "model_won": model_correct
                        })
                        st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)

                else:
                    # Reveal View
                    true_p = current_case["true_next_10d_purchase"]
                    true_persona = current_case["persona_name"]
                    c_prob = current_case["calibrated_prob"]
                    model_pred = 1 if c_prob >= 0.08 else 0
                    user_guess_p = st.session_state["p2_user_guess_purchase"]

                    user_status = "✅ Correct" if user_guess_p == true_p else "❌ Incorrect"
                    model_status = "✅ Correct" if model_pred == true_p else "❌ Incorrect"

                    st.markdown(f"""
                    <div class="takeaway-banner">
                        <strong>Ground Truth Outcome:</strong> Customer {current_case['customer_code']} 
                        <b>{'BOUGHT' if true_p == 1 else 'DID NOT BUY'}</b> in the next 10 days.<br>
                        <strong>True Persona:</strong> <span class="badge badge-explorers">{true_persona}</span>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-bottom: 1rem;">
                        <div style="background: {'#f0fdf4' if user_guess_p == true_p else '#fef2f2'}; padding: 0.8rem; border-radius: 8px; border: 1px solid {'#86efac' if user_guess_p == true_p else '#fca5a5'};">
                            <span style="font-size: 0.75rem; font-weight: 700; color: #64748b;">YOUR PREDICTION</span>
                            <div style="font-weight: 800; font-size: 1.1rem; color: {'#15803d' if user_guess_p == true_p else '#b91c1c'};">
                                {user_status}
                            </div>
                            <span style="font-size: 0.8rem; color: #475569;">Guessed: {'Purchase' if user_guess_p == 1 else 'No Purchase'}</span>
                        </div>
                        <div style="background: {'#f0fdf4' if model_pred == true_p else '#fef2f2'}; padding: 0.8rem; border-radius: 8px; border: 1px solid {'#86efac' if model_pred == true_p else '#fca5a5'};">
                            <span style="font-size: 0.75rem; font-weight: 700; color: #64748b;">CHAMPION MODEL</span>
                            <div style="font-weight: 800; font-size: 1.1rem; color: {'#15803d' if model_pred == true_p else '#b91c1c'};">
                                {model_status}
                            </div>
                            <span style="font-size: 0.8rem; color: #475569;">P(Buy) = <b>{c_prob*100:.1f}%</b></span>
                        </div>
                    </div>

                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
                        <span style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase;">Top Model Decision Drivers:</span>
                        <ul style="margin: 0.3rem 0 0 0; padding-left: 1.2rem; font-size: 0.85rem; color: #1e293b;">
                            {''.join(f'<li>{r}</li>' for r in current_case['reasons'])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Next Customer ➡️", type="primary", use_container_width=True):
                        st.session_state["p2_round"] += 1
                        st.session_state["p2_revealed"] = False
                        st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Game Completed Summary
            st.markdown(f"""
            <div class="nexus-card" style="text-align: center; padding: 2rem;">
                <h3 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem;">
                    🏁 Benchmark Challenge Completed!
                </h3>
                <p style="font-size: 1.1rem; color: #475569; margin-bottom: 1.5rem;">
                    Final Score: <b>Human: {st.session_state['p2_human_score']}</b> vs. <b>Model: {st.session_state['p2_model_score']}</b> (out of 5 rounds)
                </p>
                <div class="takeaway-banner" style="text-align: left; max-width: 800px; margin: 0 auto 1.5rem auto;">
                    <strong>Academic Honesty & Sample Size Disclosure:</strong> 
                    A 5-customer sample offers an intuitive comparison of human heuristics against algorithmic scoring, but is not statistically representative. 
                    On the full held-out test partition of <b>22,192 customers</b>, the model achieves a <b>PR-AUC of 0.2450</b> (6.3x baseline) and a <b>calibrated Brier score of 0.0331</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state["p2_history"]:
                st.subheader("Match Trajectory Recap")
                hist_df = pd.DataFrame(st.session_state["p2_history"])
                st.dataframe(hist_df, use_container_width=True, hide_index=True)

    with tab_science:
        current_case = cases[min(curr_round, len(cases) - 1)]
        st.markdown(f"""
        <div class="science-box">
            <div class="science-title">🔬 Applied Mathematics & Feature Attribution (Customer {current_case['customer_code']})</div>
            <p style="font-size: 0.88rem; color: #475569;">
                The champion model computes purchase intent using a regularized Logistic Regression classifier plugged into 
                standardized behavioral inputs and calibrated via Platt scaling:
            </p>
            <div style="background: white; padding: 0.8rem; border-radius: 6px; border: 1px solid #e2e8f0; font-family: monospace; font-size: 0.85rem; margin-bottom: 1rem;">
                logit(x) = &beta;<sub>0</sub> + &sum; &beta;<sub>j</sub> &middot; z<sub>j</sub><br>
                P<sub>raw</sub>(Y=1|x) = &sigma;(logit(x)) = 1 / (1 + e<sup>-logit(x)</sup>)<br>
                P<sub>calibrated</sub>(Y=1|x) = 1 / (1 + e<sup>(A &middot; logit + B)</sup>)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Plug-in numbers for this customer
        feat_dict = {
            "total_spend": current_case["total_spend"],
            "total_events": current_case["total_events"],
            "cart_count": current_case["cart_count"],
            "recency_days": current_case["recency_days"],
            "view_count": max(0, current_case["total_events"] - current_case["cart_count"]),
            "n_unique_products": 3
        }
        scored = score_customer_vector(feat_dict, cluster_id=current_case["cluster_id"])

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Distance to Cluster Centroids in 17D Scaled Space")
            st.latex(r"d_k = \|z - \mu_k\|_2 = \sqrt{\sum_{j=1}^{17} (z_j - \mu_{k,j})^2}")
            dists_df = pd.DataFrame([
                {"Persona": k, "Euclidean Distance": v, "Is Nearest": "🌟 NEAREST" if k == scored["the_science"]["nearest_persona"] else ""}
                for k, v in scored["the_science"]["distances_to_centroids"].items()
            ])
            st.dataframe(dists_df, use_container_width=True, hide_index=True)

        with col_m2:
            st.markdown("#### Probability Calibration Transition")
            st.markdown(f"""
            - **Raw Intercept $\\beta_0$**: `{scored['the_science']['intercept_beta_0']}`
            - **Calculated Logit**: `{scored['the_science']['calculated_logit']}`
            - **Raw Model Probability**: `{scored['the_science']['raw_sigmoid_prob']*100:.2f}%` *(uncalibrated prior)*
            - **Platt Calibrated Probability**: `{scored['the_science']['calibrated_platt_prob']*100:.2f}%` *(true empirical risk)*
            """)
            st.info("💡 **Why Platt Calibration Matters:** Because the model was trained with class balancing to counter severe 3.87% class imbalance, raw model outputs are shifted upwards. Platt scaling maps them back to true real-world conversion frequencies.")

        st.markdown("#### Top Log-Odds Feature Contributions")
        contrib_df = pd.DataFrame(scored["the_science"]["feature_contributions"]).head(8)
        st.dataframe(contrib_df[["feature", "raw_value", "z_score", "coefficient", "contribution", "odds_ratio"]], use_container_width=True, hide_index=True)

