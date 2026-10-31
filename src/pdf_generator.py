"""
src/pdf_generator.py
====================
Generates a polished 1-page Executive Campaign Brief PDF
using pure-Python fpdf2 with zero external OS/C-library dependencies.
Provides personal, downloadable output for Page 4 (Campaign Builder).
"""

from datetime import datetime
from typing import Dict, Any
from fpdf import FPDF


class CampaignBriefPDF(FPDF):
    def header(self):
        # Top banner background
        self.set_fill_color(15, 23, 42) # Slate-900
        self.rect(0, 0, 210, 24, "F")
        
        # Header text
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 13)
        self.set_xy(12, 5)
        self.cell(100, 7, "NEXUS INTELLIGENCE | CAMPAIGN DECISION BRIEF", ln=False)
        
        self.set_font("Helvetica", "", 8)
        self.set_xy(140, 5)
        self.cell(60, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="R", ln=True)

        self.set_xy(12, 12)
        self.set_font("Helvetica", "I", 8)
        self.cell(180, 5, "BSc Data Science Capstone Project | Supervised Intent & Lagrange Budget Engine", ln=True)
        self.set_y(28)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(148, 163, 184)
        self.cell(0, 8, "NEXUS E-Commerce Analytics Engine  |  Page 1 of 1  |  Observational Association, Not Causal Lift", align="C")


def generate_campaign_pdf(campaign_data: Dict[str, Any]) -> bytes:
    """
    Generate an executive 1-page PDF summarizing the campaign parameters,
    targeted audience, Lagrange budget split, and data science methodology.
    """
    pdf = CampaignBriefPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # 1. Title Block
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, f"Executive Campaign Brief: {campaign_data['goal_title']}", ln=True)
    pdf.set_draw_color(99, 102, 241) # Indigo-500
    pdf.set_line_width(0.6)
    pdf.line(12, pdf.get_y(), 198, pdf.get_y())
    pdf.ln(3)

    # 2. Executive KPI Grid (4 Cards)
    start_y = pdf.get_y()
    card_width = 44.5
    gap = 2.5

    kpis = [
        ("Total Budget", f"${campaign_data['total_budget']:,.2f}", "Capital Allocated"),
        ("Funded Targets", f"{campaign_data['target_count']:,} users", f"@ ${campaign_data['cost_per_contact']:.2f}/user"),
        ("Expected Buyers", f"{campaign_data['expected_buyers']} users", f"{campaign_data['conversion_rate_targeted_pct']}% of targets"),
        ("Projected Return", f"${campaign_data['expected_revenue']:,.2f}", f"+{campaign_data['roi_pct']}% Net ROI")
    ]

    for i, (label, val, sub) in enumerate(kpis):
        x = 12 + i * (card_width + gap)
        # Background box
        pdf.set_fill_color(248, 250, 252) # Slate-50
        pdf.set_draw_color(226, 232, 240)
        pdf.set_line_width(0.2)
        pdf.rect(x, start_y, card_width, 20, "FD")
        
        pdf.set_xy(x + 2, start_y + 2)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(card_width - 4, 4, label.upper(), align="L")
        
        pdf.set_xy(x + 2, start_y + 7)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(card_width - 4, 6, val, align="L")

        pdf.set_xy(x + 2, start_y + 14)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(79, 70, 229)
        pdf.cell(card_width - 4, 4, sub, align="L")

    pdf.set_y(start_y + 23)

    # 3. Lagrange Optimal Budget Allocation Table
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, "Lagrange Multiplier Optimal Budget Allocation (KKT Solution)", ln=True)

    # Table Header
    headers = [("Persona Segment", 58), ("Allocated ($)", 30), ("Share (%)", 24), ("Expected GMV", 36), ("Marginal Return / $1", 38)]
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(51, 65, 85)
    
    for h_name, w in headers:
        pdf.cell(w, 6, h_name, border=1, fill=True, align="C")
    pdf.ln()

    # Table Rows
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    for k_str, alloc in campaign_data["lagrange_allocations"].items():
        pdf.cell(58, 6, f" {alloc['persona_name']}", border=1, align="L")
        pdf.cell(30, 6, f"${alloc['budget_allocated']:,.2f} ", border=1, align="R")
        pdf.cell(24, 6, f"{alloc['budget_share_pct']}% ", border=1, align="R")
        pdf.cell(36, 6, f"${alloc['expected_segment_revenue']:,.2f} ", border=1, align="R")
        pdf.cell(38, 6, f"{alloc['marginal_return_per_dollar']:.4f} ", border=1, align="R")
        pdf.ln()

    pdf.ln(2)

    # 4. Strategic Execution Matrix (Recommended Channels & Guardrails)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Campaign Operational Matrix & Execution Protocol", ln=True)

    protocols = [
        ("Champions (High-Value Spenders)", "VIP Concierge & Loyalty Perks", "Frequency cap: 1x/wk. Personal phone/white-glove email. Direct credit voucher."),
        ("Active Explorers (High Engagement)", "Dynamic Cart Abandonment Nudges", "Frequency cap: 2x/wk. Automated mobile push & 48-hr expiring discount codes."),
        ("Recent Window Shoppers", "Personalized Category Retargeting", "Frequency cap: 2x/wk. Email showcase of top viewed products with social proof."),
        ("Dormant Bouncers (Cold)", "Low-Cost Automated Win-Back", "Frequency cap: 1x/month. Batch discount blast; suppress from high-cost direct mail.")
    ]

    for persona, channel, guardrail in protocols:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(62, 5, f"- {persona}:", ln=False)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(79, 70, 229)
        pdf.cell(55, 5, channel, ln=False)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(70, 5, guardrail, ln=True)

    pdf.ln(3)

    # 5. Scientific Methodology, Honesty & Governance Disclaimer
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(12, pdf.get_y(), 186, 38, "FD")
    
    box_y = pdf.get_y() + 2
    pdf.set_xy(14, box_y)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(180, 4, "DATA SCIENCE METHODOLOGY & STATISTICAL DISCLOSURES (FOR EXAMINERS & OPERATORS)", ln=True)

    notes = [
        "1. Observational Association: 'Expected Buyers' is the sum of calibrated probabilities P(Buy=1 | X). It reflects behavioral intent from days 1-20, not causal lift.",
        "2. Calibration: Raw Logistic Regression probabilities are Platt-calibrated (Brier score reduced from 0.1749 to 0.0331; PR-AUC = 0.2450 on held-out test data).",
        "3. Concave Elasticity: Budget allocation assumes diminishing returns R_i(b_i) = alpha_i * ln(1 + beta_i * b_i). Shadow price lambda* = " + str(campaign_data["the_science"]["lambda_star"]) + " marginal return/$.",
        "4. Temporal Hygiene: Features are strictly aggregated from days 1-20; conversion label is evaluated on days 21-31 (zero forward-looking data leakage)."
    ]

    for note in notes:
        pdf.set_x(14)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(100, 116, 139)
        pdf.multi_cell(182, 3.8, note)

    return bytes(pdf.output())

