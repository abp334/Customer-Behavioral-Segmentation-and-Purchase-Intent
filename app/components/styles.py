"""
app/components/styles.py
========================
High-end, bespoke CSS styling for NEXUS Customer Decision Platform.
Overrides default generic Streamlit visuals with an executive SaaS aesthetic:
- Inter font family & polished typography
- Glassmorphic hero headers
- Interactive card containers & timeline nodes
- Styled dual-tab components (Result vs The Science)
- Clean, colorblind-safe visual accents
"""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Base Reset & Typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0f172a;
}
code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Hide default streamlit decorations */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* Main container breathing room */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* Executive Brand Topbar */
.brand-topbar {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #312e81 100%);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.brand-title {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
}
.brand-subtitle {
    font-size: 0.88rem;
    color: #cbd5e1;
    margin-top: 0.3rem;
    font-weight: 400;
}
.brand-pill {
    background: rgba(99, 102, 241, 0.25);
    border: 1px solid rgba(165, 180, 252, 0.4);
    color: #e0e7ff;
    padding: 0.35rem 0.8rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Top Navigation Bar Styling */
div[data-testid="stHorizontalBlock"] .nav-btn-active {
    background: #4f46e5 !important;
    color: white !important;
    font-weight: 700 !important;
}

/* High-End Card Containers */
.nexus-card {
    background: #ffffff;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
    margin-bottom: 1.2rem;
}
.nexus-card-dark {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 14px;
    border: 1px solid #334155;
    padding: 1.5rem;
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
}

/* Takeaway Banner (Result Tab) */
.takeaway-banner {
    background: #f0fdf4;
    border-left: 4.5px solid #10b981;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin-bottom: 1.2rem;
    color: #14532d;
    font-size: 0.95rem;
    font-weight: 500;
    line-height: 1.5;
}
.takeaway-banner strong {
    color: #064e3b;
    font-weight: 700;
}

/* Science Explainer Box */
.science-box {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    margin-top: 1rem;
    margin-bottom: 1.2rem;
}
.science-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Pill Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.badge-champions { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
.badge-explorers { background: #e0e7ff; color: #4338ca; border: 1px solid #a5b4fc; }
.badge-shoppers { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
.badge-bouncers { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
.badge-calibrated { background: #ede9fe; color: #6d28d9; border: 1px solid #c4b5fd; }
.badge-verified { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }

/* Timeline Node Component for P2 */
.timeline-step {
    border-left: 2px solid #cbd5e1;
    padding-left: 1.2rem;
    position: relative;
    padding-bottom: 1rem;
    margin-left: 0.5rem;
}
.timeline-dot {
    position: absolute;
    left: -6px;
    top: 2px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #6366f1;
    border: 2px solid #ffffff;
}
.timeline-dot.purchase { background: #10b981; }
.timeline-dot.cart { background: #f59e0b; }
.timeline-date {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
}
.timeline-desc {
    font-size: 0.88rem;
    color: #1e293b;
    font-weight: 500;
    margin-top: 0.15rem;
}

/* Tab Styling Enhancements */
button[data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.4rem !important;
    border-radius: 8px 8px 0 0 !important;
}
button[aria-selected="true"] {
    color: #4f46e5 !important;
    border-bottom-color: #4f46e5 !important;
}

/* Metric Display Enhancements */
div[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    letter-spacing: -0.02em !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* Team Credits Footer */
.team-footer {
    margin-top: 3.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e2e8f0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    color: #64748b;
}
</style>
"""

