# NEXUS | Customer Behavioral Intelligence

Marketer-facing product for customer scoring, campaign targeting, and segment exploration—backed by models trained on **42.4M REES46 eCommerce clickstream events** (October 2019).

The **product UI** is React. The **API** is a thin FastAPI layer over the existing Python engines in `src/`. Deep math, model tables, and the Lagrange study live under **Methodology** for instructors.

---

## Quick start

```bash
# 1. Python env (from repo root)
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. API
uvicorn backend.main:app --reload --port 8000

# 3. UI (second terminal)
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** (Vite proxies `/api` to port 8000).

API docs: http://localhost:8000/docs

---

## Product routes

| Path | Purpose |
| --- | --- |
| `/` | Home — brand + jobs |
| `/customer` | Score one shopper (ID or simple profile) |
| `/campaign` | Budget → ranked target list + CSV/PDF |
| `/batch` | Upload CSV → scored export |
| `/segments` | Compare personas + behavioral map |
| `/traffic` | 7-day send-day outlook |
| `/methodology` | Models, stats, Lagrange, pipeline (graders) |

---

## Repository layout

```text
frontend/                 # Vite + React + CSS product UI
backend/                  # FastAPI routes wrapping src/
src/                      # Scoring, campaign, PDF, DS modules
models/                   # joblib models
data/app_data/            # Parquet/JSON contracts for the app
notebooks/                # Offline training & analysis (01–07)
reports/                  # Benchmark CSVs used by Methodology
app/                      # Legacy Streamlit (see app/LEGACY.md)
```

---

## Campaign product story

Under your **budget** and **cost per contact**, NEXUS contacts the top customers by **predicted expected value** (purchase likelihood × typical segment spend).

Predicted buyers / revenue are **observational expectations**, not causal lift. The Lagrange segment-budget study is under **Methodology → Optimization** and does not drive the product target list.

---

## Teammate setup

See [TEAMMATE_SETUP_GUIDE.md](TEAMMATE_SETUP_GUIDE.md) for Windows/macOS steps.
