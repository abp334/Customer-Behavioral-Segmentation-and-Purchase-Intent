# NEXUS | Teammate Setup Guide

## Prerequisites

- Python **3.10 or 3.11** on PATH
- Node.js **20+** and npm
- Git

## 1. Clone and enter the repo

```bash
git clone <your-repo-url>
cd DSC_GP
```

## 2. Python environment + API

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Leave this terminal running. Health check: http://localhost:8000/api/health

## 3. React frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## 4. Optional acceptance check

```bash
# from repo root, with venv active
python -c "from fastapi.testclient import TestClient; from backend.main import app; c=TestClient(app); assert c.get('/api/health').json()['status']=='ok'; print('API OK')"
```

## Product vs Methodology

- Use **Lookup / Campaign / Batch / Segments / Traffic** for the product demo.
- Open **Methodology** when an instructor asks how models, calibration, Lagrange, or the lakehouse work.

## Legacy Streamlit

The old Streamlit app under `app/` is retired. Do not run `streamlit run app/main.py` for demos.
