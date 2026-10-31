"""
NEXUS API — thin FastAPI layer over existing src/ engines.
Run from project root:
  .venv/bin/uvicorn backend.main:app --reload --port 8000
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.routes import batch, campaign, customers, forecast, methodology, personas

app = FastAPI(
    title="NEXUS API",
    description="Customer behavioral intelligence — product APIs for the React frontend.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(customers.router)
app.include_router(batch.router)
app.include_router(campaign.router)
app.include_router(personas.router)
app.include_router(forecast.router)
app.include_router(methodology.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "nexus-api", "version": "2.0.0"}
