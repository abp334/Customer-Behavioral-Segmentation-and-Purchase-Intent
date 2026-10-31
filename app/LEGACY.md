# Legacy Streamlit UI

The product surface is now **React (`frontend/`) + FastAPI (`backend/`)**.

This `app/` folder (Streamlit tabs and orphaned `views/`) is retained only for historical reference and is **not** the supported way to run NEXUS.

## Run the product instead

```bash
# Terminal 1 — API
.venv/bin/uvicorn backend.main:app --reload --port 8000

# Terminal 2 — UI
cd frontend && npm run dev
```

Open http://localhost:5173
