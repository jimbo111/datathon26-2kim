# CLAUDE.md — Datathon 2026 (Team 2Kim)

## Project
- Datathon competition project. Team: Jimmy + Alice.
- Python data analysis stack — no ML assumed unless needed.
- Demo runs on localhost: FastAPI backend + React frontend.

## Architecture
- `backend/` — FastAPI app (Plotly JSON API, data endpoints, external API integration)
  - `routes/charts.py` — Chart endpoints returning Plotly JSON
  - `routes/data.py` — Data loading/querying endpoints
  - `services/chart_service.py` — Plotly figure generation
  - `services/data_service.py` — In-memory dataset management
  - `services/external.py` — Async HTTP client for external APIs
  - `services/openai_service.py` — OpenAI integration (key via .env)
- `frontend/` — React + Vite, renders Plotly JSON from backend
  - `src/components/Chart.jsx` — Reusable Plotly chart renderer
  - `src/components/Dashboard.jsx` — Main dashboard layout
  - `src/api/client.js` — Axios API client
- `src/` — Shared Python analysis code (loaders, stats, viz, export)
- `notebooks/` — Jupyter notebooks for exploration
- `data/raw/` — Gitignored raw data

## Running
- Backend: `uvicorn backend.main:app --reload` (port 8000)
- Frontend: `cd frontend && npm run dev` (port 5173)

## Conventions
- Raw data is gitignored — never commit data files
- API keys go in `.env` (copy from `.env.example`) — never commit
- Charts flow: backend generates Plotly JSON → frontend renders with react-plotly.js
- Shared reusable code goes in `src/`, not inline in notebooks
