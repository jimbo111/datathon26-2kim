# CLAUDE.md — Datathon 2026 (Team 2Kim)

## Project
- SBU AI Community Datathon 2026. Team: Jimmy + Alice.
- **Research question:** "Your Zip Code is Your Health Sentence" — food deserts, geographic inequality, and chronic disease.
- Python data analysis stack — ML optional, only if it adds value.
- Demo runs on localhost: FastAPI backend + React frontend.

## Competition Rules
- Track: **Healthcare & Wellness**
- Submit: `.ipynb` notebook + `.pptx`/`.pdf` slideshow via Google Form
- Notebook MUST follow 7-section structure (problem → data → cleaning → EDA → stats/model → results → limitations)
- Last notebook cell MUST be "Dataset Citations (MLA 8)" with full citations
- File naming: `2kim_health_notebook.ipynb`, `2kim_health_slides.pptx`
- Judged by SBU professors. Scoring (100pts): Analysis & Evidence (25), Data Quality (20), Research Question (15), Technical Rigor (15), Presentation (15), Limitations & Ethics (10)
- Tiebreaker priority: Analysis & Evidence → Data Quality → Limitations & Ethics

## Architecture
- `backend/` — FastAPI app (Plotly JSON API, health data endpoints)
  - `routes/data.py` — Data + health-specific endpoints (food desert comparison, race matrix, HDI)
  - `services/data_service.py` — Loads master.parquet, serves analysis JSON
- `frontend/` — React + Vite, renders Plotly JSON from backend
- `src/loaders/` — Dataset downloaders and merge pipeline
  - `health_data.py` — 5 federal dataset loaders (USDA, CDC PLACES, ACS, USALEEP, HRSA)
  - `merge.py` — Joins all on FIPS → `data/processed/master.parquet`
- `src/analysis/` — Statistical analysis
  - `health_stats.py` — Phase 2-4 regressions (food access, zip code effect, race gap)
  - `health_index.py` — Phase 5B Health Disadvantage Index
- `notebooks/2kim_health_notebook.ipynb` — Competition submission (7-section + MLA citations)
- `data/raw/` — Gitignored raw data (downloaded at runtime)
- `data/processed/` — Gitignored outputs (master.parquet, analysis JSONs)

## Running
- Backend: `uvicorn backend.main:app --reload` (port 8000)
- Frontend: `cd frontend && npm run dev` (port 5173)
- Data pipeline: `python -c "from src.loaders import build_master; build_master()"`

## Conventions
- Raw data is gitignored — never commit data files
- API keys go in `.env` (copy from `.env.example`) — never commit
- Charts flow: backend generates Plotly JSON → frontend renders with react-plotly.js
- Every chart in the notebook must answer a specific sub-question
- Always distinguish correlation from causation in conclusions
