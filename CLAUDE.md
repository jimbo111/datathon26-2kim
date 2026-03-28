# CLAUDE.md — Datathon 2026 (Team 2Kim)

## Project
- SBU AI Community Datathon 2026. Team: Jimmy + Alice.
- Python data analysis stack — ML optional, only if it adds value.
- Demo runs on localhost: FastAPI backend + React frontend.

## Competition Rules
- Pick exactly ONE track: Education, Sustainability & Infrastructure, Healthcare & Wellness, Finance & Economics
- Submit: `.ipynb` notebook + `.pptx`/`.pdf` slideshow via Google Form
- Notebook MUST follow 7-section structure (problem → data → cleaning → EDA → stats/model → results → limitations)
- Last notebook cell MUST be "Dataset Citations (MLA 8)" with full citations
- File naming: `2kim_[track]_notebook.ipynb`, `2kim_[track]_slides.pptx`
- AI tools allowed; team responsible for accuracy — no fabricated results
- Judged by SBU professors. Scoring (100pts): Analysis & Evidence (25), Data Quality (20), Research Question (15), Technical Rigor (15), Presentation (15), Limitations & Ethics (10)
- Tiebreaker priority: Analysis & Evidence → Data Quality → Limitations & Ethics

## Architecture
- `backend/` — FastAPI app (Plotly JSON API, data endpoints, external API integration)
- `frontend/` — React + Vite, renders Plotly JSON from backend
- `src/` — Shared Python analysis code (loaders, stats, viz, export)
- `notebooks/01_eda.ipynb` — Competition submission template (7-section + MLA citations)
- `data/raw/` — Gitignored raw data

## Running
- Backend: `uvicorn backend.main:app --reload` (port 8000)
- Frontend: `cd frontend && npm run dev` (port 5173)

## Conventions
- Raw data is gitignored — never commit data files
- API keys go in `.env` (copy from `.env.example`) — never commit
- Charts flow: backend generates Plotly JSON → frontend renders with react-plotly.js
- Every chart in the notebook must answer a specific sub-question
- Always distinguish correlation from causation in conclusions
