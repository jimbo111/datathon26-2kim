# CLAUDE.md — Datathon 2026 (Team 2Kim)

## Project
- Datathon competition project. Team: Jimmy + Alice.
- Python data analysis stack — no ML assumed unless needed.

## Architecture
- `src/loaders/` — Universal data loader (CSV, Excel, Parquet, JSON, SQL, URLs)
- `src/analysis/` — Statistical tests (normality, chi-sq, t-test, ANOVA)
- `src/viz/` — Reusable matplotlib/seaborn plotting functions
- `src/utils/` — Submission export helpers
- `notebooks/` — Numbered Jupyter notebooks (01_eda, 02_analysis)
- `data/raw/` — Gitignored raw data; `data/processed/` for cleaned data

## Conventions
- Raw data is gitignored — never commit data files
- Shared reusable code goes in `src/`, not inline in notebooks
- Notebooks import from `src/` via `sys.path.insert(0, "..")`
