# Datathon 2026 — Team 2Kim

**Team:** Jimmy + Alice

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Project Structure

```
├── data/
│   ├── raw/            # Original datasets (gitignored)
│   ├── processed/      # Cleaned/transformed data
│   └── external/       # Supplementary data
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   └── 02_analysis.ipynb   # Deep analysis & hypothesis testing
├── src/
│   ├── loaders/        # Universal data loader (CSV, Excel, Parquet, JSON, SQL)
│   ├── analysis/       # Statistical tests & summary functions
│   ├── viz/            # Reusable plotting functions
│   └── utils/          # Export & submission helpers
├── submissions/        # Final outputs
└── requirements.txt
```

## Usage

```python
from src.loaders import load_data, describe_data

df = load_data("dataset.csv")       # auto-detects format
describe_data(df, "My Dataset")     # rich summary table
```

## Conventions

- **Data stays local** — raw data is gitignored. Share via Slack/Drive.
- **Notebooks are numbered** — `01_`, `02_`, etc. for ordering.
- **Shared code goes in `src/`** — keep notebooks clean.
