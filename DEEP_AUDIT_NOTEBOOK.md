# Deep EDA Audit — 2kim_finance_notebook.ipynb
**Audited against:** Competition README EDA requirements + JUDGING-RUBRIC.md
**Date:** 2026-03-28

---

## What Judges Will Like

1. **Shape shown for all key datasets.** Cells [6], [7], [8] print `.shape` with date ranges for NVDA, CSCO, macro, and fundamentals. Cell [9] summarizes rows/columns in a markdown table.
2. **Missing values explicitly quantified.** Cell [8] computes `isnull().mean() * 100` per column. Cell [9] table lists exact missing % per dataset (e.g., "CSCO fundamentals: 30–50%") and calls out the handling decision.
3. **Strong correlation matrix present.** Cell [53] constructs a cross-layer feature matrix and computes a full correlation heatmap via `plotly.express`. Cell [46] also runs Pearson correlation tests on sentiment vs price.
4. **Return distributions with statistics.** Cell [28] produces side-by-side histograms (NVDA vs CSCO), annotated with skew, kurtosis, and std inline. The README's histogram requirement is fully met.
5. **Every chart followed by an Interpretation cell.** 18 interpretation cells (e.g., [13], [15], [17], [23], [25], [33], [41], [47]) explain what each chart shows in plain language — directly addressing "visuals as communication tool."
6. **sys.path setup present.** Cell [4] does `sys.path.insert(0, "..")` before layer imports — required for reproducibility.
7. **Multi-layer EDA structure.** Five independent analysis layers (price, fundamentals, concentration, macro, sentiment) each follow question → chart → interpretation, which directly satisfies "thoughtful, structured EDA."
8. **Bubble Scorecard synthesizes EDA into conclusions.** Cell [72] produces a traffic-light table mapping every EDA finding to a verdict — judges can see the full evidence chain.
9. **MLA 8 citations in the last cell.** Compliance check will pass.

---

## What Judges Will Flag as Weak

1. **No df.describe() or univariate distribution for individual features.** Shape and missing values are shown, but there is no `describe()` call that shows mean/std/min/max for the raw price or macro columns before modeling. Cell [7] calls `.describe()` only on fundamentals, not on the primary price series.
2. **No explanation of how to run the notebook.** Cell [4] imports from `src/layers/` via `sys.path.insert(0, "..")` but there is no Markdown cell explaining what environment, API keys (FRED, OpenAI), or data files are required. A judge trying to reproduce the work will be blocked immediately.
3. **Zero evidence of question evolution from EDA.** The README specifically rewards "thoughtful iteration — question evolving based on EDA findings." The notebook's research question is stated in cell [1] and never revisited. No cell says "after seeing the return distribution, we refined our question to…" — the narrative is top-down, not EDA-driven.
4. **Charts rely on pre-built layer modules (`fig.show()`)** — source chart code is hidden in `src/layers/`. A judge reading only the notebook cannot verify how charts were produced, which hurts the Technical Rigor / Reproducibility score (15 pts).
5. **Volume distribution never explored directly.** Volume surge is discussed in chart [18]/[19] but only as z-scores; raw volume distribution histograms are absent.
6. **No pairplot or scatter matrix for the price features.** The correlation matrix (cell [53]) is cross-layer; there is no scatter matrix showing pairwise relationships within the primary price dataset.

---

## Specific Improvements Needed

| Priority | Fix | Rubric Impact |
|---|---|---|
| P1 | Add a "How to Run" Markdown cell after cell [4]: list required env vars (`FRED_API_KEY`, `OPENAI_API_KEY`), `pip install -r requirements.txt`, and cache behavior. | Technical Rigor +3–5 pts |
| P1 | Add one cell after cell [9] with `nvda_df.describe()` and `csco_df.describe()` — judges expect to see raw feature distributions before EDA begins. | Data Quality +2–3 pts |
| P2 | Add a brief "EDA Refinement" note in cell [10] or [54] — one sentence explaining which sub-question was sharpened based on what EDA revealed (e.g., "Layer 2 EDA showed NVDA P/E was 3–4x lower than expected, which shifted our focus to revenue growth as the key differentiator"). | Analysis & Evidence +2 pts |
| P2 | Inline the chart-generation code for at least 2–3 key charts (e.g., normalized price overlay, return histogram) directly in the notebook instead of `fig.show()` alone. This ensures reproducibility without the `src/` module dependency. | Technical Rigor +2 pts |
| P3 | Add a scatter plot of NVDA daily returns vs CSCO daily returns (aligned by trading day since breakout) to make the relationship visual, not just statistical. | Analysis & Evidence +1 pt |

---

## Comparison to Starter Notebook

The starter notebook (`finance.ipynb`) is a simpler BTC analysis that sets the EDA baseline clearly: it calls `df.shape`, prints missing percentages, produces histogram + KDE with skew/kurtosis printed, and explains every chart in plain English before any model is built. The submission notebook **exceeds the starter** in depth (5 layers, DTW, regime classifier, SHAP) but **lags behind it** on two explicit starter patterns: `describe()` before modeling and inline chart code for reproducibility. Judges familiar with the starter will notice both gaps.
