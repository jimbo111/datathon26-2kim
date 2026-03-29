# Implementation Audit — Checklist vs Actual Code

**Audit date:** 2026-03-28
**Auditor:** Code Review Agent (automated + manual verification)
**Method:** Bytecode inspection of .pyc files (source .py files for layers were deleted), direct reading of pipeline.py, feature_matrix.py, and notebook .ipynb.

---

**CRITICAL NOTE:** The source files `src/layers/layer1_price.py` through `layer5_sentiment.py` and `src/utils/cache.py` do NOT exist on disk. Only their compiled `.pyc` bytecode remains in `__pycache__/`. All Layer 1-5 and cache.py verification was done via bytecode constant/name extraction, which confirms function existence and docstrings but cannot verify line-by-line logic. The ML files (`pipeline.py`, `feature_matrix.py`) DO exist as full source and were verified at source level.

---

## Verified Correct [x]

### Layer 1 — Price Dynamics
- **`find_breakout()` exists** -- Confirmed in `layer1_price.cpython-311.pyc`. Function name `find_breakout` present.
- **50% threshold used** -- Confirmed. Docstring reads: "threshold : Minimum trailing 1-year return (0.50 = 50 %)." Numeric constant `0.5` present. Chart title includes "50%+ YoY gain".
- **4 charts generated** -- Confirmed: `chart_1_1_price_overlay`, `chart_1_2_drawdown`, `chart_1_3_rolling_returns`, `chart_1_4_volume_surge`.
- **ADF test implemented** -- Confirmed: names `adfuller`, `adf_stat`, `adf_p`, `adf_key` present.
- **Pearson test implemented** -- Confirmed: names `pearsonr`, `_pearson_ci` present.
- **Ljung-Box test implemented** -- Confirmed: name `acorr_ljungbox` present.
- **KS test implemented** -- Confirmed: names `ks_2samp`, `ks_stat`, `ks_p` present.

### Layer 2 — Fundamentals
- **`compute_peg_ratio()` exists** -- Confirmed in `layer2_fundamentals.cpython-311.pyc`. Name `compute_peg_ratio` present. Docstring: "PEG = P/E / (YoY EPS growth rate ...)".
- **6 charts generated** -- Confirmed: `chart_2_1_pe_trajectory`, `chart_2_2_revenue_growth`, `chart_2_3_mcap_vs_revenue`, `chart_2_4_fcf_yield`, `chart_2_5_rd_investment`, `chart_2_6_peg_ratio`. All 6 also exist as rendered JSON+PNG in `submissions/charts/`.
- **CPI adjustment implemented** -- Confirmed: `_fetch_cpi` function, `CPIAUCSL` FRED series fetch, docstring "Adjust nominal dollar values to real dollars at target_year".
- **log(P/E) used (not clipped P/E)** -- Confirmed: variable `log_pe_ratio` used extensively. Stat test labeled `bootstrap_log_pe`. String: "log(P/E) mean comparison: NVDA vs CSCO dot-com era."

### Layer 3 — Concentration
- **Buffett Indicator is Fed-adjusted** -- Confirmed: docstring "Compute Buffett Indicator and a Fed-adjusted variant." Names `buffett_adj`, string "Fed-Adjusted Buffett".
- **Concentration-reversibility implemented** -- Confirmed: function name `concentration_reversibility`, data key `reversibility_df` present.
- **5 charts generated** -- Confirmed: `chart_3_1_concentration_timeline`, `chart_3_2_spy_vs_rsp`, `chart_3_3_buffett_indicator`, `chart_3_4_treemap`, `chart_3_5_hhi`.

### Layer 4 — Macro
- **10 FRED series fetched** -- Confirmed: `GDP`, `FEDFUNDS`, `DGS10`, `DGS2`, `T10Y2Y`, `M2SL`, `CPIAUCSL`, `BAA10Y`, `UNRATE`, `SP500`. Exactly 10 series IDs found.
- **GDP YoY uses pct_change(4) on quarterly** -- Confirmed: docstring reads "pct_change(periods=4) on quarterly series == year-over-year." Number 4 present in constants.
- **Findings framed as associations** -- Confirmed: strings include "All statistics are ASSOCIATIONS", "Association only -- not a causal claim." (appears multiple times), "ASSOCIATION [", narrative function docstring: "Derive narrative findings from data. All framed as ASSOCIATIONS."
- **6 charts generated** -- Confirmed: `chart_4_1_fed_funds`, `chart_4_2_m2_growth`, `chart_4_3_yield_curve`, `chart_4_4_credit_spreads`, `chart_4_5_macro_dashboard`, `chart_4_6_real_rate`.

### Layer 5 — Sentiment
- **CSCO dot-com keyword counting implemented** -- Confirmed: `CSCO_DOTCOM_KEYWORDS` constant, `_fetch_csco_filings` function, `_count_keywords` function, "Fetch CSCO 10-Q/10-K filings (dot-com era) and count dot-com keywords." This is implemented as a primary feature, NOT a stretch goal.
- **OpenAI scoring is guarded** -- Confirmed: strings include "OPENAI_API_KEY not set -- skipping GPT scoring", "Returns None if OpenAI is unavailable", "OpenAI Scores Unavailable" fallback label for Chart 5.2.
- **5 charts generated** -- Confirmed: `chart_5_1_ai_mentions`, `chart_5_2_hype_vs_specificity`, `chart_5_3_google_trends`, `chart_5_4_sentiment_price_overlay`, `chart_5_6_keyword_comparison`. Note: chart 5_5 is intentionally skipped (numbering goes 5_1 through 5_4 then 5_6).

### ML Pipeline (verified from full source)
- **Price-excluded RF variant trained** -- Confirmed in `pipeline.py` lines 297-317. Drops `PRICE_FEATURES` columns, trains separate RF with same hyperparameters, reports separate CV scores.
- **DTW null distribution computed** -- Confirmed in `compute_dtw_similarity()` lines 802-811. Permutation test with `n_permutations=500`, shuffles AI-era array, computes empirical p-value.
- **BH correction applied** -- Confirmed in `apply_bh_correction()` lines 1149-1222. Uses `statsmodels.stats.multitest.multipletests` with `method="fdr_bh"`. Reports raw vs adjusted p-values and "lost significance" flag.
- **SHAP implemented** -- Confirmed in `compute_shap_analysis()` lines 1038-1141. Uses `shap.TreeExplainer` on RF, computes per-class SHAP values, extracts bubble-class importance, generates March 2026 waterfall.
- **4 ML charts** -- Confirmed: `chart_ml_1_regime_probabilities`, `chart_ml_2_dtw_similarity`, `chart_ml_3_shap_summary`, `chart_ml_4_shap_waterfall`.

### Feature Matrix (verified from full source)
- **Feature matrix combines all 5 layers** -- Confirmed in `feature_matrix.py`. Builds price (9), fundamental (7), concentration (6), macro (9), sentiment (7) feature groups.
- **Time-based splits (no random split)** -- Confirmed in `pipeline.py` lines 1620-1628. Train <= 2018, Val 2019-2021, Test 2022-Jun2024, Live Jul2024+.
- **No look-ahead bias** -- Confirmed: trailing-only features, training medians stored and applied at inference.

### Cache Utility
- **`cache_or_call` exists** -- Confirmed in `cache.cpython-311.pyc`. Function `cache_or_call` with parquet caching pattern. Docstring: "Load from parquet cache if available; otherwise call fetch_fn and cache."

### Notebook
- **7 sections present** -- Confirmed: (1) Problem Statement, (2) Dataset Description, (3) Data Cleaning, (4) EDA, (5) ML, (6) Results & Conclusions, (7) Limitations.
- **Executive Summary present** -- Confirmed: Cell 59, "### Executive Summary" with substantive content.
- **MLA 8 citations as last cell** -- Confirmed: Cell 61 (last cell), "## Dataset Citations (MLA 8)" with numbered entries.
- **62 total cells** -- Confirmed.

---

## FALSE POSITIVES -- Marked [x] But NOT Actually Implemented

### CRITICAL: Source Files Missing
- **ALL Layer source files (.py) are DELETED from disk.** Only `.pyc` bytecode remains:
  - `src/layers/layer1_price.py` -- MISSING (only .pyc exists)
  - `src/layers/layer2_fundamentals.py` -- MISSING (only .pyc exists)
  - `src/layers/layer3_concentration.py` -- MISSING (only .pyc exists)
  - `src/layers/layer4_macro.py` -- MISSING (only .pyc exists)
  - `src/layers/layer5_sentiment.py` -- MISSING (only .pyc exists)
  - `src/utils/cache.py` -- MISSING (only .pyc exists)
  - `src/layers/__init__.py` -- MISSING (only .pyc exists)
  - If the checklist claims these files exist as source, that is FALSE. The code ran at some point (bytecode was compiled), but the source has been removed from the working tree.

### HHI Scale Discrepancy
- If the checklist claims "HHI uses 0-10000 scale", this is **PARTIALLY FALSE**. The actual implementation uses a "percentage-based scale" where weights are in percentage form (e.g., 7.5 for 7.5%) and the HHI is the sum of squared percentages. The docstring explicitly states: "To convert to the standard antitrust 0-10,000 scale, multiply by 100." Values in this implementation range roughly 20-300 for a diversified index, NOT 0-10000. The chart axis label reads "HHI (percentage-based scale)" and the annotation says "20 = perfectly equal; >200 = highly concentrated."

### Missing FRED Series (if checklist claims specific ones)
- `T10Y3M` (10Y-3M spread) -- **NOT FOUND**. Only `T10Y2Y` (10Y-2Y spread) is fetched.
- `VIXCLS` (VIX) -- **NOT FOUND** in FRED series IDs. VIX may be obtained from another source, but it is not fetched via the FRED API in Layer 4. The feature matrix references `vix` as a potential column but Layer 4 does not produce it.

### Layer 5: Chart 5_5 Does Not Exist
- Layer 5 produces 5 charts numbered 5_1, 5_2, 5_3, 5_4, and 5_6. There is NO chart 5_5. If a checklist claims 6 charts for Layer 5, that is false. The count is 5 charts.

### scripts/build_slides.py Does Not Exist
- The `scripts/` directory does not exist at all. If the checklist claims `scripts/build_slides.py` is implemented, that is FALSE. No slide-building automation exists.

---

## FALSE NEGATIVES -- Marked [ ] But Actually Done

### Items That Appear Fully Implemented Despite Potential "TODO" Status
- **Breakout sensitivity analysis** -- `breakout_sensitivity` function exists in Layer 1 bytecode. Docstring: "Grid search over breakout parameter combinations." This may not be on the checklist at all.
- **Rolling beta computation** -- Layer 1 stats dict includes `rolling_beta_nvda`, `rolling_beta_csco` keys. Additional analysis beyond core requirements.
- **Multiple DTW reference periods** -- The DTW analysis compares against 4 reference periods (dotcom, housing, post_gfc, covid_rally), not just one. This exceeds a minimal checklist requirement.
- **Rolling DTW timeline** -- `_compute_rolling_dtw_timeline()` computes expanding-window DTW similarity over the AI era, producing a time series of similarity scores.
- **Ensemble classifier (RF + XGB + LR)** -- Three classifiers with weighted ensemble (0.4/0.4/0.2), not just a single Random Forest. This exceeds minimal requirements.
- **Model persistence (joblib)** -- Models, scalers, label encoders, feature matrices, and predictions are serialized to disk.
- **Cross-layer correlation matrix** -- Notebook Cell 39 builds a cross-layer correlation matrix for all 5 layers.
- **Bubble scorecard** -- Notebook Section 6.1 includes a synthesis scorecard across all 5 layers.

---

## Summary

| Metric | Count |
|--------|-------|
| Items verified correct | **34** |
| False positives (overclaimed) | **4** (source files missing, HHI scale wrong, 2 FRED series missing if claimed, chart 5_5 missing, build_slides.py missing) |
| False negatives (underclaimed) | **8** (extra features implemented beyond minimal requirements) |

### Risk Assessment

**HIGH RISK -- Source files deleted.** The most critical finding is that ALL layer source files and `cache.py` have been deleted from disk. Only compiled `.pyc` bytecode remains. This means:
1. The code cannot be reviewed, edited, or debugged without decompilation.
2. A Python version upgrade will invalidate the `.pyc` files entirely.
3. Any CI/CD pipeline that expects `.py` source files will fail.
4. The project cannot be meaningfully maintained.

**MEDIUM RISK -- HHI scale mismatch.** If presentations or the notebook claim "standard 0-10000 HHI", the actual code uses a different scale. The numbers will look wrong to anyone familiar with antitrust HHI values.

**LOW RISK -- Missing FRED series.** T10Y3M and VIXCLS are not fetched. The code still has 10 FRED series (uses DGS2 and SP500 instead), but if specific series are promised, they are not delivered.
