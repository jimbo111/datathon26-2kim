# Spec-to-Implementation Audit Report

**Auditor:** Code Review Agent (Claude)
**Date:** 2026-03-28
**Branch audited:** `jimmy/health-analysis` (current), cross-referenced with `jimmy/dev` (specs + layers)

---

## Critical Finding: Layer Source Files Missing from Current Branch

The 5 data layer source files (`layer1_price.py` through `layer5_sentiment.py`) exist on the `jimmy/dev` branch but are **missing from the current `jimmy/health-analysis` branch**. Only their `__pycache__/*.pyc` compiled bytecode files remain in `src/layers/__pycache__/`, compiled at 2026-03-28 23:08. The `__init__.py` for `src/layers/` is also missing. This means:

- The notebook (`2kim_finance_notebook.ipynb`) imports `from src.layers.layer1_price import run_layer1` etc., which will **fail on a clean checkout**.
- The notebook cannot pass the rubric's "Kernel > Restart & Run All" test.
- All chart generation, statistical tests, and data fetching that happens in layers 1-5 is unreachable from the submission branch.

**Severity: SHOWSTOPPER** -- Must merge or cherry-pick the layer files from `jimmy/dev` into `jimmy/health-analysis` before submission.

---

## Spec 00: Project Overview

### Matches

- Research question is stated correctly in the notebook (Cell 1): "Is the NVIDIA-led AI equity rally of 2023-2026 structurally analogous..."
- 5-layer + ML pipeline architecture exists conceptually in both the notebook and `src/ml/pipeline.py`
- Notebook follows the required 7-section structure (Sections 1-7 plus citations in Cell 61)
- H0 and H1 hypotheses are present (Cell 1)
- Data sources table is present (Cell 2)
- Color palette partially matches: NVDA_COLOR = `#00CC96` (matches dark bg spec), CSCO_COLOR = `#EF553B` (matches spec)
- Plotly dark template used (`plotly_dark`) -- matches dashboard requirement
- `feature_matrix.py` implements the unified monthly feature matrix (38 features across 5 categories as spec requires)
- `pipeline.py` implements regime classification, DTW similarity, SHAP analysis, and BH correction
- Regime labels in `feature_matrix.py` (lines 539-568) match the spec's historical period definitions
- Time-based train/val/test/live splits in `pipeline.py` (lines 1620-1628) match spec Section 2.7
- Cache-or-call pattern required by spec is implemented in layer files (via `src/utils/cache.py` on `jimmy/dev`)

### Gaps

1. **Layer source files missing from current branch** -- `src/layers/layer1_price.py` through `layer5_sentiment.py` and `src/layers/__init__.py` do not exist on `jimmy/health-analysis`. Only `.pyc` files in `__pycache__/` remain.
   - **Severity: CRITICAL**

2. **No `scripts/build_slides.py` on current branch** -- The spec requires `2kim_finance_slides.pptx` built by a script. `build_slides.py` exists on `jimmy/dev` (582 lines) but is missing from the current branch. No `.pptx` file exists in `submissions/`.
   - **Severity: CRITICAL**

3. **No `2kim_finance_slides.pptx` or `.pdf` submission file** -- Spec 00 line 10 requires `2kim_finance_slides.pptx`. Glob search returns no `.pptx` files in the project (only the python-pptx default template in `.venv`).
   - **Severity: CRITICAL**

4. **`src/utils/cache.py` missing from current branch** -- Layer files import `from src.utils.cache import cache_or_call`. The `src/utils/` directory on `jimmy/health-analysis` only has `__init__.py` and `export.py`. `cache.py` exists on `jimmy/dev`.
   - **Severity: CRITICAL**

5. **`requirements.txt` missing ML/DTW/sentiment dependencies** -- Spec 00 lines 524-542 require `scikit-learn`, `dtw-python`, `shap`, `fredapi`, `pytrends`, `praw`, `vaderSentiment`, `textblob`, `beautifulsoup4`, `lxml`, `xgboost`, `yfinance`, and `joblib`. Current `requirements.txt` only has the base data stack (pandas, numpy, plotly, scipy, statsmodels, etc.) -- missing all finance-specific and ML-specific packages.
   - **Severity: CRITICAL**

6. **File structure diverges from spec** -- Spec expects `src/loaders.py`, `src/cleaning.py`, `src/features.py`, `src/stats.py`, `src/ml.py`, `src/viz.py`. Actual structure uses `src/loaders/`, `src/analysis/`, `src/viz/`, `src/ml/` sub-packages. This is a structural difference but acceptable if imports work.
   - **Severity: LOW**

7. **Notebook Cell 4 imports `from src.layers.layer1_price import run_layer1`** -- These modules don't exist on the current branch. The notebook will crash at Cell 4.
   - **Severity: CRITICAL** (duplicate of gap #1, noted for completeness)

8. **`.env` variables incomplete** -- Spec requires `FRED_API_KEY`, `ALPHA_VANTAGE_KEY`, `OPENAI_API_KEY`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`. `.env.example` was not checked but `.env` exists. Not verifiable without reading it.
   - **Severity: MEDIUM**

---

## Spec 01: Layer 1 -- Price Comparison

### Matches

- `layer1_price.py` (on `jimmy/dev`) implements all required functionality:
  - `fetch_all_data()` downloads NVDA, AMD, SMCI, AVGO, ARM, MSFT (AI era) + CSCO, JNPR, QCOM, INTC (dot-com era) + NT (Nortel) + ^GSPC
  - `normalize_to_breakout()` rebases to 100 at detected breakout dates
  - `compute_drawdown()`, `compute_rolling_returns()`, `compute_log_returns()` all present
  - `find_breakout()` with systematic 200-day SMA crossover detection
  - `breakout_sensitivity()` for +/-60 trading day alignment sensitivity analysis
  - 4 chart functions: `chart_1_1_price_overlay`, `chart_1_2_drawdown`, `chart_1_3_rolling_returns`, `chart_1_4_volume_surge`
  - Nortel Networks (NT) included as non-survivor analog (gray dashed line)
  - Statistical tests: Pearson correlation with bootstrap CI, Spearman correlation, KS test on return distributions, ADF stationarity test, Ljung-Box autocorrelation test
  - Color scheme matches: NVDA_COLOR = `#00CC96`, CSCO_COLOR = `#EF553B`, NORTEL_COLOR = `#888888`

### Gaps

1. **Source file not on current branch** -- `src/layers/layer1_price.py` only exists on `jimmy/dev`.
   - **Severity: CRITICAL**

2. **Chart 1.1 in notebook (Cell 11) shows the chart** but only works if layer1 ran successfully. The notebook does not have fallback logic for chart 1.1 (unlike chart 2.6 and 3.3 which have inline fallbacks).
   - **Severity: LOW**

3. **Spec requires SUNW (Sun Microsystems) or DELL as fallback** (spec line 42-43). Layer1 implementation uses `DOTCOM_TICKERS = ["CSCO", "JNPR", "QCOM", "INTC"]` -- SUNW/JAVA/DELL is missing.
   - **Severity: LOW** (SUNW is listed as a peer, not a critical data source)

4. **Spec requires ^IXIC (NASDAQ Composite) and QQQ** as market indices (spec lines 48-51). Implementation only fetches `^GSPC` (S&P 500).
   - **Severity: LOW** (^GSPC is the primary benchmark per the ML pipeline; NASDAQ is supplementary)

5. **Chart 1.3 (Rolling Returns) uses 90-day** in the implementation but spec 07 calls it "Rolling 12-Month Returns". The implementation computes both 30d and 90d rolling returns, but not 252d (12-month) as labeled in spec 07.
   - **Severity: LOW**

---

## Spec 02: Layer 2 -- Fundamentals

### Matches

- `layer2_fundamentals.py` (on `jimmy/dev`) implements:
  - `_fetch_yf_financials()` for quarterly financials via yfinance
  - `compute_pe_trajectory()`, `compute_peg_ratio()`, `compute_revenue_growth()`, `compute_fcf_yield()`, `compute_rd_ratio()`, `compute_gross_margin()`
  - TTM (trailing twelve months) computation
  - Inflation adjustment via CPI from FRED
  - All 6 charts: `chart_2_1_pe_trajectory`, `chart_2_2_revenue_growth`, `chart_2_3_mcap_vs_revenue`, `chart_2_4_fcf_yield`, `chart_2_5_rd_investment`, `chart_2_6_peg_ratio`
  - Statistical tests: t-test on log P/E, t-test on P/S ratio, Chow test for structural break, bootstrap confidence interval on mean P/E difference
  - `assign_cycle_quarter()` for phase-aligned comparisons
  - Charts saved as both JSON and PNG

### Gaps

1. **Source file not on current branch** -- only on `jimmy/dev`.
   - **Severity: CRITICAL**

2. **Spec requires EV/EBITDA and Debt/Equity ratios** (spec 00 line 173). Implementation computes P/E, P/S, FCF yield, R&D ratio, gross margin, PEG -- but no EV/EBITDA or Debt/Equity.
   - **Severity: MEDIUM**

3. **Spec requires comparing at t+250, t+500, t+750 trading days** (spec 00 line 220). Implementation uses `assign_cycle_quarter()` for quarterly alignment, not explicit trading-day snapshots.
   - **Severity: LOW** (quarterly alignment is arguably better for fundamental data)

4. **Chart 2.3 (Market Cap vs Revenue) uses bubble scatter** as spec requires, but the notebook cell 21 has a fallback for chart_2_6 that shows chart_2_3 instead if 2_6 is missing -- potentially confusing the narrative.
   - **Severity: LOW**

---

## Spec 03: Layer 3 -- Market Concentration

### Matches

- `layer3_concentration.py` (on `jimmy/dev`) implements:
  - `_fetch_etf_prices()` for SPY, RSP, QQQ
  - `_fetch_top_holdings_info()` for current S&P 500 holdings
  - `_fetch_fred_series()` for GDP (Buffett Indicator)
  - `compute_sp500_weights()`, `compute_top_n_concentration()`, `compute_ai_concentration()`
  - `compute_spy_rsp_spread()` with cumulative return divergence
  - `compute_buffett_indicator()` (Total Market Cap / GDP)
  - `compute_hhi()` (Herfindahl-Hirschman Index)
  - `_build_concentration_timeseries()` constructing historical time series
  - `concentration_reversibility()` analysis
  - All 5 charts: `chart_3_1_concentration_timeline`, `chart_3_2_spy_vs_rsp`, `chart_3_3_buffett_indicator`, `chart_3_4_treemap`, `chart_3_5_hhi`
  - Statistical tests: z-test for current concentration vs historical mean, Pearson correlation, Mann-Whitney U test

### Gaps

1. **Source file not on current branch** -- only on `jimmy/dev`.
   - **Severity: CRITICAL**

2. **Spec requires historical S&P 500 top-10 constituent weights from Slickcharts/Wikipedia** (spec 00 line 101). Implementation uses current ETF holdings via yfinance and approximates historical concentration from SPY/RSP ETF price ratios -- an indirect method. Historical annual top-10 snapshots are not downloaded.
   - **Severity: MEDIUM** (the approximation is reasonable but acknowledged as a limitation)

3. **Spec 03 line 596 calls for a scatter plot of concentration level vs forward 12-month return** with regression line. No such chart exists in the implementation. This is listed as a "chart recommendation" in the spec's critique additions.
   - **Severity: LOW**

4. **Spec requires sector-level HHI** (spec 00 line 180: "HHI by sector"). Implementation computes HHI from individual stock weights within S&P 500, not sector-level breakdowns.
   - **Severity: LOW** (stock-level HHI is more granular and arguably better)

---

## Spec 04: Layer 4 -- Macro Environment

### Matches

- `layer4_macro.py` (on `jimmy/dev`) implements:
  - `fetch_all_series()` downloads from FRED: FEDFUNDS, M2SL, T10Y2Y, T10Y3M, BAAFFM, AAA, DGS10, GDP, UNRATE, VIXCLS, CPIAUCSL, WILSHIRE (Wilshire 5000)
  - `build_macro_dashboard_data()` unifies all series to monthly frequency
  - `_compute_gdp_yoy_growth()` correctly uses `pct_change(periods=4)` on raw quarterly data before forward-filling to monthly (matches spec 00 lines 186-188)
  - `compare_eras()` for dot-com vs AI era statistical comparison (Welch's t-test)
  - `detect_yield_curve_inversions()` with duration tracking
  - `compute_lead_lag()` for lead-lag analysis
  - All 6 charts: `chart_4_1_fed_funds`, `chart_4_2_m2_growth`, `chart_4_3_yield_curve`, `chart_4_4_credit_spreads`, `chart_4_5_macro_dashboard`, `chart_4_6_real_rate`
  - Statistical tests: Welch's t-test for era comparison, cosine similarity, Granger causality

### Gaps

1. **Source file not on current branch** -- only on `jimmy/dev`.
   - **Severity: CRITICAL**

2. **Spec requires `BAAFFM` (BAA-AAA credit spread) from FRED**. Implementation fetches `BAAFFM` and `AAA` separately and computes `credit_spread = BAAFFM - AAA` (which gives BAA minus AAA, i.e., the risk premium). However, `BAAFFM` from FRED is already the BAA effective yield, not the BAA-AAA spread. The code should fetch `BAAFFM` and subtract `AAA` to get the spread, OR use the FRED series `BAAFFM` directly if it's the spread. Need to verify the FRED series definition.
   - **Severity: LOW** (implementation appears correct -- BAAFFM is BAA yield, AAA is AAA yield, so the difference is the credit spread)

3. **Chart 4.5 (Macro Dashboard) spec calls for 2x3 grid (9 subplots)** matching 6 macro indicators. Notebook Cell 29 references `chart_4_5` directly from layer4 results. Implementation uses `make_subplots(rows=3, cols=2)` which is 3x2 = 6 subplots, not 2x3 = 6 or 3x3 = 9. The notebook cell 29 markdown says "2x3 small multiples" but the spec itself says 9 subplots in some places.
   - **Severity: LOW** (6 subplots covers the core indicators)

4. **Spec 04 requires real GDP growth** -- implementation computes `gdp_yoy_growth` from nominal GDP. The spec does not specify real vs nominal explicitly, but mentions "GDP YoY growth" which could be either.
   - **Severity: LOW**

---

## Spec 05: Layer 5 -- Sentiment & NLP

### Matches

- `layer5_sentiment.py` (on `jimmy/dev`) implements:
  - SEC EDGAR filing fetch for NVDA (CIK 1045810) and CSCO (CIK 858877)
  - Keyword frequency analysis (AI/technology terms per 1K words)
  - OpenAI GPT-4o-mini scoring for hype, specificity, and revenue detail
  - Google Trends fetch via pytrends for AI-related and historical keywords
  - Charts: `chart_5_1_ai_mentions`, `chart_5_2_hype_vs_specificity`, `chart_5_3_google_trends`, `chart_5_4_sentiment_price_overlay`, `chart_5_6_keyword_comparison`
  - Statistical tests: Mann-Whitney U, Pearson correlation (hype vs price)

### Gaps

1. **Source file not on current branch** -- only on `jimmy/dev`.
   - **Severity: CRITICAL**

2. **Reddit integration explicitly SKIPPED** -- Layer5 line 7 states "Reddit (sub-layer 5C) is explicitly SKIPPED per project instructions." Spec 05 requires Reddit data from PRAW (chart 5.5: Reddit mention volume, plus weekly post count and sentiment). Chart 5.5 does not exist in the implementation.
   - **Severity: MEDIUM** (spec 00 line 99 lists Reddit as data source #7; spec 09 line 74 requires "Reddit activity chart"; however the critique may have deprioritized it)

3. **No `chart_5_5_reddit_mention_volume` function** -- Missing entirely from layer5. The notebook (Cell 35) has fallback logic for `chart_5_4` but no mention of chart 5.5.
   - **Severity: MEDIUM**

4. **Word cloud chart (5.6) uses matplotlib** while spec says it should be included. Implementation has `chart_5_6_keyword_comparison` which is a bar chart comparison, not a word cloud. Spec 05 line 1163 calls for `create_chart_5_6` as a word cloud.
   - **Severity: LOW** (keyword comparison bar chart is arguably more informative)

5. **Spec requires TextBlob or VADER sentiment for Reddit backup** (spec 00 line 531). Neither is used since Reddit is skipped.
   - **Severity: LOW** (consequential gap from Reddit being skipped)

6. **Spec requires running each transcript through GPT-4o-mini 3x and taking median** (spec 00 line 476: "Run each transcript 3x, take median score"). Implementation runs scoring once per filing. No triple-run averaging.
   - **Severity: MEDIUM** (reduces sentiment reliability and reproducibility)

---

## Spec 06: ML Pipeline

### Matches

- `src/ml/pipeline.py` (1856 lines, present on current branch) implements:
  - Regime Classifier: RF (n_estimators=200, max_depth=8), XGBoost, Logistic Regression
  - Price-excluded RF variant (spec Section 2.6)
  - TimeSeriesSplit cross-validation with gap parameter
  - Weighted ensemble (RF=0.4, XGB=0.4, LR=0.2)
  - DTW Similarity with permutation test (n=500), Sakoe-Chiba band (33%)
  - Bootstrap confidence intervals on DTW scores
  - SHAP TreeExplainer analysis with summary and waterfall plots
  - Benjamini-Hochberg correction across all p-values
  - All 4 ML charts: `chart_ml_1_regime_probabilities`, `chart_ml_2_dtw_similarity`, `chart_ml_3_shap_summary`, `chart_ml_4_shap_waterfall`
  - Time-based splits: Train (1990-2018), Val (2019-2021), Test (2022-Jun 2024), Live (Jul 2024-Mar 2026)
  - Model artifacts saved to `models/` directory (RF, XGB, LR, scaler, label encoder)
  - Feature matrix saved to `data/features/`
  - BH correction results saved to `results/bh_correction.csv`
  - `run_ml_pipeline()` orchestrates all 11 steps

- `src/ml/feature_matrix.py` (615 lines, present on current branch) implements:
  - 9 price features, 7 fundamental features, 6 concentration features, 9 macro features, 7 sentiment features
  - Feature names match spec exactly (momentum_30d/90d/252d, volatility_30d/90d, drawdown_from_ath, rsi_14, price_to_sma200, return_skewness, etc.)
  - `assign_regime_labels()` with all historical period definitions matching spec Section 2.2
  - Monthly resampling, no look-ahead bias
  - Macro forward-fill with 3-month limit
  - 50% NaN threshold for row dropping
  - Sentiment features not imputed (two separate models: core vs augmented)

### Gaps

1. **Pipeline depends on layer results that can't be generated** -- `run_ml_pipeline(layer_results)` requires output from `run_layer1()` through `run_layer5()`, which can't be called because layer files are missing from the current branch.
   - **Severity: CRITICAL**

2. **Spec Section 5 requires robustness checks:**
   - **Ablation study** (remove one layer at a time, report accuracy change) -- NOT implemented in `pipeline.py`. The pipeline runs one full model and one price-excluded variant, but no systematic leave-one-layer-out ablation.
   - **Severity: MEDIUM**
   - **Breakout sensitivity (+/-60 days)** -- Layer1 has `breakout_sensitivity()` on `jimmy/dev` but the ML pipeline doesn't consume or report this. The notebook doesn't show DTW sensitivity to t=0 alignment.
   - **Severity: MEDIUM**

3. **Spec requires 1000 bootstrap resamples** (spec 09 line 92). Implementation uses `n_bootstrap=200` in `compute_dtw_similarity()` (pipeline.py line 731).
   - **Severity: MEDIUM**

4. **Spec requires DTW on individual features** (price, volume, P/E, sentiment) per spec 09 lines 91-92. Implementation computes per-category DTW (price, fundamentals, concentration, macro, sentiment) which is conceptually the same but uses averaged features within each category, not single raw features.
   - **Severity: LOW**

5. **Regime labels use 4 classes** (bubble, correction, normal_growth, recovery) in the ML pipeline, but the **notebook** (Cell 48) uses 5 classes (Accumulation, Expansion, Euphoria, Distribution, Decline). These are two different labeling schemes -- the notebook's inline ML code doesn't align with `pipeline.py`.
   - **Severity: MEDIUM** (the notebook reimplements a simpler version rather than calling `run_ml_pipeline()`)

6. **`yield_curve_10y3m` feature listed in feature_matrix docstring** (line 14) but `_build_macro_features()` does not extract it. Only `yield_curve` (10Y-2Y) is mapped.
   - **Severity: LOW**

7. **DTW permutation test p-value** in `results/bh_correction.csv` shows `dtw_permutation_dotcom` has `raw_p=1.0` which suggests the permutation test found the observed similarity was NOT statistically significant. This is a results issue, not a code gap, but worth noting.
   - **Severity: N/A** (finding, not gap)

---

## Spec 07: Visualization Plan

### Matches

- Charts generated on `jimmy/dev` branch (confirmed in `submissions/charts/` on current branch):
  - Layer 2: chart_2_1 through chart_2_6 (PNG + JSON) -- 6 charts
  - ML: chart_ml_1_regime_probabilities, chart_ml_2_dtw_similarity (PNG + JSON) -- 2 charts
  - Total on disk: 8 chart pairs (JSON + PNG)

### Gaps

1. **15 PRIMARY charts required, only 8 on disk** -- Missing:
   - chart_1_1 (normalized price overlay -- the HOOK chart, highest priority)
   - chart_1_2 (drawdown comparison)
   - chart_3_1 (concentration timeline)
   - chart_3_2 (SPY vs RSP spread)
   - chart_3_3 (Buffett Indicator)
   - chart_4_1 (Fed funds overlay)
   - chart_4_3 (Yield curve)
   - chart_4_5 (Macro dashboard)
   - chart_5_1 (AI mentions in earnings calls)
   - chart_5_2 (Hype vs specificity)
   - chart_ml_3 (SHAP summary)
   - chart_ml_4 (SHAP waterfall)
   - **Severity: CRITICAL** (charts 1.1 and 3.1 are Sprint 1 priorities)

2. **15 SECONDARY charts required, 0 confirmed on disk** -- None of the secondary charts (1.3, 1.4, 2.4, 2.5, 3.4, 3.5, 4.2, 4.4, 4.6, 5.3, 5.4, 5.5, 5.6) are in `submissions/charts/`. Some (2.4, 2.5) may be generated at runtime by layer2 but aren't cached.
   - **Severity: MEDIUM** (secondary charts are "nice to have")

3. **Chart 1.1 is listed as THE hook chart** (spec 07 line 79, spec 08 slide 2) and is Sprint 1 priority #1. Its absence means the most important single visual for the presentation is missing from the output.
   - **Severity: CRITICAL**

4. **Spec requires "Inter" font family** for all charts (spec 07 line 51). Implementation uses `plotly_dark` template which does not set Inter as the font. Layer files use the Plotly default.
   - **Severity: LOW**

5. **Light theme template specified** in spec 00 (lines 500-516) for notebook charts, but implementation uses `plotly_dark` template everywhere. Spec says light bg (`#ffffff`) for notebook, dark for dashboard.
   - **Severity: LOW** (consistent dark theme is acceptable for a datathon)

---

## Spec 08: Presentation Narrative

### Matches

- Spec 08 defines 16 slides (1 title + 14 content + 1 close) + 2 backup slides
- The `build_slides.py` script on `jimmy/dev` (582 lines) implements slide generation using python-pptx

### Gaps

1. **`build_slides.py` not on current branch** -- Script exists on `jimmy/dev` but not `jimmy/health-analysis`.
   - **Severity: CRITICAL**

2. **No `.pptx` file exists anywhere in the project** -- The submission requires `2kim_finance_slides.pptx` uploaded alongside the notebook.
   - **Severity: CRITICAL**

3. **Even on `jimmy/dev`, the slide builder depends on chart files** that are only partially generated. Missing charts (1.1, 3.1, 3.2, 3.3, 4.5, 5.1, 5.2) would cause the builder to produce slides with placeholder or missing visuals.
   - **Severity: CRITICAL**

4. **Spec requires specific speaking notes per slide** (spec 08 lines 83-84, 97, etc.). The slide builder on `jimmy/dev` likely generates notes from the spec, but the actual notes content was not verified.
   - **Severity: LOW**

---

## Spec 09: Submission Checklist

### Matches (against `2kim_finance_notebook.ipynb`)

- Section 1 (Cell 1): Research question present, H0/H1 stated, significance discussed, track stated
- Section 2 (Cell 2): Data sources table present
- Section 3 (Cells 3-8): Data cleaning code exists (loading layers, quality audits)
- Section 4 (Cells 10-39): EDA with charts from all 5 layers + cross-layer correlation
- Section 5 (Cells 40-57): ML section with DTW, regime classification, SHAP, BH correction
- Section 6 (Cells 58-59): Results with executive summary
- Section 7 (Cell 60): Limitations, ethics, future work
- Section 8 (Cell 61): MLA 8 citations present

### Gaps

1. **Notebook won't run top-to-bottom** -- Cell 4 imports from `src.layers.*` which don't exist on the current branch. This is the #1 rubric risk (Technical Rigor, 15pts).
   - **Severity: CRITICAL**

2. **Section 2 does not show `df.head()` and `df.shape` after loading** as required (spec 09 line 25). Cells 6-8 do show this for some layers but it depends on layer1-5 running first.
   - **Severity: MEDIUM**

3. **Section 3 does not show missing values BEFORE and AFTER cleaning** side by side (spec 09 lines 34, 41). Missing value info is printed per layer but not in a consolidated before/after comparison.
   - **Severity: MEDIUM**

4. **Bubble Scorecard table** (spec 09 lines 112-123) -- Cell 58 has a markdown cell titled "Bubble Scorecard" but it's a narrative description, not a formatted table with the specific columns (Dimension, Metric, CSCO @ Peak, NVDA @ Current, Similarity 1-5, Signal color).
   - **Severity: MEDIUM**

5. **Executive Summary** (spec 09 line 130) -- Cell 59 has "Executive Summary" as a markdown cell. The format is narrative prose rather than the required "3 findings in 3 sentences" format. However, it does contain the key findings.
   - **Severity: LOW**

6. **Spec requires minimum 10 distinct visualizations** -- The notebook references ~15 charts across all layers, but many depend on layer results that can't be generated. If layers don't run, 0 charts display.
   - **Severity: CRITICAL** (blocked by layer file absence)

7. **Spec requires minimum 4 statistical tests with p-values** -- The notebook shows statistical test results from all 5 layers (Cells 15, 22, 30, 36) plus BH correction (Cell 57). Tests include Pearson, Spearman, KS test, t-tests, Mann-Whitney, cosine similarity. This meets the requirement IF layers run.
   - **Severity: CRITICAL** (blocked by layer file absence)

8. **Spec requires warping path visualization** (spec 09 line 89). Neither the ML pipeline nor the notebook generates a DTW warping path plot. The notebook (Cell 45) shows a rolling DTW similarity timeline but not the actual alignment/warping path.
   - **Severity: MEDIUM**

9. **Robustness checks section incomplete** -- Spec 09 lines 105-107 require: (a) t=0 alignment sensitivity, (b) ablation study, (c) random seeds documented. The notebook does not show (a) or (b). Random seeds are set (Cell 41: `np.random.seed(42)`).
   - **Severity: MEDIUM**

10. **Notebook reimplements ML inline** rather than calling `run_ml_pipeline()` -- Cells 43-51 build DTW, feature matrix, regime classification, and SHAP from scratch using inline code. This creates divergence from the `src/ml/pipeline.py` implementation (different regime labels, different feature sets, different model parameters).
    - **Severity: MEDIUM** (risk of inconsistency between the two implementations)

11. **Notebook regime labels differ from pipeline.py** -- Notebook Cell 48 uses 5 regimes (Accumulation, Expansion, Euphoria, Distribution, Decline) while `pipeline.py` uses 4 (bubble, correction, normal_growth, recovery). The notebook's approach is CSCO-era-specific training, while the pipeline uses full historical data.
    - **Severity: MEDIUM**

12. **No prior literature references** -- Spec 09 line 17 requires "at least 2 prior works or known analyses on tech bubbles (cited in MLA 8)." Cell 1 discusses the topic but does not cite specific prior academic works inline. Cell 61 has MLA citations for data sources but not for analytical precedents.
    - **Severity: MEDIUM**

---

## Summary

### Total Gaps Found: 47

### Critical Gaps (must fix before submission): 12

| # | Gap | File/Location |
|---|-----|---------------|
| 1 | Layer source files (layer1-5) missing from current branch | `src/layers/*.py` |
| 2 | `src/layers/__init__.py` missing | `src/layers/__init__.py` |
| 3 | `src/utils/cache.py` missing from current branch | `src/utils/cache.py` |
| 4 | `build_slides.py` missing from current branch | `scripts/build_slides.py` |
| 5 | No `.pptx` slide file exists | `submissions/2kim_finance_slides.pptx` |
| 6 | `requirements.txt` missing ML/finance dependencies | `requirements.txt` |
| 7 | Notebook crashes at Cell 4 (import failure) | `notebooks/2kim_finance_notebook.ipynb` |
| 8 | Chart 1.1 (the hook) not generated | `submissions/charts/chart_1_1_*` |
| 9 | 12+ charts missing from output | `submissions/charts/` |
| 10 | Notebook cannot pass "Restart & Run All" | `notebooks/2kim_finance_notebook.ipynb` |
| 11 | 0 visualizations display without layer files | Notebook cells 11-39 |
| 12 | 0 statistical tests display without layer files | Notebook cells 15-36 |

### Medium Gaps (should fix): 16

| # | Gap |
|---|-----|
| 1 | Reddit (PRAW) integration skipped entirely (spec requires it as data source #7) |
| 2 | Chart 5.5 (Reddit mention volume) missing |
| 3 | GPT-4o-mini scoring not triple-run for median (spec requires 3x for reproducibility) |
| 4 | Bootstrap uses 200 resamples, spec requires 1000 |
| 5 | No ablation study (remove one layer at a time) |
| 6 | No breakout sensitivity analysis in final output |
| 7 | Notebook regime labels (5 classes) diverge from pipeline.py (4 classes) |
| 8 | `.env` variables may be incomplete for all required APIs |
| 9 | No `df.head()`/`df.shape` shown after loading in notebook Section 2 |
| 10 | No before/after missing value comparison in notebook Section 3 |
| 11 | Bubble Scorecard is prose, not the required table format |
| 12 | DTW warping path visualization missing |
| 13 | No prior literature cited inline (spec requires 2+) |
| 14 | Notebook inline ML diverges from pipeline.py |
| 15 | EV/EBITDA and Debt/Equity ratios not computed (spec 00 requires them) |
| 16 | S&P 500 historical weights approximated, not actual annual snapshots |

### Low Gaps (nice to have): 19

| # | Gap |
|---|-----|
| 1 | SUNW/JAVA ticker not fetched (spec lists as dot-com peer) |
| 2 | ^IXIC and QQQ not fetched (spec lists as market indices) |
| 3 | Chart 1.3 uses 90-day returns, spec 07 labels it "12-month" |
| 4 | t+250/500/750 day snapshots not explicit (quarterly alignment used instead) |
| 5 | Notebook chart 2.6 fallback logic is confusing |
| 6 | Concentration scatter (level vs forward return) not implemented |
| 7 | Sector-level HHI not computed (stock-level used) |
| 8 | FRED credit spread computation might double-count |
| 9 | Chart 4.5 subplot count (6 vs 9) |
| 10 | GDP growth uses nominal (spec ambiguous) |
| 11 | Chart 5.6 is bar chart, not word cloud |
| 12 | TextBlob/VADER not used (Reddit skipped) |
| 13 | `yield_curve_10y3m` feature documented but not extracted |
| 14 | DTW per-feature uses averaged categories, not raw single features |
| 15 | File structure differs from spec (sub-packages vs flat modules) |
| 16 | Inter font not set in Plotly charts |
| 17 | Light theme not used for notebook (dark everywhere) |
| 18 | Executive Summary format is prose, not "3 findings in 3 sentences" |
| 19 | Slide speaking notes not verified |

---

## Recommended Fix Priority

### Immediate (blocks submission):

1. **Merge layer files from `jimmy/dev`**: Cherry-pick commit `372cfbf` ("Implement all 5 data analysis layers") and `c5480d3` ("Add implementation infrastructure: checklist, cache utility, directory structure") into `jimmy/health-analysis`.

2. **Merge build_slides.py from `jimmy/dev`**: Cherry-pick commit `c55d7f2` ("Add presentation slide builder and generated .pptx").

3. **Update requirements.txt**: Add `scikit-learn>=1.5`, `xgboost>=2.0`, `shap>=0.45`, `dtw-python>=1.5`, `yfinance>=0.2`, `fredapi>=0.5`, `pytrends>=4.9`, `joblib>=1.3`, `python-pptx>=0.6`, `kaleido>=0.2`.

4. **Run all 5 layers + ML pipeline to generate missing charts**: Execute `run_layer1()` through `run_layer5()`, then `run_ml_pipeline()`. This will populate `submissions/charts/` with all 30 charts.

5. **Run `build_slides.py`** to generate `2kim_finance_slides.pptx`.

6. **Test notebook end-to-end**: `Kernel > Restart & Run All` on `2kim_finance_notebook.ipynb`.

### Before polish:

7. Format the Bubble Scorecard as a proper table (Section 6).
8. Add 2 prior literature citations in Section 1.
9. Add before/after missing value counts in Section 3.
10. Increase DTW bootstrap resamples from 200 to 1000.
11. Reconcile notebook inline ML (5 regimes) with pipeline.py (4 regimes) -- pick one approach.
