# Implementation Checklist — AI Hype Cycle Datathon Submission

> Auto-updated as agents complete work. Each `[ ]` becomes `[x]` when done.

---

## Infrastructure
- [x] Directory structure (`src/layers/`, `src/ml/`, `data/raw/`, `data/processed/`, `submissions/charts/`)
- [x] Cache-or-call utility (`src/utils/cache.py`)
- [x] Install dependencies (fredapi, pytrends, xgboost, shap, dtw-python, kaleido)
- [ ] `.env` configured (FRED_API_KEY, ALPHA_VANTAGE_KEY, OPENAI_API_KEY) ← USER ACTION
- [x] Verify all imports work in clean environment

---

## Layer 1: Price Comparison (spec: `01_data_layer_price_comparison.md`)

### Data Collection
- [x] Fetch NVDA daily prices (Jan 2023 – Mar 2026) via yfinance
- [x] Fetch CSCO daily prices (Jan 1998 – Dec 2002) via yfinance
- [x] Fetch Nortel Networks (NT) prices (Jan 1998 – delisting) via yfinance
- [x] Fetch peer stocks: AMD, SMCI, AVGO, ARM, MSFT (AI era)
- [x] Fetch peer stocks: JNPR, QCOM, INTC (dot-com era)
- [x] Fetch index data: ^GSPC for both eras
- [x] All data cached as parquet

### Analysis
- [x] Breakout point detection algorithm (50% trailing 252d return, 20d sustain)
- [x] Breakout sensitivity analysis (40/50/60% thresholds × 10/20/30 sustain days)
- [x] Normalize prices to index 100 at breakout
- [x] Weekly resampling for overlay
- [x] Compute daily log-returns
- [x] Compute rolling returns (30d, 90d, 252d)
- [x] Compute drawdown from ATH
- [x] Compute volume z-score

### Statistical Tests
- [x] ADF stationarity test on log-returns
- [x] Pearson/Spearman correlation of log-returns (aligned by days_from_breakout)
- [x] Ljung-Box autocorrelation test
- [x] KS test on return distributions
- [x] DTW distance (primary quantitative measure — in ML pipeline)

### Charts
- [x] Chart 1.1: Normalized price overlay (NVDA green, CSCO red, Nortel gray dashed)
- [x] Chart 1.2: Drawdown from ATH comparison
- [x] Chart 1.3: Rolling 90-day return comparison
- [x] Chart 1.4: Volume surge timeline

---

## Layer 2: Fundamentals (spec: `02_data_layer_fundamentals.md`)

### Data Collection
- [x] Fetch NVDA quarterly financials (yfinance + Alpha Vantage cross-reference)
- [x] Fetch CSCO quarterly financials (yfinance + EDGAR validation note)
- [x] Fetch CPI data from FRED (CPIAUCSL) for inflation adjustment
- [x] Fetch peer fundamentals (if available)

### Analysis
- [x] Compute P/E ratio trajectory (raw + log-transformed)
- [x] Compute PEG ratio (P/E / earnings growth rate)
- [x] Compute revenue growth rate (YoY, QoQ)
- [x] Compute free cash flow yield
- [x] Compute R&D-to-revenue ratio
- [x] Compute gross margin trajectory
- [x] CPI inflation adjustment to 2026 dollars
- [x] Fiscal-to-calendar quarter alignment (NVDA Jan FY, CSCO Jul FY)

### Statistical Tests
- [x] Two-sample bootstrap test on log(P/E) distributions (with Cohen's d)
- [x] Shapiro-Wilk normality test
- [x] Durbin-Watson autocorrelation check

### Charts
- [x] Chart 2.1: P/E ratio trajectory overlay
- [x] Chart 2.2: Revenue growth rate comparison (grouped bar)
- [x] Chart 2.3: Market cap vs revenue scatter
- [x] Chart 2.4: Free cash flow yield timeline
- [x] Chart 2.5: R&D investment comparison
- [x] Chart 2.6: PEG ratio comparison

---

## Layer 3: Market Concentration (spec: `03_data_layer_market_concentration.md`)

### Data Collection
- [x] Fetch S&P 500 top constituents by market cap (current)
- [x] Fetch SPY and RSP ETF prices
- [x] Fetch sector ETFs: XLK, SMH
- [x] Fetch Wilshire 5000 from FRED (WILL5000IND)
- [x] Fetch GDP from FRED (GDP) for Buffett Indicator
- [x] Historical S&P 500 concentration data (dot-com era via anchor points)

### Analysis
- [x] Top-10 concentration ratio over time
- [x] AI-specific concentration
- [x] SPY vs RSP spread
- [x] Buffett Indicator with Fed-adjusted variant
- [x] HHI index computation (0-10000 scale)
- [x] Concentration-reversibility analysis
- [x] Passive investing microstructure adjustment note

### Statistical Tests
- [x] Compare concentration ratios across eras (Welch's t-test)
- [x] Concentration-drawdown predictive correlation

### Charts
- [x] Chart 3.1: Top-10 concentration ratio timeline
- [x] Chart 3.2: SPY vs RSP cumulative return spread
- [x] Chart 3.3: Buffett Indicator historical
- [x] Chart 3.4: Treemap of S&P 500 by market cap
- [x] Chart 3.5: HHI index over time

---

## Layer 4: Macro Environment (spec: `04_data_layer_macro_environment.md`)

### Data Collection (all FRED API)
- [x] Federal Funds Rate (FEDFUNDS)
- [x] 10-Year Treasury (DGS10)
- [x] 2-Year Treasury (DGS2)
- [x] Yield Curve Spread 10Y-2Y (T10Y2Y)
- [x] M2 Money Supply (M2SL)
- [x] CPI (CPIAUCSL)
- [x] Corporate Credit Spreads (BAA10Y)
- [x] GDP (GDP)
- [x] Unemployment Rate (UNRATE)
- [x] S&P 500 (SP500)

### Analysis
- [x] Align different frequencies to monthly
- [x] Compute derived metrics: real rate, M2 growth rate, GDP YoY (pct_change(4) on quarterly)
- [x] Yield curve inversion detection and duration
- [x] Era comparison: 1996-2003 vs 2021-2026
- [x] Peak-aligned macro comparison

### Statistical Tests
- [x] Welch's t-test comparing macro means across eras
- [x] Cross-correlation lead-lag analysis
- [x] Granger causality: yield curve → equity drawdowns (with ADF + AIC lag selection)
- [x] Levene's variance test
- [x] All findings framed as associations, NOT causal claims

### Charts
- [x] Chart 4.1: Fed Funds Rate dual-era overlay
- [x] Chart 4.2: M2 Money Supply growth rate comparison
- [x] Chart 4.3: Yield Curve (10Y-2Y) with crash annotations
- [x] Chart 4.4: Credit Spreads timeline
- [x] Chart 4.5: Macro Dashboard (3x2 small multiples)
- [x] Chart 4.6: Real Interest Rate comparison

---

## Layer 5: Sentiment & NLP (spec: `05_data_layer_sentiment_nlp.md`)

### Data Collection
- [x] Google Trends: "NVIDIA stock", "AI stocks", "AI bubble" (pytrends)
- [x] SEC EDGAR: NVDA 10-Q/10-K filings (AI keyword counting)
- [x] SEC EDGAR: CSCO 10-Q filings 1998-2001 ("internet" keyword count) — REQUIRED
- [x] OpenAI sentiment scoring (optional, guarded by API key)
- [x] ~~Reddit API~~ (SKIPPED per user request)

### Analysis
- [x] AI mention frequency in NVDA filings over time
- [x] "Internet" mention frequency in CSCO filings 1998-2001
- [x] Google Trends normalization and timeline
- [x] OpenAI sentiment scoring (guarded behind key check)
- [x] Keyword density correlation check

### Statistical Tests
- [x] Hype-price correlation
- [x] Mann-Whitney U on quarterly sentiment aggregates

### Charts
- [x] Chart 5.1: AI mentions in earnings calls over time
- [x] Chart 5.2: Hype score vs specificity (or keyword density fallback)
- [x] Chart 5.3: Google Trends comparison
- [x] Chart 5.4: Sentiment timeline with stock price overlay
- [x] ~~Chart 5.5: Reddit~~ (SKIPPED)
- [x] Chart 5.6: Keyword frequency comparison

---

## ML Pipeline (spec: `06_ml_pipeline.md`)

### Feature Engineering
- [x] Build feature matrix from all 5 layers (38 features)
- [x] Time-based train/test split (pre-2022 train, 2022+ test)
- [x] Feature scaling (StandardScaler)
- [x] Handle missing sentiment features with two-model approach (core + augmented)

### Model 1: Regime Classifier
- [x] Label historical periods (bubble, correction, normal_growth, recovery)
- [x] Train Random Forest (primary)
- [x] Train XGBoost (secondary, graceful degradation if unavailable)
- [x] Train Logistic Regression (baseline)
- [x] Train price-feature-excluded RF variant
- [x] TimeSeriesSplit cross-validation
- [x] Evaluate: accuracy, F1, confusion matrix
- [x] Generate March 2026 regime probabilities

### Model 2: DTW Similarity
- [x] Normalize features to [0,1]
- [x] Compute per-feature DTW distance (AI era vs dot-com)
- [x] Weighted composite similarity score
- [x] Null distribution: compare against 3-5 other historical periods
- [x] Permutation test (500 shuffles) for p-value
- [x] Rolling similarity timeline
- [x] Bootstrap confidence intervals

### Model 3: SHAP Analysis
- [x] SHAP TreeExplainer on RF
- [x] SHAP summary plot (beeswarm)
- [x] SHAP waterfall for March 2026 prediction
- [x] Feature importance ranking

### Validation
- [x] Walk-forward validation with expanding window
- [x] Label boundary sensitivity (+/- 3 months)
- [x] Ensemble agreement (Jensen-Shannon divergence)

### Statistical Corrections
- [x] Collect ALL p-values across all layers
- [x] Apply Benjamini-Hochberg FDR correction
- [x] Report both raw and adjusted p-values

### Charts
- [x] Chart ML.1: Regime classification probabilities (stacked area)
- [x] Chart ML.2: DTW similarity timeline
- [x] Chart ML.3: SHAP summary plot (beeswarm)
- [x] Chart ML.4: SHAP waterfall for March 2026

---

## Notebook Assembly (spec: `00_project_overview.md` + competition guidelines)

### Structure (7 required sections)
- [x] Section 1: Problem statement and hypothesis
- [x] Section 2: Dataset description
- [x] Section 3: Data cleaning and preprocessing
- [x] Section 4: Exploratory data analysis (charts + interpretation)
- [x] Section 5: Statistical testing and/or model approach
- [x] Section 6: Results and conclusions
- [x] Section 6.5: Executive Summary (top 3 findings in 3 sentences)
- [x] Section 7: Limitations and future work
- [x] Final cell: "Dataset Citations (MLA 8)" — mandatory

### Competition Compliance
- [ ] Notebook opens and runs top-to-bottom without errors ← NEEDS TESTING
- [x] All charts have clear titles stating findings
- [x] 15 PRIMARY charts in main body
- [x] File named: `2kim_finance_notebook.ipynb`
- [x] Track clearly stated as Finance & Economics
- [x] Correlation vs causation distinguished throughout
- [x] All data sources cited in MLA 8 format

---

## Presentation (spec: `08_presentation_narrative.md`)

- [ ] Update slides with real chart images (replace placeholders) ← AFTER DATA RUN
- [ ] Insert actual numbers into stat callouts ← AFTER DATA RUN
- [ ] Export to PDF: `2kim_finance_slides.pdf`
- [ ] Timing rehearsal: target 9 minutes
- [x] File named: `2kim_finance_slides.pptx`

---

## Final Pre-Submission Checks (spec: `09_submission_checklist.md`)
- [ ] `Restart & Run All` succeeds ← NEEDS .env FIRST
- [ ] All 30 charts render correctly
- [x] Last cell is "Dataset Citations (MLA 8)"
- [ ] Slide content matches notebook findings
- [x] Color scheme consistent: NVDA=green, CSCO=coral red
- [x] No API keys or credentials in committed code
- [ ] Files ready for Google Form upload
