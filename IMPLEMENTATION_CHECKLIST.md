# Implementation Checklist — AI Hype Cycle Datathon Submission

> Auto-updated as agents complete work. Each `[ ]` becomes `[x]` when done.

---

## Infrastructure
- [x] Directory structure (`src/layers/`, `src/ml/`, `data/raw/`, `data/processed/`, `submissions/charts/`)
- [x] Cache-or-call utility (`src/utils/cache.py`)
- [x] Install dependencies (fredapi, pytrends, xgboost, shap, dtw-python, kaleido)
- [ ] `.env` configured (FRED_API_KEY, ALPHA_VANTAGE_KEY, OPENAI_API_KEY)
- [ ] Verify all imports work in clean environment

---

## Layer 1: Price Comparison (spec: `01_data_layer_price_comparison.md`)

### Data Collection
- [ ] Fetch NVDA daily prices (Jan 2023 – Mar 2026) via yfinance
- [ ] Fetch CSCO daily prices (Jan 1998 – Dec 2002) via yfinance
- [ ] Fetch Nortel Networks (NT) prices (Jan 1998 – delisting) via yfinance
- [ ] Fetch peer stocks: AMD, SMCI, AVGO, ARM, MSFT (AI era)
- [ ] Fetch peer stocks: JNPR, QCOM, INTC, SUNW (dot-com era)
- [ ] Fetch index data: QQQ/^IXIC, ^GSPC for both eras
- [ ] All data cached as parquet

### Analysis
- [ ] Breakout point detection algorithm (50% trailing 252d return, 20d sustain)
- [ ] Breakout sensitivity analysis (40/50/60% thresholds × 10/20/30 sustain days)
- [ ] Normalize prices to index 100 at breakout
- [ ] Weekly resampling for overlay
- [ ] Compute daily log-returns
- [ ] Compute rolling returns (30d, 90d, 252d)
- [ ] Compute drawdown from ATH
- [ ] Compute rolling beta vs S&P 500

### Statistical Tests
- [ ] ADF stationarity test on log-returns
- [ ] Pearson/Spearman correlation of log-returns (aligned by days_from_breakout)
- [ ] Ljung-Box autocorrelation test
- [ ] KS test on return distributions
- [ ] DTW distance (primary quantitative measure — deferred to ML layer)

### Charts
- [ ] Chart 1.1: Normalized price overlay (NVDA green, CSCO red, Nortel gray dashed)
- [ ] Chart 1.2: Drawdown from ATH comparison
- [ ] Chart 1.3: Rolling 90-day return comparison
- [ ] Chart 1.4: Volume surge timeline

---

## Layer 2: Fundamentals (spec: `02_data_layer_fundamentals.md`)

### Data Collection
- [ ] Fetch NVDA quarterly financials (yfinance + Alpha Vantage cross-reference)
- [ ] Fetch CSCO quarterly financials (yfinance + Alpha Vantage + EDGAR validation)
- [ ] Fetch CPI data from FRED (CPIAUCSL) for inflation adjustment
- [ ] Fetch peer fundamentals (AMD, MSFT, AVGO for AI era; INTC, QCOM for dot-com)

### Analysis
- [ ] Compute P/E ratio trajectory (raw + log-transformed)
- [ ] Compute PEG ratio (P/E / earnings growth rate)
- [ ] Compute revenue growth rate (YoY, QoQ)
- [ ] Compute free cash flow yield
- [ ] Compute R&D-to-revenue ratio
- [ ] Compute gross margin trajectory
- [ ] CPI inflation adjustment to 2026 dollars
- [ ] Fiscal-to-calendar quarter alignment (NVDA Jan FY, CSCO Jul FY)

### Statistical Tests
- [ ] Two-sample bootstrap test on log(P/E) distributions (with Cohen's d)
- [ ] Regression: market cap ~ f(revenue, growth, margin) per era
- [ ] Shapiro-Wilk normality test
- [ ] Durbin-Watson autocorrelation check

### Charts
- [ ] Chart 2.1: P/E ratio trajectory overlay (aligned by cycle phase)
- [ ] Chart 2.2: Revenue growth rate comparison (grouped bar)
- [ ] Chart 2.3: Market cap vs revenue scatter (bubble size = P/E)
- [ ] Chart 2.4: Free cash flow yield timeline
- [ ] Chart 2.5: R&D investment comparison
- [ ] Chart 2.6: PEG ratio comparison

---

## Layer 3: Market Concentration (spec: `03_data_layer_market_concentration.md`)

### Data Collection
- [ ] Fetch S&P 500 top constituents by market cap (current)
- [ ] Fetch SPY and RSP (equal-weight) ETF prices
- [ ] Fetch sector ETFs: XLK, SMH/SOXX
- [ ] Fetch Wilshire 5000 / total market cap from FRED (WILL5000IND)
- [ ] Fetch GDP from FRED (GDP) for Buffett Indicator
- [ ] Historical S&P 500 concentration data (dot-com era)

### Analysis
- [ ] Top-10 concentration ratio over time
- [ ] AI-specific concentration (NVDA+AMD+MSFT+GOOGL+META+AVGO as % of S&P 500)
- [ ] SPY vs RSP spread (cap-weight vs equal-weight performance gap)
- [ ] Buffett Indicator (market cap / GDP) with Fed-adjusted variant
- [ ] HHI index computation (on 0-10000 scale)
- [ ] Concentration-reversibility analysis (forward 12m returns by concentration level)
- [ ] Passive investing microstructure adjustment note

### Statistical Tests
- [ ] Compare concentration ratios across eras (z-test)
- [ ] Concentration-drawdown predictive correlation
- [ ] Granger causality: concentration → drawdowns

### Charts
- [ ] Chart 3.1: Top-10 concentration ratio timeline (2000 vs 2024-2026)
- [ ] Chart 3.2: SPY vs RSP cumulative return spread
- [ ] Chart 3.3: Buffett Indicator historical with current marker
- [ ] Chart 3.4: Treemap of S&P 500 by market cap (current)
- [ ] Chart 3.5: HHI index over time

---

## Layer 4: Macro Environment (spec: `04_data_layer_macro_environment.md`)

### Data Collection (all FRED API)
- [ ] Federal Funds Rate (FEDFUNDS)
- [ ] 10-Year Treasury (DGS10)
- [ ] 2-Year Treasury (DGS2)
- [ ] Yield Curve Spread 10Y-2Y (T10Y2Y)
- [ ] M2 Money Supply (M2SL)
- [ ] CPI (CPIAUCSL)
- [ ] Corporate Credit Spreads (BAA10Y)
- [ ] GDP (GDP)
- [ ] Unemployment Rate (UNRATE)
- [ ] S&P 500 (SP500)

### Analysis
- [ ] Align different frequencies to monthly
- [ ] Compute derived metrics: real rate, M2 growth rate, GDP YoY (pct_change(4) on quarterly)
- [ ] Yield curve inversion detection and duration
- [ ] Era comparison: 1996-2003 vs 2021-2026
- [ ] Peak-aligned macro comparison (months_from_peak)

### Statistical Tests
- [ ] Welch's t-test comparing macro means across eras
- [ ] Cross-correlation lead-lag analysis (macro → equity)
- [ ] Granger causality: yield curve → equity drawdowns (with ADF + AIC lag selection)
- [ ] Levene's variance test
- [ ] All findings framed as associations, NOT causal claims

### Charts
- [ ] Chart 4.1: Fed Funds Rate dual-era overlay
- [ ] Chart 4.2: M2 Money Supply growth rate comparison
- [ ] Chart 4.3: Yield Curve (10Y-2Y) with crash annotations
- [ ] Chart 4.4: Credit Spreads timeline
- [ ] Chart 4.5: Macro Dashboard (3x2 small multiples)
- [ ] Chart 4.6: Real Interest Rate comparison

---

## Layer 5: Sentiment & NLP (spec: `05_data_layer_sentiment_nlp.md`)

### Data Collection
- [ ] Google Trends: "NVIDIA stock", "AI stocks", "AI bubble" (pytrends)
- [ ] Google Trends: "artificial intelligence investing" (pytrends)
- [ ] SEC EDGAR: NVDA earnings call transcripts or 10-Q filings (keyword counting)
- [ ] SEC EDGAR: CSCO 10-Q filings 1998-2001 ("internet"/"e-commerce"/"web" keyword count)
- [ ] ~~Reddit API~~ (SKIPPED per user request)

### Analysis
- [ ] AI mention frequency in NVDA filings over time
- [ ] "Internet" mention frequency in CSCO filings 1998-2001
- [ ] Google Trends normalization and timeline
- [ ] OpenAI sentiment scoring of earnings call sentences (hype score 1-10, specificity 1-10)
- [ ] GPT validation: manual score 10 transcripts, compare agreement rate
- [ ] Keyword density correlation check (GPT vs simple regex)

### Statistical Tests
- [ ] Hype-price correlation
- [ ] Granger causality: Google Trends → price moves
- [ ] Mann-Whitney U on quarterly sentiment aggregates

### Charts
- [ ] Chart 5.1: AI mentions in earnings calls over time (hockey stick)
- [ ] Chart 5.2: Hype score vs revenue specificity scatter
- [ ] Chart 5.3: Google Trends comparison
- [ ] Chart 5.4: Sentiment timeline with stock price overlay
- [ ] Chart 5.5: ~~Reddit NVDA mentions~~ (SKIPPED)
- [ ] Chart 5.6: Word cloud or keyword frequency comparison

---

## ML Pipeline (spec: `06_ml_pipeline.md`)

### Feature Engineering
- [ ] Build feature matrix from all 5 layers (38 features)
- [ ] Time-based train/test split (pre-2022 train, 2022+ test)
- [ ] Feature scaling (StandardScaler)
- [ ] Handle missing sentiment features with two-model approach (core + augmented)

### Model 1: Regime Classifier
- [ ] Label historical periods (bubble, correction, normal_growth, recovery)
- [ ] Train Random Forest (primary)
- [ ] Train XGBoost (secondary)
- [ ] Train Logistic Regression (baseline)
- [ ] Train price-feature-excluded RF variant
- [ ] TimeSeriesSplit cross-validation
- [ ] Hyperparameter tuning
- [ ] Evaluate: accuracy, F1, confusion matrix
- [ ] Generate March 2026 regime probabilities

### Model 2: DTW Similarity
- [ ] Normalize features to [0,1]
- [ ] Compute per-feature DTW distance (AI era vs dot-com)
- [ ] Weighted composite similarity score
- [ ] Null distribution: compare against 3-5 other historical periods
- [ ] Permutation test (1000 shuffles) for p-value
- [ ] Rolling similarity timeline
- [ ] Bootstrap confidence intervals

### Model 3: SHAP Analysis
- [ ] SHAP TreeExplainer on RF
- [ ] SHAP summary plot (beeswarm)
- [ ] SHAP waterfall for March 2026 prediction
- [ ] Feature importance ranking

### Validation
- [ ] Walk-forward validation with expanding window
- [ ] Label boundary sensitivity (+/- 3 months)
- [ ] Ensemble agreement (Jensen-Shannon divergence)

### Statistical Corrections
- [ ] Collect ALL p-values across all layers
- [ ] Apply Benjamini-Hochberg FDR correction
- [ ] Report both raw and adjusted p-values

### Charts
- [ ] Chart ML.1: Regime classification probabilities (stacked area)
- [ ] Chart ML.2: DTW similarity timeline
- [ ] Chart ML.3: SHAP summary plot (beeswarm)
- [ ] Chart ML.4: SHAP waterfall for March 2026

---

## Notebook Assembly (spec: `00_project_overview.md` + competition guidelines)

### Structure (7 required sections)
- [ ] Section 1: Problem statement and hypothesis
- [ ] Section 2: Dataset description
- [ ] Section 3: Data cleaning and preprocessing
- [ ] Section 4: Exploratory data analysis (charts + interpretation)
- [ ] Section 5: Statistical testing and/or model approach
- [ ] Section 6: Results and conclusions
- [ ] Section 6.5: Executive Summary (top 3 findings in 3 sentences)
- [ ] Section 7: Limitations and future work
- [ ] Final cell: "Dataset Citations (MLA 8)" — mandatory

### Competition Compliance
- [ ] Notebook opens and runs top-to-bottom without errors
- [ ] All charts have clear titles stating findings (not just metric names)
- [ ] 15 PRIMARY charts in main body, SECONDARY in supplementary section
- [ ] File named: `2kim_finance_notebook.ipynb`
- [ ] Track clearly stated as Finance & Economics
- [ ] Correlation vs causation distinguished throughout
- [ ] All data sources cited in MLA 8 format

---

## Presentation (spec: `08_presentation_narrative.md`)

- [ ] Update slides with real chart images (replace placeholders)
- [ ] Insert actual numbers into stat callouts
- [ ] Export to PDF: `2kim_finance_slides.pdf`
- [ ] Timing rehearsal: target 9 minutes
- [ ] File named: `2kim_finance_slides.pptx`

---

## Final Pre-Submission Checks (spec: `09_submission_checklist.md`)
- [ ] `Restart & Run All` succeeds
- [ ] All 30 charts render correctly
- [ ] Last cell is "Dataset Citations (MLA 8)"
- [ ] Slide content matches notebook findings
- [ ] Color scheme consistent: NVDA=green, CSCO=coral red
- [ ] No API keys or credentials in committed code
- [ ] Files ready for Google Form upload
