# 00 -- Project Overview (Master Spec)

## Project Identity

| Field | Value |
|---|---|
| **Title** | The AI Hype Cycle -- Are We in a Bubble? Comparing NVIDIA 2023-2026 to Cisco 1998-2001 |
| **Track** | Finance & Economics |
| **Competition** | SBU AI Community Datathon 2026 |
| **Team** | 2Kim (Jimmy Kim + Alice Kim) |
| **File naming** | `2kim_finance_notebook.ipynb`, `2kim_finance_slides.pptx` |
| **Submission** | Google Form upload: `.ipynb` + `.pptx`/`.pdf` |

---

## Research Question

> **Plain English:** Does the current AI rally look like the dot-com bubble — and how similar is it across five independent dimensions?

> **Is the NVIDIA-led AI equity rally of 2023-2026 structurally analogous to the Cisco-led dot-com bubble of 1998-2001, and if so, at what stage of the bubble lifecycle are we today?**

Sub-questions (each maps to one data layer):

1. **Price dynamics**: How closely do NVDA's 2023-2026 returns track CSCO's 1998-2001 returns when time-aligned and normalized? Include Nortel Networks (NT) as a non-survivor dot-com analog to address survivorship bias. (Layer 1)
2. **Fundamental backing**: Does NVDA's current valuation have stronger fundamental support (revenue growth, free cash flow, R&D yield) than CSCO had at comparable price multiples? (Layer 2)
3. **Concentration risk**: Has AI-driven market concentration in the S&P 500 reached levels comparable to the late-1990s tech concentration? (Layer 3)
4. **Macro environment**: Are the macro conditions enabling the AI rally (loose monetary policy, credit expansion, yield curve shape) structurally similar to 1998-2001? (Layer 4)
5. **Sentiment temperature**: Does NLP analysis of corporate earnings calls, Google search trends, and Reddit activity show euphoria levels comparable to the dot-com peak? (Layer 5)
6. **Regime classification**: Can a machine-learning model trained on dot-com bubble features classify the current AI market into a bubble lifecycle stage? (ML Pipeline)

---

## Hypothesis

**H0 (Null):** The NVDA rally is fundamentally distinct from the CSCO bubble -- driven by real earnings growth, sustainable demand, and rational valuations -- and current market conditions do not resemble a bubble.

**H1 (Alternative):** The NVDA rally exhibits statistically significant structural similarities to the CSCO bubble across price dynamics, concentration risk, and sentiment metrics, suggesting we are in the mid-to-late expansion phase of a technology equity bubble.

**Our prior (what we expect to find):** Partial bubble signal. We expect Layers 1, 3, and 5 (price, concentration, sentiment) to show strong dot-com parallels, while Layer 2 (fundamentals) shows NVDA has materially better underlying economics than CSCO did. Layer 4 (macro) likely shows a mixed picture due to post-pandemic monetary tightening vs. the 1998-2001 easing cycle. The ML regime classifier should place the current market in "expansion" or "early euphoria" rather than "peak."

---

## Why It Matters

### For investors
- **Portfolio risk**: If AI equities are in a bubble, traditional 60/40 portfolios with heavy S&P 500 exposure carry outsized single-factor risk. Identifying the bubble stage informs hedging strategy.
- **Entry timing**: Bubbles can persist for years. Knowing whether we are in early or late expansion is actionable alpha.

### For policymakers
- **Financial stability**: The 2000 crash erased $5 trillion in market cap. An AI bubble collapse of comparable scale would today impact $10T+ given S&P 500 market cap growth.
- **Fed policy feedback**: If AI stock concentration is correlated with monetary conditions, rate decisions have second-order bubble effects worth modeling.

### For the AI community
- **Capital allocation**: Hype cycles distort R&D spending. Understanding whether current AI investment levels are sustainable affects long-term research planning.
- **Historical precedent**: Cisco survived the crash but traded flat for 15 years. Understanding what makes NVIDIA's position different (or not) has real implications for the ecosystem.

---

## Methodology Overview

### Five Data Layers + ML Pipeline

```
Layer 1 (Price)
    |
Layer 2 (Fundamentals) ----> Feature Matrix ----> ML Regime Classifier
    |                              ^                      |
Layer 3 (Concentration)            |                      v
    |                         Engineered           Bubble Stage
Layer 4 (Macro)               Features            Classification
    |                              ^                      |
Layer 5 (Sentiment/NLP)           |                      v
                              All Layers ----> DTW Similarity Score
                                                   (NVDA vs CSCO)
```

Each layer produces:
1. **A standalone visual** (chart/table answering its sub-question)
2. **Engineered features** fed into the ML pipeline
3. **A narrative paragraph** for the presentation

The ML pipeline consumes features from all 5 layers and outputs:
- A **regime classification** (accumulation / expansion / euphoria / distribution / decline)
- A **DTW similarity score** (0-1 scale: how closely NVDA's multi-dimensional trajectory matches CSCO's)
- **Feature importance rankings** (which bubble indicators matter most)

---

## Data Sources

| # | Source | API / Library | What It Provides | Ticker/Series | Time Range | Granularity |
|---|---|---|---|---|---|---|
| 1 | Yahoo Finance | `yfinance` | OHLCV price data, market cap | NVDA, CSCO, NT (Nortel Networks, non-survivor analog), ^GSPC, ^SPXEW (or RSP ETF) | CSCO: 1996-01 to 2003-12; NVDA: 2021-01 to 2026-03; NT: 1997-01 to 2009-01 (delisting) | Daily |
| 2 | Yahoo Finance | `yfinance` `.quarterly_financials`, `.quarterly_balance_sheet`, `.quarterly_cashflow` | Revenue, net income, FCF, total debt, R&D expense, total assets | NVDA, CSCO | Same as above | Quarterly |
| 3 | FRED | `fredapi` / `requests` | Fed funds rate, M2 money supply, BAA-AAA credit spread, 10Y-2Y yield curve, GDP, CPI | FEDFUNDS, M2SL, BAAFFM, T10Y2Y, GDP, CPIAUCSL | 1996-01 to 2026-03 | Monthly (some quarterly) |
| 4 | Alpha Vantage | `requests` (REST) | S&P 500 constituent weights, sector breakdowns (backup for concentration data) | S&P 500 constituents | Current snapshot + historical where available | Quarterly |
| 5 | Google Trends | `pytrends` | Search interest over time | Keywords: "NVIDIA stock", "AI bubble", "dot com bubble", "artificial intelligence" | 2004-01 to 2026-03 (Trends limit) | Weekly |
| 6 | OpenAI API | `openai` (GPT-4o-mini) | Sentiment scoring of earnings call transcripts | NVDA + CSCO earnings calls (sourced from SEC EDGAR / Seeking Alpha) | CSCO: 1998-2001; NVDA: 2023-2026 | Quarterly |
| 7 | Reddit (PRAW) | `praw` | Post titles + scores from r/wallstreetbets, r/investing, r/stocks | Subreddit search: "NVIDIA", "NVDA", "AI bubble" | 2023-01 to 2026-03 | Daily (aggregated weekly) |
| 8 | SEC EDGAR | `requests` (EDGAR full-text search API) | 10-K/10-Q filings for earnings call transcripts | CIK: NVDA=1045810, CSCO=858877 | Same quarterly ranges | Quarterly |
| 9 | Slickcharts / Wikipedia | Web scrape or static CSV | Historical S&P 500 top-10 constituent weights | Top 10 by market cap | Annual snapshots 1996-2026 | Annual |

### API Configuration

```python
# .env file (never committed)
FRED_API_KEY=<your_key>              # https://fred.stlouisfed.org/docs/api/api_key.html
ALPHA_VANTAGE_KEY=<your_key>         # https://www.alphavantage.co/support/#api-key
OPENAI_API_KEY=<your_key>            # https://platform.openai.com/api-keys
REDDIT_CLIENT_ID=<your_id>           # https://www.reddit.com/prefs/apps
REDDIT_CLIENT_SECRET=<your_secret>
REDDIT_USER_AGENT=datathon2026:v1.0 (by /u/<username>)
```

### Data Caching Convention

> **Cache-or-call pattern required for ALL API calls.** Every API call must follow: `if Path("data/raw/<file>.parquet").exists(): load from cache; else: call API, cache result, return data`. This pattern must be enforced in `src/loaders.py` and used consistently in every notebook cell that fetches data. A notebook that fails because an API is rate-limited or down during judging is a catastrophic failure.

### Key API Endpoints

| API | Endpoint / Method | Rate Limit |
|---|---|---|
| yfinance | `yf.download(tickers, start, end, interval="1d")` | Unofficial, ~2000 req/hr |
| FRED | `https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key=...&file_type=json` | 120 req/min |
| Alpha Vantage | `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NVDA&apikey=...&outputsize=full` | 25 req/day (free tier) |
| pytrends | `pytrends.build_payload(kw_list, timeframe='2004-01-01 2026-03-28')` | ~10 req/min (unofficial, rate-limited) |
| OpenAI | `client.chat.completions.create(model="gpt-4o-mini", ...)` | Tier-dependent, typically 500 RPM |
| PRAW | `subreddit.search(query, sort="relevance", time_filter="all")` | 60 req/min (OAuth) |
| EDGAR | `https://efts.sec.gov/LATEST/search-index?q=...&dateRange=...&forms=10-K,10-Q` | 10 req/sec |

---

## Analysis Pipeline (Ordered Steps)

### Phase 1: Data Collection & Storage (~Day 1)

```
Step 1.1  Download NVDA, CSCO, ^GSPC daily OHLCV via yfinance
          -> data/raw/prices_{ticker}_{start}_{end}.parquet

Step 1.2  Download quarterly financials for NVDA, CSCO via yfinance
          -> data/raw/fundamentals_{ticker}.parquet

Step 1.3  Download FRED series (FEDFUNDS, M2SL, BAAFFM, T10Y2Y, GDP, CPIAUCSL)
          -> data/raw/fred_{series_id}.parquet

Step 1.4  Download Google Trends data for keyword list
          -> data/raw/gtrends_{keyword_hash}.parquet

Step 1.5  Pull Reddit posts via PRAW
          -> data/raw/reddit_{subreddit}_{query}.parquet

Step 1.6  Collect earnings call transcripts (EDGAR or Seeking Alpha)
          -> data/raw/transcripts/{ticker}_{quarter}.txt

Step 1.7  (Optional) Alpha Vantage S&P 500 sector weights
          -> data/raw/sp500_weights.parquet
```

### Phase 2: Cleaning & Preprocessing (~Day 1-2)

```
Step 2.1  Price normalization:
          - Rebase both NVDA and CSCO to 100 at t=0
          - t=0 for CSCO: 1998-01-02; t=0 for NVDA: 2023-01-03
          - Align on "trading days since t=0" (integer index 0..N)
          - Forward-fill missing dates, drop weekends/holidays
          - Columns: [trading_day, nvda_norm, csco_norm, sp500_nvda_era, sp500_csco_era]

Step 2.2  Fundamental ratios:
          - Compute: P/E (trailing), P/S, EV/EBITDA, FCF yield, R&D/Revenue, Debt/Equity
          - Quarterly frequency, forward-fill to daily where needed for overlays
          - Handle negative earnings (P/E = NaN when EPS < 0)
          - Columns: [date, ticker, pe_ratio, ps_ratio, ev_ebitda, fcf_yield, rd_pct_rev, debt_equity]

Step 2.3  Concentration metrics:
          - Top-1, Top-5, Top-10 share of S&P 500 total market cap
          - Equal-weight vs cap-weight S&P 500 performance ratio (RSP/SPY)
          - HHI (Herfindahl-Hirschman Index) for S&P 500 by sector
          - Columns: [date, top1_pct, top5_pct, top10_pct, equal_cap_ratio, hhi_sector]

Step 2.4  Macro feature engineering:
          - Rate regime: classify Fed funds as tightening/neutral/easing (12-month delta)
          - M2 growth rate (YoY %)
          - GDP YoY growth: apply pct_change(periods=4) to the RAW quarterly GDP series
            (before any forward-fill to monthly). FRED GDP is already annualized quarterly,
            so pct_change(periods=4) gives true YoY growth. Then merge to monthly.
            Do NOT use pct_change(periods=12) on forward-filled monthly data.
          - Credit spread z-score (rolling 5-year mean/std)
          - Yield curve inversion flag (T10Y2Y < 0)
          - Columns: [date, fed_rate, fed_regime, m2_yoy, gdp_yoy, credit_spread_z, yield_curve, yc_inverted]

Step 2.5  Sentiment scores:
          - Google Trends: normalize 0-100 scale, compute rolling 4-week momentum
          - Reddit: aggregate weekly post count + median score + avg sentiment (TextBlob or VADER)
          - Earnings calls: GPT-4o-mini prompt to score on 1-10 scale for {optimism, certainty, hype}
          - Columns: [date, gtrends_nvda, gtrends_ai_bubble, reddit_post_count, reddit_avg_sentiment,
                       earnings_optimism, earnings_certainty, earnings_hype]

Step 2.6  Merge all layers into master DataFrame:
          - Primary key: date (daily granularity, lower-freq data forward-filled)
          - Secondary grouping: era (csco_bubble | nvda_rally)
          - Final shape: ~3000 rows (trading days across both eras) x ~40 columns
          -> data/processed/master_features.parquet
```

### Phase 3: Exploratory Data Analysis (~Day 2-3)

```
Step 3.1  Layer 1 -- Price overlay chart:
          - Dual-line plot: NVDA vs CSCO normalized to 100
          - X-axis: "Trading days since rally start"
          - Annotate key events: CSCO peak (Mar 2000), NVDA key dates
          - Chart type: Plotly line chart with Plotly `add_vrect` for NBER recession
          - Statistical test: DTW similarity score as primary quantitative measure; Pearson/Spearman correlation of log-returns as secondary (applied to detrended log-returns aligned by days_from_breakout, not price levels)

Step 3.2  Layer 2 -- Fundamentals comparison:
          - Side-by-side bar charts: P/E, P/S, FCF yield at comparable rally stages
          - Revenue growth trajectory overlay (indexed to 100)
          - Table: NVDA vs CSCO fundamental metrics at {t+250, t+500, t+750} trading days
          - Chart type: Plotly grouped bar + line overlay
          - Statistical test: Two-sample t-test on quarterly revenue growth rates

Step 3.3  Layer 3 -- Concentration risk:
          - Area chart: top-1 / top-5 / top-10 share of S&P 500 over time (1996-2026)
          - Line chart: equal-weight / cap-weight divergence (RSP/SPY ratio)
          - Highlight zones: dot-com era vs AI era
          - Chart type: Plotly area + line
          - Statistical test: Compare current top-5 concentration to 2000 peak (z-score)

Step 3.4  Layer 4 -- Macro environment:
          - Multi-panel chart (2x2): Fed funds, M2 growth, credit spread, yield curve
          - Each panel overlays CSCO era (1998-2001) vs NVDA era (2023-2026)
          - Chart type: Plotly subplots (make_subplots, 2 rows, 2 cols)
          - Statistical test: Cosine similarity of macro regime vectors between eras

Step 3.5  Layer 5 -- Sentiment:
          - Google Trends time series for AI-related keywords
          - Reddit activity heatmap (post volume by week)
          - Earnings call sentiment radar chart (optimism, certainty, hype) comparing NVDA vs CSCO
          - Chart type: Plotly line + heatmap + radar (scatterpolar)
          - Statistical test: Mann-Whitney U test on sentiment distributions between eras

Step 3.6  Correlation matrix:
          - Heatmap of all engineered features from master DataFrame
          - Highlight cross-layer correlations (e.g., sentiment vs price momentum)
          - Chart type: Seaborn clustermap or Plotly heatmap

Step 3.7  Multiple comparisons correction:
          - Collect all p-values from Steps 3.1-3.5 into a pre-analysis table
          - Apply Benjamini-Hochberg (FDR) correction across the full set of tests
            using `statsmodels.stats.multitest.multipletests(pvalues, method='fdr_bh')`
          - Report both raw p-values and BH-adjusted p-values for every test
          - Downgrade any finding whose adjusted p-value > 0.05 from "statistically
            significant" to "suggestive pattern"
```

### Phase 4: Statistical Modeling & ML (~Day 3-4)

```
Step 4.1  Dynamic Time Warping (DTW):
          - Compare NVDA and CSCO normalized price paths using dtw-python
          - Compute DTW distance + optimal warping path
          - Visualize alignment with warping path plot
          - Sensitivity: run DTW on {price, volume, P/E, sentiment} individually
          - Library: dtw-python (`from dtw import dtw, accelerated_dtw`)
          - Output: dtw_distance (scalar), dtw_path (array), per-feature DTW scores

Step 4.2  Regime classification:
          - Features: price_momentum_30d, price_momentum_90d, pe_ratio, ps_ratio,
                      volume_zscore, concentration_top5, credit_spread_z, gtrends_momentum,
                      reddit_sentiment, earnings_hype, m2_yoy, vix (if available)
          - Target: manually label CSCO era into 5 regimes:
            - Accumulation (1998-01 to 1998-09)
            - Expansion (1998-10 to 1999-06)
            - Euphoria (1999-07 to 2000-03)
            - Distribution (2000-04 to 2000-09)
            - Decline (2000-10 to 2001-12)
          - Model: Random Forest Classifier (n_estimators=200, max_depth=8)
          - Train on CSCO era, predict on NVDA era
          - Output: predicted regime label for each NVDA trading day
          - Validation: 5-fold time-series cross-validation on CSCO era (TimeSeriesSplit)

Step 4.3  Feature importance:
          - Extract from Random Forest `.feature_importances_`
          - Also compute SHAP values for interpretability
          - Visualize: horizontal bar chart of top-15 features
          - Library: shap (`shap.TreeExplainer`)

Step 4.4  Robustness checks:
          - Vary t=0 alignment by +/- 60 trading days, recompute DTW
          - Remove one layer at a time, re-run regime classifier (ablation study)
          - Bootstrap confidence intervals on DTW distance (1000 resamples)
```

### Phase 5: Synthesis & Conclusions (~Day 4-5)

```
Step 5.1  Compile "Bubble Scorecard":
          - Table with rows = {Price, Fundamentals, Concentration, Macro, Sentiment}
          - Columns = {Metric, CSCO @ peak, NVDA @ current, Similarity (0-5), Direction}
          - Traffic-light color coding (green/yellow/red for bubble risk)

Step 5.2  Write narrative conclusions answering each sub-question
Step 5.3  Document limitations (see Section 12 below)
Step 5.4  Format MLA 8 citations
Step 5.5  Export notebook as final `.ipynb`
Step 5.6  Build slideshow from notebook visuals
```

---

## Expected Deliverables

### Notebook Sections (Required 7-Section Structure)

| Section | Notebook Header | Content | Maps to Rubric |
|---|---|---|---|
| 1 | `## 1. Problem Statement & Research Question` | Research question, hypothesis, why it matters, prior literature | Research Question (15pts) |
| 2 | `## 2. Data Collection & Sources` | API calls, download code, raw data preview (`df.head()`, `df.info()`) | Data Quality (20pts) |
| 3 | `## 3. Data Cleaning & Preprocessing` | Null handling, normalization, feature engineering, merge pipeline | Data Quality (20pts) |
| 4 | `## 4. Exploratory Data Analysis` | All Layer 1-5 charts, correlation matrix, descriptive statistics | Analysis & Evidence (25pts) |
| 5 | `## 5. Statistical Modeling & Machine Learning` | DTW, regime classifier, SHAP, robustness checks | Analysis & Evidence (25pts), Technical Rigor (15pts) |
| 6 | `## 6. Results & Discussion` | Bubble scorecard, narrative answers to sub-questions, key findings | Presentation Clarity (15pts) |
| 6.5 | `### Executive Summary` | **Required cell immediately after Section 6.** Top 3 findings in 3 sentences (e.g., DTW similarity score, P/E divergence, concentration level). This is what judges will actually read. | Presentation Clarity (15pts) |
| 7 | `## 7. Limitations, Ethics & Future Work` | Data gaps, correlation vs causation, model assumptions, ethical considerations | Limitations (10pts) |
| 8 | `## Dataset Citations (MLA 8)` | Full MLA 8 citations for every dataset | Required (mandatory) |

### Slideshow Sections (16 slides: 1 title + 14 content + 1 close, plus 2 backup)

| Slide # | Title | Content |
|---|---|---|
| 1 | Title slide | Project title, team, track, date |
| 2 | The Hook | NVDA (green) vs CSCO (coral red) price overlay + Nortel (gray dashed) |
| 3 | Research Framework | Five-layer diagram, one-line question per layer |
| 4 | Data & Methods | Data sources table, methodology summary (15 sec) |
| 5 | P/E Comparison (Bull) | Layer 2 P/E chart, CSCO 200x vs NVDA [X]x |
| 6 | Revenue Growth | Layer 2 grouped bars, 265% vs 66% stat callout |
| 7 | Concentration (Bear) | Layer 3 area chart + SPY/RSP, Buffett Indicator stat verbal |
| 8 | Sentiment | Layer 5: AI mentions + hype vs specificity |
| 9 | Macro Wild Card | Layer 4 macro dashboard small multiples |
| 10 | ML: Regime Classification | Regime probabilities + March 2026 readout |
| 11 | ML: DTW + SHAP | DTW similarity + SHAP waterfall |
| 12 | Synthesis | Bubble scorecard with traffic-light colors |
| 13 | Key Takeaway | One-sentence verdict |
| 14 | Limitations & Ethics | Top 4 limitations, not financial advice |
| 15 | Thank You / Q&A | Notebook reference, dashboard available for demo |
| 16 (backup) | SHAP Deep Dive | SHAP beeswarm plot |
| 17 (backup) | Methodology | Walk-forward validation, confusion matrix |

---

## Story Arc

### Narrative Flow: Hook --> Evidence --> Nuance --> Answer

1. **Hook (30 seconds):** Open with the NVDA vs CSCO price overlay. The visual similarity is striking and immediately grabs attention. Pose the question: "History doesn't repeat, but does it rhyme?"

2. **Context (1 minute):** Brief overview of the dot-com bubble -- Cisco was the most valuable company in the world in March 2000 ($555B market cap), trading at 196x P/E. It lost 86% of its value by 2002. NVIDIA hit $3.6T in 2024 on AI demand.

3. **Evidence Build (3-4 minutes):** Walk through each layer. Start with the strongest parallel (price dynamics), then systematically test whether the comparison holds across fundamentals, concentration, macro, and sentiment. Each layer either strengthens or weakens the bubble thesis.

4. **The Twist (1 minute):** Layer 2 (fundamentals) is where NVDA and CSCO diverge. NVDA has real revenue growth, massive FCF, and monopolistic GPU market share. This is the critical nuance that separates informed analysis from clickbait.

5. **ML Synthesis (1 minute):** The regime classifier provides a data-driven answer. Show where the model places today's market on the bubble lifecycle. Feature importance reveals which signals matter most.

6. **Conclusion (1 minute):** Deliver a nuanced verdict. Not "yes it's a bubble" or "no it's not" -- but "here are the dimensions of similarity (X, Y) and difference (Z), and here's the probability distribution of outcomes." The bubble scorecard makes this concrete.

7. **Honesty (30 seconds):** Acknowledge limitations. Past performance doesn't predict future results. Two data points (CSCO, NVDA) don't make a statistical sample. Sentiment analysis has noise. But the framework is generalizable.

---

## Success Criteria (Per Rubric Category)

### Analysis & Evidence (25pts) -- TIEBREAKER #1
- **Target: 22-25/25**
- Minimum 10 distinct visualizations across 5 data layers
- Every chart has a title, axis labels, legend, and a 1-2 sentence caption stating the finding
- At least 4 statistical tests with reported test statistics and p-values
- DTW + regime classifier provide quantitative (not just visual) evidence
- Correlation matrix shows cross-layer relationships
- Bubble scorecard synthesizes all findings into one table

### Data Quality & Preparation (20pts) -- TIEBREAKER #2
- **Target: 18-20/20**
- 7+ distinct data sources merged on a common time axis
- Documented null-handling strategy (forward-fill for time series, drop for cross-sectional)
- Show `df.info()`, `df.describe()`, missing-value counts before and after cleaning
- Feature engineering is documented and justified (not ad hoc)
- Data pipeline is reproducible (all API calls have explicit parameters, fallback to cached data)

### Research Question & Track Fit (15pts)
- **Target: 13-15/15**
- Question is specific (not "is there a bubble?" but "does NVDA's trajectory match CSCO's across 5 measurable dimensions?")
- Question is testable (DTW distance, regime classifier accuracy, statistical tests all produce quantitative answers)
- Clearly Finance & Economics track (stock prices, P/E ratios, Fed policy, market concentration)

### Technical Rigor & Reproducibility (15pts)
- **Target: 13-15/15**
- Notebook runs top-to-bottom with `Kernel > Restart & Run All`
- All random seeds set (`np.random.seed(42)`, `random_state=42`)
- Model validation uses time-series-aware cross-validation (no data leakage)
- Results section references specific cell outputs (not external artifacts)
- Requirements pinned in `requirements.txt`

### Presentation Clarity & Storytelling (15pts)
- **Target: 12-15/15**
- Every section starts with a question and ends with an answer
- Charts use consistent color scheme: NVDA = green (#76b900 for light bg, #00CC96 for dark bg), CSCO = coral red (#EF553B). AI era annotations use the green palette; dot-com era annotations use the red/coral palette.
- Markdown headers create clear hierarchy
- Slideshow tells a story, not just dumps charts

### Limitations, Ethics & Practical Insight (10pts) -- TIEBREAKER #3
- **Target: 8-10/10**
- Explicitly state: "correlation does not imply causation" with a specific example
- Discuss survivorship bias (we're comparing NVDA to CSCO because CSCO crashed -- selection bias)
- Discuss NLP limitations (GPT-4o-mini sentiment scores are model outputs, not ground truth)
- Discuss data limitations (Google Trends only goes back to 2004, missing dot-com era sentiment)
- Discuss practical implications for investors (not just academic findings)
- Ethical note: this analysis should not be construed as financial advice

---

## Timeline & Workflow

### Day 1 (Data Collection + Cleaning) -- Jimmy leads

| Time | Task | Owner |
|---|---|---|
| Morning | Set up API keys, test all endpoints, handle rate limits | Jimmy |
| Morning | Draft slideshow template in PowerPoint/Google Slides | Alice |
| Afternoon | Download all raw data (Steps 1.1-1.7) | Jimmy |
| Afternoon | Research dot-com bubble historical context for narrative | Alice |
| Evening | Clean + preprocess (Steps 2.1-2.6), produce `master_features.parquet` | Jimmy |

### Day 2 (EDA + Visualization) -- Alice leads charts, Jimmy leads stats

| Time | Task | Owner |
|---|---|---|
| Morning | Layer 1-2 charts (price overlay, fundamentals bars) | Alice |
| Morning | Statistical tests for Layers 1-2 (Pearson, t-test) | Jimmy |
| Afternoon | Layer 3-5 charts (concentration, macro, sentiment) | Alice |
| Afternoon | Statistical tests for Layers 3-5, correlation matrix | Jimmy |
| Evening | Review all charts for consistency, add captions | Both |

### Day 3 (ML + Synthesis) -- Jimmy leads ML, Alice leads narrative

| Time | Task | Owner |
|---|---|---|
| Morning | DTW computation + warping path visualization | Jimmy |
| Morning | Begin drafting Section 6 narrative (Results & Discussion) | Alice |
| Afternoon | Regime classifier training + SHAP analysis | Jimmy |
| Afternoon | Build bubble scorecard, write Section 7 (Limitations) | Alice |
| Evening | Robustness checks (ablation, bootstrap) | Jimmy |

### Day 4 (Polish + Submit)

| Time | Task | Owner |
|---|---|---|
| Morning | Kernel > Restart & Run All -- fix any failures | Both |
| Morning | Finalize slideshow (16 slides: 1 title + 14 content + 1 close, plus 2 backup) | Alice |
| Afternoon | Final review against submission checklist (see `09_submission_checklist.md`) | Both |
| Afternoon | MLA 8 citations formatted and verified | Alice |
| Evening | Submit via Google Form | Jimmy |

---

## Risk & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Alpha Vantage 25 req/day limit** | High | Medium | Cache all responses to `data/raw/`. Use yfinance as primary, Alpha Vantage as supplement. Pre-download on Day 1. |
| **pytrends rate limiting / blocking** | Medium | Medium | Use `time.sleep(10)` between requests. Cache results. Fall back to manual Google Trends CSV export. |
| **Reddit API returns sparse data** | Medium | Low | Reddit is supplementary (Layer 5). If sparse, focus on Google Trends + earnings calls for sentiment. |
| **EDGAR transcript parsing fails** | Medium | Medium | Use Seeking Alpha transcripts as backup. Worst case, manually collect 8-12 quarterly transcripts for key periods. |
| **GPT-4o-mini sentiment inconsistency** | Medium | Low | Run each transcript 3x, take median score. Document variance. Use temperature=0 for determinism. |
| **yfinance data gaps for CSCO 1998** | Low | High | Validate CSCO data against known historical prices. Cross-reference with Alpha Vantage. Fill gaps with linear interpolation. |
| **Notebook doesn't run top-to-bottom** | Medium | Critical | Test `Restart & Run All` on Day 3 and Day 4 both. Cache intermediate results as parquet so cells are idempotent. |
| **Slideshow and notebook tell different stories** | Low | High | Alice reviews notebook -> slides alignment on Day 4. Every slide references a specific notebook section. |
| **S&P 500 historical weights unavailable** | Medium | Medium | Use annual top-10 lists from Wikipedia/Slickcharts. Compute approximate weights from market cap data via yfinance. |
| **Data looks nothing like a bubble** | Low | Medium | This is actually fine -- our research question is testable either way. A "no, it's not a bubble" finding is equally valid and publishable. Adjust narrative accordingly. |

---

## Color Palette & Visual Standards

| Element | Color | Hex |
|---|---|---|
| NVIDIA / AI era | NVIDIA Green | `#76b900` (light bg) / `#00CC96` (dark bg) |
| Cisco / Dot-com era | Coral Red | `#EF553B` |
| S&P 500 | Neutral Gray | `#888888` |
| Danger/Bubble zone | Red | `#d62728` |
| Safe/Healthy zone | Green | `#2ca02c` |
| Background | White | `#ffffff` |
| Grid lines | Light gray | `#e5e5e5` |

### Chart Template (Plotly)

```python
CHART_TEMPLATE = dict(
    layout=dict(
        font=dict(family="Inter, sans-serif", size=13),
        title_font_size=18,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=True, gridcolor="#e5e5e5", gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor="#e5e5e5", gridwidth=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=60),
    )
)

COLOR_NVDA = "#76b900"       # Use #00CC96 on dark backgrounds
COLOR_CSCO = "#EF553B"       # Coral red — dot-com era
COLOR_SP500 = "#888888"
```

---

## Dependencies (Beyond requirements.txt)

These additional packages are needed for the ML pipeline and should be added:

```
# ML / DTW
scikit-learn>=1.5
dtw-python>=1.5
shap>=0.45

# Sentiment
textblob>=0.18          # Backup sentiment (Reddit)
vaderSentiment>=3.3     # Backup sentiment (Reddit)

# Data APIs
fredapi>=0.5
pytrends>=4.9
praw>=7.7

# Web scraping (backup for transcripts)
beautifulsoup4>=4.12
lxml>=5.2
```

---

## Key Assumptions & Definitions

| Term | Definition |
|---|---|
| **t=0 (CSCO)** | 1998-01-02, start of the dot-com acceleration phase |
| **t=0 (NVDA)** | 2023-01-03, first trading day of 2023 (ChatGPT was released Nov 2022) |
| **Bubble** | A condition where asset prices significantly exceed fundamental value, sustained by speculative demand, eventually correcting sharply |
| **Rally** | A sustained price increase without the pejorative implication of irrationality |
| **Regime** | A distinct market phase with characteristic statistical properties (momentum, volatility, valuation) |
| **DTW similarity** | Dynamic Time Warping distance normalized to [0, 1] where 0 = identical trajectories, 1 = maximally different |
| **Bubble scorecard** | Summary matrix comparing CSCO-at-peak vs NVDA-at-current across all 5 layers, with similarity rating |

---

## File Structure (Final)

```
datathon26-2kim/
  specs/                          # You are here
    00_project_overview.md        # This file
    01_layer1_price.md
    02_layer2_fundamentals.md
    03_layer3_concentration.md
    04_layer4_macro.md
    05_layer5_sentiment.md
    06_ml_pipeline.md
    07_visualization_plan.md
    08_presentation_narrative.md
    09_submission_checklist.md
  data/
    raw/                          # Gitignored raw downloads
    processed/                    # Gitignored cleaned/merged data
  src/
    loaders.py                    # API data fetchers
    cleaning.py                   # Preprocessing pipeline
    features.py                   # Feature engineering
    stats.py                      # Statistical tests
    ml.py                         # DTW + regime classifier
    viz.py                        # Chart generation (Plotly)
    export.py                     # Notebook/slide export helpers
  notebooks/
    2kim_finance_notebook.ipynb   # SUBMISSION FILE
  submissions/
    2kim_finance_slides.pptx      # SUBMISSION FILE
  backend/                        # Demo app (FastAPI)
  frontend/                       # Demo app (React)
  requirements.txt
  .env                            # API keys (gitignored)
  .gitignore
  CLAUDE.md
  README.md
```
