# 09 -- Submission Checklist

This document maps every competition requirement to a concrete deliverable and provides pre-submission verification steps. Use this as the final gate before uploading.

---

## 1. Notebook Structure Checklist

The notebook MUST follow the 7-section structure plus the mandatory citation cell. Each section below includes the required header text, minimum content, and the rubric category it satisfies.

### Section 1: Problem Statement & Research Question
- [ ] Header: `## 1. Problem Statement & Research Question`
- [ ] State the research question verbatim: *"Is the NVIDIA-led AI equity rally of 2023-2026 structurally analogous to the Cisco-led dot-com bubble of 1998-2001, and if so, at what stage of the bubble lifecycle are we today?"*
- [ ] List 6 sub-questions (one per data layer + ML)
- [ ] State null hypothesis (H0) and alternative hypothesis (H1)
- [ ] Include 1-2 paragraphs on real-world significance (investors, policymakers, AI community)
- [ ] Reference at least 2 prior works or known analyses on tech bubbles (cited in MLA 8)
- [ ] Explicitly state: this is a Finance & Economics track submission
- [ ] **Rubric alignment:** Research Question & Track Fit (15pts)

### Section 2: Data Collection & Sources
- [ ] Header: `## 2. Data Collection & Sources`
- [ ] Include a summary table of all data sources (name, API, what it provides, time range, granularity)
- [ ] Show live API calls OR cached data loading with clear comments explaining the source
- [ ] For each data source, display `df.head()` and `df.shape` immediately after loading
- [ ] Print dtypes with `df.dtypes` or `df.info()` for at least 2 key DataFrames
- [ ] Document API rate limits and caching strategy in markdown cells
- [ ] Show the merge/join logic that combines sources (explicit join keys, join type)
- [ ] Minimum data sources: 7 (yfinance prices, yfinance fundamentals, FRED, Google Trends, Reddit, OpenAI/earnings, S&P 500 weights)
- [ ] **Rubric alignment:** Data Quality & Preparation (20pts)

### Section 3: Data Cleaning & Preprocessing
- [ ] Header: `## 3. Data Cleaning & Preprocessing`
- [ ] Show missing value counts BEFORE cleaning: `df.isnull().sum()`
- [ ] Document cleaning steps in order:
  - [ ] Date parsing and timezone handling
  - [ ] Null handling strategy (forward-fill for time series, documented per column)
  - [ ] Outlier detection (z-score > 3 on returns, document any removals)
  - [ ] Normalization (rebase to 100 for price overlay)
  - [ ] Feature engineering (list every derived column with formula)
- [ ] Show missing value counts AFTER cleaning: `df.isnull().sum()`
- [ ] Display `df.describe()` for the master features DataFrame
- [ ] Show the final merged DataFrame shape and column list
- [ ] Print the first and last 5 rows of the master DataFrame
- [ ] Save cleaned data: `master_features.to_parquet("data/processed/master_features.parquet")`
- [ ] **Rubric alignment:** Data Quality & Preparation (20pts)

### Section 4: Exploratory Data Analysis
- [ ] Header: `## 4. Exploratory Data Analysis`
- [ ] **Layer 1 -- Price Overlay:**
  - [ ] Dual-line chart: NVDA vs CSCO normalized to 100 (x-axis = trading days since t=0)
  - [ ] Annotate CSCO peak (Mar 27, 2000, day ~550) and any NVDA milestones
  - [ ] Report DTW similarity score as primary quantitative measure; Pearson/Spearman correlation of log-returns as secondary
  - [ ] Caption stating the finding
- [ ] **Layer 2 -- Fundamentals:**
  - [ ] Grouped bar chart: P/E, P/S at comparable rally stages (t+250, t+500, t+750)
  - [ ] Revenue growth overlay (both indexed to 100)
  - [ ] Free cash flow comparison (bar chart or line)
  - [ ] R&D spending as % of revenue (line chart)
  - [ ] Two-sample t-test on quarterly revenue growth rates, report t-statistic and p-value
  - [ ] Caption: "NVDA has [stronger/weaker] fundamental backing than CSCO at comparable stages"
- [ ] **Layer 3 -- Concentration:**
  - [ ] Area chart: top-1, top-5, top-10 % of S&P 500 market cap (1996-2026)
  - [ ] Line chart: equal-weight / cap-weight ratio (RSP/SPY or equivalent)
  - [ ] Report current top-5 concentration vs 2000 peak as z-score
  - [ ] Caption discussing concentration risk implications
- [ ] **Layer 4 -- Macro:**
  - [ ] 2x2 subplot panel: Fed funds rate, M2 growth (YoY%), credit spread (BAA-AAA), yield curve (10Y-2Y)
  - [ ] Each panel overlays CSCO era vs NVDA era
  - [ ] Report cosine similarity of macro regime vectors
  - [ ] Caption: "Macro conditions are [similar to / different from] the dot-com era because..."
- [ ] **Layer 5 -- Sentiment:**
  - [ ] Google Trends line chart for "NVIDIA stock", "AI bubble" (2004-2026)
  - [ ] Reddit activity chart (weekly post count + average sentiment)
  - [ ] Earnings call sentiment radar (scatterpolar): optimism, certainty, hype axes
  - [ ] Mann-Whitney U test comparing sentiment distributions between eras
  - [ ] Caption on each chart
- [ ] **Cross-layer correlation matrix:**
  - [ ] Heatmap of all features from master DataFrame
  - [ ] Annotate notable cross-layer correlations
- [ ] **Minimum chart count: 10 distinct visualizations**
- [ ] **Minimum statistical test count: 4 tests with test statistic + p-value**
- [ ] **Rubric alignment:** Analysis & Evidence (25pts)

### Section 5: Statistical Modeling & Machine Learning
- [ ] Header: `## 5. Statistical Modeling & Machine Learning`
- [ ] **DTW Analysis:**
  - [ ] Compute DTW distance between NVDA and CSCO normalized price series
  - [ ] Visualize warping path (alignment plot)
  - [ ] Report normalized DTW distance (0-1 scale)
  - [ ] Run DTW on individual features (price, volume, P/E, sentiment) and report per-feature distances
  - [ ] Bootstrap confidence interval on DTW distance (1000 resamples, report 95% CI)
- [ ] **Regime Classifier:**
  - [ ] Document labeling scheme for CSCO eras (5 regimes with date ranges)
  - [ ] List feature columns used (minimum 10 features)
  - [ ] Train Random Forest (n_estimators=200, max_depth=8, random_state=42)
  - [ ] Report 5-fold TimeSeriesSplit cross-validation accuracy + confusion matrix on CSCO era
  - [ ] Apply trained model to NVDA era, show predicted regime per quarter
  - [ ] Visualize: NVDA price chart color-coded by predicted regime
- [ ] **Feature Importance:**
  - [ ] Random Forest `.feature_importances_` horizontal bar chart (top 15)
  - [ ] SHAP summary plot (beeswarm or bar)
  - [ ] Discussion: which bubble indicators are most predictive?
- [ ] **Robustness Checks:**
  - [ ] Vary t=0 alignment by +/- 60 trading days, report DTW sensitivity
  - [ ] Ablation study: remove one layer at a time, report classifier accuracy change
  - [ ] Document random seeds used (`np.random.seed(42)`, `random_state=42`)
- [ ] **Rubric alignment:** Analysis & Evidence (25pts) + Technical Rigor (15pts)

### Section 6: Results & Discussion
- [ ] Header: `## 6. Results & Discussion`
- [ ] **Bubble Scorecard Table:**

  ```
  | Dimension      | Metric           | CSCO @ Peak | NVDA @ Current | Similarity (1-5) | Signal    |
  |----------------|------------------|-------------|----------------|-------------------|-----------|
  | Price          | Return from t=0  | +2,700%     | +X,XXX%        | X                 | [color]   |
  | Fundamentals   | Trailing P/E     | 196x        | XXx            | X                 | [color]   |
  | Concentration  | Top-5 % of S&P   | XX%         | XX%            | X                 | [color]   |
  | Macro          | Fed rate regime  | Easing      | [Tight/Easing] | X                 | [color]   |
  | Sentiment      | Earnings hype    | X/10        | X/10           | X                 | [color]   |
  | ML             | DTW distance     | --          | 0.XX           | X                 | [color]   |
  | ML             | Predicted regime | --          | [Stage]        | --                | [color]   |
  ```

- [ ] Answer each of the 6 sub-questions with a 2-3 sentence paragraph citing specific numbers
- [ ] Overall verdict paragraph: "Based on X of 5 layers showing strong similarity..."
- [ ] Explicitly state what makes NVDA different from CSCO (the nuance)
- [ ] Practical implications for investors (actionable, not just academic)
- [ ] **Executive Summary cell** immediately after Section 6: top 3 findings in 3 sentences (e.g., "Finding 1: DTW similarity of X/100. Finding 2: NVDA P/E is Xx lower than CSCO at equivalent stages. Finding 3: Market concentration exceeds dot-com peak."). This is what judges will actually read.
- [ ] **Rubric alignment:** Presentation Clarity & Storytelling (15pts)

### Section 7: Limitations, Ethics & Future Work
- [ ] Header: `## 7. Limitations, Ethics & Future Work`
- [ ] **Data Limitations (minimum 4):**
  - [ ] Google Trends data only available from 2004 -- missing dot-com era sentiment
  - [ ] S&P 500 historical weights are approximate (point-in-time data is expensive)
  - [ ] Reddit did not exist during the dot-com era -- asymmetric sentiment sources
  - [ ] Quarterly financial data has low granularity for fast-moving markets
  - [ ] yfinance data may contain adjusted-price artifacts for pre-2000 data
- [ ] **Methodological Limitations (minimum 3):**
  - [ ] N=2 comparison (CSCO, NVDA) -- cannot generalize to all bubbles
  - [ ] Regime labels for CSCO era are manually assigned with hindsight bias
  - [ ] DTW is sensitive to the choice of t=0 alignment (mitigated by sensitivity analysis)
  - [ ] GPT-4o-mini sentiment scores are model outputs, not validated against human annotators
  - [ ] Random Forest feature importance can be misleading with correlated features (mitigated by SHAP)
- [ ] **Correlation vs. Causation:**
  - [ ] Explicit statement: "Price similarity does not mean the same causal mechanisms are at work"
  - [ ] Give a specific example (e.g., "high P/E + high Google Trends interest both occurred, but one does not cause the other")
  - [ ] Discuss confounders: general market rally, index construction changes, monetary policy differences
- [ ] **Survivorship / Selection Bias:**
  - [ ] "We selected CSCO for comparison because it crashed -- this is textbook selection bias"
  - [ ] Acknowledge that many high-P/E stocks in 1999 did NOT crash immediately
  - [ ] Note that NVDA was selected because it's the current narrative leader, not randomly
- [ ] **Ethical Considerations:**
  - [ ] "This analysis should not be construed as financial advice"
  - [ ] Discuss potential for this type of analysis to cause panic or irrational selling if misinterpreted
  - [ ] Note that AI tools (GPT-4o-mini) were used for sentiment scoring -- outputs are not ground truth
  - [ ] Team is responsible for all claims and has verified AI-generated outputs
- [ ] **Future Work (minimum 2):**
  - [ ] Expand to N>2 historical parallels (e.g., add Japan 1985-1990, China 2014-2015)
  - [ ] Real-time dashboard that updates the bubble scorecard as new data arrives
  - [ ] Validate regime classifier against out-of-sample bubbles
- [ ] **Rubric alignment:** Limitations, Ethics & Practical Insight (10pts)

### Section 8: Dataset Citations (MLA 8)
- [ ] Header: `## Dataset Citations (MLA 8)`
- [ ] This MUST be the LAST cell in the notebook
- [ ] Cell type: Markdown (not code)
- [ ] Every data source used in the analysis must have a citation
- [ ] Format: MLA 8th Edition
- [ ] See Section 2 below for citation templates

---

## 2. MLA 8 Citation Templates

Each citation below is pre-formatted for MLA 8. Replace bracketed values with actuals before submission.

### yfinance (NVDA Price Data)
```
"NVIDIA Corporation (NVDA) Historical Data." Yahoo Finance, Verizon Media,
    finance.yahoo.com/quote/NVDA/history. Accessed 28 Mar. 2026.
```

### yfinance (CSCO Price Data)
```
"Cisco Systems, Inc. (CSCO) Historical Data." Yahoo Finance, Verizon Media,
    finance.yahoo.com/quote/CSCO/history. Accessed 28 Mar. 2026.
```

### yfinance (Nortel Networks — Non-Survivor Analog)
```
"Nortel Networks Corporation (NT) Historical Data." Yahoo Finance, Verizon Media,
    finance.yahoo.com/quote/NRTLQ/history. Accessed 28 Mar. 2026.
```

### yfinance (S&P 500 Index)
```
"S&P 500 (^GSPC) Historical Data." Yahoo Finance, Verizon Media,
    finance.yahoo.com/quote/%5EGSPC/history. Accessed 28 Mar. 2026.
```

### yfinance (Quarterly Financials -- NVDA)
```
"NVIDIA Corporation (NVDA) Financials." Yahoo Finance, Verizon Media,
    finance.yahoo.com/quote/NVDA/financials. Accessed 28 Mar. 2026.
```

### yfinance (Quarterly Financials -- CSCO)
```
"Cisco Systems, Inc. (CSCO) Financials." Yahoo Finance, Verizon Media,
    finance.yahoo.com/quote/CSCO/financials. Accessed 28 Mar. 2026.
```

### FRED -- Federal Funds Rate
```
Board of Governors of the Federal Reserve System (US). "Federal Funds Effective Rate
    [FEDFUNDS]." FRED, Federal Reserve Bank of St. Louis,
    fred.stlouisfed.org/series/FEDFUNDS. Accessed 28 Mar. 2026.
```

### FRED -- M2 Money Supply
```
Board of Governors of the Federal Reserve System (US). "M2 [M2SL]." FRED, Federal
    Reserve Bank of St. Louis, fred.stlouisfed.org/series/M2SL. Accessed 28 Mar. 2026.
```

### FRED -- BAA-AAA Credit Spread
```
"Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury
    Constant Maturity [BAAFFM]." FRED, Federal Reserve Bank of St. Louis,
    fred.stlouisfed.org/series/BAAFFM. Accessed 28 Mar. 2026.
```

### FRED -- 10-Year Minus 2-Year Treasury Spread
```
"10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
    [T10Y2Y]." FRED, Federal Reserve Bank of St. Louis,
    fred.stlouisfed.org/series/T10Y2Y. Accessed 28 Mar. 2026.
```

### FRED -- GDP
```
U.S. Bureau of Economic Analysis. "Gross Domestic Product [GDP]." FRED, Federal
    Reserve Bank of St. Louis, fred.stlouisfed.org/series/GDP. Accessed 28 Mar. 2026.
```

### FRED -- CPI
```
U.S. Bureau of Labor Statistics. "Consumer Price Index for All Urban Consumers: All
    Items in U.S. City Average [CPIAUCSL]." FRED, Federal Reserve Bank of St. Louis,
    fred.stlouisfed.org/series/CPIAUCSL. Accessed 28 Mar. 2026.
```

### Google Trends
```
"Google Trends: NVIDIA Stock, AI Bubble." Google Trends, Google,
    trends.google.com/trends/explore?q=NVIDIA+stock,AI+bubble. Accessed 28 Mar. 2026.
```

### Reddit (via PRAW)
```
"r/wallstreetbets, r/investing, r/stocks." Reddit, reddit.com. Accessed via PRAW
    (Python Reddit API Wrapper), 28 Mar. 2026.
```

### OpenAI API (Sentiment Analysis)
```
OpenAI. "GPT-4o-mini." OpenAI API, openai.com/api. Accessed 28 Mar. 2026. Used for
    sentiment scoring of earnings call transcripts.
```

### SEC EDGAR (Earnings Transcripts)
```
U.S. Securities and Exchange Commission. "EDGAR Full-Text Search." SEC.gov,
    efts.sec.gov/LATEST/search-index. Accessed 28 Mar. 2026.
```

### S&P 500 Constituent Weights
```
"S&P 500 Companies by Weight." SlickCharts, slickcharts.com/sp500. Accessed
    28 Mar. 2026.
```

### Alpha Vantage (Backup Data)
```
"Alpha Vantage API." Alpha Vantage, alphavantage.co. Accessed 28 Mar. 2026.
```

**Note:** Update all "Accessed" dates to the actual date of data download before submission. If a source was accessed on multiple dates, use the most recent date.

---

## 3. File Naming Requirements

| Deliverable | Required Filename | Format |
|---|---|---|
| Notebook | `2kim_finance_notebook.ipynb` | Jupyter Notebook (.ipynb) |
| Slideshow | `2kim_finance_slides.pptx` | PowerPoint (.pptx) or PDF (.pdf) |

### File naming rules (from competition):
- Team name prefix: `2kim`
- Track name: `finance`
- Descriptive suffix: `notebook` or `slides`
- Separator: underscore (`_`)
- All lowercase

### Pre-submission file verification:

```bash
# Run from project root
ls -la submissions/2kim_finance_notebook.ipynb
ls -la submissions/2kim_finance_slides.pptx

# Verify notebook is valid JSON (corrupt notebooks won't open)
python -c "import json; json.load(open('submissions/2kim_finance_notebook.ipynb'))"

# Verify notebook runs top-to-bottom
jupyter nbconvert --to notebook --execute submissions/2kim_finance_notebook.ipynb \
    --output /tmp/test_run.ipynb --ExecutePreprocessor.timeout=600
```

---

## 4. Slideshow Requirements Checklist

- [ ] **Format:** .pptx or .pdf (PowerPoint or Google Slides export)
- [ ] **Slide count:** 16 slides (1 title + 14 content + 1 close) + 2 backup not shown in main presentation (see `08_presentation_narrative.md`)
- [ ] **Title slide includes:** project title, team name (2Kim), track (Finance & Economics), date, SBU AI Community Datathon 2026
- [ ] **Every chart** from the notebook that appears in slides uses the same color scheme (NVDA green `#76b900` light bg / `#00CC96` dark bg, CSCO coral red `#EF553B`)
- [ ] **Every chart** has a title, axis labels, and legend
- [ ] **No wall of text** -- slides should be visual-heavy, text-light
- [ ] **Speaker notes** included for each slide (for reference during presentation, if applicable)
- [ ] **Font consistency:** Use one sans-serif font family throughout (Inter, Helvetica, or Arial)
- [ ] **Font sizes:** Title >= 28pt, body >= 18pt, footnotes >= 12pt
- [ ] **Slide numbering** enabled
- [ ] **Last slide** has references / MLA 8 citations (abbreviated form acceptable)
- [ ] **No animations** that could break in PDF conversion
- [ ] **Charts exported** from Plotly at 1200x800px minimum (`fig.write_image("chart.png", width=1200, height=800, scale=2)`)
- [ ] **Aspect ratio:** 16:9 widescreen

### Slide-by-slide content checklist:

| # | Title | Must Include |
|---|---|---|
| 1 | Title | Project title, team, track, date |
| 2 | The Hook | NVDA (green) vs CSCO (coral red) price overlay + Nortel (gray dashed) |
| 3 | Research Framework | Five-layer diagram, one-line question per layer |
| 4 | Data & Methods | Data sources table (7+ sources), methodology summary (15 sec) |
| 5 | P/E Comparison (Bull) | Layer 2 chart, CSCO 200x vs NVDA [X]x |
| 6 | Revenue Growth | Layer 2 grouped bars, 265% vs 66% stat callout |
| 7 | Concentration (Bear) | Layer 3 area chart + SPY/RSP, z-score, Buffett Indicator stat verbal |
| 8 | Sentiment | Layer 5: AI mentions + hype vs specificity charts |
| 9 | Macro Wild Card | Layer 4 macro dashboard small multiples |
| 10 | ML: Regime Classification | Regime probabilities stacked area, March 2026 readout |
| 11 | ML: DTW + SHAP | DTW similarity timeline + SHAP waterfall |
| 12 | Synthesis | Bubble scorecard table with traffic-light colors |
| 13 | Key Takeaway | One-sentence verdict, large quote text |
| 14 | Limitations & Ethics | Top 4 limitations, correlation vs causation, not financial advice |
| 15 | Thank You / Q&A | Contact info, notebook reference, dashboard available |
| 16 (backup) | SHAP Deep Dive | SHAP beeswarm plot, available if judge asks |
| 17 (backup) | Methodology | Walk-forward validation, confusion matrix |

---

## 5. Judging Rubric Alignment

### Analysis & Evidence (25pts) -- TIEBREAKER #1, MOST IMPORTANT

**What judges look for:**
- Strongest EDA with multiple visualization types
- Correct statistical tests with reported test statistics and p-values
- Every claim backed by data (no hand-waving)
- Depth of analysis (not just surface-level charts)

**Our deliverables that satisfy this:**

| Deliverable | Points toward |
|---|---|
| 10+ distinct Plotly visualizations across 5 data layers | Strongest EDA |
| DTW similarity score (primary) + Pearson/Spearman on log-returns (secondary) (Layer 1) | Correct stats |
| Two-sample t-test on revenue growth (Layer 2) | Correct stats |
| Z-score comparison of concentration (Layer 3) | Correct stats |
| Cosine similarity of macro vectors (Layer 4) | Correct stats |
| Mann-Whitney U on sentiment distributions (Layer 5) | Correct stats |
| DTW distance with bootstrap CI | Advanced method |
| Random Forest regime classifier with CV accuracy | ML rigor |
| SHAP feature importance | Interpretability |
| Bubble scorecard synthesizing all layers | Integration |

**Target: 22-25/25**

### Data Quality & Preparation (20pts) -- TIEBREAKER #2

**What judges look for:**
- Multiple data sources merged cleanly
- Documented preprocessing steps
- Missing data handled appropriately
- Feature engineering is justified

**Our deliverables that satisfy this:**

| Deliverable | Points toward |
|---|---|
| 7+ distinct data sources (yfinance x2, FRED x6, Trends, Reddit, OpenAI, EDGAR, S&P weights) | Multi-source |
| Explicit join keys and merge logic shown in code | Clean merge |
| `df.isnull().sum()` before and after cleaning | Documented handling |
| `df.describe()` on master DataFrame | Data profiling |
| Feature engineering with formulas documented (P/E, FCF yield, credit spread z-score, etc.) | Justified features |
| Data cached as .parquet for reproducibility | Reproducible pipeline |

**Target: 18-20/20**

### Research Question & Track Fit (15pts)

**What judges look for:**
- Specific, not vague
- Testable (can be answered with data)
- Clearly relevant to the chosen track

**Our deliverables that satisfy this:**

| Deliverable | Points toward |
|---|---|
| Question specifies NVDA vs CSCO (not generic "is there a bubble?") | Specific |
| 6 sub-questions, each answerable with a quantitative metric | Testable |
| H0/H1 framing allows falsification | Scientific rigor |
| Stock prices, P/E ratios, Fed policy, market concentration = clearly Finance & Economics | Track fit |

**Target: 13-15/15**

### Technical Rigor & Reproducibility (15pts)

**What judges look for:**
- Notebook runs top-to-bottom without errors
- Sound statistical methods
- No data leakage in ML
- Reproducible results

**Our deliverables that satisfy this:**

| Deliverable | Points toward |
|---|---|
| `Kernel > Restart & Run All` tested on Day 3 and Day 4 | Runs clean |
| TimeSeriesSplit for CV (no future leakage) | Sound ML |
| All random seeds set to 42 | Reproducible |
| Results reference cell outputs, not external files | Self-contained |
| Intermediate data cached as .parquet (cells are idempotent) | Robust pipeline |
| API calls wrapped in try/except with fallback to cache | Handles failures |

**Target: 13-15/15**

### Presentation Clarity & Storytelling (15pts)

**What judges look for:**
- Clear structure with logical flow
- Readable visualizations
- Story arc (not a data dump)

**Our deliverables that satisfy this:**

| Deliverable | Points toward |
|---|---|
| Every section starts with a question, ends with an answer | Clear structure |
| Consistent color scheme (NVDA green `#76b900`/`#00CC96`, CSCO coral red `#EF553B`) across all charts | Readable visuals |
| Story arc: hook -> evidence -> nuance -> verdict | Narrative flow |
| Bubble scorecard as the synthesis artifact | Clear conclusion |
| Slideshow follows the notebook structure | Consistency |
| Chart captions state the finding (not just "Figure 3") | Self-documenting |

**Target: 12-15/15**

### Limitations, Ethics & Practical Insight (10pts) -- TIEBREAKER #3

**What judges look for:**
- Honest about what the analysis can and cannot show
- Distinguishes correlation from causation
- Practical implications

**Our deliverables that satisfy this:**

| Deliverable | Points toward |
|---|---|
| 4+ data limitations with specific examples | Honest limitations |
| 3+ methodological limitations | Self-awareness |
| Explicit "correlation != causation" statement with example | Statistical honesty |
| Survivorship / selection bias discussion | Sophisticated thinking |
| "Not financial advice" disclaimer | Ethics |
| Discussion of AI tool usage and verification | Responsible AI use |
| 2+ future work directions | Intellectual honesty |

**Target: 8-10/10**

---

## 6. Pre-Submission Verification Steps

Run these checks in order on the final day. Both team members should independently verify.

### 6.1 Notebook Integrity

```bash
# 1. Verify file exists and is named correctly
ls submissions/2kim_finance_notebook.ipynb

# 2. Verify valid JSON
python3 -c "
import json
with open('submissions/2kim_finance_notebook.ipynb') as f:
    nb = json.load(f)
print(f'Cells: {len(nb[\"cells\"])}')
print(f'Kernel: {nb[\"metadata\"][\"kernelspec\"][\"display_name\"]}')
"

# 3. Restart & Run All (most critical check)
jupyter nbconvert --to notebook --execute \
    submissions/2kim_finance_notebook.ipynb \
    --output /tmp/test_execution.ipynb \
    --ExecutePreprocessor.timeout=600 \
    --ExecutePreprocessor.kernel_name=python3

# 4. Verify no empty output cells (every code cell should produce output)
python3 -c "
import json
with open('submissions/2kim_finance_notebook.ipynb') as f:
    nb = json.load(f)
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
empty = [i for i, c in enumerate(code_cells) if not c.get('outputs')]
if empty:
    print(f'WARNING: Code cells with no output: {empty}')
else:
    print('All code cells have outputs.')
"
```

### 6.2 Section Structure Verification

```python
# Run this IN the notebook as a self-check cell (delete before submission, or keep as last code cell before citations)
import json

with open('2kim_finance_notebook.ipynb') as f:
    nb = json.load(f)

required_headers = [
    "1. Problem Statement",
    "2. Data Collection",
    "3. Data Cleaning",
    "4. Exploratory Data Analysis",
    "5. Statistical Modeling",
    "6. Results & Discussion",
    "7. Limitations",
    "Dataset Citations (MLA 8)"
]

md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
all_md_text = '\n'.join([''.join(c['source']) for c in md_cells])

for header in required_headers:
    if header.lower() in all_md_text.lower():
        print(f"  FOUND: {header}")
    else:
        print(f"  MISSING: {header}")
```

### 6.3 Citation Verification

- [ ] Last cell is a Markdown cell (not code)
- [ ] Cell starts with `## Dataset Citations (MLA 8)`
- [ ] Every data source used in the notebook has a corresponding citation
- [ ] All "Accessed" dates are accurate
- [ ] MLA 8 format: Author. "Title." *Container*, Publisher, URL. Accessed DD Mon. YYYY.
- [ ] Cross-check: count unique data sources in Section 2 vs. citation count in Section 8. They must match.

### 6.4 Chart Quality Verification

For every chart in the notebook:
- [ ] Has a descriptive title (not "Figure 1" or "Plot")
- [ ] X-axis and Y-axis are labeled with units
- [ ] Legend present if multiple series
- [ ] Font size readable (>= 12pt)
- [ ] Color scheme matches the standard (NVDA = `#76b900` / `#00CC96`, CSCO = `#EF553B` coral red)
- [ ] Caption in the Markdown cell below the chart stating the key finding
- [ ] No truncated labels or overlapping text

### 6.5 Statistical Test Verification

For every statistical test reported:
- [ ] Test name stated (e.g., "Pearson correlation", "Welch's t-test")
- [ ] Test statistic reported (e.g., r = 0.73, t = 2.41)
- [ ] P-value reported (e.g., p = 0.003)
- [ ] Significance level stated (alpha = 0.05)
- [ ] Conclusion stated in plain English (e.g., "reject H0 at the 5% level")
- [ ] Assumptions checked where applicable (normality for t-test, etc.)

### 6.6 ML Model Verification

- [ ] Random seed set: `np.random.seed(42)`, `random_state=42` in all sklearn estimators
- [ ] Train/test split uses `TimeSeriesSplit` (not random split -- time series data)
- [ ] No data leakage: model trained only on CSCO era, applied to NVDA era
- [ ] Cross-validation results reported (mean +/- std accuracy)
- [ ] Confusion matrix shown for CSCO era validation
- [ ] Feature importance plot uses the same features as the model
- [ ] SHAP values computed on the correct dataset

### 6.7 Reproducibility Verification

- [ ] All API calls have explicit parameters (dates, tickers, series IDs)
- [ ] Fallback to cached .parquet files if API is unavailable
- [ ] **Cache-or-call pattern verified for ALL API calls:** every API call follows `if Path("data/raw/<file>.parquet").exists(): load cache; else: call API, cache result, return data`. Verify by disconnecting network and running `Restart & Run All` — notebook should complete from cache alone.
- [ ] `requirements.txt` is up to date with all packages used
- [ ] No hardcoded absolute paths (use relative paths or `pathlib`)
- [ ] No cells depend on execution order (each cell is idempotent or clearly sequential)
- [ ] `%matplotlib inline` or Plotly renderer set at top of notebook

### 6.8 Slideshow Verification

- [ ] File exists: `submissions/2kim_finance_slides.pptx`
- [ ] Opens correctly in PowerPoint / Google Slides / LibreOffice
- [ ] 16 slides (1 title + 14 content + 1 close) + 2 backup
- [ ] Title slide has: project title, 2Kim, Finance & Economics, date
- [ ] All charts render (no broken images)
- [ ] Font sizes: title >= 28pt, body >= 18pt
- [ ] No typos (run spellcheck)
- [ ] Last slide has references
- [ ] Export to PDF as backup: `submissions/2kim_finance_slides.pdf`

### 6.9 Final Upload Verification

- [ ] Google Form is accessible (test the link)
- [ ] Upload both files:
  - `2kim_finance_notebook.ipynb`
  - `2kim_finance_slides.pptx` (or `.pdf`)
- [ ] Verify uploads completed (check file sizes are non-zero)
- [ ] Screenshot the confirmation page
- [ ] Both team members confirm submission

---

## 7. Common Pitfalls to Avoid

### Data Pitfalls

| Pitfall | Why It's Dangerous | How to Avoid |
|---|---|---|
| **Using adjusted close without understanding splits** | CSCO had multiple stock splits in 1998-2000. Adjusted close already accounts for this, but mixing adjusted and unadjusted creates artifacts. | Always use `Adj Close` from yfinance. Document this choice. |
| **Timezone mismatch between sources** | FRED data is in UTC, yfinance in exchange-local. Merging on date string can create off-by-one errors. | Convert all timestamps to UTC, then to date-only (no time component) before merging. |
| **Forward-looking bias in fundamentals** | Financial statements are filed weeks/months after quarter-end. Using them on the report date (not filing date) creates look-ahead bias. | For the regime classifier, lag fundamental features by 45 days (typical filing delay). |
| **Google Trends normalization is relative** | Google Trends normalizes each query independently to 0-100. Comparing absolute levels across queries is meaningless. | Only compare trends over TIME within the same query. Do not say "AI bubble is searched more than NVIDIA stock" based on raw Trends values. |
| **survivorship bias in S&P 500 constituents** | Today's S&P 500 includes NVDA but 1998's did not include many companies that went bankrupt. Using current constituents to measure past concentration is wrong. | Use point-in-time constituent lists where possible. Document this limitation clearly if unavailable. |

### Statistical Pitfalls

| Pitfall | Why It's Dangerous | How to Avoid |
|---|---|---|
| **P-hacking across 5 layers** | Running many tests increases the chance of spurious significance. | Apply Benjamini-Hochberg (FDR) correction via `statsmodels.stats.multitest.multipletests(pvalues, method='fdr_bh')`. Report both raw and adjusted p-values. |
| **Pearson correlation on non-stationary series** | Stock prices are non-stationary. Pearson r on price levels is meaningless (spurious correlation). | Use DTW as primary similarity measure. If computing Pearson/Spearman, apply to **log-returns** (not price levels) aligned by days_from_breakout. |
| **Overfitting the regime classifier** | Training on ~500 days (CSCO era) with 10+ features risks overfitting, especially with tree models. | Use max_depth=8, min_samples_leaf=10, and validate with TimeSeriesSplit. Report both train and validation accuracy. |
| **Interpreting DTW distance without context** | A DTW distance of 0.45 means nothing without a baseline. | Compute DTW between NVDA and a random walk as a null baseline. Report NVDA-CSCO distance as a percentile of the null distribution. |
| **Confusing statistical and practical significance** | A p-value of 0.01 on a Pearson r of 0.08 means the correlation is "significant" but essentially zero in magnitude. | Always report effect sizes alongside p-values. |

### Presentation Pitfalls

| Pitfall | Why It's Dangerous | How to Avoid |
|---|---|---|
| **Starting with methodology instead of the question** | Judges lose interest. "We used Random Forest..." is not a hook. | Start with the price overlay chart. The visual similarity is the hook. |
| **Too many charts, no narrative** | A data dump is not analysis. 20 charts with no interpretation scores lower than 8 charts with clear insights. | Every chart must have a 1-2 sentence finding stated in the caption. If a chart doesn't answer a question, cut it. |
| **Claiming "it's a bubble" without nuance** | Judges are professors. They will penalize overconfident claims. | Use probabilistic language: "X of 5 layers suggest similarity to the dot-com era, but Y layers show key differences." |
| **Inconsistent chart styling** | Mixed colors, fonts, and axes make the presentation look amateur. | Use the `CHART_TEMPLATE` defined in `00_project_overview.md` for every Plotly chart. |
| **Forgetting to say "not financial advice"** | Ethics section is 10% of the grade. Omitting the disclaimer costs easy points. | Include it in Section 7 AND on the limitations slide. |

### Technical Pitfalls

| Pitfall | Why It's Dangerous | How to Avoid |
|---|---|---|
| **Notebook doesn't run top-to-bottom** | This is the #1 reason notebooks fail technical rigor checks. | Test `Restart & Run All` at least twice before submission. Cache API data as .parquet so cells work offline. |
| **API keys hardcoded in notebook** | Judges can see your keys. Also a security risk. | Use `os.getenv("FRED_API_KEY")`. Add a markdown cell explaining that keys must be set in `.env`. |
| **Import errors from missing packages** | If judges try to run the notebook and scikit-learn is missing, it fails immediately. | First cell should be: `!pip install -q yfinance fredapi pytrends praw openai dtw-python shap scikit-learn textblob` |
| **Large notebook file size** | Plotly charts embedded as JSON can make notebooks 50MB+. Google Form may reject large files. | Use `fig.show(renderer="png")` in the notebook for static output. Or clear outputs and re-run before export. |
| **Relative imports from `src/`** | If the notebook is in `submissions/` but imports from `src/`, the path may break. | Add `sys.path.insert(0, '..')` at the top, or copy all necessary code into the notebook itself for the submission version. |

---

## 8. Emergency Fallback Plan

If something goes critically wrong on submission day:

### API is down
- All raw data should be cached in `data/raw/` as .parquet by Day 1
- Notebook cells should have `try: fetch_from_api() except: load_from_cache()` pattern
- If cache is also missing, skip that layer and note it in Limitations

### Notebook won't run
- Isolate the failing cell
- If it's an API call, replace with cached data load
- If it's a library error, pin the exact version in the `!pip install` cell
- If it's a data shape issue, add defensive `assert df.shape[0] > 0` checks

### Running out of time
- Priority order (highest value per time spent):
  1. Sections 1, 4, 6, 8 (Problem, EDA, Results, Citations) -- these cover 55pts
  2. Sections 2, 3 (Data Collection, Cleaning) -- these cover 20pts
  3. Section 5 (ML) -- nice-to-have but not required
  4. Section 7 (Limitations) -- easy 8-10pts, write last
- Skip the regime classifier if short on time. DTW alone is sufficient for the ML section.
- The bubble scorecard can be built manually from EDA findings without ML.

### Slideshow not done
- Export key notebook charts as PNGs
- Use Google Slides (faster than PowerPoint) with a blank template
- Minimum viable slideshow: title + 5 chart slides + conclusion = 7 slides
- This is better than no slideshow

---

## 9. Post-Submission Checklist

- [ ] Screenshot the Google Form confirmation page
- [ ] Save the confirmation email (if any)
- [ ] Push final code to git: `git add -A && git commit -m "final submission" && git push`
- [ ] Verify both files are accessible in the submission system
- [ ] Celebrate -- you shipped it
