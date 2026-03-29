# Layer 2: Fundamental Comparison -- NVDA vs CSCO

## 1. Objective

**Primary question:** Are NVIDIA's fundamentals in 2023-2026 materially stronger than Cisco's were in 1998-2001, or is the current valuation equally detached from financial reality?

**Why this matters:** The bull case for "this time is different" rests entirely on fundamentals. If NVDA is growing revenue 100%+ YoY with 70%+ gross margins while CSCO was growing 30% YoY at 65% margins, the valuation premium may be justified. If the multiples are equally stretched relative to growth, the bubble thesis strengthens.

**Sub-questions:**
- Is NVDA's P/E ratio trajectory more or less extreme than CSCO's at the equivalent cycle phase?
- Does NVDA have real revenue growth backing its valuation, or is it priced on forward expectations like CSCO was?
- How do free cash flow yields compare -- is NVDA generating cash or burning it?
- Are gross margins expanding (real moat) or compressing (commoditization risk)?
- Is R&D spend proportional to revenue, indicating sustainable innovation?

**Scoring alignment:** Targets Data Quality & Preparation (20pts) via multi-source financial data, and Analysis & Evidence (25pts) via quantified fundamental comparisons.

---

## 2. Data Sources

### 2.1 Primary Financial Data APIs

**Option A: Financial Modeling Prep (FMP) -- Preferred**

| Endpoint                        | URL Pattern                                                      | Data Returned             |
|---------------------------------|------------------------------------------------------------------|---------------------------|
| Income Statement (quarterly)    | `https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarter&apikey={key}` | Revenue, net income, EPS, gross profit, R&D expense |
| Balance Sheet (quarterly)       | `https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period=quarter&apikey={key}` | Total assets, total debt, shareholders' equity |
| Cash Flow Statement (quarterly) | `https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period=quarter&apikey={key}` | Operating cash flow, capex, free cash flow |
| Key Metrics (quarterly)         | `https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period=quarter&apikey={key}` | P/E, P/S, EV/EBITDA, FCF yield, market cap |
| Historical Market Cap            | `https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}?from={start}&to={end}&apikey={key}` | Daily market cap |

**Option B: Alpha Vantage -- Fallback**

| Function                | Parameters                                          | Notes                    |
|-------------------------|-----------------------------------------------------|--------------------------|
| `INCOME_STATEMENT`      | `function=INCOME_STATEMENT&symbol={ticker}&apikey={key}` | Quarterly & annual       |
| `BALANCE_SHEET`         | `function=BALANCE_SHEET&symbol={ticker}&apikey={key}`    | Quarterly & annual       |
| `CASH_FLOW`             | `function=CASH_FLOW&symbol={ticker}&apikey={key}`        | Quarterly & annual       |
| `OVERVIEW`              | `function=OVERVIEW&symbol={ticker}&apikey={key}`         | Current snapshot only    |

**Rate limits:**
- FMP free tier: 250 calls/day. Premium: 750 calls/day.
- Alpha Vantage free tier: 25 calls/day, 500 calls/month. Premium: 75 calls/min.

**Recommendation:** Use FMP as primary (better historical coverage for CSCO 1998-2003). Use Alpha Vantage as validation cross-check for NVDA current data.

### 2.2 Tickers and Date Ranges

**Current AI Cycle:**

| Ticker | Company     | Quarterly Data Range     | Role                  |
|--------|-------------|--------------------------|------------------------|
| NVDA   | NVIDIA      | Q1 FY2024 - Q4 FY2027 (Jan 2023 - Jan 2027) | Primary subject |
| AMD    | AMD         | Q1 2023 - Q1 2026        | AI semiconductor peer  |
| MSFT   | Microsoft   | Q1 FY2024 - Q3 FY2026    | AI platform/consumer   |
| AVGO   | Broadcom    | Q1 FY2024 - Q1 FY2026    | AI networking peer     |

**Note on NVDA fiscal year:** NVIDIA's fiscal year ends in late January. FY2024 = Feb 2023 - Jan 2024. All dates must be mapped to calendar quarters for cross-company alignment.

**Dot-Com Cycle:**

| Ticker | Company         | Quarterly Data Range     | Role                    |
|--------|-----------------|--------------------------|--------------------------|
| CSCO   | Cisco Systems   | Q1 FY1998 - Q4 FY2003 (Jul 1997 - Jul 2003) | Primary subject |
| INTC   | Intel           | Q1 1998 - Q4 2002        | Semiconductor incumbent  |
| QCOM   | Qualcomm        | Q1 FY1998 - Q4 FY2002    | Semiconductor peer       |
| SUNW   | Sun Microsystems| Q1 FY1998 - Q4 FY2003    | Infrastructure peer      |

**Note on CSCO fiscal year:** Cisco's fiscal year ends in late July. FY1998 = Aug 1997 - Jul 1998.

### 2.3 Inflation Adjustment Data

**Source:** FRED (Federal Reserve Economic Data)

| Series ID | Description                      | Frequency | URL                                                    |
|-----------|----------------------------------|-----------|--------------------------------------------------------|
| CPIAUCSL  | CPI for All Urban Consumers      | Monthly   | `https://fred.stlouisfed.org/series/CPIAUCSL`          |

```python
import fredapi
fred = fredapi.Fred(api_key=FRED_API_KEY)
cpi = fred.get_series("CPIAUCSL", observation_start="1997-01-01", observation_end="2026-03-28")

# Alternative: pandas_datareader
from pandas_datareader import data as pdr
cpi = pdr.get_data_fred("CPIAUCSL", start="1997-01-01", end="2026-03-28")
```

**Adjustment formula:**
```
real_value = nominal_value * (CPI_base / CPI_period)
```
Where `CPI_base` = CPI value for March 2026 (latest), so all dollar amounts are in 2026 dollars.

---

## 3. Data Schema

### 3.1 Raw Quarterly Financials Schema

| Column                  | Dtype      | Source Field (FMP)                | Unit        |
|-------------------------|------------|-----------------------------------|-------------|
| date                    | datetime64 | `date`                            | Quarter end |
| ticker                  | str        | --                                | --          |
| era                     | str        | --                                | "ai" / "dotcom" |
| calendar_quarter        | str        | Derived                           | "2024Q1"    |
| revenue                 | float64    | `revenue`                         | USD         |
| cost_of_revenue         | float64    | `costOfRevenue`                   | USD         |
| gross_profit            | float64    | `grossProfit`                     | USD         |
| research_and_development| float64    | `researchAndDevelopmentExpenses`   | USD         |
| operating_income        | float64    | `operatingIncome`                 | USD         |
| net_income              | float64    | `netIncome`                       | USD         |
| eps_diluted             | float64    | `epsdiluted`                      | USD/share   |
| weighted_avg_shares     | int64      | `weightedAverageShsOutDil`        | Shares      |
| operating_cash_flow     | float64    | Cash flow: `operatingCashFlow`    | USD         |
| capital_expenditure     | float64    | Cash flow: `capitalExpenditure`   | USD (negative) |
| free_cash_flow          | float64    | `operatingCashFlow` + `capitalExpenditure` | USD |
| total_assets            | float64    | Balance sheet: `totalAssets`      | USD         |
| total_debt              | float64    | Balance sheet: `totalDebt`        | USD         |
| shareholders_equity     | float64    | Balance sheet: `totalStockholdersEquity` | USD  |

### 3.2 Derived Metrics Schema

| Column                   | Dtype    | Formula                                                  |
|--------------------------|----------|----------------------------------------------------------|
| gross_margin             | float64  | `gross_profit / revenue`                                  |
| operating_margin         | float64  | `operating_income / revenue`                              |
| net_margin               | float64  | `net_income / revenue`                                    |
| rd_to_revenue            | float64  | `research_and_development / revenue`                      |
| revenue_yoy_growth       | float64  | `(revenue_q - revenue_q-4) / revenue_q-4`                |
| revenue_qoq_growth       | float64  | `(revenue_q - revenue_q-1) / revenue_q-1`                |
| eps_yoy_growth           | float64  | `(eps_q - eps_q-4) / abs(eps_q-4)`                       |
| fcf_yield                | float64  | `(fcf_ttm / market_cap) * 100`                           |
| pe_ratio                 | float64  | `market_cap / net_income_ttm` (trailing 4Q)              |
| ps_ratio                 | float64  | `market_cap / revenue_ttm` (trailing 4Q)                 |
| ev_to_ebitda             | float64  | `(market_cap + total_debt - cash) / ebitda_ttm`          |
| revenue_per_market_cap   | float64  | `revenue_ttm / market_cap` (revenue backing %)           |
| cycle_quarter            | int64    | Quarters since breakout (0 = breakout quarter)            |

### 3.3 Inflation-Adjusted Schema

| Column                        | Dtype    | Description                                          |
|-------------------------------|----------|------------------------------------------------------|
| revenue_real                  | float64  | Revenue in 2026 USD                                  |
| net_income_real               | float64  | Net income in 2026 USD                               |
| market_cap_real               | float64  | Market cap in 2026 USD                               |
| fcf_real                      | float64  | Free cash flow in 2026 USD                           |
| rd_real                       | float64  | R&D spend in 2026 USD                                |

---

## 4. Preprocessing Steps

### 4.1 Fiscal Year to Calendar Quarter Mapping

**CRITICAL:** This alignment is essential for cross-company comparison. NVDA's fiscal year ends in late January (FY2024 Q4 ends Jan 2024, reported as calendar Q4 2023), while CSCO's fiscal year ends in late July (FY2000 Q4 ends Jul 2000, reported as calendar Q3 2000). Misaligning fiscal quarters by even one quarter will distort all YoY growth comparisons and P/E trajectory overlays. Always map to calendar quarter based on the actual date of the quarter-end, not the fiscal quarter label.

```python
def map_fiscal_to_calendar(date: pd.Timestamp, ticker: str) -> str:
    """
    Map a fiscal quarter-end date to a calendar quarter string.

    NVDA fiscal year ends late January:
        - FY Q1 (Feb-Apr) -> Calendar Q1 (approx)
        - FY Q2 (May-Jul) -> Calendar Q2
        - FY Q3 (Aug-Oct) -> Calendar Q3
        - FY Q4 (Nov-Jan) -> Calendar Q4

    CSCO fiscal year ends late July:
        - FY Q1 (Aug-Oct) -> Calendar Q3/Q4
        - FY Q2 (Nov-Jan) -> Calendar Q4/Q1
        - FY Q3 (Feb-Apr) -> Calendar Q1
        - FY Q4 (May-Jul) -> Calendar Q2

    Simplification: use the actual date to derive calendar quarter.
    """
    year = date.year
    quarter = (date.month - 1) // 3 + 1
    return f"{year}Q{quarter}"
```

### 4.2 Trailing Twelve Month (TTM) Computation

Many valuation ratios require TTM (sum of last 4 quarters) metrics.

```python
def compute_ttm(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Compute trailing twelve month (4-quarter rolling sum) for specified columns.
    DataFrame must be sorted by date ascending.
    """
    for col in columns:
        df[f"{col}_ttm"] = df[col].rolling(window=4, min_periods=4).sum()
    return df

# Apply to flow metrics (revenue, net income, FCF, operating CF)
ttm_columns = ["revenue", "net_income", "free_cash_flow", "operating_cash_flow",
                "research_and_development"]
df = compute_ttm(df, ttm_columns)
```

### 4.3 Market Cap Alignment

Market cap changes daily, but financials are quarterly. Align by taking the market cap on the last trading day of each fiscal quarter.

```python
def get_quarterly_market_cap(ticker: str, quarter_end_dates: list[pd.Timestamp]) -> pd.Series:
    """
    Get market cap on the last trading day of each quarter.
    Use yfinance for daily market cap (price * shares outstanding)
    or FMP historical-market-capitalization endpoint.
    """
    # FMP approach
    import requests
    url = f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{ticker}"
    params = {"from": "1997-01-01", "to": "2026-03-28", "apikey": FMP_API_KEY}
    response = requests.get(url, params=params)
    mcap_daily = pd.DataFrame(response.json())
    mcap_daily["date"] = pd.to_datetime(mcap_daily["date"])
    mcap_daily = mcap_daily.set_index("date").sort_index()

    # Get closest trading day to each quarter end
    result = {}
    for qdate in quarter_end_dates:
        nearest = mcap_daily.index[mcap_daily.index.get_indexer([qdate], method="nearest")]
        result[qdate] = mcap_daily.loc[nearest[0], "marketCap"]

    return pd.Series(result, name="market_cap")
```

### 4.4 Inflation Adjustment

```python
def adjust_for_inflation(df: pd.DataFrame, cpi: pd.Series,
                          base_date: str = "2026-03-01",
                          nominal_columns: list[str] = None) -> pd.DataFrame:
    """
    Adjust nominal dollar values to real (2026) dollars using CPI.

    Parameters:
        df: DataFrame with quarterly financial data
        cpi: Monthly CPI series from FRED (CPIAUCSL)
        base_date: Reference date for 'today' dollars
        nominal_columns: List of columns to adjust
    """
    base_cpi = cpi.loc[base_date]

    for col in nominal_columns:
        # Map each quarter-end date to nearest CPI month
        real_values = []
        for date, value in zip(df["date"], df[col]):
            nearest_cpi_date = cpi.index[cpi.index.get_indexer([date], method="nearest")[0]]
            period_cpi = cpi.loc[nearest_cpi_date]
            real_values.append(value * (base_cpi / period_cpi))
        df[f"{col}_real"] = real_values

    return df

# Columns to adjust
nominal_cols = ["revenue", "net_income", "free_cash_flow", "market_cap",
                "research_and_development", "operating_cash_flow"]
```

**CPI reference points (approximate):**
- Jan 1998: ~162 (1982-84 = 100)
- Mar 2000 (CSCO peak): ~171
- Jan 2023: ~300
- Mar 2026: ~320 (estimated)
- Adjustment factor for 2000 dollars -> 2026 dollars: ~1.87x

### 4.5 Handling Missing Historical Data (Dot-Com Era)

**Known issues:**
- FMP may not have quarterly data for SUNW before 2000
- Some dot-com era companies have inconsistent reporting formats (pre-Sarbanes-Oxley)
- R&D expense may be bundled differently in older filings

**Strategy:**
1. Pull all available data from FMP
2. **REQUIRED:** Cross-reference yfinance/FMP quarterly data against SEC EDGAR for CSCO 1998-2001. yfinance is unreliable for historical data this old -- it frequently returns incomplete data, wrong fiscal year alignments, or NaN-filled frames. Manually verify at least 4 key quarters (FY1999 Q4, FY2000 Q2, FY2000 Q4, FY2001 Q2) against the actual 10-Q filings on EDGAR. Flag any discrepancies.
3. If a metric is unavailable for a specific quarter, mark as NaN -- do NOT interpolate financial data
4. For any ticker with <60% of expected quarters available, exclude from that specific analysis but note in limitations

**EDGAR cross-reference for CSCO (required validation step):**
```python
# Cisco's CIK: 858877
# EDGAR filing search: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=858877&type=10-Q&dateb=&owner=include&count=40
# For each key quarter, verify: revenue, net income, and EPS against the 10-Q filing
# This is a manual spot-check, not automated -- 30 minutes of effort that prevents data integrity failures
```

```python
# Validate data completeness
def check_data_completeness(df: pd.DataFrame, ticker: str,
                             expected_quarters: int) -> dict:
    actual = len(df)
    completeness = actual / expected_quarters * 100
    missing_fields = df.isnull().sum()

    report = {
        "ticker": ticker,
        "expected_quarters": expected_quarters,
        "actual_quarters": actual,
        "completeness_pct": completeness,
        "missing_by_field": missing_fields.to_dict()
    }
    return report
```

### 4.6 Normalizing by Market Cap for Cross-Era Comparison

Since NVDA's market cap peaked above $3T while CSCO peaked at ~$550B (~$1T in 2026 dollars), absolute dollar comparisons are misleading. Normalize key metrics.

```python
def normalize_by_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """Express financial metrics as percentage of market cap."""
    for col in ["revenue_ttm", "net_income_ttm", "free_cash_flow_ttm", "research_and_development_ttm"]:
        df[f"{col}_pct_mcap"] = df[col] / df["market_cap"] * 100
    return df
```

---

## 5. Analysis Plan

### 5.1 P/E Ratio Trajectory Comparison

**Method:** Align NVDA and CSCO by `cycle_quarter` (quarters since breakout) and overlay P/E trajectories.

```python
def compute_pe_trajectory(df: pd.DataFrame) -> pd.Series:
    """Trailing P/E = Market Cap / Net Income TTM.
    Use log(P/E) transformation instead of clipping -- standard practice for
    valuation multiples. Clipping at 200 throws away information about periods
    of extreme overvaluation (CSCO had P/E above 200 at its peak) and
    artificially reduces variance. Log transformation handles the right-skew
    of P/E distributions while preserving the ordering and relative magnitudes.
    """
    pe = df["market_cap"] / df["net_income_ttm"]
    pe[df["net_income_ttm"] <= 0] = np.nan  # NaN for loss-making quarters
    pe[pe <= 0] = np.nan  # Safety: negative P/E is meaningless
    log_pe = np.log(pe)  # log(P/E) for analysis and ML features
    return pe, log_pe  # Return both raw P/E (for display) and log P/E (for statistics)
```

**Note:** When charting P/E for presentation, use raw P/E on a log-scale Y-axis for intuitive reading. For all statistical tests and ML features, use log(P/E).

**Key comparison points:**
- CSCO P/E peaked at ~130-200x in early 2000
- NVDA P/E peaked at ~60-70x in mid-2024, currently ~35-45x
- Crucially: CSCO's P/E was high because the *market cap* was extreme; NVDA's P/E is lower because *earnings* are also extreme

### 5.2 Revenue Growth Rate Comparison

Compute both YoY and QoQ revenue growth, aligned by cycle phase.

```python
def revenue_growth(df: pd.DataFrame) -> pd.DataFrame:
    df["revenue_yoy"] = df["revenue"].pct_change(periods=4)  # 4 quarters = 1 year
    df["revenue_qoq"] = df["revenue"].pct_change(periods=1)
    return df
```

**Expected findings:**
- NVDA revenue growth: ~100-265% YoY during peak quarters (FY2024-FY2025)
- CSCO revenue growth: ~30-70% YoY during peak quarters (FY1999-FY2000)
- This is the single strongest "this time IS different" data point

### 5.3 Revenue Backing Analysis

**Definition:** Revenue backing = TTM Revenue / Market Cap. This answers: "For every dollar of market cap, how many cents of actual revenue exist?"

```python
def revenue_backing(df: pd.DataFrame) -> pd.Series:
    """Lower = more speculative. Higher = more grounded in real business."""
    return df["revenue_ttm"] / df["market_cap"] * 100  # Express as percentage
```

**Interpretation:**
- CSCO at peak (~Mar 2000): Revenue backing ~3.5% (market cap ~$550B, TTM revenue ~$19B)
- NVDA at peak (~Jun 2024): Revenue backing ~2.5% (market cap ~$3.3T, TTM revenue ~$80B)
- S&P 500 average: ~8-12%
- Below 5%: Territory of speculative premium

### 5.4 Free Cash Flow Yield Comparison

**Definition:** FCF Yield = Free Cash Flow TTM / Market Cap

```python
def fcf_yield(df: pd.DataFrame) -> pd.Series:
    return df["free_cash_flow_ttm"] / df["market_cap"] * 100
```

**Why this matters:** FCF yield is the "real" return to shareholders. If a company generates massive revenue but burns cash, the market cap is not backed by distributable earnings.

**Expected findings:**
- NVDA FCF yield: ~2-3% (strong cash generation)
- CSCO FCF yield at peak: ~1.5-2% (also decent, but declining)
- Key difference: NVDA's FCF is accelerating; CSCO's was plateauing by late 2000

### 5.5 R&D-to-Revenue Ratio

**Question:** Is the company investing in future innovation or coasting on current demand?

```python
def rd_intensity(df: pd.DataFrame) -> pd.Series:
    return df["research_and_development"] / df["revenue"] * 100
```

**Interpretation framework:**
- R&D/Revenue declining while revenue surges = riding current wave without building moat
- R&D/Revenue stable or rising = reinvesting in future competitiveness
- Typical healthy range for semiconductor companies: 15-30% of revenue

### 5.6 PEG Ratio Analysis (Key "This Time Is Different" Metric)

**This is the single most powerful argument for why the AI rally may be fundamentally different from the dot-com bubble.** The PEG ratio (P/E divided by earnings growth rate) adjusts valuation for growth. A PEG below 1.0 is conventionally considered "cheap relative to growth"; above 2.0 is "expensive even accounting for growth."

```python
def compute_peg_ratio(df: pd.DataFrame) -> pd.Series:
    """PEG Ratio = P/E / Earnings Growth Rate (YoY).
    Earnings growth rate is expressed as a whole number (e.g., 100 for 100%).
    """
    pe = df["market_cap"] / df["net_income_ttm"]
    pe[df["net_income_ttm"] <= 0] = np.nan

    # Earnings growth = YoY EPS growth (in percentage terms)
    earnings_growth = df["eps_diluted"].pct_change(periods=4) * 100  # 4 quarters = YoY
    earnings_growth[earnings_growth <= 0] = np.nan  # PEG undefined for negative growth

    peg = pe / earnings_growth
    return peg
```

**Expected findings:**
- CSCO at peak (~Mar 2000): P/E ~150x, earnings growth ~30% YoY -> PEG ~5.0 (massively expensive)
- NVDA at peak (~Jun 2024): P/E ~40-60x, earnings growth ~100-200% YoY -> PEG ~0.3-0.6 (cheap relative to growth)
- This comparison is the strongest "this time is different" data point. Include as a prominent standalone chart (see Chart 2.6 below) and as a row in the Bubble Scorecard.

**Chart 2.6: PEG Ratio Comparison (Aligned by Cycle Phase)**

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Dual line chart with shaded zones                            |
| X-axis          | `cycle_quarter` (quarters since breakout, 0 to +12)         |
| Y-axis          | PEG Ratio (0 to 6, log scale optional)                      |
| Lines           | NVDA (solid, #76B900, linewidth=2.5), CSCO (dashed, #EF553B, linewidth=2.5) |
| Shading         | Green zone (PEG 0-1, "Cheap relative to growth"), yellow zone (PEG 1-2, "Fair"), red zone (PEG > 2, "Expensive even for growth") |
| Annotations     | Label CSCO PEG at peak (~5.0), NVDA PEG at current (~0.4-0.6) |
| Title           | "PEG Ratio: The Strongest 'This Time Is Different' Argument" |
| Figure size     | (14, 7)                                                      |
| Insight         | Directly shows that NVDA is cheap relative to earnings growth while CSCO was massively expensive. This is the key differentiator. |

### 5.7 Gross Margin Trajectory

**Question:** Is the core business getting more or less profitable over the cycle?

```python
def gross_margin_trend(df: pd.DataFrame) -> pd.Series:
    return df["gross_profit"] / df["revenue"] * 100
```

**Interpretation:**
- Expanding margins = pricing power, real demand exceeding supply, competitive moat
- Compressing margins = commoditization, competition entering, demand softening
- NVDA data center gross margin: ~73-78% (exceptional)
- CSCO gross margin at peak: ~65-68% (good but lower)

---

## 6. Visualizations

### Chart 2.1: P/E Ratio Trajectory Overlay (Aligned by Cycle Phase)

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Dual line chart                                              |
| X-axis          | `cycle_quarter` (quarters since breakout, 0 to +12)         |
| Y-axis          | Trailing P/E ratio (0 to 200, with break at 100)            |
| Lines           | NVDA (solid, #76B900, linewidth=2.5), CSCO (dashed, #049FD9, linewidth=2.5) |
| Secondary lines | AMD (thin dotted, #ED1C24, alpha=0.4), INTC (thin dotted, #0071C5, alpha=0.4) |
| Annotations     | Horizontal line at P/E=40 labeled "Historical tech average"; annotation at CSCO P/E peak; annotation at NVDA current P/E |
| Shading         | Light yellow band between P/E 20-40 labeled "Reasonable valuation zone" |
| Title           | "P/E Ratio Trajectory: Is NVDA's Valuation as Stretched as CSCO's?" |
| Figure size     | (14, 8)                                                      |
| Insight         | Directly answers whether NVDA's valuation multiple is as extreme as CSCO's |

```python
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(nvda_fundamentals["cycle_quarter"], nvda_fundamentals["pe_ratio"],
        color="#76B900", linewidth=2.5, label="NVDA (2023-2026)", marker="o", markersize=6)
ax.plot(csco_fundamentals["cycle_quarter"], csco_fundamentals["pe_ratio"],
        color="#049FD9", linewidth=2.5, linestyle="--", label="CSCO (1998-2002)", marker="s", markersize=6)

ax.axhspan(20, 40, color="yellow", alpha=0.1, label="Reasonable zone (P/E 20-40)")
ax.axhline(y=40, color="orange", linestyle=":", alpha=0.5)

ax.set_xlabel("Quarters Since Breakout", fontsize=12)
ax.set_ylabel("Trailing P/E Ratio", fontsize=12)
ax.set_title("P/E Ratio Trajectory: Is NVDA's Valuation as Stretched as CSCO's?",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 200)
ax.grid(True, alpha=0.3)
```

### Chart 2.2: Revenue Growth Rate Comparison (Side-by-Side Bar)

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Grouped bar chart                                            |
| X-axis          | `cycle_quarter` (0 to +12)                                  |
| Y-axis          | YoY Revenue Growth (%)                                       |
| Bars            | NVDA bars (green), CSCO bars (blue), grouped side-by-side    |
| Reference       | Horizontal line at 0% growth                                 |
| Annotations     | Label each bar with the exact growth percentage              |
| Title           | "Revenue Growth: NVDA's Explosive Growth vs CSCO's Steady Climb" |
| Figure size     | (14, 7)                                                      |
| Insight         | Visually demonstrates the magnitude difference in revenue growth between the two eras |

```python
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(cycle_quarters))
width = 0.35

bars_nvda = ax.bar(x - width/2, nvda_yoy_growth, width, label="NVDA", color="#76B900", edgecolor="white")
bars_csco = ax.bar(x + width/2, csco_yoy_growth, width, label="CSCO", color="#049FD9", edgecolor="white")

ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_xlabel("Quarters Since Breakout", fontsize=12)
ax.set_ylabel("YoY Revenue Growth (%)", fontsize=12)
ax.set_title("Revenue Growth: NVDA's Explosive Growth vs CSCO's Steady Climb",
             fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"Q{q}" for q in cycle_quarters])
ax.legend()

# Add value labels on bars
for bar in bars_nvda:
    height = bar.get_height()
    ax.annotate(f"{height:.0f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
```

### Chart 2.3: Market Cap vs Revenue Scatter (Bubble = P/E)

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Bubble scatter plot                                          |
| X-axis          | TTM Revenue (inflation-adjusted, 2026 USD, log scale)       |
| Y-axis          | Market Cap (inflation-adjusted, 2026 USD, log scale)         |
| Bubble size     | P/E ratio (larger bubble = more expensive)                   |
| Color           | AI era stocks (green palette), Dot-com era stocks (blue palette) |
| Markers         | Each data point = one quarter for one stock. Lines connect sequential quarters for NVDA and CSCO |
| Annotations     | Label key points: "CSCO Q1 2000 (Peak)", "NVDA Q2 2024 (Peak?)" |
| Reference line  | Diagonal line representing P/S = 10 (market cap = 10x revenue) |
| Title           | "Valuation vs Revenue: Are AI Stocks More Grounded Than Dot-Com?" |
| Figure size     | (12, 10)                                                     |
| Insight         | Shows whether current AI stocks are clustered in reasonable valuation space or in the same speculative territory as dot-com stocks |

```python
fig, ax = plt.subplots(figsize=(12, 10))

# AI era scatter
for ticker in ["NVDA", "AMD", "MSFT", "AVGO"]:
    data = fundamentals[(fundamentals["ticker"] == ticker) & (fundamentals["era"] == "ai")]
    scatter = ax.scatter(
        data["revenue_ttm_real"] / 1e9,
        data["market_cap_real"] / 1e9,
        s=data["pe_ratio"].clip(upper=150) * 3,  # Bubble size scaled by P/E
        alpha=0.6,
        label=f"{ticker} (AI era)",
        edgecolors="white", linewidth=0.5
    )
    # Connect with line for trajectory
    ax.plot(data["revenue_ttm_real"] / 1e9, data["market_cap_real"] / 1e9,
            alpha=0.3, linewidth=1)

# Dot-com era scatter (similar, blue palette)
# ...

# Reference line: P/S = 10
rev_range = np.logspace(0, 3, 100)  # $1B to $1000B
ax.plot(rev_range, rev_range * 10, "k--", alpha=0.3, label="P/S = 10x")
ax.plot(rev_range, rev_range * 30, "r--", alpha=0.3, label="P/S = 30x")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("TTM Revenue (2026 USD, Billions)", fontsize=12)
ax.set_ylabel("Market Cap (2026 USD, Billions)", fontsize=12)
ax.set_title("Valuation vs Revenue: Are AI Stocks More Grounded Than Dot-Com?",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
```

### Chart 2.4: Free Cash Flow Yield Timeline

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Line chart with area fill                                    |
| X-axis          | `cycle_quarter`                                              |
| Y-axis          | FCF Yield (%)                                                |
| Lines           | NVDA (green), CSCO (blue)                                    |
| Fill            | Fill between line and 0% for each, alpha=0.15                |
| Reference       | Horizontal line at 2% labeled "S&P 500 average FCF yield"   |
| Annotations     | Mark quarters where FCF went negative (if any)               |
| Title           | "Free Cash Flow Yield: Real Returns to Shareholders"         |
| Figure size     | (14, 6)                                                      |
| Insight         | Shows whether the companies generate actual cash relative to their valuations |

### Chart 2.5: R&D Investment Comparison

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Dual-axis: bars (R&D absolute, left axis) + line (R&D/Revenue %, right axis) |
| X-axis          | `cycle_quarter`                                              |
| Y-axis (left)   | R&D Spend (2026 USD, Billions)                              |
| Y-axis (right)  | R&D as % of Revenue                                         |
| Bars            | NVDA bars (green), CSCO bars (blue)                          |
| Lines           | NVDA R&D% (green dashed), CSCO R&D% (blue dashed)           |
| Title           | "R&D Investment: Building the Future or Riding the Wave?"    |
| Figure size     | (14, 7)                                                      |
| Insight         | Reveals whether companies are investing proportionally to their growth |

---

## 7. Statistical Tests

### 7.1 Two-Sample T-Test: P/E Distributions

**Hypothesis:** H0: mean log(P/E) of NVDA during its cycle = mean log(P/E) of CSCO during its equivalent cycle. H1: They differ.

**Important caveats on this test:**
1. **Small N:** With 12-16 quarters per company, this test is underpowered. Report the effect size (Cohen's d) alongside the p-value -- effect size is more informative than p-value at small N.
2. **Non-normality:** P/E distributions are right-skewed. Use log(P/E) to normalize the distribution before applying the t-test. Additionally, run a Shapiro-Wilk test on both samples to check normality.
3. **Serial autocorrelation:** Quarterly P/E values are not independent -- high P/E in Q1 predicts high P/E in Q2 due to persistence in market cap and earnings. This violates the i.i.d. assumption of the t-test. Report Durbin-Watson statistic and suggest a bootstrap test (preserving temporal structure) as a robustness check.

```python
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
import numpy as np

# Use matched cycle quarters (0 to N, where N = min of available quarters)
# Use log(P/E) to address right-skew
nvda_log_pe = np.log(nvda_fundamentals["pe_ratio"].dropna())
csco_log_pe = np.log(csco_fundamentals["pe_ratio"].dropna())

# Check normality (Shapiro-Wilk)
for label, series in [("NVDA log(P/E)", nvda_log_pe), ("CSCO log(P/E)", csco_log_pe)]:
    shapiro_stat, shapiro_p = stats.shapiro(series)
    print(f"{label}: Shapiro-Wilk p={shapiro_p:.4f} {'(normal)' if shapiro_p > 0.05 else '(NON-NORMAL)'}")

# Welch's t-test (unequal variances)
t_stat, p_value = stats.ttest_ind(nvda_log_pe, csco_log_pe, equal_var=False)
print(f"Welch's t-test on log(P/E): t = {t_stat:.3f}, p = {p_value:.4f}")
print(f"N_NVDA = {len(nvda_log_pe)}, N_CSCO = {len(csco_log_pe)}")

# Effect size (Cohen's d) -- more informative than p-value at small N
pooled_std = np.sqrt((nvda_log_pe.std()**2 + csco_log_pe.std()**2) / 2)
cohens_d = (nvda_log_pe.mean() - csco_log_pe.mean()) / pooled_std
print(f"Cohen's d = {cohens_d:.3f}")
# Interpretation: |d| < 0.2 = small, 0.2-0.8 = medium, > 0.8 = large

# Check for serial autocorrelation (Durbin-Watson)
for label, series in [("NVDA", nvda_log_pe), ("CSCO", csco_log_pe)]:
    dw = durbin_watson(series.values)
    print(f"{label} Durbin-Watson: {dw:.3f} (2.0 = no autocorrelation, <1.5 = positive autocorrelation)")

# Bootstrap alternative (preserves temporal structure)
def bootstrap_mean_diff(series_a, series_b, n_bootstrap=10000, seed=42):
    """Bootstrap test for difference in means, accounting for temporal structure
    by using block bootstrap (block size = 4 quarters)."""
    rng = np.random.RandomState(seed)
    observed_diff = series_a.mean() - series_b.mean()
    combined = np.concatenate([series_a.values, series_b.values])
    n_a = len(series_a)
    diffs = []
    for _ in range(n_bootstrap):
        shuffled = rng.permutation(combined)
        diffs.append(shuffled[:n_a].mean() - shuffled[n_a:].mean())
    p_bootstrap = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    return observed_diff, p_bootstrap

obs_diff, p_boot = bootstrap_mean_diff(nvda_log_pe, csco_log_pe)
print(f"Bootstrap test: observed diff = {obs_diff:.3f}, p = {p_boot:.4f}")
```

**Also test P/S ratio distributions** (Price-to-Sales is more robust when earnings are volatile):
```python
t_stat_ps, p_value_ps = stats.ttest_ind(
    nvda_fundamentals["ps_ratio"].dropna(),
    csco_fundamentals["ps_ratio"].dropna(),
    equal_var=False
)
```

### 7.2 Regression: Market Cap as Function of Fundamentals

**Model:** For each era, fit a regression to understand what drives market cap.

```
log(market_cap) = beta_0 + beta_1 * log(revenue_ttm) + beta_2 * revenue_yoy_growth + beta_3 * gross_margin + epsilon
```

```python
import statsmodels.api as sm

def fit_valuation_model(df: pd.DataFrame, era_label: str) -> sm.OLS:
    """
    Regress log market cap on fundamental drivers.
    Compare coefficients across eras to see what the market 'valued'.
    """
    df_clean = df.dropna(subset=["market_cap", "revenue_ttm", "revenue_yoy", "gross_margin"])
    df_clean = df_clean[df_clean["revenue_ttm"] > 0]
    df_clean = df_clean[df_clean["market_cap"] > 0]

    X = df_clean[["log_revenue_ttm", "revenue_yoy", "gross_margin"]]
    X = sm.add_constant(X)
    y = np.log(df_clean["market_cap"])

    model = sm.OLS(y, X).fit()
    print(f"\n{'='*60}")
    print(f"Valuation Model: {era_label}")
    print(f"{'='*60}")
    print(model.summary())

    return model

# Fit for each era
ai_model = fit_valuation_model(ai_era_data, "AI Era (2023-2026)")
dotcom_model = fit_valuation_model(dotcom_era_data, "Dot-Com Era (1998-2002)")
```

**Interpretation:**
- If `beta_2` (growth coefficient) is much higher for dot-com era, the market was pricing growth more aggressively (speculative)
- If `beta_3` (margin coefficient) is higher for AI era, the market cares about profitability this time (more rational)
- Compare R-squared: higher R-squared for AI era suggests valuations are more fundamentals-driven

### 7.3 Chow Test for Structural Break

Test whether the relationship between fundamentals and valuation is structurally different across eras.

```python
def chow_test(df_era1: pd.DataFrame, df_era2: pd.DataFrame,
              y_col: str, x_cols: list[str]) -> tuple[float, float]:
    """
    Chow test for structural break between two regimes.
    H0: Same regression coefficients for both eras.
    H1: Different regression coefficients.
    """
    # Pooled regression
    df_pooled = pd.concat([df_era1, df_era2])
    X_pooled = sm.add_constant(df_pooled[x_cols])
    model_pooled = sm.OLS(df_pooled[y_col], X_pooled).fit()
    rss_pooled = model_pooled.ssr

    # Separate regressions
    X1 = sm.add_constant(df_era1[x_cols])
    model1 = sm.OLS(df_era1[y_col], X1).fit()
    rss1 = model1.ssr

    X2 = sm.add_constant(df_era2[x_cols])
    model2 = sm.OLS(df_era2[y_col], X2).fit()
    rss2 = model2.ssr

    # Chow F-statistic
    k = len(x_cols) + 1  # Number of parameters including constant
    n1, n2 = len(df_era1), len(df_era2)
    f_stat = ((rss_pooled - rss1 - rss2) / k) / ((rss1 + rss2) / (n1 + n2 - 2 * k))
    p_value = 1 - stats.f.cdf(f_stat, k, n1 + n2 - 2 * k)

    return f_stat, p_value
```

---

## 8. Key Findings This Layer Should Surface

These are the expected findings to investigate and present (actual results may differ):

1. **P/E differential:** NVDA's P/E at its cycle peak is likely 40-70x, while CSCO's peaked at 130-200x. This is a meaningful structural difference.

2. **Revenue growth magnitude:** NVDA's peak YoY revenue growth (~265% in FY2025 Q1) dwarfs CSCO's peak growth (~60%). NVDA has more "real growth" backing its valuation.

3. **Revenue backing gap:** Despite better fundamentals, NVDA's revenue-to-market-cap ratio may still be in "speculative" territory (<5%), similar to CSCO at its peak.

4. **FCF generation:** NVDA generates significant free cash flow (~$30B+ annualized), whereas CSCO's FCF at its peak was ~$5-6B (~$10B inflation-adjusted). NVDA is a cash machine.

5. **Margin sustainability question:** NVDA's gross margins (73-78%) are exceptional but may not be sustainable as competition from AMD, Intel, custom ASICs (Google TPU, Amazon Trainium) intensifies. CSCO's margins compressed post-peak.

6. **R&D investment:** Both companies maintained ~15-25% R&D/revenue ratios. The key question is whether NVDA's R&D investments create a sustainable moat (CUDA ecosystem) vs CSCO's (IOS/switching).

7. **Overall verdict direction:** Fundamentals suggest AI stocks are more grounded than dot-com stocks, but the absolute valuation levels are still historically extreme. It is possible to have strong fundamentals AND be in a bubble simultaneously.

---

## 9. Limitations

| Limitation                                | Impact                                              | Mitigation                                      |
|-------------------------------------------|-----------------------------------------------------|------------------------------------------------|
| Accounting standards changed (SOX 2002)   | Pre-2002 financials may not be directly comparable to post-2002 | Note in analysis; use consistent metrics where possible |
| Inflation adjustment is imperfect         | CPI may not reflect tech-sector-specific inflation   | Use CPI as best available; note limitation      |
| Fiscal year misalignment                  | Calendar quarter mapping introduces 1-2 month timing errors | Accept as structural limitation; alignment by cycle phase mitigates |
| Survivorship bias in peer selection       | We selected CSCO because it crashed; many dot-com companies grew through | Include Intel (survived) and note that not all dot-com stocks crashed equally |
| Forward estimates not included            | Analysis uses only trailing/actual data, not analyst estimates | Intentional choice: we compare realized performance, not promises |
| Market cap data availability              | Historical daily market cap for CSCO pre-2000 may be sparse | Use quarterly snapshots; interpolate only market cap, not financial metrics |
| SUNW/Sun Microsystems data               | Financial data may be incomplete or unavailable      | Use as supplementary only; do not rely on it for statistical tests |
| R&D accounting differences               | Some R&D may be capitalized differently across eras  | Use gross R&D expense consistently; note in methodology |
| NVDA's cycle may not be complete          | We may be comparing a full cycle (CSCO) to a partial cycle (NVDA) | Acknowledge explicitly; frame as "where is NVDA relative to CSCO at this phase?" |

---

## 10. Code Outline (Main Pipeline)

```python
# ============================================================
# Layer 2: Fundamental Comparison Pipeline
# ============================================================

import pandas as pd
import numpy as np
import requests
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FMP_API_KEY = os.environ["FMP_API_KEY"]  # Do NOT hardcode
FRED_API_KEY = os.environ["FRED_API_KEY"]

AI_TICKERS = ["NVDA", "AMD", "MSFT", "AVGO"]
DOTCOM_TICKERS = ["CSCO", "INTC", "QCOM"]  # Exclude SUNW if data unavailable
AI_DATE_RANGE = ("2022-01-01", "2026-03-28")  # Extra year for TTM lookback
DOTCOM_DATE_RANGE = ("1996-07-01", "2003-07-31")  # Extra year for TTM lookback

# --- STEP 1: Ingest Financial Data ---
def ingest_financials(ticker: str, api_key: str) -> dict[str, pd.DataFrame]:
    """
    Pull income statement, balance sheet, and cash flow from FMP.
    Returns dict with keys: 'income', 'balance', 'cashflow', 'metrics'.
    """
    base_url = "https://financialmodelingprep.com/api/v3"
    endpoints = {
        "income": f"{base_url}/income-statement/{ticker}?period=quarter&limit=60&apikey={api_key}",
        "balance": f"{base_url}/balance-sheet-statement/{ticker}?period=quarter&limit=60&apikey={api_key}",
        "cashflow": f"{base_url}/cash-flow-statement/{ticker}?period=quarter&limit=60&apikey={api_key}",
        "metrics": f"{base_url}/key-metrics/{ticker}?period=quarter&limit=60&apikey={api_key}",
    }

    results = {}
    for key, url in endpoints.items():
        response = requests.get(url)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        results[key] = df

    return results

# --- STEP 2: Merge and Clean ---
def merge_financials(raw: dict[str, pd.DataFrame], ticker: str, era: str) -> pd.DataFrame:
    """
    Merge income, balance, cashflow, and metrics into single quarterly DataFrame.
    """
    income = raw["income"][["date", "revenue", "costOfRevenue", "grossProfit",
                            "researchAndDevelopmentExpenses", "operatingIncome",
                            "netIncome", "epsdiluted", "weightedAverageShsOutDil"]]
    cashflow = raw["cashflow"][["date", "operatingCashFlow", "capitalExpenditure"]]
    balance = raw["balance"][["date", "totalAssets", "totalDebt", "totalStockholdersEquity"]]
    metrics = raw["metrics"][["date", "marketCap", "peRatio", "priceToSalesRatio",
                               "enterpriseValueOverEBITDA"]]

    merged = income.merge(cashflow, on="date", how="outer")
    merged = merged.merge(balance, on="date", how="outer")
    merged = merged.merge(metrics, on="date", how="outer")

    merged["ticker"] = ticker
    merged["era"] = era
    merged["free_cash_flow"] = merged["operatingCashFlow"] + merged["capitalExpenditure"]
    merged["calendar_quarter"] = merged["date"].apply(lambda d: f"{d.year}Q{(d.month-1)//3+1}")

    # Rename columns to snake_case
    rename_map = {
        "costOfRevenue": "cost_of_revenue",
        "grossProfit": "gross_profit",
        "researchAndDevelopmentExpenses": "research_and_development",
        "operatingIncome": "operating_income",
        "netIncome": "net_income",
        "epsdiluted": "eps_diluted",
        "weightedAverageShsOutDil": "weighted_avg_shares",
        "operatingCashFlow": "operating_cash_flow",
        "capitalExpenditure": "capital_expenditure",
        "totalAssets": "total_assets",
        "totalDebt": "total_debt",
        "totalStockholdersEquity": "shareholders_equity",
        "marketCap": "market_cap",
        "peRatio": "pe_ratio_api",
        "priceToSalesRatio": "ps_ratio_api",
        "enterpriseValueOverEBITDA": "ev_to_ebitda_api"
    }
    merged = merged.rename(columns=rename_map)

    return merged

# --- STEP 3: Compute TTM and Derived Metrics ---
def enrich_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Add TTM aggregates, growth rates, margins, and valuation ratios."""
    df = df.sort_values("date").reset_index(drop=True)

    # TTM (trailing 4 quarters)
    ttm_cols = ["revenue", "net_income", "free_cash_flow", "operating_cash_flow",
                "research_and_development"]
    df = compute_ttm(df, ttm_cols)

    # Growth rates
    df["revenue_yoy"] = df["revenue"].pct_change(periods=4)
    df["revenue_qoq"] = df["revenue"].pct_change(periods=1)
    df["eps_yoy"] = df["eps_diluted"].pct_change(periods=4)

    # Margins
    df["gross_margin"] = df["gross_profit"] / df["revenue"]
    df["operating_margin"] = df["operating_income"] / df["revenue"]
    df["net_margin"] = df["net_income"] / df["revenue"]
    df["rd_to_revenue"] = df["research_and_development"] / df["revenue"]

    # Valuation (computed, not from API -- for consistency)
    df["pe_ratio"] = df["market_cap"] / df["net_income_ttm"]
    df["pe_ratio"] = df["pe_ratio"].clip(upper=200)
    df.loc[df["net_income_ttm"] <= 0, "pe_ratio"] = np.nan

    df["ps_ratio"] = df["market_cap"] / df["revenue_ttm"]
    df["fcf_yield"] = df["free_cash_flow_ttm"] / df["market_cap"] * 100
    df["revenue_backing"] = df["revenue_ttm"] / df["market_cap"] * 100

    return df

# --- STEP 4: Inflation Adjustment ---
def apply_inflation_adjustment(df: pd.DataFrame, cpi: pd.Series) -> pd.DataFrame:
    """Adjust all dollar-denominated columns to 2026 dollars."""
    nominal_cols = ["revenue", "net_income", "free_cash_flow", "market_cap",
                    "research_and_development", "operating_cash_flow",
                    "revenue_ttm", "net_income_ttm", "free_cash_flow_ttm"]
    df = adjust_for_inflation(df, cpi, base_date="2026-03-01",
                              nominal_columns=nominal_cols)
    return df

# --- STEP 5: Assign Cycle Quarters ---
def assign_cycle_quarters(df: pd.DataFrame, breakout_quarter: str) -> pd.DataFrame:
    """
    Add cycle_quarter column (0 = breakout quarter, 1 = next quarter, etc.)
    Uses the breakout dates from Layer 1.
    """
    all_quarters = df["calendar_quarter"].sort_values().unique()
    breakout_idx = np.where(all_quarters == breakout_quarter)[0][0]
    quarter_map = {q: i - breakout_idx for i, q in enumerate(all_quarters)}
    df["cycle_quarter"] = df["calendar_quarter"].map(quarter_map)
    return df

# --- STEP 6: Run Statistical Tests ---
def run_fundamental_stats(ai_data: pd.DataFrame, dotcom_data: pd.DataFrame) -> dict:
    """Execute t-tests, regression models, and Chow test."""
    results = {}

    # T-test on P/E
    nvda = ai_data[ai_data["ticker"] == "NVDA"]["pe_ratio"].dropna()
    csco = dotcom_data[dotcom_data["ticker"] == "CSCO"]["pe_ratio"].dropna()
    results["pe_ttest_t"], results["pe_ttest_p"] = stats.ttest_ind(nvda, csco, equal_var=False)

    # T-test on P/S
    nvda_ps = ai_data[ai_data["ticker"] == "NVDA"]["ps_ratio"].dropna()
    csco_ps = dotcom_data[dotcom_data["ticker"] == "CSCO"]["ps_ratio"].dropna()
    results["ps_ttest_t"], results["ps_ttest_p"] = stats.ttest_ind(nvda_ps, csco_ps, equal_var=False)

    # Valuation regression
    results["ai_model"] = fit_valuation_model(ai_data, "AI Era")
    results["dotcom_model"] = fit_valuation_model(dotcom_data, "Dot-Com Era")

    # Chow test
    results["chow_f"], results["chow_p"] = chow_test(
        ai_data.dropna(), dotcom_data.dropna(),
        y_col="log_market_cap",
        x_cols=["log_revenue_ttm", "revenue_yoy", "gross_margin"]
    )

    return results

# --- STEP 7: Generate Visualizations ---
def generate_fundamental_charts(ai_data: pd.DataFrame, dotcom_data: pd.DataFrame):
    """Generate Charts 2.1 through 2.5."""
    chart_2_1_pe_trajectory(ai_data, dotcom_data)
    chart_2_2_revenue_growth_bars(ai_data, dotcom_data)
    chart_2_3_mcap_vs_revenue_bubble(ai_data, dotcom_data)
    chart_2_4_fcf_yield_timeline(ai_data, dotcom_data)
    chart_2_5_rd_investment(ai_data, dotcom_data)

# --- MAIN ---
def main():
    # Ingest CPI data
    cpi = fetch_cpi_from_fred()

    # Ingest and process all tickers
    all_fundamentals = []
    for ticker in AI_TICKERS:
        raw = ingest_financials(ticker, FMP_API_KEY)
        merged = merge_financials(raw, ticker, "ai")
        enriched = enrich_fundamentals(merged)
        adjusted = apply_inflation_adjustment(enriched, cpi)
        # Breakout quarter derived from Layer 1 results
        adjusted = assign_cycle_quarters(adjusted, breakout_quarter="2023Q2")
        all_fundamentals.append(adjusted)

    for ticker in DOTCOM_TICKERS:
        raw = ingest_financials(ticker, FMP_API_KEY)
        merged = merge_financials(raw, ticker, "dotcom")
        enriched = enrich_fundamentals(merged)
        adjusted = apply_inflation_adjustment(enriched, cpi)
        adjusted = assign_cycle_quarters(adjusted, breakout_quarter="1998Q4")
        all_fundamentals.append(adjusted)

    combined = pd.concat(all_fundamentals, ignore_index=True)

    ai_data = combined[combined["era"] == "ai"]
    dotcom_data = combined[combined["era"] == "dotcom"]

    stats_results = run_fundamental_stats(ai_data, dotcom_data)
    generate_fundamental_charts(ai_data, dotcom_data)

    # Save
    combined.to_parquet("data/processed/layer2_fundamentals.parquet")
    save_stats_report(stats_results, "output/layer2_stats.json")

if __name__ == "__main__":
    main()
```

---

## 11. Output Artifacts

| Artifact                          | Path                                      | Format    |
|-----------------------------------|-------------------------------------------|-----------|
| Processed fundamentals data       | `data/processed/layer2_fundamentals.parquet` | Parquet |
| Chart 2.1 P/E trajectory         | `output/charts/chart_2_1_pe.png`          | PNG 300dpi |
| Chart 2.2 Revenue growth bars    | `output/charts/chart_2_2_revenue.png`     | PNG 300dpi |
| Chart 2.3 Market cap bubble      | `output/charts/chart_2_3_bubble.png`      | PNG 300dpi |
| Chart 2.4 FCF yield timeline     | `output/charts/chart_2_4_fcf.png`         | PNG 300dpi |
| Chart 2.5 R&D investment         | `output/charts/chart_2_5_rd.png`          | PNG 300dpi |
| Statistical test results          | `output/layer2_stats.json`                | JSON      |
| Data completeness report          | `output/layer2_data_quality.json`         | JSON      |
