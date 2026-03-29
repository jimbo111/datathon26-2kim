# Layer 4: Macro Environment Comparison

> **Status:** Spec Complete | **Owner:** Team 2Kim
> **Scoring targets:** Analysis & Evidence (25pts), Data Quality (20pts), Technical Rigor (15pts)
> **Upstream deps:** Layer 1 (price data for peak alignment) | **Downstream:** Layer 7 (ML features), Layer 8 (presentation narrative)

---

## 1. Objective

Determine whether the macroeconomic conditions surrounding the AI boom (2021-2026) structurally resemble or diverge from those of the dot-com bubble (1996-2003). The core question: **Is the macro regime enabling speculation, or is it a fundamentally different liquidity/rate environment?**

This layer provides the "backdrop" for the bubble comparison. Even if price and valuation metrics look similar (Layers 1-3), different macro conditions could mean entirely different outcomes. The dot-com bust happened while the Fed was tightening into an overheating economy. The AI boom started in a post-COVID zero-rate world with unprecedented M2 expansion -- and then the Fed embarked on the fastest hiking cycle in 40 years.

### Sub-questions this layer answers:
1. How do interest rate environments compare at equivalent points in each cycle?
2. Was/is money "free" or "expensive" (real interest rates)?
3. Did/does the yield curve warn of recession before the market peaked?
4. How do credit conditions (risk appetite) compare?
5. Is M2 growth (liquidity) fueling speculation?
6. Is the economy overheating or cooling in each era?

---

## 2. Data Sources -- FRED API

All data is sourced from the Federal Reserve Economic Data (FRED) API. Base URL: `https://api.stlouisfed.org/fred/series/observations`.

### 2.1 Series Inventory

| # | Indicator | FRED Series ID | Frequency | Unit | Seasonally Adj? | Notes |
|---|-----------|---------------|-----------|------|-----------------|-------|
| 1 | Federal Funds Effective Rate | `FEDFUNDS` | Monthly | Percent | N/A | Target rate proxy; the actual effective rate |
| 2 | 10-Year Treasury Yield | `DGS10` | Daily | Percent | N/A | Constant maturity; used for yield curve |
| 3 | 2-Year Treasury Yield | `DGS2` | Daily | Percent | N/A | Constant maturity; short end of curve |
| 4 | Yield Curve Spread (10Y-2Y) | `T10Y2Y` | Daily | Percent | N/A | Pre-computed by FRED; 10Y minus 2Y |
| 5 | M2 Money Supply | `M2SL` | Monthly | Billions USD | Seasonally Adj | Broad money supply |
| 6 | CPI (All Urban Consumers) | `CPIAUCSL` | Monthly | Index (1982-84=100) | Seasonally Adj | Used to compute inflation rate |
| 7 | Corporate Credit Spread | `BAA10Y` | Daily | Percent | N/A | Moody's Baa yield minus 10Y Treasury; measures credit risk premium |
| 8 | GDP (Nominal) | `GDP` | Quarterly | Billions USD | Seasonally Adj Annual Rate | Headline GDP |
| 9 | Unemployment Rate | `UNRATE` | Monthly | Percent | Seasonally Adj | Civilian unemployment rate |
| 10 | S&P 500 Index | `SP500` | Daily | Index | N/A | Context only; primary price data in Layer 1 |

### 2.2 Date Ranges

| Era | Label | FRED `observation_start` | FRED `observation_end` | Rationale |
|-----|-------|--------------------------|------------------------|-----------|
| Dot-com | `dotcom` | `1996-01-01` | `2003-12-31` | Captures full run-up (1996), peak (Mar 2000), crash, and recovery start |
| AI | `ai` | `2021-01-01` | `2026-03-28` | Captures post-COVID recovery, rate hikes, AI boom, and current state |

### 2.3 FRED API Parameters

```
# Common parameters for every FRED API call
params = {
    "api_key": FRED_API_KEY,        # from .env
    "file_type": "json",             # always JSON
    "observation_start": "...",      # per era
    "observation_end": "...",        # per era
    "series_id": "...",              # per indicator
    "units": "lin",                  # linear (default); "pch" for percent change
    "frequency": None,               # None = native frequency; override for alignment
    "aggregation_method": "avg",     # when downsampling: avg, sum, eop
}
```

**Rate limiting:** FRED allows 120 requests per minute. With 10 series x 2 eras = 20 calls, no throttling needed. Cache all responses to `data/raw/fred/` as `{series_id}_{era}.json`.

---

## 3. Data Schema

### 3.1 Raw Data (per series, per era)

Each FRED API response returns:

```json
{
  "observations": [
    {"date": "1996-01-01", "value": "5.33"},
    {"date": "1996-02-01", "value": "5.22"},
    ...
  ]
}
```

**Important:** FRED returns `value` as a string. The string `"."` means missing data. Must handle this explicitly.

### 3.2 Parsed DataFrame Schema (per series)

| Column | dtype | Description |
|--------|-------|-------------|
| `date` | `datetime64[ns]` | Observation date |
| `value` | `float64` | Numeric value (NaN for missing "." entries) |
| `series_id` | `str` (categorical) | FRED series ID |
| `era` | `str` (categorical) | `"dotcom"` or `"ai"` |

### 3.3 Unified Macro DataFrame (after alignment)

| Column | dtype | Description |
|--------|-------|-------------|
| `date` | `datetime64[ns]` | Observation date (monthly, last business day) |
| `era` | `str` | `"dotcom"` or `"ai"` |
| `months_from_peak` | `int` | Signed integer: negative = before market peak, positive = after |
| `fed_funds_rate` | `float64` | FEDFUNDS value (%) |
| `treasury_10y` | `float64` | DGS10 monthly average (%) |
| `treasury_2y` | `float64` | DGS2 monthly average (%) |
| `yield_curve_spread` | `float64` | T10Y2Y monthly average (%) |
| `m2_supply` | `float64` | M2SL value (billions USD) |
| `m2_yoy_growth` | `float64` | M2 year-over-year percent change |
| `cpi_index` | `float64` | CPIAUCSL index value |
| `cpi_yoy_inflation` | `float64` | CPI year-over-year percent change (inflation rate) |
| `real_interest_rate` | `float64` | fed_funds_rate - cpi_yoy_inflation |
| `credit_spread` | `float64` | BAA10Y monthly average (%) |
| `gdp` | `float64` | GDP value (billions USD, forward-filled to monthly) |
| `gdp_yoy_growth` | `float64` | GDP year-over-year percent change |
| `unemployment` | `float64` | UNRATE (%) |
| `sp500` | `float64` | SP500 monthly average (index level) |

---

## 4. Preprocessing Pipeline

### 4.1 Step 1: Raw Fetch and Parse

```python
# pseudocode: fetch_fred_series(series_id, era)
import requests
import pandas as pd

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
ERAS = {
    "dotcom": ("1996-01-01", "2003-12-31"),
    "ai":     ("2021-01-01", "2026-03-28"),
}
SERIES_IDS = [
    "FEDFUNDS", "DGS10", "DGS2", "T10Y2Y", "M2SL",
    "CPIAUCSL", "BAA10Y", "GDP", "UNRATE", "SP500"
]

def fetch_fred(series_id: str, era: str, api_key: str) -> pd.DataFrame:
    start, end = ERAS[era]
    resp = requests.get(FRED_BASE, params={
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    })
    resp.raise_for_status()
    obs = resp.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." -> NaN
    df["series_id"] = series_id
    df["era"] = era
    return df

# Fetch all
raw_frames = {}
for sid in SERIES_IDS:
    for era in ERAS:
        key = f"{sid}_{era}"
        raw_frames[key] = fetch_fred(sid, era, FRED_API_KEY)
        # Cache to disk
        raw_frames[key].to_parquet(f"data/raw/fred/{key}.parquet")
```

### 4.2 Step 2: Frequency Alignment

The core challenge: data arrives at daily, monthly, and quarterly frequencies. We align everything to **monthly** as the common denominator.

| Source Frequency | Series | Alignment Method |
|-----------------|--------|-----------------|
| Daily | DGS10, DGS2, T10Y2Y, BAA10Y, SP500 | Resample to monthly using **mean** (average of all trading days in the month) |
| Monthly | FEDFUNDS, M2SL, CPIAUCSL, UNRATE | Already monthly; no resampling needed |
| Quarterly | GDP | Forward-fill to monthly (Q1 value fills Jan, Feb, Mar) |

```python
def align_to_monthly(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Resample a single-series DataFrame to monthly frequency."""
    df = df.set_index("date").sort_index()

    DAILY_SERIES = {"DGS10", "DGS2", "T10Y2Y", "BAA10Y", "SP500"}
    QUARTERLY_SERIES = {"GDP"}

    if series_id in DAILY_SERIES:
        # Resample daily -> monthly mean
        df_monthly = df["value"].resample("ME").mean().to_frame()
    elif series_id in QUARTERLY_SERIES:
        # Forward-fill quarterly -> monthly
        df_monthly = df["value"].resample("ME").ffill().to_frame()
    else:
        # Already monthly; ensure end-of-month alignment
        df_monthly = df["value"].resample("ME").last().to_frame()

    df_monthly = df_monthly.reset_index()
    df_monthly.columns = ["date", "value"]
    return df_monthly
```

### 4.3 Step 3: Derived Metrics

```python
def compute_derived_metrics(unified: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns to the unified monthly DataFrame."""
    # Group by era for rolling calculations
    for era in ["dotcom", "ai"]:
        mask = unified["era"] == era

        # M2 year-over-year growth rate
        # M2SL is level; compute pct change over 12 months
        unified.loc[mask, "m2_yoy_growth"] = (
            unified.loc[mask, "m2_supply"].pct_change(periods=12) * 100
        )

        # CPI year-over-year inflation rate
        unified.loc[mask, "cpi_yoy_inflation"] = (
            unified.loc[mask, "cpi_index"].pct_change(periods=12) * 100
        )

        # Real interest rate = nominal fed funds - inflation
        unified.loc[mask, "real_interest_rate"] = (
            unified.loc[mask, "fed_funds_rate"] - unified.loc[mask, "cpi_yoy_inflation"]
        )

        # GDP year-over-year growth rate
        unified.loc[mask, "gdp_yoy_growth"] = (
            unified.loc[mask, "gdp"].pct_change(periods=12) * 100
        )

    return unified
```

### 4.4 Step 4: Peak Alignment (Critical)

To overlay both eras on the same x-axis, we compute `months_from_peak` -- the number of months before (-) or after (+) the market peak in each era.

**Peak dates (from Layer 1 price analysis):**
- Dot-com peak: **March 2000** (S&P 500 intraday high: March 24, 2000)
- AI era peak: **TBD from Layer 1** -- if NVDA/S&P has not yet peaked, use latest date as provisional "current"

```python
PEAK_DATES = {
    "dotcom": pd.Timestamp("2000-03-31"),
    "ai":     pd.Timestamp("2025-01-31"),  # Provisional; update from Layer 1
}

def add_months_from_peak(df: pd.DataFrame) -> pd.DataFrame:
    """Add column: signed integer months from era peak."""
    for era, peak in PEAK_DATES.items():
        mask = df["era"] == era
        df.loc[mask, "months_from_peak"] = (
            (df.loc[mask, "date"].dt.year - peak.year) * 12
            + (df.loc[mask, "date"].dt.month - peak.month)
        )
    df["months_from_peak"] = df["months_from_peak"].astype(int)
    return df
```

### 4.5 Step 5: Normalization for Cross-Era Comparison

Some metrics need normalization because absolute levels differ (e.g., M2 was ~$4T in 2000 vs ~$21T in 2024).

| Metric | Normalization | Rationale |
|--------|--------------|-----------|
| Fed Funds Rate | None (already %) | Directly comparable |
| Treasury yields | None (already %) | Directly comparable |
| Yield curve spread | None (already %) | Directly comparable |
| M2 Money Supply | Use YoY growth rate (%) | Absolute levels not comparable; growth rate is |
| CPI | Use YoY inflation rate (%) | Index levels not comparable |
| GDP | Use YoY growth rate (%) | Absolute levels not comparable |
| Credit spreads | None (already %) | Directly comparable |
| Unemployment | None (already %) | Directly comparable |
| Real interest rate | None (already %) | Already derived as difference |
| S&P 500 | Index to 100 at peak date | For overlay alignment |

### 4.6 Step 6: Yield Curve Inversion Duration

Compute continuous inversion periods (T10Y2Y < 0) and measure their duration.

```python
def compute_inversion_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Identify yield curve inversion periods and their durations."""
    inversions = []
    for era in ["dotcom", "ai"]:
        era_df = df[df["era"] == era].sort_values("date")
        inverted = era_df["yield_curve_spread"] < 0
        # Group consecutive inverted months
        groups = (inverted != inverted.shift()).cumsum()
        for group_id, group_df in era_df[inverted].groupby(groups[inverted]):
            inversions.append({
                "era": era,
                "inversion_start": group_df["date"].min(),
                "inversion_end": group_df["date"].max(),
                "duration_months": len(group_df),
                "max_inversion_depth": group_df["yield_curve_spread"].min(),
            })
    return pd.DataFrame(inversions)
```

---

## 5. Analysis Plan

### 5.1 Interest Rate Environment Comparison

**Question:** Was the Fed tightening or easing at the equivalent point in each cycle?

| Aspect | Dot-com Era | AI Era |
|--------|-------------|--------|
| Rate trajectory | Hiking from 4.75% (Jan 1999) to 6.50% (May 2000) | Hiking from 0.25% (Mar 2022) to 5.50% (Jul 2023), then cuts starting Sep 2024 |
| Rate at market peak | ~6.00% (Mar 2000) | TBD |
| Direction at peak | Still hiking (last hike May 2000) | Holding / beginning cuts |

**Method:**
1. Plot fed funds rate for both eras aligned by `months_from_peak`
2. Annotate rate hike/cut dates from FOMC meeting history
3. Compare: level, trajectory, and distance from neutral rate

### 5.2 Liquidity Comparison (M2 Growth)

**Question:** Is there more "fuel" (money) available to inflate asset prices?

**Method:**
1. Compute M2 YoY growth rate for both eras
2. Highlight the COVID-era M2 spike (Feb 2020 - Feb 2022: ~40% cumulative growth)
3. Note the unprecedented M2 contraction in 2022-2023 (first YoY decline since the 1930s)
4. Compare M2 growth at equivalent `months_from_peak` points

**Expected finding:** The AI era started with a massive liquidity injection (COVID stimulus) but then saw aggressive quantitative tightening. The dot-com era had steady, moderate M2 growth throughout.

### 5.3 Yield Curve Analysis

**Question:** Did the yield curve invert before the market peak, and by how long?

**Method:**
1. Identify all inversion periods in both eras (Section 4.6)
2. Measure lead time: months between first inversion and market peak
3. Measure lead time: months between first inversion and recession start
4. Compare inversion depth and duration

**Historical context:**
- Dot-com: Yield curve inverted in ~Feb 2000, one month before market peak. Recession started Mar 2001.
- AI era: Yield curve inverted July 2022, longest continuous inversion in modern history (~26 months). Un-inverted in ~September 2024.

### 5.4 Credit Conditions Comparison

**Question:** Are credit markets signaling stress or complacency?

**Method:**
1. Plot BAA10Y spread for both eras
2. Annotate bubble and crash phases
3. Tight spreads (< 2%) = complacency / risk-seeking (bull market fuel)
4. Wide spreads (> 3%) = stress / risk aversion (crash indicator)
5. Compare spread levels at equivalent `months_from_peak` points

**Interpretation guide:**
- Spreads tightening during the run-up = investors are not pricing in risk (classic bubble behavior)
- Spreads widening sharply = the crash is in progress or imminent

### 5.5 Real Interest Rate Comparison

**Question:** Is money "free" (negative real rates) or "expensive" (positive real rates)?

**Method:**
1. Real rate = FEDFUNDS - CPI YoY inflation
2. Negative real rates incentivize risk-taking (borrow cheap, buy assets)
3. Positive real rates make holding cash attractive (money flows out of risk assets)

**Expected finding:**
- Dot-com peak: Real rate was ~2-3% (nominal 6%, inflation ~3%). Money was NOT free.
- AI era early (2021-2022): Real rate was deeply negative (-5% to -7%). Money was essentially free. This fueled the NVDA run.
- AI era late (2023-2025): Real rate turned positive (~2%). Money is no longer free.

### 5.6 GDP Growth Context

**Question:** Is the economy overheating, stalling, or in recession at the peak?

**Method:**
1. Plot GDP YoY growth for both eras
2. Annotate NBER recession dates (Mar 2001 - Nov 2001 for dot-com)
3. Compare: was the economy strong enough to support elevated valuations?

---

## 6. Visualizations

### Chart 4.1: Fed Funds Rate -- Dual-Era Overlay

**Type:** Line chart with two traces
**X-axis:** `months_from_peak` (range: -48 to +36), label: "Months from Market Peak"
**Y-axis:** Federal Funds Rate (%), range: 0-8, label: "Federal Funds Rate (%)"
**Traces:**
- Dot-com era: solid line, color `#1f77b4` (blue), linewidth 2.5
- AI era: solid line, color `#d62728` (red), linewidth 2.5

**Annotations:**
- Vertical dashed line at x=0, label "Market Peak", color gray, alpha 0.7
- Arrow annotation at dot-com rate hiking period: "Fed hiking into bubble peak"
- Arrow annotation at AI era rate cuts: "Fed begins easing"
- Shaded region for NBER recession (dot-com only): light gray, alpha 0.2

**Title:** "Federal Funds Rate: Dot-Com vs AI Era (Aligned to Market Peak)"
**Legend:** Top-right, two entries: "Dot-Com (1996-2003)" and "AI Era (2021-2026)"
**Grid:** Light gray horizontal gridlines only
**Figure size:** 12 x 6 inches (matplotlib) or 900 x 500 px (Plotly)

```python
# pseudocode
fig = go.Figure()
for era, color, label in [("dotcom", "#1f77b4", "Dot-Com"), ("ai", "#d62728", "AI Era")]:
    era_df = unified[unified["era"] == era]
    fig.add_trace(go.Scatter(
        x=era_df["months_from_peak"],
        y=era_df["fed_funds_rate"],
        mode="lines",
        name=label,
        line=dict(color=color, width=2.5),
    ))
fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Market Peak")
fig.update_layout(
    title="Federal Funds Rate: Dot-Com vs AI Era (Aligned to Market Peak)",
    xaxis_title="Months from Market Peak",
    yaxis_title="Federal Funds Rate (%)",
    template="plotly_white",
    width=900, height=500,
)
```

### Chart 4.2: M2 Money Supply Growth Rate Comparison

**Type:** Line chart with two traces + area fill
**X-axis:** `months_from_peak`, label: "Months from Market Peak"
**Y-axis:** M2 YoY Growth Rate (%), range: -5 to 30, label: "M2 YoY Growth (%)"
**Traces:**
- Dot-com era: solid line, color `#1f77b4`, linewidth 2
- AI era: solid line, color `#d62728`, linewidth 2
- AI era fill between 0 and trace: light red fill, alpha 0.15 (to emphasize the COVID spike)

**Annotations:**
- Horizontal dashed line at y=0, label "Zero Growth", color black, alpha 0.5
- Arrow pointing to COVID M2 spike peak (~26% YoY): "COVID stimulus: unprecedented M2 expansion"
- Arrow pointing to M2 contraction trough (~-3%): "First M2 contraction since 1930s"
- Vertical dashed line at x=0: "Market Peak"

**Title:** "M2 Money Supply Growth: The Liquidity Story"
**Figure size:** 12 x 6 inches / 900 x 500 px

### Chart 4.3: Yield Curve (10Y-2Y) with Crash Annotations

**Type:** Line chart with shaded inversion regions
**X-axis:** `date` (actual dates, not aligned), label: "Date"
**Y-axis:** 10Y-2Y Spread (%), range: -1.5 to 3.0, label: "10Y-2Y Spread (%)"
**Layout:** Two subplots, stacked vertically (one per era), sharing Y-axis scale

**Subplot 1 (top): Dot-Com Era (1996-2003)**
- Line trace: color `#1f77b4`, linewidth 1.5
- Shaded red region where spread < 0 (inversion): color `#ff7f7f`, alpha 0.3
- Horizontal dashed line at y=0: "Inversion Threshold"
- Vertical dashed red line at March 2000: "S&P 500 Peak"
- Vertical dashed orange line at March 2001: "Recession Start"
- Annotation: "Inversion led crash by ~13 months"

**Subplot 2 (bottom): AI Era (2021-2026)**
- Line trace: color `#d62728`, linewidth 1.5
- Same shading/annotation scheme
- Vertical dashed line at provisional peak date
- Annotation: "Longest inversion in modern history (~26 months)"

**Title:** "Yield Curve (10Y-2Y): Recession Warning Signal"
**Figure size:** 12 x 10 inches / 900 x 800 px

### Chart 4.4: Credit Spreads Timeline

**Type:** Line chart with shaded bubble/crash periods
**X-axis:** `months_from_peak`, label: "Months from Market Peak"
**Y-axis:** Moody's Baa - 10Y Treasury Spread (%), range: 1 to 5, label: "Credit Spread (%)"
**Traces:**
- Dot-com era: solid line, color `#1f77b4`, linewidth 2
- AI era: solid line, color `#d62728`, linewidth 2

**Shaded regions:**
- "Complacency Zone" below 2%: light green, alpha 0.1
- "Stress Zone" above 3.5%: light red, alpha 0.1

**Annotations:**
- Arrow at dot-com crash spread widening: "Credit stress emerges"
- Arrow at AI era tight spreads: "Risk appetite remains strong"
- Vertical dashed line at x=0: "Market Peak"

**Title:** "Corporate Credit Spreads: Risk Appetite vs Stress"
**Figure size:** 12 x 6 inches / 900 x 500 px

### Chart 4.5: Macro Dashboard -- Small Multiples

**Type:** 3x3 grid of small line charts (faceted / small multiples)
**Layout:** `plotly.subplots.make_subplots(rows=3, cols=3)`

| Position | Indicator | Y-axis Label |
|----------|-----------|-------------|
| (1,1) | Fed Funds Rate | Rate (%) |
| (1,2) | 10Y Treasury Yield | Yield (%) |
| (1,3) | Yield Curve Spread | Spread (%) |
| (2,1) | M2 YoY Growth | Growth (%) |
| (2,2) | CPI YoY Inflation | Inflation (%) |
| (2,3) | Real Interest Rate | Rate (%) |
| (3,1) | Credit Spread | Spread (%) |
| (3,2) | GDP YoY Growth | Growth (%) |
| (3,3) | Unemployment Rate | Rate (%) |

**Each subplot:**
- X-axis: `months_from_peak` (shared range: -48 to +36)
- Two traces: dot-com (blue `#1f77b4`) and AI era (red `#d62728`)
- Vertical dashed line at x=0 (market peak)
- Compact title above each subplot
- No individual legends; single shared legend at bottom

**Title:** "Macro Environment Dashboard: Dot-Com vs AI Era"
**Figure size:** 18 x 14 inches / 1400 x 1000 px
**Color scheme:** Consistent blue/red across all subplots
**Grid:** Light gray, alpha 0.3

```python
# pseudocode
from plotly.subplots import make_subplots
indicators = [
    ("fed_funds_rate", "Fed Funds Rate (%)"),
    ("treasury_10y", "10Y Treasury (%)"),
    ("yield_curve_spread", "Yield Curve (%)"),
    ("m2_yoy_growth", "M2 Growth (%)"),
    ("cpi_yoy_inflation", "CPI Inflation (%)"),
    ("real_interest_rate", "Real Rate (%)"),
    ("credit_spread", "Credit Spread (%)"),
    ("gdp_yoy_growth", "GDP Growth (%)"),
    ("unemployment", "Unemployment (%)"),
]

fig = make_subplots(rows=3, cols=3, subplot_titles=[t for _, t in indicators])
for idx, (col, title) in enumerate(indicators):
    row, col_pos = divmod(idx, 3)
    for era, color in [("dotcom", "#1f77b4"), ("ai", "#d62728")]:
        era_df = unified[unified["era"] == era].sort_values("months_from_peak")
        fig.add_trace(
            go.Scatter(x=era_df["months_from_peak"], y=era_df[col],
                       mode="lines", line=dict(color=color, width=1.5),
                       showlegend=(idx == 0),
                       name="Dot-Com" if era == "dotcom" else "AI Era"),
            row=row+1, col=col_pos+1
        )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=row+1, col=col_pos+1)

fig.update_layout(height=1000, width=1400, template="plotly_white",
                  title="Macro Environment Dashboard: Dot-Com vs AI Era")
```

### Chart 4.6: Real Interest Rate Comparison

**Type:** Area chart with two traces
**X-axis:** `months_from_peak`, label: "Months from Market Peak"
**Y-axis:** Real Interest Rate (%), range: -8 to 5, label: "Real Interest Rate (%)"
**Traces:**
- Dot-com era: filled area below line, color `#1f77b4`, fill alpha 0.2, line width 2
- AI era: filled area below line, color `#d62728`, fill alpha 0.2, line width 2

**Annotations:**
- Horizontal line at y=0: thick black dashed, label "Zero Real Rate"
- Shaded region y < 0: very light red background, label "Free Money Zone"
- Shaded region y > 0: very light green background, label "Restrictive Zone"
- Arrow at AI era trough (~-7%): "Deepest negative real rates in modern history"
- Arrow at dot-com peak real rate (~+3%): "Money was expensive at the peak"
- Vertical dashed at x=0: "Market Peak"

**Title:** "Real Interest Rates: Is Money Free or Expensive?"
**Key insight annotation:** "Dot-com peaked with expensive money (+3%). AI boom started with the freest money in history (-7%)."
**Figure size:** 12 x 7 inches / 900 x 550 px

---

## 7. Statistical Tests

### 7.1 Descriptive Comparison: Mean/Variance Across Eras

Compute summary statistics for each indicator in a comparable window around each peak (e.g., -24 to +12 months).

```python
# pseudocode
window = unified[unified["months_from_peak"].between(-24, 12)]
indicators = ["fed_funds_rate", "yield_curve_spread", "m2_yoy_growth",
              "cpi_yoy_inflation", "real_interest_rate", "credit_spread",
              "gdp_yoy_growth", "unemployment"]

summary = window.groupby("era")[indicators].agg(["mean", "std", "min", "max"])
# Output as formatted table in notebook
```

**Expected output table:**

| Indicator | Dot-Com Mean | Dot-Com Std | AI Mean | AI Std | Diff (AI - DC) |
|-----------|-------------|-------------|---------|--------|----------------|
| Fed Funds Rate | ~5.5% | ~0.8 | ~4.5% | ~1.5 | ~ -1.0% |
| Yield Curve Spread | ~0.1% | ~0.5 | ~-0.5% | ~0.6 | ~ -0.6% |
| M2 YoY Growth | ~6% | ~1.5 | ~5% | ~12 | Higher variance |
| CPI Inflation | ~2.5% | ~0.5 | ~5% | ~2.5 | Higher inflation |
| Real Interest Rate | ~+3% | ~0.5 | ~-1% | ~3 | Much lower |
| Credit Spread | ~2.2% | ~0.5 | ~1.8% | ~0.3 | Tighter |
| GDP Growth | ~4.5% | ~1.5 | ~3% | ~2 | Slower |
| Unemployment | ~4.0% | ~0.3 | ~3.8% | ~0.5 | Similar |

### 7.2 Welch's t-test for Mean Differences

For each indicator, test whether the mean is statistically different between the two eras (within the -24 to +12 month window).

```python
from scipy import stats

results = []
for col in indicators:
    dotcom_vals = window[window["era"] == "dotcom"][col].dropna()
    ai_vals = window[window["era"] == "ai"][col].dropna()
    t_stat, p_val = stats.ttest_ind(dotcom_vals, ai_vals, equal_var=False)
    results.append({
        "indicator": col,
        "dotcom_mean": dotcom_vals.mean(),
        "ai_mean": ai_vals.mean(),
        "t_statistic": t_stat,
        "p_value": p_val,
        "significant_at_05": p_val < 0.05,
    })
summary_df = pd.DataFrame(results)
```

**Hypothesis:** H0: Mean(dot-com) = Mean(AI era) for each indicator. Expect to reject H0 for real interest rate, M2 growth, and CPI inflation.

### 7.3 Lead-Lag Analysis: Macro Indicators Leading Market Peaks

**Question:** Do macro indicators shift before or after the equity market peaks?

**Method:** Cross-correlation between each macro indicator's time series and the S&P 500 (or NVDA) drawdown series, with varying lags from -12 to +12 months.

```python
import numpy as np

def cross_correlation(x: pd.Series, y: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    """Compute cross-correlation between x and y at various lags.
    Positive lag = x leads y by that many periods.
    """
    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr = x.iloc[:-lag].reset_index(drop=True).corr(
                y.iloc[lag:].reset_index(drop=True)
            )
        elif lag < 0:
            corr = x.iloc[-lag:].reset_index(drop=True).corr(
                y.iloc[:lag].reset_index(drop=True)
            )
        else:
            corr = x.corr(y)
        results.append({"lag_months": lag, "correlation": corr})
    return pd.DataFrame(results)

# Run for each indicator against sp500 monthly returns
for col in indicators:
    for era in ["dotcom", "ai"]:
        era_df = unified[unified["era"] == era].sort_values("date")
        sp500_returns = era_df["sp500"].pct_change()
        indicator_series = era_df[col]
        cc = cross_correlation(indicator_series, sp500_returns, max_lag=12)
        # Find lag with max absolute correlation
        best_lag = cc.loc[cc["correlation"].abs().idxmax(), "lag_months"]
        # Report: "{col} leads/lags S&P 500 by {best_lag} months in {era}"
```

**Expected findings:**
- Yield curve inversion leads equity drawdown by 12-18 months (well-established in literature)
- Credit spread widening is coincident or slightly lagging (confirms rather than predicts)
- Fed funds rate changes lead equity moves by 6-9 months (monetary policy transmission lag)

### 7.4 Granger Causality: Yield Curve -> Equity Drawdowns

**Question:** Does the yield curve spread Granger-cause equity market drawdowns?

```python
from statsmodels.tsa.stattools import grangercausalitytests

for era in ["dotcom", "ai"]:
    era_df = unified[unified["era"] == era].sort_values("date").dropna(
        subset=["yield_curve_spread", "sp500"]
    )
    sp500_returns = era_df["sp500"].pct_change().dropna()
    yc = era_df["yield_curve_spread"].iloc[1:]  # align with returns

    # Granger test: does yield_curve_spread Granger-cause sp500_returns?
    # Test up to 6 lags (6 months)
    data = pd.DataFrame({
        "sp500_returns": sp500_returns.values,
        "yield_curve": yc.values,
    }).dropna()

    results = grangercausalitytests(data[["sp500_returns", "yield_curve"]], maxlag=6)
    # Extract p-values for each lag
    for lag in range(1, 7):
        f_stat = results[lag][0]["ssr_ftest"][0]
        p_val = results[lag][0]["ssr_ftest"][1]
        # Report: "Lag {lag}: F={f_stat:.2f}, p={p_val:.4f}"
```

**Interpretation:**
- p < 0.05 at lag k means: yield curve spread k months ago has statistically significant predictive power for current equity returns, beyond what past equity returns alone can explain.
- We expect significance at lags 3-6 for the dot-com era (well-documented).
- For the AI era, the result is the finding: if significant, the yield curve is warning. If not, markets may be ignoring the signal.

### 7.5 Levene's Test for Variance Differences

Test whether the volatility of macro indicators differs between eras.

```python
for col in indicators:
    dotcom_vals = window[window["era"] == "dotcom"][col].dropna()
    ai_vals = window[window["era"] == "ai"][col].dropna()
    stat, p_val = stats.levene(dotcom_vals, ai_vals)
    # Report: "{col}: Levene stat={stat:.2f}, p={p_val:.4f}"
```

**Hypothesis:** AI era has higher variance for M2 growth and CPI inflation due to COVID disruptions.

---

## 8. Key Narrative

The macro environment narrative to build into the final presentation:

> "The macro setup is fundamentally different. In 2000, the Fed was tightening into an overheating economy -- money was expensive (real rates +3%), yet speculation was rampant. The bubble burst when the economy couldn't sustain 6% fed funds rates.
>
> The AI era began in the opposite extreme: the deepest negative real rates in modern history (-7%) and the largest M2 injection ever recorded (COVID stimulus). Money was not just free -- you were being paid to borrow. This liquidity tsunami fueled the initial AI rally.
>
> But by 2023-2024, the macro picture shifted dramatically. The fastest hiking cycle in 40 years pushed real rates back to positive territory. M2 contracted for the first time since the 1930s. The yield curve inverted for a record 26 months.
>
> Yet NVIDIA and AI stocks kept rallying through restrictive monetary conditions -- a notable divergence from the dot-com playbook, where the bubble peaked as tightening intensified. Either the AI fundamentals are strong enough to overcome tight money, or the market is ignoring a macro regime that historically kills bull markets."

---

## 9. Limitations

### 9.1 Structural Breaks in Monetary Policy

- The Fed's toolkit has expanded dramatically since 2000 (quantitative easing, reverse repo, standing lending facility).
- Comparing fed funds rates across eras is an apples-to-oranges comparison in some respects -- the transmission mechanism has changed.
- The "neutral rate" (r*) may differ between eras, making absolute rate comparisons less meaningful.

### 9.2 Small Sample Problem

- We are comparing exactly two bubble episodes. N=2 is not a statistical basis for generalizable conclusions.
- Any observed pattern could be coincidence. Acknowledge this explicitly.
- Other bubbles (1929, Japan 1989, China 2015, Crypto 2021) could be added for robustness but are out of scope.

### 9.3 Data Limitations

- FRED data is retrospectively revised. GDP, CPI, and M2 figures for recent months are preliminary estimates.
- The "current" data point (March 2026) may be revised significantly.
- The T10Y2Y spread is a FRED convenience series; in academic research, the 10Y-3M spread is more commonly used as a recession predictor.

### 9.4 Regime Change Caveat

- Post-2008 unconventional monetary policy (QE, ZIRP) means the relationship between short rates, long rates, and equity markets may have permanently changed.
- The yield curve's predictive power may be diminished in an era of central bank balance sheet management.

### 9.5 Endogeneity

- Macro conditions and market prices are endogenous. The Fed responds to market conditions; markets respond to the Fed.
- Granger causality is NOT true causality. It only measures predictive power in time series.
- We should frame all findings as "associations" or "leading indicators," never as causal claims.

### 9.6 Cherry-picking Risk

- By choosing specific start/end dates and peak alignment, we can inadvertently (or intentionally) make the comparison look more similar or more different.
- Mitigation: show sensitivity to peak date choice (shift +/- 3 months and re-run).

---

## 10. Code Outline (Full Implementation Pseudocode)

```python
"""
Layer 4: Macro Environment Comparison
File: src/layers/macro_environment.py
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import json
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
RAW_DIR = Path("data/raw/fred")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONSTANTS
# ============================================================
SERIES_CONFIG = {
    "FEDFUNDS":  {"freq": "monthly",   "col_name": "fed_funds_rate"},
    "DGS10":     {"freq": "daily",     "col_name": "treasury_10y"},
    "DGS2":      {"freq": "daily",     "col_name": "treasury_2y"},
    "T10Y2Y":    {"freq": "daily",     "col_name": "yield_curve_spread"},
    "M2SL":      {"freq": "monthly",   "col_name": "m2_supply"},
    "CPIAUCSL":  {"freq": "monthly",   "col_name": "cpi_index"},
    "BAA10Y":    {"freq": "daily",     "col_name": "credit_spread"},
    "GDP":       {"freq": "quarterly", "col_name": "gdp"},
    "UNRATE":    {"freq": "monthly",   "col_name": "unemployment"},
    "SP500":     {"freq": "daily",     "col_name": "sp500"},
}

ERAS = {
    "dotcom": {"start": "1996-01-01", "end": "2003-12-31", "peak": "2000-03-31"},
    "ai":     {"start": "2021-01-01", "end": "2026-03-28", "peak": "2025-01-31"},  # provisional
}

# ============================================================
# STEP 1: FETCH DATA FROM FRED
# ============================================================
def fetch_fred_series(series_id: str, start: str, end: str) -> pd.DataFrame:
    """Fetch a single FRED series and return as DataFrame."""
    cache_path = RAW_DIR / f"{series_id}_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    resp = requests.get(FRED_BASE_URL, params={
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    })
    resp.raise_for_status()
    observations = resp.json()["observations"]

    df = pd.DataFrame(observations)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." -> NaN
    df.to_parquet(cache_path)
    return df


def fetch_all_series() -> dict[str, pd.DataFrame]:
    """Fetch all FRED series for all eras. Returns dict keyed by '{series_id}_{era}'."""
    all_data = {}
    for series_id in SERIES_CONFIG:
        for era_name, era_cfg in ERAS.items():
            key = f"{series_id}_{era_name}"
            all_data[key] = fetch_fred_series(
                series_id, era_cfg["start"], era_cfg["end"]
            )
            all_data[key]["era"] = era_name
            all_data[key]["series_id"] = series_id
    return all_data


# ============================================================
# STEP 2: ALIGN TO MONTHLY FREQUENCY
# ============================================================
def align_to_monthly(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Resample a single-series DataFrame to monthly (end-of-month) frequency."""
    df = df.set_index("date").sort_index()
    freq_type = SERIES_CONFIG[series_id]["freq"]

    if freq_type == "daily":
        monthly = df["value"].resample("ME").mean()
    elif freq_type == "quarterly":
        monthly = df["value"].resample("ME").ffill()
    else:  # monthly
        monthly = df["value"].resample("ME").last()

    return monthly.to_frame().reset_index()


# ============================================================
# STEP 3: BUILD UNIFIED DATAFRAME
# ============================================================
def build_unified_dataframe(all_data: dict) -> pd.DataFrame:
    """Merge all series into a single wide-format DataFrame."""
    unified_parts = []

    for era_name in ERAS:
        era_frames = {}
        for series_id, cfg in SERIES_CONFIG.items():
            key = f"{series_id}_{era_name}"
            monthly = align_to_monthly(all_data[key], series_id)
            monthly.columns = ["date", cfg["col_name"]]
            era_frames[series_id] = monthly

        # Merge all series on date (outer join to keep all dates)
        merged = era_frames[list(era_frames.keys())[0]]
        for sid in list(era_frames.keys())[1:]:
            merged = merged.merge(era_frames[sid], on="date", how="outer")

        merged["era"] = era_name
        merged = merged.sort_values("date").reset_index(drop=True)
        unified_parts.append(merged)

    unified = pd.concat(unified_parts, ignore_index=True)
    return unified


# ============================================================
# STEP 4: COMPUTE DERIVED METRICS
# ============================================================
def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns: YoY growth rates, real rate, months_from_peak."""
    for era_name, era_cfg in ERAS.items():
        mask = df["era"] == era_name

        # M2 YoY growth
        df.loc[mask, "m2_yoy_growth"] = (
            df.loc[mask, "m2_supply"].pct_change(periods=12) * 100
        )
        # CPI YoY inflation
        df.loc[mask, "cpi_yoy_inflation"] = (
            df.loc[mask, "cpi_index"].pct_change(periods=12) * 100
        )
        # Real interest rate
        df.loc[mask, "real_interest_rate"] = (
            df.loc[mask, "fed_funds_rate"] - df.loc[mask, "cpi_yoy_inflation"]
        )
        # GDP YoY growth
        df.loc[mask, "gdp_yoy_growth"] = (
            df.loc[mask, "gdp"].pct_change(periods=12) * 100
        )
        # Months from peak
        peak = pd.Timestamp(era_cfg["peak"])
        df.loc[mask, "months_from_peak"] = (
            (df.loc[mask, "date"].dt.year - peak.year) * 12
            + (df.loc[mask, "date"].dt.month - peak.month)
        )

    df["months_from_peak"] = df["months_from_peak"].astype("Int64")
    return df


# ============================================================
# STEP 5: YIELD CURVE INVERSION ANALYSIS
# ============================================================
def analyze_inversions(df: pd.DataFrame) -> pd.DataFrame:
    """Identify yield curve inversion periods and durations."""
    inversions = []
    for era in ["dotcom", "ai"]:
        era_df = df[df["era"] == era].sort_values("date").dropna(
            subset=["yield_curve_spread"]
        )
        inverted = era_df["yield_curve_spread"] < 0
        groups = (inverted != inverted.shift()).cumsum()
        for _, group_df in era_df[inverted].groupby(groups[inverted]):
            inversions.append({
                "era": era,
                "start": group_df["date"].min(),
                "end": group_df["date"].max(),
                "duration_months": len(group_df),
                "max_depth": group_df["yield_curve_spread"].min(),
            })
    return pd.DataFrame(inversions)


# ============================================================
# STEP 6: STATISTICAL TESTS
# ============================================================
def run_statistical_tests(df: pd.DataFrame) -> dict:
    """Run all statistical tests. Returns dict of result DataFrames."""
    window = df[df["months_from_peak"].between(-24, 12)]
    indicators = [
        "fed_funds_rate", "yield_curve_spread", "m2_yoy_growth",
        "cpi_yoy_inflation", "real_interest_rate", "credit_spread",
        "gdp_yoy_growth", "unemployment",
    ]

    # --- Welch's t-test ---
    ttest_results = []
    for col in indicators:
        dc = window[window["era"] == "dotcom"][col].dropna()
        ai = window[window["era"] == "ai"][col].dropna()
        if len(dc) < 3 or len(ai) < 3:
            continue
        t_stat, p_val = stats.ttest_ind(dc, ai, equal_var=False)
        ttest_results.append({
            "indicator": col,
            "dotcom_mean": dc.mean(), "dotcom_std": dc.std(),
            "ai_mean": ai.mean(), "ai_std": ai.std(),
            "t_stat": t_stat, "p_value": p_val,
            "significant": p_val < 0.05,
        })

    # --- Levene's test for variance ---
    levene_results = []
    for col in indicators:
        dc = window[window["era"] == "dotcom"][col].dropna()
        ai = window[window["era"] == "ai"][col].dropna()
        if len(dc) < 3 or len(ai) < 3:
            continue
        stat, p_val = stats.levene(dc, ai)
        levene_results.append({
            "indicator": col,
            "levene_stat": stat, "p_value": p_val,
            "significant": p_val < 0.05,
        })

    # --- Granger causality ---
    granger_results = []
    for era in ["dotcom", "ai"]:
        era_df = df[df["era"] == era].sort_values("date").dropna(
            subset=["yield_curve_spread", "sp500"]
        )
        returns = era_df["sp500"].pct_change().dropna()
        yc = era_df["yield_curve_spread"].iloc[1:].reset_index(drop=True)
        returns = returns.reset_index(drop=True)
        min_len = min(len(returns), len(yc))
        data = pd.DataFrame({
            "returns": returns.iloc[:min_len].values,
            "yield_curve": yc.iloc[:min_len].values,
        }).dropna()
        if len(data) < 20:
            continue
        try:
            gc = grangercausalitytests(data[["returns", "yield_curve"]], maxlag=6, verbose=False)
            for lag in range(1, 7):
                f_stat = gc[lag][0]["ssr_ftest"][0]
                p_val = gc[lag][0]["ssr_ftest"][1]
                granger_results.append({
                    "era": era, "lag": lag,
                    "f_stat": f_stat, "p_value": p_val,
                    "significant": p_val < 0.05,
                })
        except Exception as e:
            granger_results.append({"era": era, "error": str(e)})

    return {
        "ttest": pd.DataFrame(ttest_results),
        "levene": pd.DataFrame(levene_results),
        "granger": pd.DataFrame(granger_results),
    }


# ============================================================
# STEP 7: VISUALIZATIONS
# ============================================================
def create_chart_4_1(df: pd.DataFrame) -> go.Figure:
    """Fed Funds Rate dual-era overlay."""
    fig = go.Figure()
    colors = {"dotcom": "#1f77b4", "ai": "#d62728"}
    labels = {"dotcom": "Dot-Com (1996-2003)", "ai": "AI Era (2021-2026)"}
    for era in ["dotcom", "ai"]:
        era_df = df[df["era"] == era].sort_values("months_from_peak")
        fig.add_trace(go.Scatter(
            x=era_df["months_from_peak"], y=era_df["fed_funds_rate"],
            mode="lines", name=labels[era],
            line=dict(color=colors[era], width=2.5),
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                  annotation_text="Market Peak")
    fig.update_layout(
        title="Federal Funds Rate: Dot-Com vs AI Era (Aligned to Market Peak)",
        xaxis_title="Months from Market Peak",
        yaxis_title="Federal Funds Rate (%)",
        template="plotly_white", width=900, height=500,
    )
    return fig


def create_chart_4_2(df: pd.DataFrame) -> go.Figure:
    """M2 Money Supply growth rate comparison."""
    fig = go.Figure()
    colors = {"dotcom": "#1f77b4", "ai": "#d62728"}
    labels = {"dotcom": "Dot-Com (1996-2003)", "ai": "AI Era (2021-2026)"}
    for era in ["dotcom", "ai"]:
        era_df = df[df["era"] == era].sort_values("months_from_peak")
        fig.add_trace(go.Scatter(
            x=era_df["months_from_peak"], y=era_df["m2_yoy_growth"],
            mode="lines", name=labels[era],
            line=dict(color=colors[era], width=2),
            fill="tozeroy" if era == "ai" else None,
            fillcolor="rgba(214,39,40,0.1)" if era == "ai" else None,
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="black",
                  annotation_text="Zero Growth")
    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                  annotation_text="Market Peak")
    fig.update_layout(
        title="M2 Money Supply Growth: The Liquidity Story",
        xaxis_title="Months from Market Peak",
        yaxis_title="M2 YoY Growth (%)",
        template="plotly_white", width=900, height=500,
    )
    return fig


def create_chart_4_3(df: pd.DataFrame) -> go.Figure:
    """Yield Curve with crash annotations -- dual subplot."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, shared_yaxes=True,
                        subplot_titles=["Dot-Com Era (1996-2003)", "AI Era (2021-2026)"],
                        vertical_spacing=0.12)
    for idx, (era, color) in enumerate([("dotcom", "#1f77b4"), ("ai", "#d62728")]):
        era_df = df[df["era"] == era].sort_values("date")
        fig.add_trace(go.Scatter(
            x=era_df["date"], y=era_df["yield_curve_spread"],
            mode="lines", name=era, line=dict(color=color, width=1.5),
        ), row=idx+1, col=1)
        # Shade inversion regions
        inverted = era_df[era_df["yield_curve_spread"] < 0]
        if not inverted.empty:
            fig.add_trace(go.Scatter(
                x=inverted["date"], y=inverted["yield_curve_spread"],
                fill="tozeroy", fillcolor="rgba(255,127,127,0.3)",
                line=dict(width=0), showlegend=False,
            ), row=idx+1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="black",
                      row=idx+1, col=1)
    fig.update_layout(
        title="Yield Curve (10Y-2Y): Recession Warning Signal",
        template="plotly_white", width=900, height=800,
    )
    return fig


def create_chart_4_4(df: pd.DataFrame) -> go.Figure:
    """Credit Spreads timeline."""
    fig = go.Figure()
    colors = {"dotcom": "#1f77b4", "ai": "#d62728"}
    labels = {"dotcom": "Dot-Com", "ai": "AI Era"}
    for era in ["dotcom", "ai"]:
        era_df = df[df["era"] == era].sort_values("months_from_peak")
        fig.add_trace(go.Scatter(
            x=era_df["months_from_peak"], y=era_df["credit_spread"],
            mode="lines", name=labels[era],
            line=dict(color=colors[era], width=2),
        ))
    # Complacency zone
    fig.add_hrect(y0=0, y1=2.0, fillcolor="lightgreen", opacity=0.1,
                  annotation_text="Complacency Zone", annotation_position="top left")
    # Stress zone
    fig.add_hrect(y0=3.5, y1=6.0, fillcolor="lightcoral", opacity=0.1,
                  annotation_text="Stress Zone", annotation_position="top left")
    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                  annotation_text="Market Peak")
    fig.update_layout(
        title="Corporate Credit Spreads: Risk Appetite vs Stress",
        xaxis_title="Months from Market Peak",
        yaxis_title="Credit Spread (Baa - 10Y) (%)",
        template="plotly_white", width=900, height=500,
    )
    return fig


def create_chart_4_5(df: pd.DataFrame) -> go.Figure:
    """Macro Dashboard -- 3x3 small multiples."""
    indicators = [
        ("fed_funds_rate", "Fed Funds Rate (%)"),
        ("treasury_10y", "10Y Treasury (%)"),
        ("yield_curve_spread", "Yield Curve (%)"),
        ("m2_yoy_growth", "M2 Growth (%)"),
        ("cpi_yoy_inflation", "CPI Inflation (%)"),
        ("real_interest_rate", "Real Rate (%)"),
        ("credit_spread", "Credit Spread (%)"),
        ("gdp_yoy_growth", "GDP Growth (%)"),
        ("unemployment", "Unemployment (%)"),
    ]
    fig = make_subplots(rows=3, cols=3,
                        subplot_titles=[t for _, t in indicators],
                        vertical_spacing=0.08, horizontal_spacing=0.06)
    colors = {"dotcom": "#1f77b4", "ai": "#d62728"}
    for i, (col_name, _) in enumerate(indicators):
        row, col = divmod(i, 3)
        for era in ["dotcom", "ai"]:
            era_df = df[df["era"] == era].sort_values("months_from_peak")
            fig.add_trace(go.Scatter(
                x=era_df["months_from_peak"], y=era_df[col_name],
                mode="lines", line=dict(color=colors[era], width=1.5),
                showlegend=(i == 0), name="Dot-Com" if era == "dotcom" else "AI Era",
            ), row=row+1, col=col+1)
        fig.add_vline(x=0, line_dash="dash", line_color="gray",
                      opacity=0.5, row=row+1, col=col+1)
    fig.update_layout(
        title="Macro Environment Dashboard: Dot-Com vs AI Era",
        template="plotly_white", width=1400, height=1000,
    )
    return fig


def create_chart_4_6(df: pd.DataFrame) -> go.Figure:
    """Real Interest Rate comparison with shaded zones."""
    fig = go.Figure()
    # Shaded zones
    fig.add_hrect(y0=-10, y1=0, fillcolor="rgba(255,200,200,0.15)",
                  annotation_text="Free Money Zone", annotation_position="bottom left")
    fig.add_hrect(y0=0, y1=10, fillcolor="rgba(200,255,200,0.15)",
                  annotation_text="Restrictive Zone", annotation_position="top left")
    colors = {"dotcom": "#1f77b4", "ai": "#d62728"}
    labels = {"dotcom": "Dot-Com", "ai": "AI Era"}
    for era in ["dotcom", "ai"]:
        era_df = df[df["era"] == era].sort_values("months_from_peak")
        fig.add_trace(go.Scatter(
            x=era_df["months_from_peak"], y=era_df["real_interest_rate"],
            mode="lines", name=labels[era],
            line=dict(color=colors[era], width=2),
            fill="tozeroy",
            fillcolor=f"rgba({39 if era=='dotcom' else 214},"
                      f"{119 if era=='dotcom' else 39},"
                      f"{180 if era=='dotcom' else 40},0.15)",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=2)
    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                  annotation_text="Market Peak")
    fig.update_layout(
        title="Real Interest Rates: Is Money Free or Expensive?",
        xaxis_title="Months from Market Peak",
        yaxis_title="Real Interest Rate (%)",
        template="plotly_white", width=900, height=550,
        yaxis_range=[-8, 5],
    )
    return fig


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_layer_4():
    """Execute the full Layer 4 pipeline."""
    # 1. Fetch
    all_data = fetch_all_series()

    # 2. Build unified DataFrame
    unified = build_unified_dataframe(all_data)

    # 3. Compute derived metrics + peak alignment
    unified = compute_derived(unified)

    # 4. Analyze yield curve inversions
    inversions = analyze_inversions(unified)

    # 5. Run statistical tests
    test_results = run_statistical_tests(unified)

    # 6. Generate all charts
    charts = {
        "4.1": create_chart_4_1(unified),
        "4.2": create_chart_4_2(unified),
        "4.3": create_chart_4_3(unified),
        "4.4": create_chart_4_4(unified),
        "4.5": create_chart_4_5(unified),
        "4.6": create_chart_4_6(unified),
    }

    return {
        "unified": unified,
        "inversions": inversions,
        "test_results": test_results,
        "charts": charts,
    }
```

---

## 11. Output Artifacts

| Artifact | Path | Format | Description |
|----------|------|--------|-------------|
| Raw FRED data | `data/raw/fred/{series_id}_{era}.parquet` | Parquet | Cached API responses |
| Unified DataFrame | `data/processed/macro_unified.parquet` | Parquet | All indicators, monthly, both eras |
| Inversions table | `data/processed/yield_curve_inversions.csv` | CSV | Inversion periods with durations |
| Statistical tests | `data/processed/macro_stat_tests.json` | JSON | t-test, Levene, Granger results |
| Charts 4.1-4.6 | `submissions/charts/chart_4_*.html` | HTML | Interactive Plotly charts |
| Charts 4.1-4.6 (static) | `submissions/charts/chart_4_*.png` | PNG | Static exports for slides |

---

## 12. Acceptance Criteria

- [ ] All 10 FRED series fetched for both eras without errors
- [ ] Missing data (".") handled as NaN, not silently dropped
- [ ] Frequency alignment produces no duplicate dates
- [ ] `months_from_peak` correctly computed (negative before peak, 0 at peak, positive after)
- [ ] Derived metrics (M2 growth, CPI inflation, real rate) match spot-checks against FRED website
- [ ] All 6 charts render without errors
- [ ] Chart 4.5 dashboard has all 9 subplots populated
- [ ] Welch's t-test results for all indicators have p-values
- [ ] Granger causality runs for both eras with at least 3 lags
- [ ] Narrative written and ready for presentation layer
