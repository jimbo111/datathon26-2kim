# 07 — Complete Visualization Plan

> **Project:** The AI Hype Cycle — Are we in a bubble? NVIDIA 2023-2026 vs Cisco 1998-2001
> **Team:** 2Kim | SBU AI Community Datathon 2026 — Finance & Economics Track
> **Last updated:** 2026-03-28

---

## Table of Contents

1. [Design System](#1-design-system)
2. [Layer 1 Charts — Price Comparison](#2-layer-1-charts--price-comparison)
3. [Layer 2 Charts — Fundamentals](#3-layer-2-charts--fundamentals)
4. [Layer 3 Charts — Market Concentration](#4-layer-3-charts--market-concentration)
5. [Layer 4 Charts — Macro Environment](#5-layer-4-charts--macro-environment)
6. [Layer 5 Charts — Sentiment & NLP](#6-layer-5-charts--sentiment--nlp)
7. [ML Charts — Model Outputs](#7-ml-charts--model-outputs)
8. [Chart Inventory Summary](#8-chart-inventory-summary)
9. [Plotly Template Configuration](#9-plotly-template-configuration)

---

## 1. Design System

### 1.1 Color Palette

All charts use a consistent palette. Two semantic color groups: **dot-com era** vs **AI era**, plus functional colors.

| Role | Hex | Name | Usage |
|---|---|---|---|
| Dot-com era primary | `#EF553B` | Coral Red | CSCO price, dot-com period annotations |
| AI era primary | `#76b900` (light bg) / `#00CC96` (dark bg) | NVIDIA Green | NVDA price, AI period annotations |
| Bubble signal | `#FFA15A` | Warm Orange | Bubble regime, hype metrics, danger zones |
| Safe/normal signal | `#00CC96` | Teal Green | Normal growth regime, fundamental support |
| Correction signal | `#AB63FA` | Purple | Correction regime, crash periods |
| Recovery signal | `#19D3F3` | Cyan | Recovery regime |
| Neutral/grid | `#2D2D2D` | Charcoal | Grid lines (dark theme) |
| Background (dark) | `#111111` | Near Black | Dashboard background |
| Background (light) | `#FFFFFF` | White | Notebook/slides background |
| Text primary | `#E8E8E8` | Light Gray | Dark theme text |
| Text secondary | `#888888` | Mid Gray | Axis labels, annotations |
| Shaded regions | `rgba(239,85,59,0.15)` | — | Dot-com bubble shading |
| Shaded regions | `rgba(118,185,0,0.15)` | — | AI era shading |

**Sequential palette (heatmaps):** Plotly `Viridis` for quantitative heatmaps, `RdYlGn_r` for diverging (red = danger, green = safe).

### 1.2 Typography

| Element | Font | Size (slides) | Size (notebook) | Size (dashboard) |
|---|---|---|---|---|
| Chart title | Inter Bold | 20px | 16px | 18px |
| Subtitle/annotation | Inter Regular | 14px | 12px | 13px |
| Axis labels | Inter Medium | 13px | 11px | 12px |
| Tick labels | Inter Regular | 11px | 10px | 10px |
| Legend | Inter Regular | 12px | 10px | 11px |
| Data callout | Inter Bold | 16px | 14px | 14px |

**Fallback stack:** Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif

### 1.3 Layout Standards

- **Aspect ratio:** 16:9 for slides, flexible for notebook (auto height), 4:3 or 16:9 for dashboard cards
- **Margins:** `{"l": 60, "r": 30, "t": 80, "b": 60}` (standard); adjust for legends
- **Legend position:** Top-right inside plot area or below chart for > 4 items
- **Grid:** Light dashed gridlines on y-axis only; no x-axis grid for time series
- **Annotations:** Key events annotated with vertical dashed lines + text (e.g., "CSCO peak Mar 2000", "ChatGPT launch Nov 2022")
- **Hover:** Show date, value, and % change on hover (dashboard only)

### 1.4 Dark Theme vs Light Theme

- **Dark theme:** React dashboard (background `#111111`, text `#E8E8E8`)
- **Light theme:** Notebook and slides (background `#FFFFFF`, text `#333333`)
- Both themes use the same color palette for data series; only background/text/grid colors change

---

## 2. Layer 1 Charts — Price Comparison

### Chart 1.1: Normalized Price Overlay (THE Hook Chart)

| Property | Value |
|---|---|
| **Chart ID** | 1.1 |
| **Title** | "NVIDIA 2023-2026 vs Cisco 1998-2001: Normalized Price Trajectories" |
| **Type** | Dual-axis line chart |
| **X-axis** | "Months Since Breakout" (0 to 48) |
| **Y-axis** | "Price Indexed to 100 at Breakout" |
| **Series 1** | CSCO normalized (coral red, `#EF553B`, solid line, 2.5px) |
| **Series 2** | NVDA normalized (green, `#76b900` light bg / `#00CC96` dark bg, solid line, 2.5px) |
| **Series 3** | Nortel Networks (NT) normalized (gray, `#888888`, dashed line, 1.5px, label: "Nortel Networks (bankrupt 2009)") — non-survivor dot-com analog addressing survivorship bias |
| **Annotations** | Vertical dashed line at CSCO peak (month ~27); "CSCO Peak: +1,200%" label. Vertical dashed line at current NVDA month; "NVDA Now: +X%" label |
| **Shading** | Light red region for CSCO crash phase (months 27-48). Question mark overlay on NVDA future region |
| **Data sources** | yfinance daily OHLCV for CSCO (Jan 1998 - Dec 2001), NVDA (Jan 2023 - Mar 2026), NT (Jan 1998 - delisting ~2009), resampled to monthly close, normalized to 100 at month 0 |
| **Key insight** | "The price trajectories are strikingly similar in the run-up phase. NVDA is currently at the same point where CSCO was 6 months before its peak." |
| **Plotly config** | `go.Scatter`, mode='lines', hovertemplate='Month %{x}: %{y:.0f}<extra>%{fullData.name}</extra>' |
| **Destination** | Notebook, Slides (Slide 2 — THE hook), Dashboard (hero card) |
| **Priority** | **PRIMARY** — this is the single most important chart |

### Chart 1.2: Drawdown From All-Time High

| Property | Value |
|---|---|
| **Chart ID** | 1.2 |
| **Title** | "Drawdown From All-Time High: CSCO (1998-2003) vs NVDA (2023-2026)" |
| **Type** | Line chart (area fill below zero) |
| **X-axis** | "Months Since Breakout" |
| **Y-axis** | "Drawdown %" (0% at top, -80% at bottom) |
| **Series 1** | CSCO drawdown (coral red `#EF553B` area fill, opacity 0.3) |
| **Series 2** | NVDA drawdown (green `#76b900` area fill, opacity 0.3) |
| **Annotations** | "CSCO max drawdown: -89%"; current NVDA drawdown value |
| **Data sources** | Derived from daily price data; `(price / cummax) - 1` |
| **Key insight** | "NVDA has experienced drawdowns of X% during this period — smaller than CSCO's corrections but significant. The CSCO drawdown only began 27 months in." |
| **Plotly config** | `go.Scatter` with `fill='tozeroy'` |
| **Destination** | Notebook, Slides (backup), Dashboard |
| **Priority** | PRIMARY |

### Chart 1.3: Rolling 12-Month Returns Comparison

| Property | Value |
|---|---|
| **Chart ID** | 1.3 |
| **Title** | "Rolling 12-Month Returns: Late-Stage Bull Markets" |
| **Type** | Line chart |
| **X-axis** | "Months Since Breakout" |
| **Y-axis** | "Trailing 12-Month Return (%)" |
| **Series 1** | CSCO rolling return (coral red `#EF553B`) |
| **Series 2** | NVDA rolling return (green `#76b900`) |
| **Reference line** | Horizontal line at 0% (black dashed) |
| **Reference line** | Horizontal line at 100% (orange dashed, label: "2x in 12 months") |
| **Data sources** | Derived: `(price_t / price_{t-252}) - 1` from daily data |
| **Key insight** | "Both stocks sustained >100% annual returns for extended periods — a historically rare phenomenon that preceded corrections in every prior instance." |
| **Plotly config** | `go.Scatter`, mode='lines' |
| **Destination** | Notebook, Dashboard |
| **Priority** | SECONDARY |

### Chart 1.4: Volume Profile Comparison

| Property | Value |
|---|---|
| **Chart ID** | 1.4 |
| **Title** | "Trading Volume: Euphoria in Numbers" |
| **Type** | Dual-panel bar chart (side by side) |
| **X-axis** | Date (monthly) |
| **Y-axis** | "Monthly Average Daily Volume (millions of shares)" |
| **Panel 1** | CSCO volume bars (coral red gradient — lighter = earlier, darker = peak) |
| **Panel 2** | NVDA volume bars (green gradient) |
| **Annotations** | Volume at peak months highlighted with callout |
| **Data sources** | yfinance daily volume, aggregated to monthly average |
| **Key insight** | "Volume surged X-fold in the final 6 months of the CSCO bubble. NVDA volume has [surged/remained stable], suggesting [similar/different] retail participation." |
| **Plotly config** | `make_subplots(rows=1, cols=2, shared_yaxes=True)`, `go.Bar` |
| **Destination** | Notebook, Dashboard |
| **Priority** | SECONDARY |

---

## 3. Layer 2 Charts — Fundamentals

### Chart 2.1: P/E Ratio Comparison

| Property | Value |
|---|---|
| **Chart ID** | 2.1 |
| **Title** | "P/E Ratios: Where Fundamentals Diverge" |
| **Type** | Dual-axis line chart with shaded danger zone |
| **X-axis** | "Months Since Breakout" |
| **Y-axis** | "Trailing P/E Ratio" |
| **Series 1** | CSCO P/E (coral red `#EF553B`) — shows explosive rise to 200+ |
| **Series 2** | NVDA P/E (green `#76b900`) — shows moderate/declining from peak |
| **Shading** | Horizontal band above P/E = 50 shaded orange ("Historically Unsustainable Zone") |
| **Reference line** | S&P 500 long-term median P/E (~17) as gray dashed |
| **Data sources** | yfinance quarterly earnings + daily price; trailing 4-quarter EPS |
| **Key insight** | "This is the single strongest counter-argument to the bubble thesis. CSCO traded at 200x earnings with decelerating revenue. NVDA trades at Xx with accelerating revenue." |
| **Plotly config** | `go.Scatter` + `go.Scatter(fill='tozeroy')` for danger zone |
| **Destination** | Notebook, Slides (Slide 7 — Bull Case), Dashboard |
| **Priority** | **PRIMARY** |

### Chart 2.2: Revenue Growth Rate

| Property | Value |
|---|---|
| **Chart ID** | 2.2 |
| **Title** | "Revenue Growth: Real Demand vs Hype" |
| **Type** | Grouped bar chart |
| **X-axis** | Quarter (relative: Q-8 to Q0 from breakout) |
| **Y-axis** | "YoY Revenue Growth (%)" |
| **Color** | Coral red (`#EF553B`) bars = CSCO, Green (`#76b900`) bars = NVDA |
| **Annotations** | Arrow pointing to NVDA's peak growth quarter with value |
| **Data sources** | yfinance quarterly financials; SEC EDGAR if needed |
| **Key insight** | "NVDA revenue grew 265% YoY at peak vs CSCO's 66%. This is fundamentally different — NVDA's growth was 4x faster and driven by measurable enterprise AI spending." |
| **Plotly config** | `go.Bar(barmode='group')` |
| **Destination** | Notebook, Slides (Slide 7), Dashboard |
| **Priority** | **PRIMARY** |

### Chart 2.3: Market Cap vs Revenue Scatter

| Property | Value |
|---|---|
| **Chart ID** | 2.3 |
| **Title** | "Market Cap vs Revenue: The Valuation Sanity Check" |
| **Type** | Scatter plot with trajectory arrows |
| **X-axis** | "Annual Revenue ($B)" — log scale |
| **Y-axis** | "Market Cap ($B)" — log scale |
| **Points** | Each point = one quarter. CSCO path in coral red `#EF553B` (connected by arrows), NVDA path in green `#76b900` (connected by arrows). Point size = P/E ratio. |
| **Reference line** | Diagonal lines for P/S ratios (1x, 5x, 10x, 20x, 50x) as light gray |
| **Annotations** | "CSCO peak: $556B cap / $18B rev (31x P/S)" and "NVDA now: $XB cap / $XB rev (Xx P/S)" |
| **Data sources** | yfinance quarterly financials + market cap |
| **Key insight** | "Both stocks reached extreme price-to-sales ratios, but NVDA's revenue base is significantly larger. The question is whether the current P/S ratio is sustainable." |
| **Plotly config** | `go.Scatter`, mode='markers+lines', `xaxis_type='log'`, `yaxis_type='log'` |
| **Destination** | Notebook, Dashboard |
| **Priority** | PRIMARY |

### Chart 2.4: Free Cash Flow Yield

| Property | Value |
|---|---|
| **Chart ID** | 2.4 |
| **Title** | "Free Cash Flow Yield: Can the Company Fund Its Own Growth?" |
| **Type** | Bar chart with line overlay |
| **X-axis** | Quarter (relative) |
| **Y-axis Left** | "FCF ($B)" — bars |
| **Y-axis Right** | "FCF Yield (%)" — line |
| **Color** | Coral red `#EF553B` (CSCO) / Green `#76b900` (NVDA) side by side |
| **Reference line** | FCF Yield = 0% highlighted (companies burning cash) |
| **Data sources** | yfinance quarterly cash flow statements |
| **Key insight** | "NVDA generates $X billion in free cash flow quarterly — Cisco was also profitable, but NVDA's FCF yield at this market cap is [higher/lower], suggesting [more/less] fundamental support." |
| **Plotly config** | `make_subplots(specs=[[{"secondary_y": True}]])` |
| **Destination** | Notebook, Dashboard |
| **Priority** | SECONDARY |

### Chart 2.5: R&D Spending as % of Revenue

| Property | Value |
|---|---|
| **Chart ID** | 2.5 |
| **Title** | "R&D Investment: Building the Future or Burning Cash?" |
| **Type** | Area chart |
| **X-axis** | Quarter (relative) |
| **Y-axis** | "R&D as % of Revenue" |
| **Series** | Coral red area (CSCO, `#EF553B`), Green area (NVDA, `#76b900`) |
| **Annotations** | Industry medians for reference |
| **Data sources** | yfinance quarterly financials |
| **Key insight** | "High R&D spending indicates long-term investment in product moats. NVDA spends X% of revenue on R&D, creating a defensible competitive position that dot-com companies lacked." |
| **Plotly config** | `go.Scatter(fill='tozeroy')` |
| **Destination** | Notebook, Dashboard |
| **Priority** | SECONDARY |

---

## 4. Layer 3 Charts — Market Concentration

### Chart 3.1: Top-N Concentration Ratio Timeline

| Property | Value |
|---|---|
| **Chart ID** | 3.1 |
| **Title** | "S&P 500 Concentration: Then and Now" |
| **Type** | Stacked area chart |
| **X-axis** | Date (1995-2026) |
| **Y-axis** | "Weight in S&P 500 (%)" |
| **Areas** | Top 5 stocks stacked, each a different shade. Bottom 495 in gray. |
| **Shading** | Vertical bands for dot-com era (light coral red) and AI era (light green) |
| **Annotations** | "Top 5 = X% (Mar 2000)" and "Top 5 = X% (Mar 2026)" with horizontal reference lines |
| **Data sources** | S&P 500 constituent weights (via Slickcharts, Wikipedia historical, or custom computation from market caps) |
| **Key insight** | "Market concentration has returned to dot-com levels. The top 5 stocks now account for ~28% of the S&P 500, compared to ~18% at the dot-com peak. This is a structural similarity that even strong fundamentals cannot dismiss." |
| **Plotly config** | `go.Scatter(stackgroup='one')` |
| **Destination** | Notebook, Slides (Slide 9 — Bear Case), Dashboard |
| **Priority** | **PRIMARY** |

### Chart 3.2: SPY vs RSP (Equal-Weight) Performance Spread

| Property | Value |
|---|---|
| **Chart ID** | 3.2 |
| **Title** | "Cap-Weighted vs Equal-Weight: How Narrow Is This Rally?" |
| **Type** | Dual-line chart with spread area |
| **X-axis** | Date (2020-2026) |
| **Y-axis** | "Cumulative Return (%)" |
| **Series 1** | SPY cumulative return (solid line) |
| **Series 2** | RSP cumulative return (dashed line) |
| **Fill** | Area between SPY and RSP lines, colored orange when SPY > RSP |
| **Annotations** | "Gap = X percentage points — largest since 2000" |
| **Data sources** | yfinance ETF data: SPY, RSP |
| **Key insight** | "The average stock has dramatically underperformed the index — a hallmark of bubble-era concentration. In 2000, this spread collapsed violently as leadership rotated." |
| **Plotly config** | `go.Scatter` for both, `fill='tonexty'` for spread area |
| **Destination** | Notebook, Slides (Slide 9), Dashboard |
| **Priority** | **PRIMARY** |

### Chart 3.3: Buffett Indicator (Total Market Cap / GDP)

| Property | Value |
|---|---|
| **Chart ID** | 3.3 |
| **Title** | "The Buffett Indicator: Total Market Capitalization vs GDP" |
| **Type** | Line chart with threshold bands |
| **X-axis** | Date (1990-2026) |
| **Y-axis** | "Total Market Cap / GDP (%)" |
| **Thresholds** | < 80% green band ("Undervalued"), 80-120% yellow ("Fair Value"), 120-150% orange ("Overvalued"), >150% red ("Extreme") |
| **Annotations** | Dot-com peak value, GFC trough value, current value |
| **Data sources** | FRED: Wilshire 5000 full cap (WILL5000INDFC) / GDP (GDP) |
| **Key insight** | "The Buffett Indicator currently stands at ~190%, exceeding the dot-com peak of ~140%. However, critics note this metric doesn't account for globalization of US corporate revenue." |
| **Plotly config** | `go.Scatter` for line, `go.Scatter(fill='tozeroy')` for threshold bands with decreasing opacity |
| **Destination** | Notebook, Slides (Slide 10), Dashboard |
| **Priority** | PRIMARY |

### Chart 3.4: S&P 500 Sector Treemap

| Property | Value |
|---|---|
| **Chart ID** | 3.4 |
| **Title** | "S&P 500 Sector Composition: 2000 vs 2026" |
| **Type** | Side-by-side treemaps |
| **Layout** | Two treemaps: left = Mar 2000, right = Mar 2026 |
| **Hierarchy** | Sector > Top 3 stocks in each sector |
| **Color** | Sector-specific colors consistent across both panels; tech sector in NVIDIA green (`#76b900`) |
| **Size** | Proportional to market cap weight |
| **Data sources** | S&P 500 sector weights — historical (2000: Wikipedia/academic sources) and current (Slickcharts) |
| **Key insight** | "Tech + Communication Services now represent ~40% of the S&P 500, similar to Tech's ~35% dominance at the dot-com peak. The concentration is comparable, even if the companies are more profitable." |
| **Plotly config** | `go.Treemap`, `make_subplots(rows=1, cols=2)` with treemaps |
| **Destination** | Notebook, Slides (Slide 9 or backup), Dashboard |
| **Priority** | SECONDARY |

### Chart 3.5: HHI (Herfindahl-Hirschman Index) Over Time

| Property | Value |
|---|---|
| **Chart ID** | 3.5 |
| **Title** | "Market Concentration Index (HHI) for S&P 500" |
| **Type** | Line chart |
| **X-axis** | Date (1995-2026) |
| **Y-axis** | "HHI (sum of squared weights x 10,000)" |
| **Reference lines** | DOJ thresholds: < 1500 ("Unconcentrated"), 1500-2500 ("Moderate"), > 2500 ("Concentrated") adapted to equity context |
| **Annotations** | Dot-com peak HHI, current HHI |
| **Data sources** | Derived from S&P 500 weights: `sum(w_i^2)` x 10,000 |
| **Key insight** | "The HHI quantifies what the treemap shows visually — concentration risk is at multi-decade highs." |
| **Plotly config** | `go.Scatter` with horizontal `hline` shapes |
| **Destination** | Notebook, Dashboard |
| **Priority** | SECONDARY |

---

## 5. Layer 4 Charts — Macro Environment

### Chart 4.1: Fed Funds Rate Overlay

| Property | Value |
|---|---|
| **Chart ID** | 4.1 |
| **Title** | "The Rate Environment: A Key Difference" |
| **Type** | Step chart with S&P 500 overlay |
| **X-axis** | Date (1995-2026) |
| **Y-axis Left** | "Fed Funds Rate (%)" — step line |
| **Y-axis Right** | "S&P 500 Level" — area chart (light fill) |
| **Shading** | Dot-com era (light coral red), AI era (light green) |
| **Annotations** | "1999-2000: Rate hikes popped the bubble" and "2023-2024: Rates held high — market rallied anyway" |
| **Data sources** | FRED: DFF (fed funds rate), yfinance: ^GSPC (S&P 500) |
| **Key insight** | "The 1999-2000 rate hikes helped trigger the dot-com crash. Today, rates are at similar levels — but the market has rallied through them, which is historically unusual." |
| **Plotly config** | `make_subplots(specs=[[{"secondary_y": True}]])`, step line via `go.Scatter(line_shape='hv')` |
| **Destination** | Notebook, Slides (Slide 11 — Macro Wild Card), Dashboard |
| **Priority** | **PRIMARY** |

### Chart 4.2: M2 Money Supply Growth

| Property | Value |
|---|---|
| **Chart ID** | 4.2 |
| **Title** | "Money Supply Growth: The Liquidity Factor" |
| **Type** | Line chart with shaded regions |
| **X-axis** | Date (1995-2026) |
| **Y-axis** | "M2 YoY Growth (%)" |
| **Reference line** | 0% horizontal (contraction zone below) |
| **Shading** | Periods of > 10% growth highlighted in green (liquidity tailwind) |
| **Annotations** | "COVID stimulus: M2 grew 27% — largest since WWII" and "2022-23: M2 contracted for first time since 1930s" |
| **Data sources** | FRED: M2SL |
| **Key insight** | "The dot-com bubble was fueled by moderate M2 growth. The current rally occurred DESPITE M2 contraction — suggesting it's more earnings-driven than liquidity-driven, which is actually a healthier sign." |
| **Plotly config** | `go.Scatter(fill='tozeroy')` with conditional coloring via separate traces above/below 0 |
| **Destination** | Notebook, Slides (backup), Dashboard |
| **Priority** | SECONDARY |

### Chart 4.3: Yield Curve (10Y-2Y) with Recession Bands

| Property | Value |
|---|---|
| **Chart ID** | 4.3 |
| **Title** | "Yield Curve: The Recession Predictor" |
| **Type** | Line chart with recession shading |
| **X-axis** | Date (1990-2026) |
| **Y-axis** | "10Y - 2Y Treasury Spread (bps)" |
| **Reference line** | 0 bps horizontal (red dashed, "Inversion Threshold") |
| **Shading** | NBER recession periods in gray vertical bands |
| **Annotations** | "Inverted Jul 2022 - [date]. Every inversion since 1970 preceded a recession within 6-24 months." |
| **Data sources** | FRED: T10Y2Y (or compute from DGS10 - DGS2) |
| **Key insight** | "The yield curve inverted for the longest period in history (2022-2024). The historical batting average is 100% for recessions following inversion. The question is timing." |
| **Plotly config** | `go.Scatter` + `go.Scatter(fill='tozeroy')` for sub-zero area colored red |
| **Destination** | Notebook, Slides (Slide 11), Dashboard |
| **Priority** | **PRIMARY** |

### Chart 4.4: Credit Spreads (BAA-10Y)

| Property | Value |
|---|---|
| **Chart ID** | 4.4 |
| **Title** | "Credit Spreads: Is Stress Building?" |
| **Type** | Line chart |
| **X-axis** | Date (1995-2026) |
| **Y-axis** | "BAA Corporate - 10Y Treasury Spread (bps)" |
| **Thresholds** | < 200bps green zone ("Complacent"), 200-400 yellow ("Elevated"), > 400 red ("Stress") |
| **Shading** | NBER recessions in gray |
| **Annotations** | GFC peak spread value, current spread value |
| **Data sources** | FRED: BAA10Y or computed from DBAA - DGS10 |
| **Key insight** | "Credit spreads remain tight at ~Xbps, suggesting no systemic stress. In 2000, spreads started widening 6 months before the crash accelerated. This is currently NOT flashing red." |
| **Plotly config** | `go.Scatter` with threshold shapes |
| **Destination** | Notebook, Dashboard |
| **Priority** | SECONDARY |

### Chart 4.5: Macro Dashboard (Small Multiples)

| Property | Value |
|---|---|
| **Chart ID** | 4.5 |
| **Title** | "Macro Conditions Dashboard: Then vs Now" |
| **Type** | 2x3 small multiples grid |
| **Subplots** | (1) Fed funds rate, (2) M2 growth, (3) Yield curve, (4) Credit spread, (5) CPI YoY, (6) VIX |
| **Each subplot** | Coral red dot = dot-com era value at equivalent month, Green dot = current AI era value, Gray line = full history |
| **Data sources** | All FRED macro series |
| **Key insight** | "At a glance: the macro environment is fundamentally different. Rates are higher, M2 is tighter, but credit conditions are benign. This is neither the permissive 1999 environment nor a crisis setup." |
| **Plotly config** | `make_subplots(rows=2, cols=3, subplot_titles=[...])` |
| **Destination** | Notebook, Slides (Slide 11 or backup), Dashboard |
| **Priority** | PRIMARY |

### Chart 4.6: Real Interest Rate

| Property | Value |
|---|---|
| **Chart ID** | 4.6 |
| **Title** | "Real Interest Rates: Truly Restrictive?" |
| **Type** | Line chart |
| **X-axis** | Date (1990-2026) |
| **Y-axis** | "Real Rate (Fed Funds - CPI YoY, %)" |
| **Reference line** | 0% horizontal |
| **Shading** | Positive real rate periods shaded red (restrictive), negative shaded green (accommodative) |
| **Data sources** | FRED: DFF - CPIAUCSL (YoY) |
| **Key insight** | "Real rates are currently positive at ~X%, meaning monetary policy is genuinely restrictive. In the dot-com era, real rates were also positive (3-4%). Both periods feature real-rate tightening alongside equity euphoria." |
| **Plotly config** | `go.Scatter` with conditional fill |
| **Destination** | Notebook, Dashboard |
| **Priority** | SECONDARY |

---

## 6. Layer 5 Charts — Sentiment & NLP

### Chart 5.1: AI Mention Rate in Earnings Calls

| Property | Value |
|---|---|
| **Chart ID** | 5.1 |
| **Title** | "AI Mentions in NVIDIA Earnings Calls Over Time" |
| **Type** | Bar chart with trend line |
| **X-axis** | Earnings call date (quarterly, 2019-2026) |
| **Y-axis** | "AI Mentions per 10,000 Words" |
| **Color** | Gradient from teal (low) to orange (high) based on count |
| **Trend line** | Exponential fit overlay (dashed) |
| **Annotations** | "ChatGPT launch" vertical line, "Jensen's GTC 2024 keynote" marker |
| **Data sources** | NVDA earnings call transcripts (Seeking Alpha / SEC EDGAR 8-K), parsed with regex for AI-related terms |
| **Key insight** | "AI mentions in NVDA earnings calls increased Xx from 2019 to 2026. This mirrors the 'internet' mention surge in Cisco calls from 1998-2000, though NVDA's mentions are backed by specific revenue figures." |
| **Plotly config** | `go.Bar` + `go.Scatter` for trend line |
| **Destination** | Notebook, Slides (Slide 8 — Bear Case intro), Dashboard |
| **Priority** | **PRIMARY** |

### Chart 5.2: Hype vs Specificity Score (NLP Analysis)

| Property | Value |
|---|---|
| **Chart ID** | 5.2 |
| **Title** | "Hype vs Substance: NLP Analysis of Earnings Call Language" |
| **Type** | Stacked bar chart |
| **X-axis** | Earnings call date (quarterly) |
| **Y-axis** | "Proportion of Sentences (%)" |
| **Stacked bars** | Orange portion = "Hype" (vague/promotional), Teal portion = "Substantive" (specific metrics/revenue/customers) |
| **Annotations** | Arrow at transition point where hype ratio starts increasing |
| **Data sources** | OpenAI classification of each sentence in earnings call transcripts |
| **Key insight** | "Unlike pure hype companies, NVDA's earnings calls maintain a high proportion of substantive content (specific revenue numbers, customer names, deployment metrics). However, the hype ratio has been [increasing/stable] recently." |
| **Plotly config** | `go.Bar(barmode='stack')` |
| **Destination** | Notebook, Slides (Slide 8), Dashboard |
| **Priority** | **PRIMARY** |

### Chart 5.3: Google Trends — "Artificial Intelligence" vs "Internet" (1998)

| Property | Value |
|---|---|
| **Chart ID** | 5.3 |
| **Title** | "Public Interest: 'Artificial Intelligence' (2023-26) vs 'Internet' (1998-01)" |
| **Type** | Dual line chart (normalized) |
| **X-axis** | "Months Since Breakout" |
| **Y-axis** | "Google Trends Index (normalized 0-100)" |
| **Series 1** | "internet" search interest 1998-2001 (coral red `#EF553B`) — Note: Google Trends only goes back to 2004, so use "Zeitgeist" or proxy |
| **Series 2** | "artificial intelligence" search interest 2023-2026 (green `#76b900`) |
| **Data sources** | Google Trends API (pytrends) |
| **Key insight** | "Public search interest in AI has followed a similar surge pattern to 'internet' searches during the dot-com era, though AI interest has shown [more/less] persistence." |
| **Plotly config** | `go.Scatter`, mode='lines' |
| **Destination** | Notebook, Dashboard |
| **Priority** | SECONDARY |

### Chart 5.4: Multi-Signal Sentiment Timeline

| Property | Value |
|---|---|
| **Chart ID** | 5.4 |
| **Title** | "Sentiment Dashboard: Multiple Signals Over Time" |
| **Type** | Multi-line chart with Y-axis normalized to z-scores |
| **X-axis** | Date (2022-2026) |
| **Y-axis** | "Z-Score (standard deviations from mean)" |
| **Series** | (1) Google Trends "AI" — green `#76b900`, (2) Reddit sentiment — cyan `#19D3F3`, (3) Earnings call hype score — orange `#FFA15A`, (4) VIX inverted — purple `#AB63FA` |
| **Reference bands** | +/- 2 standard deviations shaded (extreme territory) |
| **Data sources** | Google Trends, Reddit API, NLP pipeline output, CBOE VIX |
| **Key insight** | "Multiple sentiment indicators are elevated simultaneously, which historically precedes periods of mean reversion. The correlation between these independent signals is itself a warning sign." |
| **Plotly config** | `go.Scatter` for each series, `go.Scatter(fill='tozeroy')` for z-score bands |
| **Destination** | Notebook, Slides (Slide 8 backup), Dashboard |
| **Priority** | SECONDARY |

### Chart 5.5: Reddit Mention Volume (NVDA + AI)

| Property | Value |
|---|---|
| **Chart ID** | 5.5 |
| **Title** | "Reddit Buzz: NVDA and AI Mentions on r/wallstreetbets" |
| **Type** | Bar chart with price overlay |
| **X-axis** | Week (2022-2026) |
| **Y-axis Left** | "Weekly Post Count Mentioning NVDA or AI" — bars |
| **Y-axis Right** | "NVDA Price ($)" — line overlay |
| **Color** | Bars colored by sentiment (green = bullish, red = bearish, gray = neutral) |
| **Data sources** | Reddit API (PRAW) or Pushshift for historical, VADER sentiment |
| **Key insight** | "Retail investor enthusiasm on Reddit peaked during [period], with mention volume Xx higher than 2022 baseline. Historically, peak retail enthusiasm coincides with local market tops." |
| **Plotly config** | `make_subplots(specs=[[{"secondary_y": True}]])`, `go.Bar` + `go.Scatter` |
| **Destination** | Notebook, Dashboard |
| **Priority** | SECONDARY |

### Chart 5.6: Word Cloud — Earnings Call Evolution

| Property | Value |
|---|---|
| **Chart ID** | 5.6 |
| **Title** | "Earnings Call Language: 2020 vs 2024 vs 2026" |
| **Type** | Three word clouds (side by side) |
| **Layout** | 1x3 grid of word clouds |
| **Data** | Top 50 non-stopword terms from NVDA earnings calls, sized by TF-IDF |
| **Color** | Orange for hype-classified terms, teal for substance-classified terms |
| **Data sources** | NLP pipeline: tokenized earnings call transcripts, TF-IDF scoring |
| **Key insight** | "The vocabulary has shifted from 'gaming' and 'GeForce' (2020) to 'data center', 'inference', and 'sovereign AI' (2024-2026). The language evolution mirrors real business transformation." |
| **Plotly config** | Note: Plotly does not natively support word clouds. Use `wordcloud` library to generate PNG, embed as `go.Image` or use as static image. Alternative: horizontal bar chart of top terms. |
| **Destination** | Notebook, Slides (backup) |
| **Priority** | SECONDARY |

---

## 7. ML Charts — Model Outputs

### Chart ML.1: Regime Classification Probabilities Over Time

| Property | Value |
|---|---|
| **Chart ID** | ML.1 |
| **Title** | "Market Regime Classification: Jul 2024 - Mar 2026" |
| **Type** | Stacked area chart |
| **X-axis** | Month (Jul 2024 - Mar 2026) |
| **Y-axis** | "Probability (%)" (0-100%) |
| **Stacked areas** | Orange = bubble, Purple = correction, Green = normal_growth, Cyan = recovery |
| **Annotations** | Latest month's probabilities as text callout: "Mar 2026: Bubble X%, Normal X%" |
| **Vertical line** | "Today" marker |
| **Data sources** | `results/regime_probabilities.csv` from ML pipeline |
| **Key insight** | "The ensemble classifier assigns X% probability to 'bubble' for March 2026 — elevated but not conclusive. The bubble probability has been [trending up/stable/declining] since mid-2024." |
| **Plotly config** | `go.Scatter(stackgroup='one', groupnorm='percent')` for each class |
| **Destination** | Notebook, Slides (Slide 13 — ML Verdict), Dashboard (hero card) |
| **Priority** | **PRIMARY** |

### Chart ML.2: DTW Similarity Score Timeline

| Property | Value |
|---|---|
| **Chart ID** | ML.2 |
| **Title** | "How Similar Is the AI Cycle to the Dot-Com Bubble? (DTW Score)" |
| **Type** | Line chart with confidence band |
| **X-axis** | Month (Jun 2023 - Mar 2026) |
| **Y-axis** | "Similarity Score (0-100)" |
| **Line** | Green solid line (`#76b900`) = composite DTW similarity score |
| **Confidence band** | Light green shaded area = 95% bootstrap CI |
| **Threshold lines** | Horizontal dashed lines at 40 ("Moderate"), 60 ("Strong"), 80 ("Very Strong") with labels |
| **Secondary Y (optional)** | NVDA price as faint gray line for context |
| **Annotations** | Latest score with CI: "Mar 2026: X [CI: Y-Z]" |
| **Data sources** | `results/dtw_similarity.csv` from DTW pipeline |
| **Key insight** | "The similarity score is currently X out of 100, placing us in the [moderate/strong] similarity zone. The score has been [increasing/plateauing], suggesting [convergence with/divergence from] the dot-com pattern." |
| **Plotly config** | `go.Scatter` for center line, `go.Scatter(fill='tonexty')` for CI band |
| **Destination** | Notebook, Slides (Slide 13), Dashboard |
| **Priority** | **PRIMARY** |

### Chart ML.3: SHAP Summary Plot (Bubble Class)

| Property | Value |
|---|---|
| **Chart ID** | ML.3 |
| **Title** | "What Drives the Bubble Signal? SHAP Feature Importance" |
| **Type** | SHAP beeswarm / dot plot |
| **X-axis** | "SHAP Value (impact on bubble probability)" |
| **Y-axis** | Feature name (ranked by mean |SHAP|, top 15) |
| **Color** | Feature value: low (blue) to high (red) using Plotly `RdBu` diverging scale |
| **Each dot** | One observation; x-position = SHAP value, color = feature value |
| **Annotations** | Bracket grouping: "Bubble signals (push right)" and "Fundamental support (push left)" |
| **Data sources** | SHAP values from `results/shap_values.json` |
| **Key insight** | "Concentration and sentiment features are the strongest bubble signals. Revenue growth and FCF are the strongest counter-signals. The model sees a genuine tug-of-war." |
| **Plotly config** | Note: Use `shap.summary_plot()` to generate matplotlib figure, then convert to Plotly via `plotly.tools.mpl_to_plotly()` or re-implement as `go.Scatter` with jitter. For dashboard: custom Plotly implementation with `go.Scatter` per feature row. |
| **Destination** | Notebook, Slides (Slide 14), Dashboard |
| **Priority** | **PRIMARY** |

### Chart ML.4: SHAP Waterfall for Today's Prediction

| Property | Value |
|---|---|
| **Chart ID** | ML.4 |
| **Title** | "Why the Model Says 'X%' Bubble Risk for March 2026" |
| **Type** | Waterfall chart |
| **X-axis** | "Contribution to Bubble Probability" |
| **Y-axis** | Feature name (ordered by |contribution|, top 12) |
| **Bars** | Red bars = features pushing toward bubble, Blue bars = features pushing away from bubble |
| **Base** | Starts at base rate (~14%, the prior probability of bubble class) |
| **Final** | Ends at model's predicted probability (X%) |
| **Annotations** | "Base rate: 14%" on left, "Final: X%" on right, connected by the waterfall |
| **Data sources** | SHAP values for the March 2026 observation |
| **Key insight** | "This decomposition shows exactly WHY the model classifies the current period as it does. Top-10 concentration adds +X% to bubble probability, while revenue growth subtracts -X%." |
| **Plotly config** | `go.Waterfall(measure=['relative']*N + ['total'], x=values, y=feature_names)` |
| **Destination** | Notebook, Slides (Slide 14), Dashboard |
| **Priority** | **PRIMARY** |

---

## 8. Chart Inventory Summary

### Executive Summary: PRIMARY vs SECONDARY Charts

**15 PRIMARY charts** (must-have for notebook main body — these are the charts judges will evaluate):

1. **1.1** — Normalized Price Overlay (THE hook)
2. **1.2** — Drawdown From ATH
3. **2.1** — P/E Ratio Comparison
4. **2.2** — Revenue Growth Rate
5. **2.3** — Market Cap vs Revenue Scatter
6. **3.1** — Top-N Concentration Ratio Timeline
7. **3.2** — SPY vs RSP Spread
8. **3.3** — Buffett Indicator
9. **4.1** — Fed Funds Rate Overlay
10. **4.3** — Yield Curve with Recession Bands
11. **4.5** — Macro Dashboard (Small Multiples)
12. **5.1** — AI Mention Rate in Earnings Calls
13. **5.2** — Hype vs Specificity Score
14. **ML.1** — Regime Classification Probabilities
15. **ML.2** — DTW Similarity Score Timeline

**15 SECONDARY charts** (supplementary/appendix — include if time permits, mark clearly as "Supplementary Analysis"):

1. **1.3** — Rolling 12-Month Returns
2. **1.4** — Volume Profile
3. **2.4** — Free Cash Flow Yield
4. **2.5** — R&D Spending %
5. **3.4** — S&P 500 Sector Treemap
6. **3.5** — HHI Over Time
7. **4.2** — M2 Money Supply Growth
8. **4.4** — Credit Spreads
9. **4.6** — Real Interest Rate
10. **5.3** — Google Trends AI vs Internet
11. **5.4** — Multi-Signal Sentiment Timeline
12. **5.5** — Reddit Mention Volume
13. **5.6** — Word Cloud
14. **ML.3** — SHAP Summary Plot (Beeswarm)
15. **ML.4** — SHAP Waterfall for Today

> **Notebook rule:** The 15 PRIMARY charts go in the main EDA / ML sections. The 15 SECONDARY charts go in a clearly labeled "Supplementary Analysis" subsection or appendix. This respects judge time and signals analytical prioritization.

### By Layer

| Layer | Chart Count | Primary | Secondary |
|---|---|---|---|
| Layer 1: Price | 4 | 2 (1.1, 1.2) | 2 (1.3, 1.4) |
| Layer 2: Fundamentals | 5 | 3 (2.1, 2.2, 2.3) | 2 (2.4, 2.5) |
| Layer 3: Concentration | 5 | 3 (3.1, 3.2, 3.3) | 2 (3.4, 3.5) |
| Layer 4: Macro | 6 | 3 (4.1, 4.3, 4.5) | 3 (4.2, 4.4, 4.6) |
| Layer 5: Sentiment | 6 | 2 (5.1, 5.2) | 4 (5.3, 5.4, 5.5, 5.6) |
| ML Outputs | 4 | 2 (ML.1, ML.2) | 2 (ML.3, ML.4) |
| **Total** | **30** | **15** | **15** |

### By Destination

| Destination | Charts |
|---|---|
| Notebook (all) | All 30 |
| Slides (must-have) | 1.1, 2.1, 2.2, 3.1, 3.2, 3.3, 4.1, 4.3, 5.1, 5.2, ML.1, ML.2, ML.3, ML.4 = **14 charts** |
| Slides (backup) | 1.2, 4.5, 5.4, 5.6 = 4 charts |
| Dashboard | All 30 (interactive versions) |

### Chart Production Priority

**Sprint 1 (must have for submission):** Charts 1.1, 2.1, 2.2, 3.1, 3.2, ML.1, ML.2, ML.3, ML.4 = 9 charts

**Sprint 2 (must have for polish):** Charts 1.2, 2.3, 3.3, 4.1, 4.3, 4.5, 5.1, 5.2 = 8 charts

**Sprint 3 (nice to have):** All remaining = 13 charts

---

## 9. Plotly Template Configuration

### 9.1 Dark Theme Template (Dashboard)

```python
import plotly.graph_objects as go
import plotly.io as pio

dark_template = go.layout.Template(
    layout=go.Layout(
        # Background
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',

        # Fonts
        font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            size=12,
            color='#E8E8E8',
        ),
        title=dict(
            font=dict(size=18, color='#E8E8E8'),
            x=0.5,
            xanchor='center',
        ),

        # Axes
        xaxis=dict(
            gridcolor='#2D2D2D',
            gridwidth=0.5,
            linecolor='#444444',
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=10, color='#888888'),
            title_font=dict(size=12, color='#AAAAAA'),
        ),
        yaxis=dict(
            gridcolor='#2D2D2D',
            gridwidth=0.5,
            linecolor='#444444',
            showgrid=True,
            griddash='dot',
            zeroline=False,
            tickfont=dict(size=10, color='#888888'),
            title_font=dict(size=12, color='#AAAAAA'),
        ),

        # Legend
        legend=dict(
            bgcolor='rgba(17,17,17,0.8)',
            bordercolor='#333333',
            borderwidth=1,
            font=dict(size=11, color='#CCCCCC'),
        ),

        # Margins
        margin=dict(l=60, r=30, t=80, b=60),

        # Color sequence
        colorway=[
            '#00CC96',  # NVIDIA Green (AI era — dark bg variant)
            '#EF553B',  # Coral Red (Dot-com era)
            '#FFA15A',  # Warm Orange
            '#AB63FA',  # Purple
            '#19D3F3',  # Cyan
            '#636EFA',  # Indigo Blue
            '#FF6692',  # Pink
            '#B6E880',  # Lime
        ],

        # Hover
        hoverlabel=dict(
            bgcolor='#1E1E1E',
            bordercolor='#444444',
            font=dict(size=12, color='#E8E8E8'),
        ),

        # Annotations default
        annotationdefaults=dict(
            font=dict(size=12, color='#CCCCCC'),
            arrowcolor='#666666',
        ),
    ),
)

pio.templates['datathon_dark'] = dark_template
```

### 9.2 Light Theme Template (Notebook & Slides)

```python
light_template = go.layout.Template(
    layout=go.Layout(
        # Background
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',

        # Fonts
        font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            size=12,
            color='#333333',
        ),
        title=dict(
            font=dict(size=16, color='#222222'),
            x=0.5,
            xanchor='center',
        ),

        # Axes
        xaxis=dict(
            gridcolor='#E8E8E8',
            gridwidth=0.5,
            linecolor='#CCCCCC',
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=10, color='#666666'),
            title_font=dict(size=12, color='#444444'),
        ),
        yaxis=dict(
            gridcolor='#E8E8E8',
            gridwidth=0.5,
            linecolor='#CCCCCC',
            showgrid=True,
            griddash='dot',
            zeroline=False,
            tickfont=dict(size=10, color='#666666'),
            title_font=dict(size=12, color='#444444'),
        ),

        # Legend
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#CCCCCC',
            borderwidth=1,
            font=dict(size=11, color='#444444'),
        ),

        # Margins
        margin=dict(l=60, r=30, t=80, b=60),

        # Color sequence — NVDA green first, CSCO coral red second
        colorway=[
            '#76b900',  # NVIDIA Green (AI era — light bg variant)
            '#EF553B',  # Coral Red (Dot-com era)
            '#FFA15A',  # Warm Orange
            '#AB63FA',  # Purple
            '#19D3F3',  # Cyan
            '#636EFA',  # Indigo Blue
            '#FF6692',  # Pink
            '#B6E880',  # Lime
        ],

        # Hover
        hoverlabel=dict(
            bgcolor='#FAFAFA',
            bordercolor='#CCCCCC',
            font=dict(size=12, color='#333333'),
        ),

        # Annotations default
        annotationdefaults=dict(
            font=dict(size=12, color='#555555'),
            arrowcolor='#999999',
        ),
    ),
)

pio.templates['datathon_light'] = light_template
```

### 9.3 Usage in Code

```python
# Notebook / slides
fig.update_layout(template='datathon_light')

# Dashboard
fig.update_layout(template='datathon_dark')

# Export for slides (high-res static)
fig.write_image('charts/chart_1_1.png', width=1920, height=1080, scale=2)

# Export as interactive HTML (dashboard)
fig.write_html('charts/chart_1_1.html', include_plotlyjs='cdn')

# Export as JSON (for React frontend)
fig.to_json()  # Backend returns this; frontend renders with react-plotly.js
```

### 9.4 Responsive Sizing for Dashboard Cards

```python
# Dashboard card sizes
CARD_SIZES = {
    'hero':     {'width': 1200, 'height': 600},   # Full-width featured chart
    'half':     {'width': 580,  'height': 400},    # Half-width card
    'third':    {'width': 380,  'height': 350},    # Third-width card
    'quarter':  {'width': 280,  'height': 300},    # Quarter-width card (small multiples)
}

# Apply responsive sizing
def size_for_card(fig, card_type='half'):
    dims = CARD_SIZES[card_type]
    fig.update_layout(
        width=dims['width'],
        height=dims['height'],
        margin=dict(l=40, r=20, t=60, b=40) if card_type != 'hero'
               else dict(l=60, r=30, t=80, b=60),
    )
    return fig
```

### 9.5 Annotation Helpers

```python
def add_era_shading(fig, era='dotcom'):
    """Add vertical shading for historical era."""
    if era == 'dotcom':
        fig.add_vrect(x0='1998-01-01', x1='2000-03-31',
                      fillcolor='rgba(239,85,59,0.08)', layer='below',
                      line_width=0,
                      annotation_text='Dot-Com Bubble', annotation_position='top left',
                      annotation_font_size=10, annotation_font_color='#EF553B')
        fig.add_vrect(x0='2000-04-01', x1='2002-10-31',
                      fillcolor='rgba(171,99,250,0.08)', layer='below',
                      line_width=0,
                      annotation_text='Dot-Com Crash', annotation_position='top left',
                      annotation_font_size=10, annotation_font_color='#AB63FA')
    elif era == 'ai':
        fig.add_vrect(x0='2023-01-01', x1='2026-03-31',
                      fillcolor='rgba(118,185,0,0.08)', layer='below',
                      line_width=0,
                      annotation_text='AI Era', annotation_position='top left',
                      annotation_font_size=10, annotation_font_color='#76b900')
    return fig

def add_recession_bands(fig):
    """Add NBER recession shading."""
    recessions = [
        ('1990-07-01', '1991-03-31'),
        ('2001-03-01', '2001-11-30'),
        ('2007-12-01', '2009-06-30'),
        ('2020-02-01', '2020-04-30'),
    ]
    for start, end in recessions:
        fig.add_vrect(x0=start, x1=end,
                      fillcolor='rgba(128,128,128,0.15)', layer='below',
                      line_width=0)
    return fig

def add_event_marker(fig, date, text, y_position='top'):
    """Add vertical line with text for key events."""
    fig.add_vline(x=date, line_dash='dash', line_color='#666666', line_width=1)
    fig.add_annotation(
        x=date, y=1 if y_position == 'top' else 0,
        yref='paper', text=text,
        showarrow=False, font=dict(size=10, color='#888888'),
        textangle=-90, xanchor='left',
    )
    return fig
```
