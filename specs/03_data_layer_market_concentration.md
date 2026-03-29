# Layer 3: Market Concentration & Systemic Risk

## 1. Objective

**Primary question:** Is today's AI-driven market concentration more dangerous than the dot-com era's tech concentration, and does it represent systemic risk for the broader market?

**Why this matters:** Even if NVDA's fundamentals are strong (Layer 2), the broader market could still be in a bubble if too much capital is concentrated in a handful of AI stocks. A healthy market has broad participation; a bubble-prone market has narrow leadership. This layer measures that breadth.

**Sub-questions:**
- What percentage of the S&P 500's market cap is concentrated in the top 10 stocks today vs the dot-com peak?
- How much of the S&P 500 is specifically AI-related (NVDA, AMD, MSFT, GOOGL, META, AVGO)?
- Is the gap between cap-weighted (SPY) and equal-weighted (RSP) performance at historically extreme levels?
- Does the Buffett Indicator (total market cap / GDP) signal overvaluation?
- How does the HHI (Herfindahl-Hirschman Index) of the S&P 500 compare across eras?
- Historically, has extreme concentration predicted subsequent drawdowns?

**Scoring alignment:** Targets Research Question (15pts) by addressing systemic risk, and Limitations & Ethics (10pts) by acknowledging the broader implications of market concentration.

---

## 2. Data Sources

### 2.1 ETF and Index Price Data (yfinance)

| Ticker | Name                          | Purpose                                    | Date Range                |
|--------|-------------------------------|--------------------------------------------|---------------------------|
| SPY    | SPDR S&P 500 ETF              | Cap-weighted S&P 500 proxy                | 1998-01-01 to 2026-03-28  |
| RSP    | Invesco S&P 500 Equal Weight  | Equal-weight S&P 500 (inception 2003-04-30)| 2003-04-30 to 2026-03-28 |
| ^GSPC  | S&P 500 Index                 | Broad market benchmark                     | 1998-01-01 to 2026-03-28  |
| XLK    | Technology Select Sector SPDR | Tech sector ETF                            | 1998-12-22 to 2026-03-28  |
| SMH    | VanEck Semiconductor ETF      | Semiconductor sub-sector                   | 2000-05-05 to 2026-03-28  |
| SOXX   | iShares Semiconductor ETF     | Semiconductor sub-sector (alternative)     | 2001-07-10 to 2026-03-28  |
| QQQ    | Invesco QQQ Trust             | NASDAQ-100 proxy                           | 1999-03-10 to 2026-03-28  |

```python
import yfinance as yf

etf_tickers = ["SPY", "RSP", "^GSPC", "XLK", "SMH", "SOXX", "QQQ"]
etf_data = yf.download(
    tickers=etf_tickers,
    start="1998-01-01",
    end="2026-03-28",
    interval="1d",
    auto_adjust=True,
    group_by="ticker",
    threads=True
)
```

### 2.2 S&P 500 Constituent Market Caps

**Current constituents (2024-2026):**

Source: yfinance can pull current market cap for each S&P 500 constituent.

```python
import pandas as pd

# S&P 500 constituent list (use Wikipedia or a maintained CSV)
# Wikipedia table: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_table = pd.read_html(sp500_url)[0]
sp500_tickers = sp500_table["Symbol"].str.replace(".", "-").tolist()  # BRK.B -> BRK-B

# Pull current market cap for each constituent
def get_current_market_caps(tickers: list[str]) -> pd.DataFrame:
    """Get current market cap for each ticker using yfinance."""
    results = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            results.append({
                "ticker": ticker,
                "company": info.get("longName", ""),
                "market_cap": info.get("marketCap", 0),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", "")
            })
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    return pd.DataFrame(results)

# For historical market caps (dot-com era), use FMP or manually constructed dataset
```

**Historical constituent weights (dot-com era):**

Historical S&P 500 weights are not freely available at daily granularity. Strategies:
1. **FMP historical market cap endpoint** (preferred): Pull daily market cap for top-50 S&P 500 stocks of each era
2. **Approximation:** Use known composition from financial archives + yfinance historical prices * shares outstanding
3. **Academic datasets:** Compustat via Wharton Research Data Services (WRDS) -- if available

```python
# For current era: yfinance batch pull
def get_historical_market_caps_batch(tickers: list[str], date: str) -> pd.DataFrame:
    """
    Approximate market cap on a historical date.
    Market cap = Close price * Shares Outstanding.
    """
    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=date, end=pd.to_datetime(date) + pd.Timedelta(days=5))
            if not hist.empty:
                close = hist["Close"].iloc[0]
                shares = stock.info.get("sharesOutstanding", 0)
                results.append({
                    "ticker": ticker,
                    "date": date,
                    "close": close,
                    "shares_outstanding": shares,
                    "market_cap_approx": close * shares
                })
        except Exception:
            pass
    return pd.DataFrame(results)
```

### 2.3 Top Stocks for AI Concentration Analysis

**AI-linked S&P 500 stocks (current era):**

| Ticker | Company   | AI Role                                    |
|--------|-----------|--------------------------------------------|
| NVDA   | NVIDIA    | GPU/AI chip maker                          |
| MSFT   | Microsoft | OpenAI investor, Azure AI, Copilot         |
| GOOGL  | Alphabet  | Gemini, DeepMind, cloud AI                 |
| META   | Meta      | Llama models, AI-driven ads                |
| AMZN   | Amazon    | AWS AI, Trainium/Inferentia chips          |
| AVGO   | Broadcom  | Custom AI accelerators, networking          |
| AMD    | AMD       | MI300X AI GPUs                             |
| AAPL   | Apple     | Apple Intelligence, on-device AI           |
| ORCL   | Oracle    | Cloud AI infrastructure                    |
| TSM    | TSMC (ADR)| Fabrication of all AI chips                |

**Dot-com era top-10 (approximate, circa March 2000):**

| Ticker | Company            | Market Cap (approx) |
|--------|--------------------|---------------------|
| MSFT   | Microsoft          | ~$580B              |
| GE     | General Electric   | ~$475B              |
| CSCO   | Cisco Systems      | ~$550B              |
| INTC   | Intel              | ~$395B              |
| WMT    | Walmart            | ~$260B              |
| XOM    | Exxon Mobil        | ~$255B              |
| LU     | Lucent Technologies| ~$230B              |
| IBM    | IBM                | ~$195B              |
| C      | Citigroup          | ~$185B              |
| QCOM   | Qualcomm           | ~$165B              |

**Note:** These are approximate. Exact historical figures should be sourced from FMP or cross-referenced with archived financial data.

### 2.4 FRED Macro Data

| Series ID     | Description                                | Frequency | Purpose                        |
|---------------|--------------------------------------------|-----------|--------------------------------|
| WILL5000IND   | Wilshire 5000 Total Market Full Cap Index  | Daily     | Total US equity market cap     |
| WILL5000INDFC | Wilshire 5000 Full Cap Price Index         | Daily     | Alternative total market cap   |
| GDP           | Gross Domestic Product                     | Quarterly | Denominator for Buffett Indicator |
| GDPC1         | Real GDP (chained 2017 dollars)            | Quarterly | For real-adjusted Buffett Indicator |

```python
from fredapi import Fred
fred = Fred(api_key=FRED_API_KEY)

wilshire = fred.get_series("WILL5000IND", observation_start="1997-01-01", observation_end="2026-03-28")
gdp = fred.get_series("GDP", observation_start="1997-01-01", observation_end="2026-03-28")

# GDP is quarterly; Wilshire is daily. Align by forward-filling GDP.
```

**Buffett Indicator calculation:**
```
Buffett_Indicator = (Wilshire 5000 Total Market Cap) / (GDP) * 100
```

**Note:** The Wilshire 5000 is an index value, not a dollar amount. To convert to total market cap, use the known relationship: as of a reference date, the Wilshire 5000 index value corresponds to approximately the total US market cap in billions. Use the FRED series `WILL5000IND` directly as a proxy for total market cap, or use the separate series for total market cap in USD.

Alternative: Use the Fed Z.1 Financial Accounts data (series FL073164105.Q for corporate equities market value) or the World Bank total market cap data.

```python
# More precise: use FRED 'MKTCAP' or compute from Wilshire index
# The Wilshire 5000 index * scaling factor ≈ total market cap
# As of 2024, Wilshire 5000 index ~45000, total market cap ~$55T
# Scaling factor ≈ 1.22B per index point (approximate)

# Better approach: use total market cap directly
# FRED series: BOGZ1FL073164005Q (Market value of equities)
total_mcap = fred.get_series("BOGZ1FL073164005Q",
                              observation_start="1997-01-01",
                              observation_end="2026-03-28")
```

---

## 3. Data Schema

### 3.1 Constituent Market Cap Table

| Column          | Dtype      | Description                                   |
|-----------------|------------|-----------------------------------------------|
| date            | datetime64 | Observation date                              |
| ticker          | str        | Stock ticker                                  |
| company_name    | str        | Company name                                  |
| sector          | str        | GICS sector                                   |
| market_cap      | float64    | Market capitalization in USD                  |
| sp500_weight    | float64    | Weight in S&P 500 (market_cap / total_sp500_mcap) |
| is_ai_linked    | bool       | Manually flagged as AI-linked                 |
| is_top_10       | bool       | In top 10 by market cap on this date          |
| era             | str        | "current" or "dotcom"                         |

### 3.2 Concentration Metrics Table (Time Series)

| Column                  | Dtype      | Description                                          |
|-------------------------|------------|------------------------------------------------------|
| date                    | datetime64 | Observation date (quarterly or monthly)               |
| top10_concentration     | float64    | Sum of top-10 weights in S&P 500 (%)                 |
| top5_concentration      | float64    | Sum of top-5 weights in S&P 500 (%)                  |
| ai_concentration        | float64    | Sum of AI-linked stock weights in S&P 500 (%)        |
| hhi_index               | float64    | Herfindahl-Hirschman Index of S&P 500 weights        |
| spy_return_cum          | float64    | Cumulative return of SPY from reference date          |
| rsp_return_cum          | float64    | Cumulative return of RSP from reference date          |
| spy_rsp_spread          | float64    | SPY cumulative return - RSP cumulative return (pp)    |
| buffett_indicator       | float64    | Total market cap / GDP * 100                          |

### 3.3 Buffett Indicator Table

| Column              | Dtype      | Description                                     |
|---------------------|------------|-------------------------------------------------|
| date                | datetime64 | Quarter end date                                |
| total_market_cap    | float64    | Total US equity market cap (USD, trillions)     |
| gdp                 | float64    | Quarterly GDP annualized (USD, trillions)       |
| buffett_indicator   | float64    | (total_market_cap / gdp) * 100                  |
| era_label           | str        | "dotcom_buildup", "dotcom_crash", "ai_era", etc.|

---

## 4. Preprocessing Steps

### 4.1 S&P 500 Weight Computation

```python
def compute_sp500_weights(constituent_mcaps: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Compute S&P 500 weight for each constituent on a given date.
    Weight = company market cap / sum of all constituent market caps.

    NOTE: The actual S&P 500 uses float-adjusted market cap (excluding
    insider-held shares). Our approximation uses total market cap.
    This introduces a small bias but is acceptable for concentration analysis.
    """
    date_data = constituent_mcaps[constituent_mcaps["date"] == date].copy()
    total_mcap = date_data["market_cap"].sum()
    date_data["sp500_weight"] = date_data["market_cap"] / total_mcap * 100
    date_data = date_data.sort_values("market_cap", ascending=False)
    date_data["rank"] = range(1, len(date_data) + 1)
    date_data["is_top_10"] = date_data["rank"] <= 10
    return date_data
```

### 4.2 AI-Linked Stock Flagging

```python
AI_LINKED_TICKERS = {
    "NVDA", "AMD", "AVGO", "MSFT", "GOOGL", "GOOG", "META", "AMZN",
    "AAPL", "ORCL", "TSM", "SMCI", "ARM", "MRVL", "SNPS", "CDNS",
    "PLTR", "CRM", "NOW", "ADBE"
}

def flag_ai_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """Flag stocks that are AI-linked for concentration sub-analysis."""
    df["is_ai_linked"] = df["ticker"].isin(AI_LINKED_TICKERS)
    return df
```

**Note on AI definition:** The boundary of "AI stock" is subjective. We use a conservative core list (NVDA, AMD, AVGO, MSFT, GOOGL, META, AMZN) for the primary metric and an extended list for sensitivity analysis. Document the list explicitly in the methodology.

### 4.3 HHI Computation

The Herfindahl-Hirschman Index measures market concentration. For the S&P 500, it is the sum of squared weights.

```python
def compute_hhi(weights: pd.Series) -> float:
    """
    Compute HHI from a series of market share percentages.

    HHI ranges:
        - 10,000 = perfect monopoly (one stock = 100%)
        - 20 = perfectly equal (500 stocks at 0.2% each -> 500 * 0.04 = 20)
        - <100 = highly competitive / diversified
        - >250 = moderately concentrated
        - >1000 = highly concentrated

    For the S&P 500, weights are in percentage (0-100) form.
    """
    # Weights should be in percentage form (e.g., 7.5 for 7.5%)
    return (weights ** 2).sum()
```

**Historical reference points:**
- S&P 500 HHI in 1990s (broad market): ~60-80
- S&P 500 HHI at dot-com peak (2000): ~120-150
- S&P 500 HHI current (2025-2026): ~200-250 (driven by Mag 7)
- These are estimates; actual computation will use real data

### 4.4 SPY vs RSP Spread Computation

RSP (equal-weight S&P 500 ETF) only has data from April 2003, so the direct comparison only covers the current era. For the dot-com era, we must construct a synthetic equal-weight index.

```python
def compute_spy_rsp_spread(spy_prices: pd.Series, rsp_prices: pd.Series,
                            start_date: str) -> pd.DataFrame:
    """
    Compute cumulative return spread between cap-weighted and equal-weighted S&P 500.
    A widening spread indicates increasing concentration advantage.
    """
    # Normalize both to 100 at start_date
    spy_norm = spy_prices / spy_prices.loc[start_date] * 100
    rsp_norm = rsp_prices / rsp_prices.loc[start_date] * 100

    spread = pd.DataFrame({
        "spy_cum_return": (spy_norm / 100 - 1) * 100,
        "rsp_cum_return": (rsp_norm / 100 - 1) * 100,
        "spread": (spy_norm - rsp_norm)  # In index points
    })

    return spread
```

**Synthetic equal-weight for dot-com era:**

```python
def synthetic_equal_weight_sp500(constituent_prices: pd.DataFrame,
                                  rebalance_freq: str = "QS") -> pd.Series:
    """
    Construct a synthetic equal-weight S&P 500 index for the dot-com era.

    Strategy: At each rebalance date, assign equal weight (1/N) to all
    constituents with available price data. Between rebalances, let
    weights drift with price changes.

    Parameters:
        constituent_prices: DataFrame with columns = tickers, index = dates, values = adjusted close
        rebalance_freq: Rebalance frequency ('QS' = quarterly, 'MS' = monthly)
    """
    returns = constituent_prices.pct_change()
    rebalance_dates = pd.date_range(
        start=constituent_prices.index[0],
        end=constituent_prices.index[-1],
        freq=rebalance_freq
    )

    portfolio_value = [100.0]  # Start at 100

    for i in range(1, len(constituent_prices)):
        date = constituent_prices.index[i]
        daily_returns = returns.iloc[i].dropna()
        n_stocks = len(daily_returns)

        if n_stocks == 0:
            portfolio_value.append(portfolio_value[-1])
            continue

        # Equal weight: 1/N for each available stock
        equal_weight_return = daily_returns.mean()
        portfolio_value.append(portfolio_value[-1] * (1 + equal_weight_return))

    return pd.Series(portfolio_value, index=constituent_prices.index, name="EW_SP500")
```

### 4.5 Buffett Indicator Computation

```python
def compute_buffett_indicator(total_market_cap: pd.Series,
                                gdp: pd.Series) -> pd.DataFrame:
    """
    Compute Buffett Indicator (total market cap / GDP).

    GDP is quarterly; market cap is daily (or quarterly).
    Forward-fill GDP to align with market cap dates.

    Interpretation:
        < 80%: Undervalued
        80-100%: Fairly valued
        100-120%: Moderately overvalued
        > 120%: Significantly overvalued
        > 150%: Extreme overvaluation / bubble territory

    Historical reference points:
        - Dot-com peak (Mar 2000): ~145%
        - GFC trough (Mar 2009): ~57%
        - Pre-COVID (Feb 2020): ~152%
        - Current (Mar 2026): ~195% (estimated)
    """
    # Resample GDP to daily by forward-filling
    gdp_daily = gdp.resample("D").ffill()

    # Annualize quarterly GDP (GDP series is already annualized in FRED)
    # If using quarterly GDP that is NOT annualized, multiply by 4
    buffett = (total_market_cap / gdp_daily * 100).dropna()

    return pd.DataFrame({
        "buffett_indicator": buffett,
        "total_market_cap": total_market_cap.reindex(buffett.index, method="ffill"),
        "gdp": gdp_daily.reindex(buffett.index)
    })
```

---

## 5. Analysis Plan

### 5.1 Top-10 Concentration Ratio Over Time

**Method:** For each quarter from 1998 to 2026, compute the sum of the top-10 S&P 500 constituents' weights.

```python
def top_n_concentration_timeseries(constituent_weights: pd.DataFrame,
                                     n: int = 10,
                                     freq: str = "QS") -> pd.Series:
    """
    Compute top-N concentration at each rebalance date.

    Parameters:
        constituent_weights: DataFrame with columns [date, ticker, sp500_weight]
        n: Number of top stocks to sum
        freq: Frequency of measurement points
    """
    dates = constituent_weights["date"].unique()
    results = {}

    for date in sorted(dates):
        snapshot = constituent_weights[constituent_weights["date"] == date]
        top_n_weight = snapshot.nlargest(n, "sp500_weight")["sp500_weight"].sum()
        results[date] = top_n_weight

    return pd.Series(results, name=f"top_{n}_concentration")
```

**Expected findings:**
- Dot-com peak top-10 concentration: ~25-27% of S&P 500
- Current top-10 concentration: ~35-38% (historically unprecedented)
- The current concentration is more extreme than the dot-com era

### 5.2 AI-Specific Concentration

```python
def ai_concentration_timeseries(constituent_weights: pd.DataFrame) -> pd.Series:
    """Compute total weight of AI-linked stocks in S&P 500 over time."""
    dates = constituent_weights["date"].unique()
    results = {}

    for date in sorted(dates):
        snapshot = constituent_weights[constituent_weights["date"] == date]
        ai_weight = snapshot[snapshot["is_ai_linked"]]["sp500_weight"].sum()
        results[date] = ai_weight

    return pd.Series(results, name="ai_concentration")
```

**Expected findings:**
- AI-linked stocks currently represent ~30-35% of the S&P 500
- During the dot-com era, "internet-linked" stocks represented ~20-25%
- The current AI concentration exceeds the dot-com internet concentration

### 5.3 SPY vs RSP Spread Analysis

**Method:** Compare cumulative returns of cap-weighted (SPY) vs equal-weighted (RSP) S&P 500.

**Interpretation:**
- When SPY outperforms RSP, large-cap stocks are leading (narrow market, concentration advantage)
- When RSP outperforms SPY, the rally is broad-based (healthy market breadth)
- Sustained SPY > RSP spread is a warning sign of concentration risk

```python
# For the period 2023-2026
spread = compute_spy_rsp_spread(spy_daily, rsp_daily, start_date="2023-01-01")

# Key metric: maximum spread during the period
max_spread = spread["spread"].max()
current_spread = spread["spread"].iloc[-1]
```

### 5.4 Buffett Indicator Analysis

Compare the Buffett Indicator at key dates:

| Date         | Context                    | Expected Buffett Indicator |
|--------------|----------------------------|---------------------------|
| Jan 1998     | Pre dot-com rally          | ~110%                     |
| Mar 2000     | Dot-com peak               | ~145%                     |
| Oct 2002     | Dot-com trough             | ~72%                      |
| Oct 2007     | Pre-GFC peak               | ~110%                     |
| Mar 2009     | GFC trough                 | ~57%                      |
| Feb 2020     | Pre-COVID                  | ~152%                     |
| Mar 2020     | COVID trough               | ~117%                     |
| Jan 2023     | Pre-AI rally               | ~150%                     |
| Mar 2026     | Current                    | ~195% (estimated)         |

### 5.5 HHI Index Over Time

```python
def hhi_timeseries(constituent_weights: pd.DataFrame) -> pd.Series:
    """Compute HHI at each observation date."""
    dates = constituent_weights["date"].unique()
    results = {}

    for date in sorted(dates):
        snapshot = constituent_weights[constituent_weights["date"] == date]
        results[date] = compute_hhi(snapshot["sp500_weight"])

    return pd.Series(results, name="hhi")
```

### 5.6 Concentration vs Subsequent Drawdown Correlation

**Key analysis:** Does high concentration predict future drawdowns?

```python
def concentration_drawdown_correlation(concentration_ts: pd.Series,
                                         sp500_prices: pd.Series,
                                         forward_windows: list[int] = [90, 180, 365]) -> pd.DataFrame:
    """
    For each observation date, compute:
        - Concentration level on that date
        - Forward max drawdown over the next N days

    Then compute correlation between concentration and forward drawdown.
    """
    results = []

    for date, concentration in concentration_ts.items():
        for window in forward_windows:
            future_end = date + pd.Timedelta(days=window)
            future_prices = sp500_prices.loc[date:future_end]

            if len(future_prices) < window * 0.5:  # Need at least half the window
                continue

            peak = future_prices.iloc[0]
            max_drawdown = (future_prices / peak - 1).min()

            results.append({
                "date": date,
                "concentration": concentration,
                "forward_window_days": window,
                "forward_max_drawdown": max_drawdown
            })

    df = pd.DataFrame(results)

    # Compute correlation for each window
    for window in forward_windows:
        subset = df[df["forward_window_days"] == window]
        r, p = stats.pearsonr(subset["concentration"], subset["forward_max_drawdown"])
        print(f"Window={window}d: Pearson r={r:.3f}, p={p:.4f}")

    return df
```

**Expected finding:** Higher concentration is weakly correlated with larger subsequent drawdowns, but the relationship is noisy. This is an honest finding -- concentration is a necessary but not sufficient condition for a crash.

---

## 6. Visualizations

### Chart 3.1: Top-10 Concentration Ratio Timeline

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Area chart with line overlay                                 |
| X-axis          | Date (1998 to 2026)                                          |
| Y-axis          | Top-10 weight in S&P 500 (%)                                 |
| Area            | Filled area under the concentration line, gradient fill (light when low, dark when high) |
| Color           | Green gradient for current era, blue gradient for dot-com era |
| Annotations     | Vertical lines at dot-com peak (Mar 2000) and current date; horizontal line at all-time high concentration |
| Shading         | Red-tinted band above 30% labeled "Historically extreme concentration" |
| Secondary line  | Top-5 concentration (dashed) overlaid                        |
| Title           | "S&P 500 Top-10 Concentration: More Extreme Than Dot-Com"   |
| Figure size     | (16, 7)                                                      |
| Insight         | Immediately shows that current concentration exceeds dot-com levels |

```python
fig, ax = plt.subplots(figsize=(16, 7))

ax.fill_between(concentration_ts.index, concentration_ts.values, 0,
                color="#1f77b4", alpha=0.3)
ax.plot(concentration_ts.index, concentration_ts.values,
        color="#1f77b4", linewidth=2, label="Top-10 Concentration")
ax.plot(top5_ts.index, top5_ts.values,
        color="#ff7f0e", linewidth=1.5, linestyle="--", label="Top-5 Concentration")

ax.axhspan(30, 45, color="red", alpha=0.05, label="Historically extreme (>30%)")
ax.axvline(x=pd.Timestamp("2000-03-24"), color="gray", linestyle=":", alpha=0.7)
ax.annotate("Dot-com peak\nMar 2000", xy=(pd.Timestamp("2000-03-24"), 27),
            fontsize=9, ha="center")
ax.axvline(x=pd.Timestamp("2026-03-28"), color="red", linestyle=":", alpha=0.7)
ax.annotate("Current\nMar 2026", xy=(pd.Timestamp("2026-03-28"), 37),
            fontsize=9, ha="center")

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Top-10 Weight in S&P 500 (%)", fontsize=12)
ax.set_title("S&P 500 Top-10 Concentration: More Extreme Than Dot-Com",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(15, 45)
```

### Chart 3.2: SPY vs RSP Cumulative Return Spread

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Dual-panel: top panel = cumulative returns, bottom panel = spread |
| X-axis          | Date (2003 to 2026, limited by RSP inception)               |
| Y-axis (top)    | Cumulative return (%)                                       |
| Y-axis (bottom) | Spread (SPY - RSP cumulative return, percentage points)      |
| Top panel lines | SPY (blue, solid), RSP (orange, solid)                      |
| Bottom panel    | Spread as area chart (green when SPY leads, red when RSP leads) |
| Annotations     | Mark the AI era (2023+) with vertical shading; mark dot-com era with vertical shading |
| Title           | "Cap-Weight vs Equal-Weight: The Concentration Premium"      |
| Figure size     | (16, 10)                                                     |
| Insight         | Shows that the current cap-weight advantage is historically extreme, indicating narrow market leadership |

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1], sharex=True)

# Top panel: cumulative returns
ax1.plot(spy_cum.index, spy_cum.values, color="#1f77b4", linewidth=1.5, label="SPY (Cap-Weighted)")
ax1.plot(rsp_cum.index, rsp_cum.values, color="#ff7f0e", linewidth=1.5, label="RSP (Equal-Weight)")
ax1.set_ylabel("Cumulative Return (%)")
ax1.legend(fontsize=10)
ax1.set_title("Cap-Weight vs Equal-Weight: The Concentration Premium",
              fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)

# Bottom panel: spread
spread = spy_cum - rsp_cum
ax2.fill_between(spread.index, spread.values, 0,
                 where=spread >= 0, color="green", alpha=0.3, label="SPY leads")
ax2.fill_between(spread.index, spread.values, 0,
                 where=spread < 0, color="red", alpha=0.3, label="RSP leads")
ax2.axhline(y=0, color="black", linewidth=0.8)
ax2.set_xlabel("Date")
ax2.set_ylabel("Spread (pp)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
```

### Chart 3.3: Buffett Indicator Historical with Current Marker

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Line chart with colored zones                                |
| X-axis          | Date (1997 to 2026)                                          |
| Y-axis          | Buffett Indicator (%)                                        |
| Line            | Dark navy, linewidth=2                                       |
| Background zones| Green (<80% -- undervalued), Yellow (80-120% -- fair), Orange (120-150% -- overvalued), Red (>150% -- extreme) |
| Annotations     | Mark dot-com peak, GFC trough, COVID peak, current value with labeled points |
| Current marker  | Large red dot at current date with value label               |
| Title           | "Buffett Indicator: Total Market Cap / GDP"                  |
| Subtitle        | "Warren Buffett's preferred valuation metric for the total market" |
| Figure size     | (16, 8)                                                      |
| Insight         | Shows the current market is at or near all-time-high valuations relative to GDP |

```python
fig, ax = plt.subplots(figsize=(16, 8))

# Background zones
ax.axhspan(0, 80, color="green", alpha=0.08, label="Undervalued (<80%)")
ax.axhspan(80, 120, color="yellow", alpha=0.08, label="Fair Value (80-120%)")
ax.axhspan(120, 150, color="orange", alpha=0.08, label="Overvalued (120-150%)")
ax.axhspan(150, 250, color="red", alpha=0.08, label="Extreme (>150%)")

ax.plot(buffett_ts.index, buffett_ts.values, color="navy", linewidth=2)

# Annotate key points
key_points = {
    "2000-03-24": ("Dot-com peak\n~145%", 145),
    "2009-03-09": ("GFC trough\n~57%", 57),
    "2020-02-19": ("Pre-COVID\n~152%", 152),
    "2026-03-28": ("CURRENT\n~195%", 195),
}
for date_str, (label, y_approx) in key_points.items():
    date = pd.Timestamp(date_str)
    ax.annotate(label, xy=(date, y_approx), fontsize=9, ha="center",
                fontweight="bold" if "CURRENT" in label else "normal",
                color="red" if "CURRENT" in label else "black",
                arrowprops=dict(arrowstyle="->", color="gray"))

ax.scatter([pd.Timestamp("2026-03-28")], [195], color="red", s=150, zorder=5)

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Buffett Indicator (Total Market Cap / GDP, %)", fontsize=12)
ax.set_title("Buffett Indicator: Total Market Cap / GDP", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(40, 220)
```

### Chart 3.4: Treemap of S&P 500 by Market Cap (Current)

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Treemap (squarified)                                         |
| Size            | Proportional to market cap                                   |
| Color           | Gradient by sector (Tech = blue, Healthcare = green, Finance = purple, etc.) |
| Labels          | Show ticker + weight% for stocks > 1% weight; only ticker for 0.5-1% |
| Highlighting    | Bold border around AI-linked stocks                          |
| Title           | "S&P 500 Today: How Much is AI?"                             |
| Figure size     | (16, 12)                                                     |
| Library         | `squarify` or `plotly.express.treemap`                       |
| Insight         | Visceral visual showing how a few AI stocks dominate the index |

```python
import squarify
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(16, 12))

# Prepare data: top 50 stocks (rest grouped as "Other")
top50 = sp500_weights.nlargest(50, "market_cap")
other_weight = 100 - top50["sp500_weight"].sum()
top50 = pd.concat([top50, pd.DataFrame([{
    "ticker": "OTHER\n(450 stocks)",
    "sp500_weight": other_weight,
    "sector": "Other",
    "is_ai_linked": False
}])], ignore_index=True)

# Color by sector
sector_colors = {
    "Technology": "#1f77b4",
    "Communication Services": "#2ca02c",
    "Consumer Discretionary": "#ff7f0e",
    "Healthcare": "#d62728",
    "Financials": "#9467bd",
    "Other": "#cccccc"
}
colors = [sector_colors.get(s, "#cccccc") for s in top50["sector"]]

# Plot treemap
squarify.plot(
    sizes=top50["sp500_weight"],
    label=[f"{t}\n{w:.1f}%" if w > 1 else t
           for t, w in zip(top50["ticker"], top50["sp500_weight"])],
    color=colors, alpha=0.85, ax=ax,
    text_kwargs={"fontsize": 8, "fontweight": "bold"}
)

ax.set_title("S&P 500 Today: How Much is AI?", fontsize=16, fontweight="bold")
ax.axis("off")
```

### Chart 3.5: HHI Index Over Time

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Line chart                                                   |
| X-axis          | Date (1998 to 2026)                                          |
| Y-axis          | HHI of S&P 500 (unitless)                                    |
| Line            | Dark blue, linewidth=2                                       |
| Reference bands | Shade below 100 as "well diversified", 100-200 as "moderately concentrated", above 200 as "highly concentrated" |
| Annotations     | Label dot-com peak HHI and current HHI                       |
| Title           | "S&P 500 Concentration (HHI): Are We in Uncharted Territory?" |
| Figure size     | (14, 7)                                                      |
| Insight         | Quantifies concentration using a standardized economic metric |

```python
fig, ax = plt.subplots(figsize=(14, 7))

ax.axhspan(0, 100, color="green", alpha=0.05, label="Well Diversified (<100)")
ax.axhspan(100, 200, color="yellow", alpha=0.05, label="Moderate Concentration (100-200)")
ax.axhspan(200, 400, color="red", alpha=0.05, label="High Concentration (>200)")

ax.plot(hhi_ts.index, hhi_ts.values, color="darkblue", linewidth=2)

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Herfindahl-Hirschman Index (HHI)", fontsize=12)
ax.set_title("S&P 500 Concentration (HHI): Are We in Uncharted Territory?",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
```

---

## 7. Statistical Tests

### 7.1 Cross-Era Concentration Comparison

**Test:** Are concentration levels today statistically different from the dot-com era?

```python
# Compare quarterly concentration ratios across eras
dotcom_conc = concentration_ts.loc["1998":"2002"]  # 20 quarterly observations
current_conc = concentration_ts.loc["2023":"2026"]  # 13 quarterly observations

# Mann-Whitney U test (non-parametric, appropriate for small samples)
u_stat, p_value = stats.mannwhitneyu(
    current_conc.values,
    dotcom_conc.values,
    alternative="greater"  # One-sided: current > dotcom
)
print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_value:.4f}")
print(f"Current mean: {current_conc.mean():.1f}%, Dot-com mean: {dotcom_conc.mean():.1f}%")

# Bootstrap confidence interval for the difference
from scipy.stats import bootstrap
diff = current_conc.mean() - dotcom_conc.mean()
print(f"Mean difference: {diff:.1f} percentage points")
```

### 7.2 Correlation Between Concentration and Subsequent Drawdowns

```python
# Compute for each quarter: concentration level + max drawdown in next 12 months
def concentration_predicts_drawdown(concentration_quarterly: pd.Series,
                                      sp500_daily: pd.Series) -> dict:
    results = []
    for date, conc in concentration_quarterly.items():
        future_start = date
        future_end = date + pd.DateOffset(months=12)
        future_prices = sp500_daily.loc[future_start:future_end]

        if len(future_prices) < 60:  # Need at least 60 trading days
            continue

        peak = future_prices.iloc[0]
        max_dd = (future_prices / peak - 1).min() * 100  # In percentage

        results.append({"date": date, "concentration": conc, "forward_12m_max_dd": max_dd})

    df = pd.DataFrame(results)
    r, p = stats.pearsonr(df["concentration"], df["forward_12m_max_dd"])

    return {
        "pearson_r": r,
        "p_value": p,
        "data": df,
        "interpretation": (
            "Negative correlation means higher concentration is associated with "
            "larger future drawdowns (more negative drawdown values)."
        )
    }
```

### 7.3 Granger Causality Test

Does concentration "Granger-cause" subsequent market drawdowns?

```python
from statsmodels.tsa.stattools import grangercausalitytests

def test_granger_causality(concentration_ts: pd.Series,
                            drawdown_ts: pd.Series,
                            maxlag: int = 4) -> dict:
    """
    Test if concentration changes Granger-cause drawdown changes.
    Both series should be quarterly and stationary.
    """
    # First, make stationary by differencing
    conc_diff = concentration_ts.diff().dropna()
    dd_diff = drawdown_ts.diff().dropna()

    # Align indices
    aligned = pd.concat([dd_diff, conc_diff], axis=1).dropna()
    aligned.columns = ["drawdown_change", "concentration_change"]

    # Granger test: does concentration change -> drawdown change?
    results = grangercausalitytests(aligned, maxlag=maxlag, verbose=True)

    return results
```

---

## 8. Limitations

| Limitation                                  | Impact                                               | Mitigation                                     |
|---------------------------------------------|------------------------------------------------------|------------------------------------------------|
| S&P 500 composition changed significantly   | Comparing 2000 S&P to 2025 S&P is apples-to-oranges | Acknowledge; use top-N approach rather than exact composition matching |
| RSP data only from 2003                     | No direct equal-weight comparison for dot-com era    | Construct synthetic equal-weight index from constituent prices |
| Market cap data quality for 1998-2002       | Historical daily market caps may be approximate      | Use multiple sources; cross-validate with known benchmarks |
| "AI-linked" definition is subjective        | Different definitions yield different concentration  | Provide results for core (7 stocks) and extended (20 stocks) definitions |
| Buffett Indicator has structural biases     | Multinational company revenue is global; GDP is domestic | Note the limitation; indicator is still directionally useful |
| HHI assumes S&P 500 is the "market"        | S&P 500 is only ~80% of total US market cap          | Note; Wilshire 5000 concentration would be more accurate but harder to compute |
| Float-adjusted vs total market cap          | S&P uses float-adjusted; we use total market cap     | Note the methodology difference; directionally equivalent for concentration trends |
| Small sample size for cross-era comparison  | Only ~20 quarterly observations per era              | Use non-parametric tests (Mann-Whitney U); acknowledge low statistical power |
| Survivorship bias in index composition      | S&P 500 only includes current survivors              | Historical constituent lists from S&P would be ideal but are paywalled |
| Correlation != causation                    | Concentration may correlate with drawdowns but not cause them | Explicitly state in narrative; use Granger test as directional evidence only |

---

## 9. Code Outline (Main Pipeline)

```python
# ============================================================
# Layer 3: Market Concentration & Systemic Risk Pipeline
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from scipy import stats
import matplotlib.pyplot as plt
import squarify

# --- CONFIGURATION ---
FRED_API_KEY = os.environ["FRED_API_KEY"]
FMP_API_KEY = os.environ["FMP_API_KEY"]

# --- STEP 1: Ingest ETF/Index Data ---
def ingest_etf_data() -> dict[str, pd.DataFrame]:
    """Pull daily prices for SPY, RSP, ^GSPC, XLK, SMH, SOXX, QQQ."""
    etf_tickers = ["SPY", "RSP", "^GSPC", "XLK", "SMH", "SOXX", "QQQ"]
    data = {}
    for ticker in etf_tickers:
        df = yf.Ticker(ticker).history(start="1997-01-01", end="2026-03-28",
                                        auto_adjust=True)
        df["ticker"] = ticker
        data[ticker] = df
    return data

# --- STEP 2: Get S&P 500 Constituent List ---
def get_sp500_constituents() -> pd.DataFrame:
    """Scrape current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    constituents = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]
    constituents.columns = ["ticker", "company", "sector", "industry"]
    constituents["ticker"] = constituents["ticker"].str.replace(".", "-")
    return constituents

# --- STEP 3: Compute Current Market Caps ---
def compute_current_constituent_weights() -> pd.DataFrame:
    """Get market cap and compute weight for each S&P 500 constituent."""
    constituents = get_sp500_constituents()
    mcaps = get_current_market_caps(constituents["ticker"].tolist())

    merged = constituents.merge(mcaps, on="ticker", how="left")
    merged = merged.dropna(subset=["market_cap"])
    total_mcap = merged["market_cap"].sum()
    merged["sp500_weight"] = merged["market_cap"] / total_mcap * 100
    merged = flag_ai_stocks(merged)
    merged = merged.sort_values("market_cap", ascending=False)
    merged["rank"] = range(1, len(merged) + 1)
    merged["is_top_10"] = merged["rank"] <= 10

    return merged

# --- STEP 4: Compute Historical Concentration (Dot-Com Era) ---
def compute_dotcom_concentration() -> pd.DataFrame:
    """
    Approximate S&P 500 concentration for 1998-2002.
    Use known top stocks + FMP historical market cap.
    """
    dotcom_top_tickers = ["MSFT", "GE", "CSCO", "INTC", "WMT", "XOM",
                          "IBM", "C", "QCOM", "ORCL", "LU", "JNJ",
                          "PFE", "AIG", "MRK"]

    quarterly_dates = pd.date_range("1998-01-01", "2002-12-31", freq="QS")
    results = []

    for date in quarterly_dates:
        date_str = date.strftime("%Y-%m-%d")
        snapshot = get_historical_market_caps_batch(dotcom_top_tickers, date_str)
        # Approximate total S&P 500 market cap from ^GSPC level
        # (use known relationship: S&P 500 level * multiplier)
        # Better: use FMP or FRED total market cap data
        results.append(snapshot)

    return pd.concat(results, ignore_index=True)

# --- STEP 5: Compute FRED Macro Data ---
def compute_buffett_indicator_ts() -> pd.DataFrame:
    """Pull Wilshire 5000 and GDP from FRED, compute Buffett Indicator."""
    fred = Fred(api_key=FRED_API_KEY)

    # Total market cap proxy
    wilshire = fred.get_series("WILL5000IND",
                                observation_start="1997-01-01",
                                observation_end="2026-03-28")

    # GDP (quarterly, seasonally adjusted annual rate)
    gdp = fred.get_series("GDP",
                           observation_start="1997-01-01",
                           observation_end="2026-03-28")

    # Alternative: use total market cap series if available
    try:
        total_mcap = fred.get_series("BOGZ1FL073164005Q",
                                      observation_start="1997-01-01",
                                      observation_end="2026-03-28")
    except Exception:
        # Fallback: use Wilshire 5000 with scaling factor
        # Index level * ~1.2B per point (approximate as of recent data)
        total_mcap = wilshire * 1.2e9 / 1e12  # In trillions

    buffett = compute_buffett_indicator(total_mcap, gdp / 1e12)
    return buffett

# --- STEP 6: Compute All Concentration Metrics ---
def compute_all_concentration_metrics(
    current_weights: pd.DataFrame,
    dotcom_weights: pd.DataFrame,
    spy_prices: pd.Series,
    rsp_prices: pd.Series,
    buffett_ts: pd.DataFrame
) -> pd.DataFrame:
    """Combine all concentration metrics into a single time series."""

    metrics = {
        "top10_concentration": top_n_concentration_timeseries(
            pd.concat([current_weights, dotcom_weights]), n=10
        ),
        "top5_concentration": top_n_concentration_timeseries(
            pd.concat([current_weights, dotcom_weights]), n=5
        ),
        "ai_concentration": ai_concentration_timeseries(current_weights),
        "hhi": hhi_timeseries(pd.concat([current_weights, dotcom_weights])),
    }

    # SPY vs RSP spread (2003+)
    if rsp_prices is not None:
        spread = compute_spy_rsp_spread(spy_prices, rsp_prices, start_date="2003-04-30")
        metrics["spy_rsp_spread"] = spread["spread"]

    # Buffett Indicator
    metrics["buffett_indicator"] = buffett_ts["buffett_indicator"]

    return metrics

# --- STEP 7: Run Statistical Tests ---
def run_concentration_stats(metrics: dict, sp500_daily: pd.Series) -> dict:
    """Execute cross-era comparison and predictive correlation tests."""
    results = {}

    # 7.1 Cross-era concentration comparison (Mann-Whitney U)
    conc_ts = metrics["top10_concentration"]
    dotcom_conc = conc_ts.loc["1998":"2002"]
    current_conc = conc_ts.loc["2023":"2026"]

    if len(dotcom_conc) > 0 and len(current_conc) > 0:
        results["mannwhitney_u"], results["mannwhitney_p"] = stats.mannwhitneyu(
            current_conc.values, dotcom_conc.values, alternative="greater"
        )
        results["current_conc_mean"] = current_conc.mean()
        results["dotcom_conc_mean"] = dotcom_conc.mean()

    # 7.2 Concentration vs subsequent drawdown correlation
    results["conc_drawdown"] = concentration_predicts_drawdown(
        conc_ts.resample("QS").mean(), sp500_daily
    )

    # 7.3 Granger causality
    try:
        drawdown_quarterly = sp500_daily.resample("QS").apply(
            lambda x: (x / x.expanding().max() - 1).min()
        )
        results["granger"] = test_granger_causality(
            conc_ts.resample("QS").mean(),
            drawdown_quarterly,
            maxlag=4
        )
    except Exception as e:
        results["granger_error"] = str(e)

    return results

# --- STEP 8: Generate Visualizations ---
def generate_concentration_charts(metrics: dict,
                                    current_weights: pd.DataFrame,
                                    etf_data: dict):
    """Generate Charts 3.1 through 3.5."""
    chart_3_1_top10_concentration(metrics["top10_concentration"], metrics["top5_concentration"])
    chart_3_2_spy_rsp_spread(etf_data["SPY"], etf_data["RSP"])
    chart_3_3_buffett_indicator(metrics["buffett_indicator"])
    chart_3_4_sp500_treemap(current_weights)
    chart_3_5_hhi_timeline(metrics["hhi"])

# --- MAIN ---
def main():
    # Step 1: ETF data
    etf_data = ingest_etf_data()

    # Step 2-3: Current S&P 500 weights
    current_weights = compute_current_constituent_weights()

    # Step 4: Historical (dot-com) weights
    dotcom_weights = compute_dotcom_concentration()

    # Step 5: FRED macro data
    buffett_ts = compute_buffett_indicator_ts()

    # Step 6: Compute all metrics
    spy_prices = etf_data["SPY"]["Close"]
    rsp_prices = etf_data.get("RSP", {}).get("Close", None)

    metrics = compute_all_concentration_metrics(
        current_weights, dotcom_weights, spy_prices, rsp_prices, buffett_ts
    )

    # Step 7: Statistical tests
    sp500_daily = etf_data["^GSPC"]["Close"]
    stats_results = run_concentration_stats(metrics, sp500_daily)

    # Step 8: Charts
    generate_concentration_charts(metrics, current_weights, etf_data)

    # Save
    current_weights.to_parquet("data/processed/layer3_current_weights.parquet")
    save_metrics_to_parquet(metrics, "data/processed/layer3_concentration_metrics.parquet")
    save_stats_report(stats_results, "output/layer3_stats.json")

    # Print summary
    print("\n" + "="*60)
    print("LAYER 3 SUMMARY")
    print("="*60)
    print(f"Current top-10 concentration: {metrics['top10_concentration'].iloc[-1]:.1f}%")
    print(f"Dot-com peak top-10 concentration: {metrics['top10_concentration'].loc['1999':'2001'].max():.1f}%")
    print(f"Current AI-linked concentration: {metrics['ai_concentration'].iloc[-1]:.1f}%")
    print(f"Current HHI: {metrics['hhi'].iloc[-1]:.0f}")
    print(f"Buffett Indicator: {metrics['buffett_indicator'].iloc[-1]:.0f}%")
    if "mannwhitney_p" in stats_results:
        print(f"Concentration difference p-value: {stats_results['mannwhitney_p']:.4f}")

if __name__ == "__main__":
    main()
```

---

## 10. Output Artifacts

| Artifact                              | Path                                            | Format    |
|---------------------------------------|-------------------------------------------------|-----------|
| Current S&P 500 constituent weights   | `data/processed/layer3_current_weights.parquet`  | Parquet   |
| Historical concentration metrics      | `data/processed/layer3_concentration_metrics.parquet` | Parquet |
| Chart 3.1 top-10 concentration       | `output/charts/chart_3_1_concentration.png`      | PNG 300dpi|
| Chart 3.2 SPY vs RSP spread          | `output/charts/chart_3_2_spy_rsp.png`            | PNG 300dpi|
| Chart 3.3 Buffett Indicator          | `output/charts/chart_3_3_buffett.png`            | PNG 300dpi|
| Chart 3.4 S&P 500 treemap           | `output/charts/chart_3_4_treemap.png`            | PNG 300dpi|
| Chart 3.5 HHI timeline              | `output/charts/chart_3_5_hhi.png`                | PNG 300dpi|
| Statistical test results              | `output/layer3_stats.json`                       | JSON      |

---

## 11. Cross-Layer Integration Points

This layer connects to the other layers as follows:

| Integration                                | From Layer | To Layer | How                                                    |
|--------------------------------------------|------------|----------|--------------------------------------------------------|
| Breakout dates                             | Layer 1    | Layer 3  | Use Layer 1's breakout dates to define era boundaries for concentration analysis |
| "Is it a bubble?" verdict                  | Layer 2    | Layer 3  | Layer 2 says fundamentals are strong; Layer 3 asks "but is the market structure still dangerous?" |
| Combined risk score                        | All        | Presentation | Combine price trajectory similarity (Layer 1), fundamental strength (Layer 2), and concentration risk (Layer 3) into a qualitative "bubble risk dashboard" |
| Narrative arc                              | All        | Presentation | Layer 1: "The price trajectories look similar." Layer 2: "But the fundamentals are different." Layer 3: "However, the systemic concentration risk may be even worse this time." |
