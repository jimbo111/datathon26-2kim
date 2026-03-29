# Layer 1: Historical Price Overlay -- NVDA vs CSCO

## 1. Objective

**Primary question:** Does NVIDIA's price trajectory from 2023-2026 structurally mirror Cisco's trajectory from 1998-2001, and if so, where are we on the curve?

**Sub-questions this layer answers:**
- How closely correlated are the two normalized price curves?
- Is the magnitude and velocity of NVDA's rally comparable to CSCO's dot-com run?
- Are drawdown patterns showing early warning signs that match CSCO's pre-crash behavior?
- How do rolling return profiles compare -- is NVDA's run "hotter" or "cooler" than CSCO's equivalent phase?
- Is volume behavior (euphoria spikes) similar across both eras?

**Scoring alignment:** This layer directly targets Analysis & Evidence (25pts) and Technical Rigor (15pts) by producing statistically grounded visual comparisons with quantified correlation metrics.

---

## 2. Data Sources

### 2.1 Current AI Cycle Stocks

| Ticker | Name                  | Role in Analysis         | Date Range          |
|--------|-----------------------|--------------------------|---------------------|
| NVDA   | NVIDIA Corp           | Primary subject          | 2023-01-01 to 2026-03-28 |
| AMD    | Advanced Micro Devices| AI semiconductor peer    | 2023-01-01 to 2026-03-28 |
| SMCI   | Super Micro Computer  | AI infrastructure        | 2023-01-01 to 2026-03-28 |
| AVGO   | Broadcom Inc          | AI networking/custom ASIC| 2023-01-01 to 2026-03-28 |
| ARM    | Arm Holdings          | AI chip architecture     | 2023-09-14 (IPO) to 2026-03-28 |
| MSFT   | Microsoft Corp        | AI platform (OpenAI investor) | 2023-01-01 to 2026-03-28 |

### 2.2 Dot-Com Cycle Stocks

| Ticker | Name                   | Role in Analysis         | Date Range          |
|--------|------------------------|--------------------------|---------------------|
| CSCO   | Cisco Systems          | Primary subject          | 1998-01-01 to 2002-12-31 |
| JNPR   | Juniper Networks       | Networking peer          | 1999-06-25 (IPO) to 2002-12-31 |
| QCOM   | Qualcomm Inc           | Semiconductor peer       | 1998-01-01 to 2002-12-31 |
| INTC   | Intel Corp             | Semiconductor incumbent  | 1998-01-01 to 2002-12-31 |
| SUNW   | Sun Microsystems       | Infrastructure (use JAVA after ticker change, or COMS as proxy if unavailable) | 1998-01-01 to 2002-12-31 |

**Note on SUNW:** Sun Microsystems traded under SUNW until 2007 when it became JAVA. yfinance may not have SUNW data. Fallback: use `JAVA` ticker and filter to the date range 1998-2002. If unavailable, substitute with `DELL` (Dell Technologies, traded as DELL in that era).

### 2.3 Market Indices

| Ticker | Name                | Purpose                          | Date Range (both eras) |
|--------|---------------------|----------------------------------|------------------------|
| ^IXIC  | NASDAQ Composite    | Tech-heavy benchmark             | Both eras              |
| QQQ    | Invesco QQQ Trust   | NASDAQ-100 ETF (inception 1999-03-10) | Both eras (note: no QQQ data before 1999-03) |
| ^GSPC  | S&P 500 Index       | Broad market benchmark for beta  | Both eras              |

### 2.4 API Configuration

```python
import yfinance as yf

# Primary data pull function
def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Parameters:
        ticker: Stock ticker symbol
        start: "YYYY-MM-DD" format
        end: "YYYY-MM-DD" format
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
    """
    stock = yf.Ticker(ticker)
    df = stock.history(
        start=start,
        end=end,
        interval="1d",       # Daily frequency
        auto_adjust=True,    # Adjusts for splits and dividends automatically
        actions=True         # Include dividends and stock splits columns
    )
    return df
```

**Rate limiting:** yfinance does not enforce strict rate limits but batching is recommended. Use `yf.download()` for bulk pulls:

```python
# Bulk download for all AI-era tickers
ai_tickers = ["NVDA", "AMD", "SMCI", "AVGO", "ARM", "MSFT"]
ai_data = yf.download(
    tickers=ai_tickers,
    start="2023-01-01",
    end="2026-03-28",
    interval="1d",
    auto_adjust=True,
    group_by="ticker",
    threads=True
)

# Bulk download for all dot-com tickers
dotcom_tickers = ["CSCO", "JNPR", "QCOM", "INTC", "SUNW"]
dotcom_data = yf.download(
    tickers=dotcom_tickers,
    start="1998-01-01",
    end="2002-12-31",
    interval="1d",
    auto_adjust=True,
    group_by="ticker",
    threads=True
)
```

---

## 3. Data Schema

### 3.1 Raw Data Schema (per ticker, daily)

| Column       | Dtype      | Description                                 |
|--------------|------------|---------------------------------------------|
| Date         | datetime64 | Trading date (index)                        |
| Open         | float64    | Opening price (split-adjusted)              |
| High         | float64    | Intraday high (split-adjusted)              |
| Low          | float64    | Intraday low (split-adjusted)               |
| Close        | float64    | Closing price (split-adjusted)              |
| Volume       | int64      | Shares traded                               |
| Dividends    | float64    | Dividend paid on date (0.0 if none)         |
| Stock Splits | float64    | Split ratio on date (0.0 if none)           |

### 3.2 Processed Data Schema (normalized)

| Column           | Dtype      | Description                                     |
|------------------|------------|-------------------------------------------------|
| date             | datetime64 | Trading date                                    |
| ticker           | str        | Ticker symbol                                   |
| era              | str        | "ai_cycle" or "dotcom_cycle"                    |
| close            | float64    | Adjusted close price                            |
| volume           | int64      | Raw volume                                      |
| days_from_breakout | int64    | Trading days since breakout point (Day 0)       |
| normalized_price | float64    | Price indexed to 100 at breakout                |
| daily_return     | float64    | (close_t / close_{t-1}) - 1                     |
| log_return       | float64    | ln(close_t / close_{t-1})                       |
| cumulative_return| float64    | Cumulative return from breakout                 |
| drawdown         | float64    | Current drawdown from running max (negative %)  |
| rolling_30d_ret  | float64    | 30-trading-day rolling return                   |
| rolling_90d_ret  | float64    | 90-trading-day rolling return                   |
| rolling_252d_ret | float64    | 252-trading-day (1 year) rolling return         |
| volume_zscore    | float64    | Z-score of volume vs 60-day rolling mean/std    |

### 3.3 Weekly Resampled Schema

| Column           | Dtype      | Description                                     |
|------------------|------------|-------------------------------------------------|
| week_end_date    | datetime64 | Friday of each trading week                     |
| ticker           | str        | Ticker symbol                                   |
| era              | str        | "ai_cycle" or "dotcom_cycle"                    |
| close            | float64    | Last close of the week                          |
| weekly_high      | float64    | Highest intraday high of the week               |
| weekly_low       | float64    | Lowest intraday low of the week                 |
| weekly_volume    | int64      | Sum of daily volume for the week                |
| normalized_price | float64    | Price indexed to 100 at breakout (weekly close) |
| weekly_return    | float64    | Week-over-week return                           |

---

## 4. Preprocessing Steps

### 4.1 Split and Dividend Adjustment

yfinance `auto_adjust=True` handles this automatically by back-adjusting historical prices for:
- Stock splits (e.g., NVDA 10:1 split on June 10, 2024)
- Dividend payments

**Verification step:** After pulling data, verify the NVDA 10:1 split is properly adjusted by checking that the close price on 2024-06-07 (pre-split) and 2024-06-10 (post-split) are continuous (no 10x jump).

```python
# Verification
nvda_split_check = nvda_df.loc["2024-06-05":"2024-06-12", "Close"]
assert nvda_split_check.pct_change().abs().max() < 0.15, "Split not properly adjusted"
```

### 4.2 Missing Data Handling

**Strategy for each scenario:**

| Scenario                  | Handling                                              |
|---------------------------|-------------------------------------------------------|
| Market holidays           | No action needed -- these dates won't exist in data   |
| Trading halts (< 3 days)  | Forward-fill close price, set volume to 0             |
| Ticker not yet trading    | NaN until IPO date (ARM: 2023-09-14, JNPR: 1999-06-25)|
| Delisted tickers (SUNW)   | Use available data up to delisting, note in limitations|
| Sparse early data         | Drop ticker from analysis if <80% of expected trading days present |

```python
# Forward-fill gaps of up to 3 trading days
df["Close"] = df["Close"].fillna(method="ffill", limit=3)

# Flag any remaining NaNs
missing_pct = df["Close"].isna().sum() / len(df) * 100
if missing_pct > 5:
    print(f"WARNING: {ticker} has {missing_pct:.1f}% missing close prices")
```

### 4.3 Breakout Point Definition

The "breakout" is the alignment anchor that lets us overlay NVDA and CSCO on the same time axis. The definition must be systematic and reproducible.

**Definition:** The breakout point is the **first trading day** where the stock's trailing 252-day (1-year) return exceeds 50% AND this return stays above 50% for at least 20 consecutive trading days (sustained breakout, not a single-day spike).

```python
def find_breakout(df: pd.DataFrame, threshold: float = 0.50, sustain_days: int = 20) -> pd.Timestamp:
    """
    Find the first date where trailing 252-day return exceeds threshold
    and remains above threshold for at least sustain_days consecutive days.

    Parameters:
        df: DataFrame with 'Close' column and DatetimeIndex
        threshold: Minimum trailing 1-year return (0.50 = 50%)
        sustain_days: Minimum consecutive days above threshold

    Returns:
        Timestamp of breakout date
    """
    trailing_return = df["Close"].pct_change(periods=252)
    above_threshold = trailing_return > threshold

    # Find first sustained period
    consecutive = above_threshold.rolling(window=sustain_days).sum()
    breakout_candidates = consecutive[consecutive == sustain_days]

    if breakout_candidates.empty:
        raise ValueError(f"No sustained breakout found for threshold={threshold}")

    # The breakout day is sustain_days before the first confirmation day
    first_confirmation = breakout_candidates.index[0]
    breakout_day = df.index[df.index.get_loc(first_confirmation) - sustain_days + 1]

    return breakout_day
```

**Expected breakout dates (approximate):**
- **NVDA:** ~May-June 2023 (when the ChatGPT/AI demand became undeniable in price)
- **CSCO:** ~October-November 1998 (early internet infrastructure boom acceleration)
- These will be computed, not hardcoded. The above are sanity-check estimates.

### 4.4 Normalization to Index 100

```python
def normalize_to_breakout(df: pd.DataFrame, breakout_date: pd.Timestamp) -> pd.DataFrame:
    """
    Normalize prices so that breakout_date close = 100.
    Add days_from_breakout column (trading days, not calendar days).
    """
    breakout_price = df.loc[breakout_date, "Close"]
    df["normalized_price"] = (df["Close"] / breakout_price) * 100

    breakout_idx = df.index.get_loc(breakout_date)
    df["days_from_breakout"] = range(-breakout_idx, len(df) - breakout_idx)

    return df
```

### 4.5 Weekly Resampling

```python
def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily data to weekly (ending Friday).
    """
    weekly = df.resample("W-FRI").agg({
        "Close": "last",        # Friday close
        "High": "max",          # Weekly high
        "Low": "min",           # Weekly low
        "Volume": "sum",        # Total weekly volume
        "normalized_price": "last",
        "days_from_breakout": "last"
    }).dropna(subset=["Close"])

    weekly["weekly_return"] = weekly["Close"].pct_change()
    return weekly
```

---

## 5. Analysis Plan

### 5.1 Normalized Price Overlay (Primary Analysis)

**Method:** Align NVDA and CSCO (and peers) by `days_from_breakout`, plot normalized price curves on the same axis.

```python
# Merge on days_from_breakout
merged = pd.merge(
    nvda_norm[["days_from_breakout", "normalized_price"]].rename(columns={"normalized_price": "NVDA"}),
    csco_norm[["days_from_breakout", "normalized_price"]].rename(columns={"normalized_price": "CSCO"}),
    on="days_from_breakout",
    how="outer"
)
```

**Interpretation framework:**
- If NVDA tracks above CSCO at equivalent days, the AI rally is *more aggressive* than dot-com
- If NVDA tracks below, the rally is *more measured*
- If NVDA begins diverging downward after tracking closely, that is an early warning signal

### 5.2 Drawdown Analysis

**Definition:** Drawdown_t = (Price_t / Running_Max_t) - 1, where Running_Max_t = max(Price_0, ..., Price_t)

```python
def compute_drawdown(prices: pd.Series) -> pd.Series:
    """Compute drawdown series from a price series."""
    running_max = prices.expanding().max()
    drawdown = (prices / running_max) - 1.0
    return drawdown
```

**Key comparisons:**
- Maximum drawdown during CSCO's crash: ~-89% (from ~$80 to ~$8.60)
- Current NVDA maximum drawdown from its all-time high
- Time-to-recovery patterns
- Frequency and depth of intermediate drawdowns during the *rally* phase (pre-peak)

### 5.3 Volume Profile Comparison

**Method:** Compute volume z-scores relative to a 60-day rolling window. Identify volume "euphoria spikes" (z > 2.0) and "capitulation spikes" during selloffs.

```python
def volume_zscore(volume: pd.Series, window: int = 60) -> pd.Series:
    rolling_mean = volume.rolling(window=window).mean()
    rolling_std = volume.rolling(window=window).std()
    return (volume - rolling_mean) / rolling_std
```

**Volume surge events to flag:**
- Days with volume z-score > 2.5 (extreme euphoria or panic)
- Whether these cluster around the same relative cycle points in both eras

### 5.4 Rolling Returns Comparison

Compute rolling returns for three windows and compare distributions:

| Window   | Trading Days | Purpose                                  |
|----------|-------------|------------------------------------------|
| 30-day   | 30          | Short-term momentum / noise              |
| 90-day   | 90          | Medium-term trend strength               |
| 252-day  | 252         | Annual return -- the "headline" number   |

```python
for window in [30, 90, 252]:
    df[f"rolling_{window}d_ret"] = df["Close"].pct_change(periods=window)
```

### 5.5 Sharpe Ratio Comparison

Annualized Sharpe using the risk-free rate of each era:

```python
def rolling_sharpe(returns: pd.Series, window: int = 252, risk_free_annual: float = 0.05) -> pd.Series:
    """
    Rolling annualized Sharpe ratio.

    Parameters:
        returns: Daily return series
        window: Rolling window in trading days
        risk_free_annual: Annualized risk-free rate (use ~5% for dot-com, ~4.5% for current)
    """
    rf_daily = (1 + risk_free_annual) ** (1/252) - 1
    excess = returns - rf_daily
    rolling_mean = excess.rolling(window=window).mean() * 252
    rolling_std = returns.rolling(window=window).std() * (252 ** 0.5)
    return rolling_mean / rolling_std
```

**Risk-free rates:**
- Dot-com era (1998-2002): Use 10-Year Treasury yield (~5.0-6.5%). Source: FRED series `DGS10`.
- AI era (2023-2026): Use 10-Year Treasury yield (~3.5-4.8%). Source: FRED series `DGS10`.

---

## 6. Visualizations

### Chart 1.1: Normalized Price Overlay (Dual-Era)

**This is the hero chart of the entire project.**

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Line chart                                                   |
| X-axis          | `days_from_breakout` (trading days since breakout), range: -60 to +780 |
| Y-axis          | Normalized price (100 = breakout day price), log scale       |
| Lines           | NVDA (solid, #76B900 -- NVIDIA green), CSCO (dashed, #049FD9 -- Cisco blue) |
| Peer lines      | AMD (dotted, thin, #ED1C24), QCOM (dotted, thin, #3253DC), alpha=0.4 |
| Annotations     | Vertical dashed line at Day 0 labeled "Breakout"; annotation at CSCO peak (Day ~500) labeled "CSCO Peak: Mar 2000"; annotation at NVDA current position |
| Title           | "AI Hype vs Dot-Com: Normalized Price Trajectories"          |
| Subtitle        | "Day 0 = first sustained 50%+ YoY gain. Prices indexed to 100." |
| Grid            | Light gray, major only                                       |
| Legend           | Upper left, outside plot area                                |
| Figure size     | (14, 8)                                                      |
| Shading         | Light red shading on the CSCO crash region (peak to -50%)    |
| Insight         | Shows whether NVDA is tracking CSCO's trajectory, ahead of it, or behind it |

```python
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(nvda_weekly["days_from_breakout"], nvda_weekly["normalized_price"],
        color="#76B900", linewidth=2.5, label="NVDA (2023-2026)", zorder=5)
ax.plot(csco_weekly["days_from_breakout"], csco_weekly["normalized_price"],
        color="#049FD9", linewidth=2.5, linestyle="--", label="CSCO (1998-2002)", zorder=4)

# Peer lines (thinner, more transparent)
for ticker, color in [("AMD", "#ED1C24"), ("QCOM", "#3253DC")]:
    peer_data = peers_weekly[peers_weekly["ticker"] == ticker]
    ax.plot(peer_data["days_from_breakout"], peer_data["normalized_price"],
            color=color, linewidth=1.0, linestyle=":", alpha=0.4, label=ticker)

ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7, label="Breakout (Day 0)")
ax.set_yscale("log")
ax.set_xlabel("Trading Days from Breakout", fontsize=12)
ax.set_ylabel("Normalized Price (100 = Breakout Day)", fontsize=12)
ax.set_title("AI Hype vs Dot-Com: Normalized Price Trajectories", fontsize=16, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

### Chart 1.2: Drawdown from All-Time High Comparison

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Area chart (filled below zero)                               |
| X-axis          | `days_from_breakout`                                         |
| Y-axis          | Drawdown percentage (0% at top, -90% at bottom)              |
| Areas           | NVDA drawdown (green fill, alpha=0.3), CSCO drawdown (blue fill, alpha=0.3) |
| Annotations     | Horizontal line at -20% labeled "Bear market threshold"; label CSCO max drawdown (-89%) |
| Title           | "Drawdown Comparison: How Far From the Peak?"                |
| Figure size     | (14, 6)                                                      |
| Insight         | Reveals whether NVDA has experienced any significant drawdowns vs CSCO's catastrophic crash |

```python
fig, ax = plt.subplots(figsize=(14, 6))

ax.fill_between(nvda_weekly["days_from_breakout"], nvda_weekly["drawdown"] * 100, 0,
                color="#76B900", alpha=0.3, label="NVDA Drawdown")
ax.fill_between(csco_weekly["days_from_breakout"], csco_weekly["drawdown"] * 100, 0,
                color="#049FD9", alpha=0.3, label="CSCO Drawdown")

ax.axhline(y=-20, color="red", linestyle="--", alpha=0.5, label="Bear Market (-20%)")
ax.set_xlabel("Trading Days from Breakout")
ax.set_ylabel("Drawdown from Running Peak (%)")
ax.set_title("Drawdown Comparison: How Far From the Peak?", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
```

### Chart 1.3: Rolling 90-Day Return Comparison

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Dual line chart with shared x-axis                           |
| X-axis          | `days_from_breakout`                                         |
| Y-axis          | 90-day rolling return (percentage)                           |
| Lines           | NVDA (green), CSCO (blue)                                    |
| Shading         | Shade area where CSCO 90d return went negative (crash onset) |
| Reference       | Horizontal line at 0%                                        |
| Title           | "Rolling 90-Day Returns: Momentum Comparison"                |
| Figure size     | (14, 6)                                                      |
| Insight         | Shows when CSCO's momentum broke down and whether NVDA shows similar deceleration patterns |

### Chart 1.4: Volume Surge Timeline

| Attribute       | Specification                                                |
|-----------------|--------------------------------------------------------------|
| Chart type      | Scatter plot overlaid on normalized price line               |
| X-axis          | `days_from_breakout`                                         |
| Y-axis (left)   | Normalized price                                            |
| Y-axis (right)  | Volume z-score                                              |
| Scatter points  | Only days with volume z-score > 2.0, sized by z-score magnitude, colored red (sell-off) or green (rally) based on daily return sign |
| Lines           | NVDA and CSCO normalized price in background (thin, alpha=0.5) |
| Title           | "Volume Euphoria Spikes: When Did the Crowd Rush In?"        |
| Figure size     | (14, 7)                                                      |
| Insight         | Identifies whether volume euphoria events cluster at similar cycle points |

---

## 7. Statistical Tests

### 7.1 Pearson and Spearman Correlation

**What:** Correlation between NVDA's and CSCO's normalized price curves over the overlapping `days_from_breakout` range.

```python
from scipy import stats

# Align the two series on days_from_breakout
overlap = merged.dropna(subset=["NVDA", "CSCO"])

pearson_r, pearson_p = stats.pearsonr(overlap["NVDA"], overlap["CSCO"])
spearman_r, spearman_p = stats.spearmanr(overlap["NVDA"], overlap["CSCO"])

print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.2e}")
print(f"Spearman rho = {spearman_r:.4f}, p = {spearman_p:.2e}")
```

**Interpretation:**
- Pearson r > 0.85: Trajectories are highly similar in magnitude
- Spearman rho > 0.90: Trajectories are nearly identical in rank-order (direction)
- Divergence between Pearson and Spearman indicates non-linear scaling differences

**Report:** Include both coefficients with p-values. State confidence intervals using Fisher z-transformation:

```python
from scipy.stats import norm
import numpy as np

def pearson_ci(r, n, alpha=0.05):
    """Compute confidence interval for Pearson r using Fisher z-transform."""
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    ci_low = np.tanh(z - z_crit * se)
    ci_high = np.tanh(z + z_crit * se)
    return ci_low, ci_high
```

### 7.2 Kolmogorov-Smirnov Test on Return Distributions

**What:** Test whether NVDA's daily return distribution differs significantly from CSCO's at the equivalent cycle phase.

```python
# Compare daily returns over overlapping breakout-aligned period
nvda_returns = nvda_daily.loc[nvda_daily["days_from_breakout"].between(0, 500), "daily_return"]
csco_returns = csco_daily.loc[csco_daily["days_from_breakout"].between(0, 500), "daily_return"]

ks_stat, ks_p = stats.ks_2samp(nvda_returns.dropna(), csco_returns.dropna())
print(f"KS statistic = {ks_stat:.4f}, p = {ks_p:.2e}")
```

**Interpretation:**
- p < 0.05: Return distributions are significantly different (NVDA is not just a replay of CSCO)
- p >= 0.05: Cannot reject hypothesis that return distributions are drawn from the same underlying distribution

**Additional distribution comparison:**
- Compare skewness and kurtosis of both return distributions
- Plot overlaid histograms or KDE plots of daily returns for visual comparison

```python
from scipy.stats import skew, kurtosis

for label, returns in [("NVDA", nvda_returns), ("CSCO", csco_returns)]:
    print(f"{label}: mean={returns.mean():.5f}, std={returns.std():.5f}, "
          f"skew={skew(returns):.3f}, kurtosis={kurtosis(returns):.3f}")
```

### 7.3 Rolling Beta vs S&P 500

**What:** How sensitive is each stock to broad market moves at each cycle phase?

```python
def rolling_beta(stock_returns: pd.Series, market_returns: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute rolling beta of stock vs market.
    Beta = Cov(stock, market) / Var(market)
    """
    cov = stock_returns.rolling(window=window).cov(market_returns)
    var = market_returns.rolling(window=window).var()
    return cov / var

# Compute for both eras
nvda_beta = rolling_beta(nvda_daily["daily_return"], sp500_ai["daily_return"], window=60)
csco_beta = rolling_beta(csco_daily["daily_return"], sp500_dotcom["daily_return"], window=60)
```

**Interpretation:**
- Beta > 1.5: Stock is significantly more volatile than the market (speculative behavior)
- Beta rising over time: Increasing speculation / momentum-driven trading
- Compare the beta trajectory of NVDA vs CSCO at equivalent cycle phases

---

## 8. Key Sub-Questions This Layer Answers

1. **Trajectory match:** Are NVDA and CSCO following the same normalized price path?
2. **Velocity comparison:** Is the AI rally faster or slower than the dot-com rally?
3. **Drawdown similarity:** Do intermediate drawdowns during the rally phase match in depth and frequency?
4. **Return distribution:** Are the daily return characteristics (volatility, skew, fat tails) similar?
5. **Market sensitivity:** Is NVDA more or less correlated with the broader market than CSCO was?
6. **Volume behavior:** Are volume euphoria spikes occurring at similar relative cycle positions?
7. **Current position:** Based on the overlay, where does NVDA currently sit on the CSCO trajectory?

---

## 9. Limitations Specific to This Layer

| Limitation                          | Impact                                      | Mitigation                                    |
|-------------------------------------|---------------------------------------------|-----------------------------------------------|
| Breakout definition is arbitrary    | Different thresholds shift alignment by weeks/months | Sensitivity analysis: test 40%, 50%, 60% thresholds and show overlay is robust |
| Survivorship bias                   | We know CSCO crashed; selecting it biases the comparison toward finding a bubble | Acknowledge in narrative; include peers that did NOT crash as severely |
| Market microstructure changes       | 2023 markets have algo trading, options flow, retail via Robinhood that 1998 did not | Note that volume comparisons are directional, not absolute |
| Different monetary regimes          | Fed funds rate ~5% in 1999 vs ~4.5% in 2024 | Layer 3 addresses macro context; note in this layer |
| Index composition changes           | S&P 500 in 2000 vs 2025 has vastly different sector weights | Use sector ETFs (XLK) in addition to broad index |
| SUNW data availability              | May be incomplete or unavailable in yfinance | Use JAVA ticker or substitute with DELL       |
| ARM has limited history             | IPO in Sep 2023, so only ~2.5 years of data | Note shorter series; exclude from tests requiring full window |

---

## 10. Code Outline (Main Pipeline)

```python
# ============================================================
# Layer 1: Historical Price Overlay Pipeline
# ============================================================

# --- STEP 1: Data Ingestion ---
def ingest_all_price_data() -> dict[str, pd.DataFrame]:
    """Pull daily OHLCV for all tickers across both eras."""
    tickers_config = {
        "ai_cycle": {
            "tickers": ["NVDA", "AMD", "SMCI", "AVGO", "ARM", "MSFT"],
            "start": "2022-01-01",   # Extra year for 252-day lookback
            "end": "2026-03-28"
        },
        "dotcom_cycle": {
            "tickers": ["CSCO", "JNPR", "QCOM", "INTC", "SUNW"],
            "start": "1997-01-01",   # Extra year for 252-day lookback
            "end": "2002-12-31"
        },
        "indices": {
            "tickers": ["^IXIC", "^GSPC"],
            "start": "1997-01-01",
            "end": "2026-03-28"
        }
    }

    all_data = {}
    for era, config in tickers_config.items():
        for ticker in config["tickers"]:
            df = fetch_price_data(ticker, config["start"], config["end"])
            df["ticker"] = ticker
            df["era"] = era
            all_data[ticker] = df

    return all_data

# --- STEP 2: Validate & Clean ---
def validate_and_clean(all_data: dict) -> dict:
    """Handle missing data, verify split adjustments, compute coverage stats."""
    for ticker, df in all_data.items():
        # Forward-fill small gaps
        df["Close"] = df["Close"].fillna(method="ffill", limit=3)

        # Log coverage
        expected_days = np.busday_count(df.index[0].date(), df.index[-1].date())
        actual_days = len(df)
        coverage = actual_days / expected_days * 100
        print(f"{ticker}: {actual_days}/{expected_days} trading days ({coverage:.1f}%)")

        # Verify no extreme jumps (>50% in a day) which indicate split errors
        daily_changes = df["Close"].pct_change().abs()
        if daily_changes.max() > 0.50:
            suspect_dates = daily_changes[daily_changes > 0.50].index
            print(f"WARNING: {ticker} has suspicious jumps on {suspect_dates.tolist()}")

    return all_data

# --- STEP 3: Find Breakout Points ---
def compute_breakouts(all_data: dict) -> dict[str, pd.Timestamp]:
    """Find breakout date for each primary and peer ticker."""
    breakouts = {}
    for ticker, df in all_data.items():
        if ticker.startswith("^"):  # Skip indices
            continue
        try:
            breakouts[ticker] = find_breakout(df, threshold=0.50, sustain_days=20)
            print(f"{ticker} breakout: {breakouts[ticker].strftime('%Y-%m-%d')}")
        except ValueError as e:
            print(f"{ticker}: {e}")
    return breakouts

# --- STEP 4: Normalize ---
def normalize_all(all_data: dict, breakouts: dict) -> dict:
    """Apply normalization to each ticker using its breakout point."""
    for ticker, df in all_data.items():
        if ticker in breakouts:
            all_data[ticker] = normalize_to_breakout(df, breakouts[ticker])
    return all_data

# --- STEP 5: Compute Derived Metrics ---
def compute_metrics(all_data: dict) -> dict:
    """Add returns, drawdowns, volume z-scores, rolling stats."""
    for ticker, df in all_data.items():
        df["daily_return"] = df["Close"].pct_change()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["drawdown"] = compute_drawdown(df["Close"])
        df["volume_zscore"] = volume_zscore(df["Volume"], window=60)

        for window in [30, 90, 252]:
            df[f"rolling_{window}d_ret"] = df["Close"].pct_change(periods=window)

    return all_data

# --- STEP 6: Weekly Resample ---
def resample_all_weekly(all_data: dict) -> dict:
    """Create weekly versions for smoother chart overlays."""
    weekly_data = {}
    for ticker, df in all_data.items():
        weekly_data[ticker] = resample_to_weekly(df)
    return weekly_data

# --- STEP 7: Statistical Tests ---
def run_statistical_tests(all_data: dict) -> dict:
    """Execute all stat tests, return results dict for reporting."""
    results = {}

    # 7.1 Correlation
    overlap = get_overlapping_normalized(all_data["NVDA"], all_data["CSCO"])
    results["pearson_r"], results["pearson_p"] = stats.pearsonr(overlap["NVDA"], overlap["CSCO"])
    results["spearman_r"], results["spearman_p"] = stats.spearmanr(overlap["NVDA"], overlap["CSCO"])

    # 7.2 KS test
    nvda_ret = all_data["NVDA"]["daily_return"].dropna()
    csco_ret = all_data["CSCO"]["daily_return"].dropna()
    results["ks_stat"], results["ks_p"] = stats.ks_2samp(nvda_ret, csco_ret)

    # 7.3 Rolling beta
    results["nvda_beta_series"] = rolling_beta(
        all_data["NVDA"]["daily_return"],
        all_data["^GSPC"]["daily_return"],
        window=60
    )
    results["csco_beta_series"] = rolling_beta(
        all_data["CSCO"]["daily_return"],
        all_data["^GSPC"]["daily_return"],
        window=60
    )

    return results

# --- STEP 8: Generate Visualizations ---
def generate_all_charts(weekly_data: dict, daily_data: dict, stats_results: dict):
    """Produce Charts 1.1 through 1.4 and save to output/charts/."""
    chart_1_1_normalized_overlay(weekly_data)
    chart_1_2_drawdown_comparison(weekly_data)
    chart_1_3_rolling_returns(daily_data)
    chart_1_4_volume_surges(daily_data)

# --- MAIN ---
def main():
    raw = ingest_all_price_data()
    clean = validate_and_clean(raw)
    breakouts = compute_breakouts(clean)
    normalized = normalize_all(clean, breakouts)
    enriched = compute_metrics(normalized)
    weekly = resample_all_weekly(enriched)
    stats_results = run_statistical_tests(enriched)
    generate_all_charts(weekly, enriched, stats_results)

    # Save processed data
    save_to_parquet(enriched, "data/processed/layer1_daily.parquet")
    save_to_parquet(weekly, "data/processed/layer1_weekly.parquet")
    save_stats_report(stats_results, "output/layer1_stats.json")

if __name__ == "__main__":
    main()
```

---

## 11. Output Artifacts

| Artifact                          | Path                                    | Format   |
|-----------------------------------|-----------------------------------------|----------|
| Processed daily data              | `data/processed/layer1_daily.parquet`   | Parquet  |
| Processed weekly data             | `data/processed/layer1_weekly.parquet`  | Parquet  |
| Chart 1.1 normalized overlay      | `output/charts/chart_1_1_overlay.png`   | PNG 300dpi |
| Chart 1.2 drawdown comparison     | `output/charts/chart_1_2_drawdown.png`  | PNG 300dpi |
| Chart 1.3 rolling returns         | `output/charts/chart_1_3_rolling.png`   | PNG 300dpi |
| Chart 1.4 volume surges           | `output/charts/chart_1_4_volume.png`    | PNG 300dpi |
| Statistical test results          | `output/layer1_stats.json`              | JSON     |
| Breakout dates log                | `output/layer1_breakouts.json`          | JSON     |
