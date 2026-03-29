"""Layer 1: Historical Price Comparison — NVDA vs CSCO (dot-com analog).

Fetches OHLCV data for both eras, normalizes prices to a systematic breakout
anchor, runs statistical tests, and generates four publication-quality charts.

Entry point: run_layer1(force_refresh=False)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.cache import cache_or_call

warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE = "plotly_dark"
NVDA_COLOR = "#00CC96"
CSCO_COLOR = "#EF553B"
NORTEL_COLOR = "#888888"

CHART_DIR = Path(__file__).resolve().parents[2] / "submissions" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Data windows — one extra year before the analysis window for 252-day lookback
AI_TICKERS = ["NVDA", "AMD", "SMCI", "AVGO", "ARM", "MSFT"]
AI_START = "2022-01-01"
AI_END = "2026-03-28"

DOTCOM_TICKERS = ["CSCO", "JNPR", "QCOM", "INTC"]
DOTCOM_START = "1997-01-01"
DOTCOM_END = "2003-12-31"

NORTEL_TICKER = "NT"
NORTEL_START = "1997-01-01"
NORTEL_END = "2009-12-31"

INDEX_TICKER = "^GSPC"


# ---------------------------------------------------------------------------
# 1. Data Fetching
# ---------------------------------------------------------------------------


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Thin wrapper around yfinance.download; returns a clean single-ticker DF."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"yfinance returned empty DataFrame for {ticker}")
    # yfinance may return MultiIndex columns for a single ticker — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def fetch_all_data(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch all required tickers using the cache-or-call pattern.

    Returns a dict keyed by ticker symbol. Each value is a raw OHLCV DataFrame
    with a DatetimeIndex.
    """
    console.rule("[bold cyan]Layer 1 — Data Ingestion")

    tickers_config: list[tuple[str, str, str]] = []

    # AI-era tickers
    for t in AI_TICKERS:
        tickers_config.append((t, AI_START, AI_END))

    # Dot-com era tickers
    for t in DOTCOM_TICKERS:
        tickers_config.append((t, DOTCOM_START, DOTCOM_END))

    # Nortel (non-survivor analog, longer window)
    tickers_config.append((NORTEL_TICKER, NORTEL_START, NORTEL_END))

    # S&P 500 — covers both eras in one pull
    tickers_config.append((INDEX_TICKER, "1997-01-01", AI_END))

    all_data: dict[str, pd.DataFrame] = {}

    for ticker, start, end in tickers_config:
        safe_key = ticker.replace("^", "idx_")
        cache_key = f"layer1_{safe_key}_{start}_{end}"
        try:
            df = cache_or_call(
                key=cache_key,
                fetch_fn=lambda t=ticker, s=start, e=end: _download(t, s, e),
                subdir="raw",
                force_refresh=force_refresh,
            )
            all_data[ticker] = df
        except Exception as exc:
            console.print(f"[red]SKIP {ticker}: {exc}[/]")

    return all_data


# ---------------------------------------------------------------------------
# 2. Preprocessing & Validation
# ---------------------------------------------------------------------------


def _clean_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Forward-fill small gaps, verify no pathological jumps, return copy."""
    df = df.copy()

    # Standardize column name to 'Close' (yfinance may return 'Close' or 'Adj Close')
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    # Forward-fill gaps of up to 3 trading days (halts, thin markets)
    df["Close"] = df["Close"].ffill(limit=3)

    # Coverage check
    if len(df) > 0:
        expected = np.busday_count(df.index[0].date(), df.index[-1].date())
        coverage = len(df) / max(expected, 1) * 100
        if coverage < 80:
            console.print(
                f"[yellow]WARNING {ticker}: only {coverage:.1f}% trading-day coverage[/]"
            )

    # Suspicious jump check (>50% single-day change — likely a split artifact)
    jumps = df["Close"].pct_change().abs()
    bad = jumps[jumps > 0.50]
    if not bad.empty:
        console.print(
            f"[yellow]WARNING {ticker}: suspicious price jumps on {bad.index.tolist()}[/]"
        )

    return df


def validate_and_clean(all_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean all tickers in-place."""
    console.rule("[bold cyan]Layer 1 — Validation & Cleaning")
    cleaned = {}
    for ticker, df in all_data.items():
        try:
            cleaned[ticker] = _clean_ticker(df, ticker)
        except Exception as exc:
            console.print(f"[red]ERROR cleaning {ticker}: {exc}[/]")
    return cleaned


# ---------------------------------------------------------------------------
# 3. Analysis Functions
# ---------------------------------------------------------------------------


def find_breakout(
    df: pd.DataFrame,
    threshold: float = 0.5,
    sustain_days: int = 20,
) -> pd.Timestamp:
    """Find the first trading day where the trailing 252-day return exceeds
    *threshold* and remains above it for at least *sustain_days* consecutive
    trading days.

    Parameters
    ----------
    df : DataFrame with 'Close' column and DatetimeIndex.
    threshold : Minimum trailing 1-year return (0.50 = 50 %).
    sustain_days : Minimum consecutive days above threshold.

    Returns
    -------
    pd.Timestamp — the breakout date (start of the sustained period).

    Raises
    ------
    ValueError if no qualifying sustained period is found.
    """
    trailing_return = df["Close"].pct_change(periods=252)
    above = (trailing_return > threshold).astype(int)

    # rolling sum of 1s — equals sustain_days when every day in window is above
    consecutive = above.rolling(window=sustain_days).sum()
    candidates = consecutive[consecutive == sustain_days]

    if candidates.empty:
        raise ValueError(
            f"No sustained breakout found (threshold={threshold}, "
            f"sustain_days={sustain_days})"
        )

    first_confirmation_loc = df.index.get_loc(candidates.index[0])
    breakout_loc = first_confirmation_loc - sustain_days + 1
    return df.index[breakout_loc]


def normalize_to_breakout(
    df: pd.DataFrame,
    breakout_date: pd.Timestamp,
) -> pd.DataFrame:
    """Index price to 100 at breakout_date and add days_from_breakout column.

    Parameters
    ----------
    df : Clean OHLCV DataFrame with DatetimeIndex.
    breakout_date : The alignment anchor returned by find_breakout.

    Returns
    -------
    DataFrame with two new columns: normalized_price, days_from_breakout.
    """
    df = df.copy()
    breakout_price = df.loc[breakout_date, "Close"]
    df["normalized_price"] = (df["Close"] / breakout_price) * 100.0

    breakout_idx = df.index.get_loc(breakout_date)
    df["days_from_breakout"] = np.arange(len(df)) - breakout_idx

    return df


def compute_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'drawdown' column: (price / running_max) - 1 (negative values).

    Operates on the 'Close' column; returns a copy.
    """
    df = df.copy()
    running_max = df["Close"].expanding().max()
    df["drawdown"] = (df["Close"] / running_max) - 1.0
    return df


def compute_rolling_returns(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling_Xd_ret columns for each window (percentage, not decimal)."""
    if windows is None:
        windows = [30, 90, 252]
    df = df.copy()
    for w in windows:
        df[f"rolling_{w}d_ret"] = df["Close"].pct_change(periods=w) * 100.0
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily_return and log_return columns; return copy."""
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def _volume_zscore(volume: pd.Series, window: int = 60) -> pd.Series:
    """Z-score of volume relative to a rolling mean/std window."""
    rolling_mean = volume.rolling(window=window).mean()
    rolling_std = volume.rolling(window=window).std()
    return (volume - rolling_mean) / rolling_std.replace(0, np.nan)


def breakout_sensitivity(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
    sustain_days_list: list[int] | None = None,
) -> pd.DataFrame:
    """Grid search over breakout parameter combinations.

    Returns a DataFrame with columns: threshold, sustain_days, breakout_date.
    """
    if thresholds is None:
        thresholds = [0.4, 0.5, 0.6]
    if sustain_days_list is None:
        sustain_days_list = [10, 20, 30]

    rows = []
    for thresh in thresholds:
        for sustain in sustain_days_list:
            try:
                bd = find_breakout(df, threshold=thresh, sustain_days=sustain)
                rows.append(
                    {"threshold": thresh, "sustain_days": sustain, "breakout_date": bd}
                )
            except ValueError:
                rows.append(
                    {"threshold": thresh, "sustain_days": sustain, "breakout_date": None}
                )
    return pd.DataFrame(rows)


def _rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_annual: float = 0.05,
) -> pd.Series:
    rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
    excess = returns - rf_daily
    rolling_mean = excess.rolling(window=window).mean() * 252
    rolling_std = returns.rolling(window=window).std() * (252**0.5)
    return rolling_mean / rolling_std.replace(0, np.nan)


def _rolling_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    cov = stock_returns.rolling(window=window).cov(market_returns)
    var = market_returns.rolling(window=window).var()
    return cov / var.replace(0, np.nan)


def _pearson_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher z-transform confidence interval for Pearson r."""
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    z_crit = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))


# ---------------------------------------------------------------------------
# 4. Full Metric Pipeline
# ---------------------------------------------------------------------------


def _enrich(df: pd.DataFrame, breakout_date: pd.Timestamp | None) -> pd.DataFrame:
    """Apply the complete metric pipeline to a single ticker DataFrame."""
    df = compute_log_returns(df)
    df = compute_drawdown(df)
    df = compute_rolling_returns(df)
    if "Volume" in df.columns:
        df["volume_zscore"] = _volume_zscore(df["Volume"])
    if breakout_date is not None:
        df = normalize_to_breakout(df, breakout_date)
    return df


# ---------------------------------------------------------------------------
# 5. Statistical Tests
# ---------------------------------------------------------------------------


def run_statistical_tests(
    nvda: pd.DataFrame,
    csco: pd.DataFrame,
    sp500_ai: pd.DataFrame | None = None,
    sp500_dotcom: pd.DataFrame | None = None,
) -> dict:
    """Execute all Layer 1 statistical tests.

    Parameters
    ----------
    nvda, csco : Enriched DataFrames with log_return and days_from_breakout.
    sp500_ai, sp500_dotcom : Optional S&P 500 slices for rolling beta.

    Returns
    -------
    dict with keys: adf_nvda, adf_csco, pearson, spearman, ljungbox_nvda,
    ljungbox_csco, ks_test, distribution_stats, sensitivity_nvda,
    sensitivity_csco, rolling_beta_nvda, rolling_beta_csco.
    """
    console.rule("[bold cyan]Layer 1 — Statistical Tests")
    results: dict = {}

    # --- Align log-returns on days_from_breakout ---
    if "days_from_breakout" in nvda.columns and "days_from_breakout" in csco.columns:
        nvda_lr = nvda.set_index("days_from_breakout")["log_return"].dropna()
        csco_lr = csco.set_index("days_from_breakout")["log_return"].dropna()
        overlap_idx = nvda_lr.index.intersection(csco_lr.index)
        nvda_aligned = nvda_lr.loc[overlap_idx]
        csco_aligned = csco_lr.loc[overlap_idx]
    else:
        # Fallback: align on calendar date (best-effort)
        nvda_lr = nvda["log_return"].dropna()
        csco_lr = csco["log_return"].dropna()
        overlap_idx = nvda_lr.index.intersection(csco_lr.index)
        nvda_aligned = nvda_lr.loc[overlap_idx] if len(overlap_idx) > 0 else nvda_lr.iloc[:min(len(nvda_lr), len(csco_lr))]
        csco_aligned = csco_lr.loc[overlap_idx] if len(overlap_idx) > 0 else csco_lr.iloc[:min(len(nvda_lr), len(csco_lr))]

    min_len = min(len(nvda_aligned), len(csco_aligned))
    nvda_aligned = nvda_aligned.iloc[:min_len]
    csco_aligned = csco_aligned.iloc[:min_len]

    console.print(f"Aligned series length: {min_len} observations")

    # --- ADF stationarity tests ---
    for label, series in [("NVDA", nvda_aligned), ("CSCO", csco_aligned)]:
        if len(series) < 10:
            results[f"adf_{label.lower()}"] = {"error": "insufficient data"}
            continue
        try:
            adf_stat, adf_p, _, _, _, _ = adfuller(series.values)
            results[f"adf_{label.lower()}"] = {
                "statistic": float(adf_stat),
                "p_value": float(adf_p),
                "stationary": bool(adf_p < 0.05),
            }
            console.print(
                f"ADF {label}: stat={adf_stat:.4f}, p={adf_p:.2e} "
                f"({'stationary' if adf_p < 0.05 else 'NON-STATIONARY'})"
            )
        except Exception as exc:
            results[f"adf_{label.lower()}"] = {"error": str(exc)}

    # --- Pearson & Spearman on log-returns ---
    if min_len >= 10:
        try:
            pr, pp = stats.pearsonr(nvda_aligned.values, csco_aligned.values)
            sr, sp_val = stats.spearmanr(nvda_aligned.values, csco_aligned.values)
            ci_low, ci_high = _pearson_ci(pr, min_len)
            results["pearson"] = {
                "r": float(pr),
                "p_value": float(pp),
                "ci_95": [ci_low, ci_high],
                "n": min_len,
            }
            results["spearman"] = {
                "rho": float(sr),
                "p_value": float(sp_val),
                "n": min_len,
            }
            console.print(
                f"Pearson r={pr:.4f} (p={pp:.2e}), "
                f"95% CI [{ci_low:.4f}, {ci_high:.4f}]"
            )
            console.print(f"Spearman rho={sr:.4f} (p={sp_val:.2e})")
        except Exception as exc:
            results["pearson"] = {"error": str(exc)}
            results["spearman"] = {"error": str(exc)}

    # --- Ljung-Box autocorrelation test ---
    for label, series in [("nvda", nvda_aligned), ("csco", csco_aligned)]:
        if len(series) < 15:
            results[f"ljungbox_{label}"] = {"error": "insufficient data"}
            continue
        try:
            lb = acorr_ljungbox(series.values, lags=[10], return_df=True)
            lb_stat = float(lb["lb_stat"].iloc[0])
            lb_p = float(lb["lb_pvalue"].iloc[0])
            results[f"ljungbox_{label}"] = {
                "statistic": lb_stat,
                "p_value": lb_p,
                "autocorrelated": bool(lb_p < 0.05),
            }
            console.print(
                f"Ljung-Box {label.upper()}: stat={lb_stat:.4f}, p={lb_p:.4f} "
                f"({'autocorrelated' if lb_p < 0.05 else 'no autocorrelation'})"
            )
        except Exception as exc:
            results[f"ljungbox_{label}"] = {"error": str(exc)}

    # --- KS test on daily return distributions (days 0–500 relative to breakout) ---
    try:
        if "days_from_breakout" in nvda.columns:
            nvda_ret = nvda.loc[
                nvda["days_from_breakout"].between(0, 500), "daily_return"
            ].dropna()
            csco_ret = csco.loc[
                csco["days_from_breakout"].between(0, 500), "daily_return"
            ].dropna()
        else:
            nvda_ret = nvda["daily_return"].dropna()
            csco_ret = csco["daily_return"].dropna()

        if len(nvda_ret) > 5 and len(csco_ret) > 5:
            ks_stat, ks_p = stats.ks_2samp(nvda_ret.values, csco_ret.values)
            results["ks_test"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
                "distributions_differ": bool(ks_p < 0.05),
            }
            console.print(
                f"KS test: stat={ks_stat:.4f}, p={ks_p:.2e} "
                f"({'different distributions' if ks_p < 0.05 else 'same distribution'})"
            )

            # Distribution descriptive stats
            for label, series in [("nvda", nvda_ret), ("csco", csco_ret)]:
                results[f"dist_{label}"] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "skewness": float(stats.skew(series.values)),
                    "kurtosis": float(stats.kurtosis(series.values)),
                    "n": len(series),
                }
    except Exception as exc:
        results["ks_test"] = {"error": str(exc)}

    # --- Sensitivity analysis ---
    try:
        results["sensitivity_nvda"] = breakout_sensitivity(nvda).to_dict("records")
    except Exception as exc:
        results["sensitivity_nvda"] = {"error": str(exc)}

    try:
        results["sensitivity_csco"] = breakout_sensitivity(csco).to_dict("records")
    except Exception as exc:
        results["sensitivity_csco"] = {"error": str(exc)}

    # --- Rolling beta vs S&P 500 ---
    if sp500_ai is not None and "daily_return" in nvda.columns:
        try:
            # Align on shared calendar dates
            mkt = sp500_ai["daily_return"].dropna()
            stk = nvda["daily_return"].dropna()
            common = stk.index.intersection(mkt.index)
            if len(common) > 60:
                beta_series = _rolling_beta(stk.loc[common], mkt.loc[common])
                results["rolling_beta_nvda"] = {
                    "mean": float(beta_series.mean()),
                    "max": float(beta_series.max()),
                    "current": float(beta_series.dropna().iloc[-1]),
                }
        except Exception as exc:
            results["rolling_beta_nvda"] = {"error": str(exc)}

    if sp500_dotcom is not None and "daily_return" in csco.columns:
        try:
            mkt = sp500_dotcom["daily_return"].dropna()
            stk = csco["daily_return"].dropna()
            common = stk.index.intersection(mkt.index)
            if len(common) > 60:
                beta_series = _rolling_beta(stk.loc[common], mkt.loc[common])
                results["rolling_beta_csco"] = {
                    "mean": float(beta_series.mean()),
                    "max": float(beta_series.max()),
                    "current": float(beta_series.dropna().iloc[-1]),
                }
        except Exception as exc:
            results["rolling_beta_csco"] = {"error": str(exc)}

    return results


# ---------------------------------------------------------------------------
# 6. Chart Generation
# ---------------------------------------------------------------------------


def _save_chart(fig: go.Figure, name: str) -> None:
    """Write chart as JSON and PNG to CHART_DIR."""
    json_path = CHART_DIR / f"{name}.json"
    png_path = CHART_DIR / f"{name}.png"

    fig.write_json(str(json_path))
    console.print(f"[green]Saved:[/] {json_path.name}")

    try:
        fig.write_image(str(png_path), width=1600, height=900, scale=2)
        console.print(f"[green]Saved:[/] {png_path.name}")
    except Exception as exc:
        console.print(
            f"[yellow]PNG export skipped ({exc}). "
            "Install kaleido: pip install kaleido[/]"
        )


def chart_1_1_price_overlay(
    nvda: pd.DataFrame,
    csco: pd.DataFrame,
    nortel: pd.DataFrame | None = None,
    peers_ai: dict[str, pd.DataFrame] | None = None,
    peers_dotcom: dict[str, pd.DataFrame] | None = None,
) -> go.Figure:
    """Chart 1.1 — Normalized price overlay, log scale, day-aligned.

    NVDA (solid green), CSCO (dashed red), Nortel (gray dashed),
    optional thin peer lines.
    """
    fig = go.Figure()

    # --- Peer lines first (lowest z-order) ---
    ai_peer_colors = {"AMD": "#ED1C24", "SMCI": "#FF7F00", "AVGO": "#9467BD", "ARM": "#8C564B", "MSFT": "#1F77B4"}
    dotcom_peer_colors = {"QCOM": "#3253DC", "JNPR": "#BCBD22", "INTC": "#17BECF"}

    if peers_ai:
        for ticker, df_peer in peers_ai.items():
            if "days_from_breakout" not in df_peer.columns or "normalized_price" not in df_peer.columns:
                continue
            p = df_peer.dropna(subset=["normalized_price"]).sort_values("days_from_breakout")
            fig.add_trace(
                go.Scatter(
                    x=p["days_from_breakout"],
                    y=p["normalized_price"],
                    mode="lines",
                    name=f"{ticker} (AI)",
                    line=dict(color=ai_peer_colors.get(ticker, "#AAAAAA"), width=1, dash="dot"),
                    opacity=0.4,
                    legendgroup="ai_peers",
                    legendgrouptitle_text="AI-Era Peers",
                )
            )

    if peers_dotcom:
        for ticker, df_peer in peers_dotcom.items():
            if "days_from_breakout" not in df_peer.columns or "normalized_price" not in df_peer.columns:
                continue
            p = df_peer.dropna(subset=["normalized_price"]).sort_values("days_from_breakout")
            fig.add_trace(
                go.Scatter(
                    x=p["days_from_breakout"],
                    y=p["normalized_price"],
                    mode="lines",
                    name=f"{ticker} (Dot-com)",
                    line=dict(color=dotcom_peer_colors.get(ticker, "#AAAAAA"), width=1, dash="dot"),
                    opacity=0.4,
                    legendgroup="dotcom_peers",
                    legendgrouptitle_text="Dot-Com Peers",
                )
            )

    # --- Nortel (non-survivor) ---
    if nortel is not None and "days_from_breakout" in nortel.columns and "normalized_price" in nortel.columns:
        nt = nortel.dropna(subset=["normalized_price"]).sort_values("days_from_breakout")
        fig.add_trace(
            go.Scatter(
                x=nt["days_from_breakout"],
                y=nt["normalized_price"],
                mode="lines",
                name="Nortel / NT (bankrupt 2009)",
                line=dict(color=NORTEL_COLOR, width=1.5, dash="dash"),
                opacity=0.7,
            )
        )

    # --- CSCO ---
    if "days_from_breakout" in csco.columns and "normalized_price" in csco.columns:
        c = csco.dropna(subset=["normalized_price"]).sort_values("days_from_breakout")
        fig.add_trace(
            go.Scatter(
                x=c["days_from_breakout"],
                y=c["normalized_price"],
                mode="lines",
                name="CSCO (1998–2002)",
                line=dict(color=CSCO_COLOR, width=2.5, dash="dash"),
            )
        )

        # Shade the CSCO crash region (from ATH to -50% after it)
        if "drawdown" in csco.columns:
            peak_loc = c["normalized_price"].idxmax()
            peak_day = c.loc[peak_loc, "days_from_breakout"] if "days_from_breakout" in c.columns else None
            if peak_day is not None:
                crash_mask = (c["days_from_breakout"] >= peak_day) & (c["normalized_price"] <= c["normalized_price"].max() * 0.5)
                crash_region = c[crash_mask]
                if not crash_region.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=pd.concat([crash_region["days_from_breakout"], crash_region["days_from_breakout"].iloc[::-1]]),
                            y=pd.concat([crash_region["normalized_price"], pd.Series([100.0] * len(crash_region))]),
                            fill="toself",
                            fillcolor="rgba(239,85,59,0.08)",
                            line=dict(color="rgba(0,0,0,0)"),
                            showlegend=True,
                            name="CSCO Crash Zone",
                            hoverinfo="skip",
                        )
                    )

    # --- NVDA (hero line, on top) ---
    if "days_from_breakout" in nvda.columns and "normalized_price" in nvda.columns:
        n = nvda.dropna(subset=["normalized_price"]).sort_values("days_from_breakout")
        fig.add_trace(
            go.Scatter(
                x=n["days_from_breakout"],
                y=n["normalized_price"],
                mode="lines",
                name="NVDA (2023–2026)",
                line=dict(color=NVDA_COLOR, width=3),
            )
        )

        # Annotation: current NVDA position
        last = n.iloc[-1]
        fig.add_annotation(
            x=last["days_from_breakout"],
            y=last["normalized_price"],
            text=f"NVDA Now<br>Day {int(last['days_from_breakout'])}",
            showarrow=True,
            arrowhead=2,
            font=dict(color=NVDA_COLOR, size=11),
            arrowcolor=NVDA_COLOR,
            ax=30,
            ay=-40,
        )

    # --- Breakout annotation ---
    fig.add_vline(
        x=0,
        line=dict(color="gray", dash="dash", width=1.5),
        annotation_text="Breakout (Day 0)",
        annotation_position="top right",
        annotation_font_color="gray",
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=(
                "AI Hype vs Dot-Com: Normalized Price Trajectories"
                "<br><sup>Day 0 = first sustained 50%+ YoY gain. "
                "Prices indexed to 100. Log scale.</sup>"
            ),
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Trading Days from Breakout",
            range=[-60, 820],
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="Normalized Price (100 = Breakout Day)",
            type="log",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)", bordercolor="gray", borderwidth=1),
        hovermode="x unified",
        height=700,
    )

    _save_chart(fig, "chart_1_1")
    return fig


def chart_1_2_drawdown(
    nvda: pd.DataFrame,
    csco: pd.DataFrame,
) -> go.Figure:
    """Chart 1.2 — Drawdown from ATH, area chart."""
    fig = go.Figure()

    if "days_from_breakout" in nvda.columns and "drawdown" in nvda.columns:
        n = nvda.dropna(subset=["drawdown"]).sort_values("days_from_breakout")
        fig.add_trace(
            go.Scatter(
                x=n["days_from_breakout"],
                y=n["drawdown"] * 100,
                mode="lines",
                fill="tozeroy",
                name="NVDA Drawdown",
                line=dict(color=NVDA_COLOR, width=1.5),
                fillcolor=f"rgba(0,204,150,0.25)",
            )
        )
        nvda_max_dd = float(n["drawdown"].min() * 100)

    if "days_from_breakout" in csco.columns and "drawdown" in csco.columns:
        c = csco.dropna(subset=["drawdown"]).sort_values("days_from_breakout")
        fig.add_trace(
            go.Scatter(
                x=c["days_from_breakout"],
                y=c["drawdown"] * 100,
                mode="lines",
                fill="tozeroy",
                name="CSCO Drawdown",
                line=dict(color=CSCO_COLOR, width=1.5),
                fillcolor="rgba(239,85,59,0.25)",
            )
        )
        csco_max_dd = float(c["drawdown"].min() * 100)

        # Annotate CSCO max drawdown
        csco_dd_idx = c["drawdown"].idxmin()
        fig.add_annotation(
            x=c.loc[csco_dd_idx, "days_from_breakout"],
            y=csco_max_dd,
            text=f"CSCO Peak Drawdown<br>{csco_max_dd:.1f}%",
            showarrow=True,
            arrowhead=2,
            font=dict(color=CSCO_COLOR, size=10),
            arrowcolor=CSCO_COLOR,
            ax=-60,
            ay=40,
        )

    # Bear market threshold line
    fig.add_hline(
        y=-20,
        line=dict(color="red", dash="dash", width=1),
        annotation_text="Bear Market Threshold (-20%)",
        annotation_position="top right",
        annotation_font_color="red",
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text="Drawdown Comparison: How Far From the Peak?",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Trading Days from Breakout",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="Drawdown from Running Peak (%)",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        legend=dict(x=0.01, y=0.05, bgcolor="rgba(0,0,0,0.4)"),
        hovermode="x unified",
        height=600,
    )

    _save_chart(fig, "chart_1_2")
    return fig


def chart_1_3_rolling_returns(
    nvda: pd.DataFrame,
    csco: pd.DataFrame,
) -> go.Figure:
    """Chart 1.3 — Rolling 90-day return comparison."""
    fig = go.Figure()

    col = "rolling_90d_ret"

    if "days_from_breakout" in nvda.columns and col in nvda.columns:
        n = nvda.dropna(subset=[col]).sort_values("days_from_breakout")
        fig.add_trace(
            go.Scatter(
                x=n["days_from_breakout"],
                y=n[col],
                mode="lines",
                name="NVDA 90-Day Rolling Return",
                line=dict(color=NVDA_COLOR, width=2),
            )
        )

    if "days_from_breakout" in csco.columns and col in csco.columns:
        c = csco.dropna(subset=[col]).sort_values("days_from_breakout")
        fig.add_trace(
            go.Scatter(
                x=c["days_from_breakout"],
                y=c[col],
                mode="lines",
                name="CSCO 90-Day Rolling Return",
                line=dict(color=CSCO_COLOR, width=2, dash="dash"),
            )
        )

        # Shade region where CSCO 90d return went negative
        csco_neg = c[c[col] < 0]
        if not csco_neg.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([csco_neg["days_from_breakout"], csco_neg["days_from_breakout"].iloc[::-1]]),
                    y=pd.concat([csco_neg[col], pd.Series([0.0] * len(csco_neg))]),
                    fill="toself",
                    fillcolor="rgba(239,85,59,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="CSCO Negative Momentum",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    # Zero line
    fig.add_hline(
        y=0,
        line=dict(color="white", dash="dot", width=1),
        annotation_text="0% Return",
        annotation_position="top left",
        annotation_font_color="gray",
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text="Rolling 90-Day Returns: Momentum Comparison",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Trading Days from Breakout",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="90-Day Rolling Return (%)",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
        hovermode="x unified",
        height=600,
    )

    _save_chart(fig, "chart_1_3")
    return fig


def chart_1_4_volume_surge(
    nvda: pd.DataFrame,
    csco: pd.DataFrame,
) -> go.Figure:
    """Chart 1.4 — Volume euphoria spikes overlaid on normalized price."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
        subplot_titles=("Normalized Price", "Volume Z-Score"),
    )

    for ticker, df, color in [
        ("NVDA", nvda, NVDA_COLOR),
        ("CSCO", csco, CSCO_COLOR),
    ]:
        if "days_from_breakout" not in df.columns or "normalized_price" not in df.columns:
            continue

        d = df.dropna(subset=["normalized_price"]).sort_values("days_from_breakout")

        # Background price line (thin)
        fig.add_trace(
            go.Scatter(
                x=d["days_from_breakout"],
                y=d["normalized_price"],
                mode="lines",
                name=f"{ticker} Price",
                line=dict(color=color, width=1.5),
                opacity=0.6,
                legendgroup=ticker,
            ),
            row=1,
            col=1,
        )

        # Volume z-score bar
        if "volume_zscore" in d.columns:
            fig.add_trace(
                go.Bar(
                    x=d["days_from_breakout"],
                    y=d["volume_zscore"],
                    name=f"{ticker} Vol Z-Score",
                    marker_color=color,
                    opacity=0.5,
                    legendgroup=ticker,
                ),
                row=2,
                col=1,
            )

        # Euphoria spike scatter (|z| > 2.0) on price panel
        if "volume_zscore" in d.columns and "daily_return" in d.columns:
            spikes = d[d["volume_zscore"].abs() > 2.0].copy()
            if not spikes.empty:
                spike_colors = spikes["daily_return"].apply(
                    lambda r: "rgba(0,204,150,0.85)" if r >= 0 else "rgba(239,85,59,0.85)"
                )
                fig.add_trace(
                    go.Scatter(
                        x=spikes["days_from_breakout"],
                        y=spikes["normalized_price"],
                        mode="markers",
                        name=f"{ticker} Vol Spike",
                        marker=dict(
                            size=spikes["volume_zscore"].abs().clip(upper=10) * 2,
                            color=spike_colors.tolist(),
                            line=dict(width=1, color="white"),
                        ),
                        legendgroup=ticker,
                    ),
                    row=1,
                    col=1,
                )

    # Threshold line on z-score panel
    fig.add_hline(
        y=2.5,
        line=dict(color="yellow", dash="dot", width=1),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=-2.5,
        line=dict(color="yellow", dash="dot", width=1),
        row=2,
        col=1,
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text="Volume Euphoria Spikes: When Did the Crowd Rush In?",
            font=dict(size=18),
        ),
        hovermode="x unified",
        height=750,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
    )
    fig.update_xaxes(title_text="Trading Days from Breakout", row=2, col=1)
    fig.update_yaxes(title_text="Normalized Price (log)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Volume Z-Score", row=2, col=1)

    _save_chart(fig, "chart_1_4")
    return fig


# ---------------------------------------------------------------------------
# 7. Key Findings Extraction
# ---------------------------------------------------------------------------


def _extract_findings(
    nvda: pd.DataFrame,
    csco: pd.DataFrame,
    nortel: pd.DataFrame | None,
    stats_results: dict,
    nvda_breakout: pd.Timestamp,
    csco_breakout: pd.Timestamp,
) -> dict[str, str]:
    """Derive plain-language findings from computed data."""
    findings: dict[str, str] = {}

    # Breakout dates
    findings["nvda_breakout_date"] = nvda_breakout.strftime("%Y-%m-%d")
    findings["csco_breakout_date"] = csco_breakout.strftime("%Y-%m-%d")

    # Current NVDA position on cycle
    if "days_from_breakout" in nvda.columns:
        current_day = int(nvda["days_from_breakout"].iloc[-1])
        findings["nvda_current_cycle_day"] = str(current_day)
        findings["nvda_position"] = (
            f"NVDA is currently at Day {current_day} relative to its breakout. "
            f"CSCO peaked around Day ~500 in its cycle."
        )

    # Peak gains
    if "normalized_price" in nvda.columns:
        nvda_peak = float(nvda["normalized_price"].max())
        findings["nvda_peak_normalized"] = f"{nvda_peak:.1f}x (from breakout = 100)"

    if "normalized_price" in csco.columns:
        csco_peak = float(csco["normalized_price"].max())
        findings["csco_peak_normalized"] = f"{csco_peak:.1f}x (from breakout = 100)"

    # Max drawdowns
    if "drawdown" in nvda.columns:
        nvda_max_dd = float(nvda["drawdown"].min() * 100)
        findings["nvda_max_drawdown"] = f"{nvda_max_dd:.1f}%"

    if "drawdown" in csco.columns:
        csco_max_dd = float(csco["drawdown"].min() * 100)
        findings["csco_max_drawdown"] = f"{csco_max_dd:.1f}%"

    # Correlation
    if "pearson" in stats_results and "error" not in stats_results["pearson"]:
        r = stats_results["pearson"]["r"]
        p = stats_results["pearson"]["p_value"]
        findings["log_return_correlation"] = (
            f"Pearson r={r:.4f} (p={p:.2e}) on breakout-aligned log-returns. "
            "Near-zero is expected — eras are independent; DTW shape similarity is the primary metric."
        )

    # KS test interpretation
    if "ks_test" in stats_results and "error" not in stats_results["ks_test"]:
        ks_p = stats_results["ks_test"]["p_value"]
        differ = stats_results["ks_test"]["distributions_differ"]
        findings["ks_interpretation"] = (
            f"KS test p={ks_p:.2e}: return distributions are "
            f"{'significantly different' if differ else 'not significantly different'} "
            f"(alpha=0.05)."
        )

    # Stationarity check warning
    for label in ["nvda", "csco"]:
        adf_key = f"adf_{label}"
        if adf_key in stats_results and "stationary" in stats_results[adf_key]:
            if not stats_results[adf_key]["stationary"]:
                findings[f"WARNING_{label}_nonstationarity"] = (
                    f"{label.upper()} log-returns failed ADF stationarity test "
                    "(p >= 0.05). Correlation p-values are approximate."
                )

    # Volume euphoria
    if "volume_zscore" in nvda.columns:
        nvda_spikes = int((nvda["volume_zscore"].abs() > 2.5).sum())
        findings["nvda_volume_spikes"] = f"{nvda_spikes} trading days with |volume z-score| > 2.5"

    if "volume_zscore" in csco.columns:
        csco_spikes = int((csco["volume_zscore"].abs() > 2.5).sum())
        findings["csco_volume_spikes"] = f"{csco_spikes} trading days with |volume z-score| > 2.5"

    return findings


# ---------------------------------------------------------------------------
# 8. Main Runner
# ---------------------------------------------------------------------------


def run_layer1(force_refresh: bool = False) -> dict:
    """Execute the complete Layer 1 pipeline.

    Parameters
    ----------
    force_refresh : If True, bypass cache and re-download all data.

    Returns
    -------
    dict with keys:
        "data"     — dict of enriched DataFrames keyed by ticker
        "stats"    — dict of statistical test results
        "charts"   — dict of Plotly figures
        "findings" — dict of key findings as human-readable strings
    """
    console.rule("[bold green]Layer 1: Historical Price Comparison")

    # --- Step 1: Fetch ---
    raw = fetch_all_data(force_refresh=force_refresh)

    # --- Step 2: Clean ---
    data = validate_and_clean(raw)

    # --- Step 3: Find breakout points ---
    console.rule("[bold cyan]Layer 1 — Breakout Detection")
    breakouts: dict[str, pd.Timestamp] = {}

    for ticker in ["NVDA", "CSCO", "NT"] + [t for t in data if t not in ("NVDA", "CSCO", "NT", "^GSPC")]:
        if ticker not in data:
            continue
        try:
            bd = find_breakout(data[ticker], threshold=0.5, sustain_days=20)
            breakouts[ticker] = bd
            console.print(f"  {ticker} breakout: [bold]{bd.strftime('%Y-%m-%d')}[/]")
        except ValueError as exc:
            console.print(f"  [yellow]{ticker}: {exc}[/]")

    if "NVDA" not in breakouts or "CSCO" not in breakouts:
        console.print(
            "[red]CRITICAL: Could not find breakout for NVDA or CSCO. "
            "Charts will be partial.[/]"
        )

    # --- Step 4: Enrich all tickers ---
    console.rule("[bold cyan]Layer 1 — Feature Engineering")
    enriched: dict[str, pd.DataFrame] = {}
    for ticker, df in data.items():
        try:
            bd = breakouts.get(ticker)
            enriched[ticker] = _enrich(df, bd)
            console.print(f"  Enriched {ticker} ({len(df):,} rows)")
        except Exception as exc:
            console.print(f"  [red]ERROR enriching {ticker}: {exc}[/]")
            enriched[ticker] = df

    # --- Step 5: Statistical tests ---
    nvda_df = enriched.get("NVDA", pd.DataFrame())
    csco_df = enriched.get("CSCO", pd.DataFrame())
    gspc = enriched.get("^GSPC", None)

    # Split S&P 500 into two era slices for rolling beta
    sp500_ai = sp500_dotcom = None
    if gspc is not None and "daily_return" in gspc.columns:
        sp500_ai = gspc.loc[gspc.index >= "2022-01-01"]
        sp500_dotcom = gspc.loc[(gspc.index >= "1997-01-01") & (gspc.index <= "2003-12-31")]

    stat_results: dict = {}
    if not nvda_df.empty and not csco_df.empty:
        stat_results = run_statistical_tests(nvda_df, csco_df, sp500_ai, sp500_dotcom)
    else:
        console.print("[yellow]Skipping statistical tests — NVDA or CSCO data missing[/]")

    # --- Step 6: Generate charts ---
    console.rule("[bold cyan]Layer 1 — Chart Generation")

    nortel_df = enriched.get("NT")

    ai_peer_tickers = [t for t in AI_TICKERS if t not in ("NVDA",) and t in enriched]
    dotcom_peer_tickers = [t for t in DOTCOM_TICKERS if t not in ("CSCO",) and t in enriched]

    peers_ai_map = {t: enriched[t] for t in ai_peer_tickers}
    peers_dotcom_map = {t: enriched[t] for t in dotcom_peer_tickers}

    charts: dict[str, go.Figure] = {}

    try:
        charts["chart_1_1"] = chart_1_1_price_overlay(
            nvda_df, csco_df, nortel_df, peers_ai_map, peers_dotcom_map
        )
    except Exception as exc:
        console.print(f"[red]chart_1_1 failed: {exc}[/]")

    try:
        charts["chart_1_2"] = chart_1_2_drawdown(nvda_df, csco_df)
    except Exception as exc:
        console.print(f"[red]chart_1_2 failed: {exc}[/]")

    try:
        charts["chart_1_3"] = chart_1_3_rolling_returns(nvda_df, csco_df)
    except Exception as exc:
        console.print(f"[red]chart_1_3 failed: {exc}[/]")

    try:
        charts["chart_1_4"] = chart_1_4_volume_surge(nvda_df, csco_df)
    except Exception as exc:
        console.print(f"[red]chart_1_4 failed: {exc}[/]")

    # --- Step 7: Extract findings ---
    nvda_bd = breakouts.get("NVDA", pd.Timestamp("2023-05-01"))
    csco_bd = breakouts.get("CSCO", pd.Timestamp("1998-10-01"))

    findings = _extract_findings(nvda_df, csco_df, nortel_df, stat_results, nvda_bd, csco_bd)

    # --- Summary ---
    console.rule("[bold green]Layer 1 — Complete")
    for k, v in findings.items():
        console.print(f"  [cyan]{k}:[/] {v}")

    return {
        "data": enriched,
        "stats": stat_results,
        "charts": charts,
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    force = "--refresh" in sys.argv
    result = run_layer1(force_refresh=force)
    console.print(f"\n[bold green]Done.[/] Charts saved to {CHART_DIR}")
