"""Layer 2: Fundamental Comparison — NVDA vs CSCO.

Primary question: Are NVIDIA's fundamentals in 2023-2026 materially stronger
than Cisco's were in 1998-2001, or is the current valuation equally detached
from financial reality?

Data sources:
  - yfinance: quarterly financials + price history for NVDA and CSCO
  - FRED (fredapi): CPI series CPIAUCSL for inflation adjustment

Fiscal year notes:
  - NVDA: fiscal year ends late January (FY2024 Q4 = Jan 2024)
  - CSCO: fiscal year ends late July  (FY2000 Q4 = Jul 2000)
  All dates are mapped to calendar quarters via actual quarter-end date.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

# ── internal utilities ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.cache import cache_or_call, cache_json, load_json

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

console = Console()

# ── constants ─────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
NVDA_COLOR = "#00CC96"
CSCO_COLOR = "#EF553B"
CHART_DIR = Path(__file__).resolve().parents[2] / "submissions" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Cycle breakout anchors (calendar quarter-end dates)
NVDA_BREAKOUT = pd.Timestamp("2023-01-31")   # NVDA FY2024 Q4 earnings signal
CSCO_BREAKOUT = pd.Timestamp("1998-07-31")   # CSCO FY1998 Q4, start of mega-run

# ── helper: fiscal-to-calendar quarter ───────────────────────────────────────

def map_fiscal_to_calendar(date: pd.Timestamp) -> str:
    """Return calendar quarter string from a quarter-end date, e.g. '2024Q1'."""
    q = (date.month - 1) // 3 + 1
    return f"{date.year}Q{q}"


# ── helper: TTM rolling sum ───────────────────────────────────────────────────

def compute_ttm(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add trailing twelve-month (4-quarter rolling sum) columns.

    DataFrame must be sorted ascending by date and indexed by date.
    Requires at least 4 consecutive quarters; earlier rows get NaN.
    """
    df = df.sort_index()
    for col in columns:
        if col in df.columns:
            df[f"{col}_ttm"] = df[col].rolling(window=4, min_periods=4).sum()
    return df


# ── data fetchers ─────────────────────────────────────────────────────────────

def _fetch_yf_financials(ticker: str) -> pd.DataFrame:
    """Fetch quarterly income statement + cash flow + balance sheet via yfinance.

    Returns a single wide DataFrame indexed by quarter-end date, sorted ascending.
    Handles sparse/missing data gracefully — any unavailable field becomes NaN.
    """
    import yfinance as yf

    t = yf.Ticker(ticker)

    # ---- income statement ----
    try:
        inc = t.quarterly_financials.T.copy()
        inc.index = pd.to_datetime(inc.index)
        inc.index.name = "date"
    except Exception as exc:
        console.print(f"[yellow]  income statement failed for {ticker}: {exc}[/]")
        inc = pd.DataFrame()

    # ---- cash flow ----
    try:
        cf = t.quarterly_cashflow.T.copy()
        cf.index = pd.to_datetime(cf.index)
        cf.index.name = "date"
    except Exception as exc:
        console.print(f"[yellow]  cashflow failed for {ticker}: {exc}[/]")
        cf = pd.DataFrame()

    # ---- balance sheet ----
    try:
        bs = t.quarterly_balance_sheet.T.copy()
        bs.index = pd.to_datetime(bs.index)
        bs.index.name = "date"
    except Exception as exc:
        console.print(f"[yellow]  balance sheet failed for {ticker}: {exc}[/]")
        bs = pd.DataFrame()

    # ---- merge all three on date index ----
    frames = [f for f in [inc, cf, bs] if not f.empty]
    if not frames:
        return pd.DataFrame()

    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, how="outer", rsuffix="_dup")
    # drop duplicate columns from rsuffix
    df = df[[c for c in df.columns if not c.endswith("_dup")]]

    df = df.sort_index()
    df.index.name = "date"
    return df


def _fetch_yf_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily adjusted close + info.sharesOutstanding via yfinance."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    hist = t.history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        return pd.DataFrame()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    hist.index.name = "date"
    return hist[["Close", "Volume"]].copy()


def _fetch_cpi() -> pd.DataFrame:
    """Fetch CPIAUCSL from FRED (1997-01-01 to present)."""
    from fredapi import Fred

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Add it to your .env file."
        )
    fred = Fred(api_key=api_key)
    series = fred.get_series(
        "CPIAUCSL",
        observation_start="1997-01-01",
        observation_end="2026-03-28",
    )
    df = series.to_frame(name="cpi")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


# ── canonical column mapping from yfinance raw names ─────────────────────────

_INC_MAP = {
    "Total Revenue": "revenue",
    "Cost Of Revenue": "cost_of_revenue",
    "Gross Profit": "gross_profit",
    "Research And Development": "research_and_development",
    "Operating Income": "operating_income",
    "Net Income": "net_income",
    "Diluted EPS": "eps_diluted",
    "Diluted Average Shares": "weighted_avg_shares",
}

_CF_MAP = {
    "Operating Cash Flow": "operating_cash_flow",
    "Capital Expenditure": "capital_expenditure",
    "Free Cash Flow": "free_cash_flow_raw",
}

_BS_MAP = {
    "Total Assets": "total_assets",
    "Total Debt": "total_debt",
    "Stockholders Equity": "shareholders_equity",
    "Cash And Cash Equivalents": "cash",
}


def _remap_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename available columns per mapping, ignore missing ones."""
    rename = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=rename)


def _standardize_financials(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Apply all column mappings and add derived columns."""
    df = raw.copy()
    for mapping in [_INC_MAP, _CF_MAP, _BS_MAP]:
        df = _remap_columns(df, mapping)

    # Ensure expected columns exist (fill with NaN if absent)
    expected = list(_INC_MAP.values()) + list(_CF_MAP.values()) + list(_BS_MAP.values())
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    # Derive FCF = opCF + capex (capex is typically negative in yfinance)
    if "free_cash_flow_raw" in df.columns:
        df["free_cash_flow"] = df["free_cash_flow_raw"]
    else:
        df["free_cash_flow"] = df["operating_cash_flow"] + df["capital_expenditure"]

    df["ticker"] = ticker
    df["era"] = "ai" if ticker == "NVDA" else "dotcom"
    df["calendar_quarter"] = [map_fiscal_to_calendar(d) for d in df.index]

    df.index.name = "date"
    return df


# ── market cap from price * shares ───────────────────────────────────────────

def _build_market_cap_series(
    price_df: pd.DataFrame,
    financials_df: pd.DataFrame,
    ticker: str,
) -> pd.Series:
    """Approximate market cap at each quarter-end date.

    Strategy: use yfinance Ticker.info for current shares outstanding,
    then fall back to weighted_avg_shares from the income statement.
    Market cap = close_price * shares_outstanding on the nearest trading day
    to each quarter-end.
    """
    import yfinance as yf

    if price_df.empty or financials_df.empty:
        return pd.Series(dtype=float, name="market_cap")

    # Attempt to get shares from yfinance info (best effort)
    try:
        info = yf.Ticker(ticker).info
        shares_current = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    except Exception:
        shares_current = None

    quarter_ends = financials_df.index
    market_caps = {}

    for qdate in quarter_ends:
        # Find nearest trading day in price history
        idx = price_df.index.get_indexer([qdate], method="nearest")
        if idx[0] == -1 or idx[0] >= len(price_df):
            market_caps[qdate] = np.nan
            continue

        price = price_df.iloc[idx[0]]["Close"]

        # Prefer quarterly weighted avg shares if available
        shares = financials_df.loc[qdate, "weighted_avg_shares"] \
            if "weighted_avg_shares" in financials_df.columns else np.nan

        if pd.isna(shares) or shares <= 0:
            shares = shares_current

        if shares and not pd.isna(price):
            market_caps[qdate] = float(price) * float(shares)
        else:
            market_caps[qdate] = np.nan

    return pd.Series(market_caps, name="market_cap")


# ── inflation adjustment ──────────────────────────────────────────────────────

def adjust_for_inflation(
    values: pd.Series,
    cpi_df: pd.DataFrame,
    target_year: int = 2026,
) -> pd.Series:
    """Adjust nominal dollar values to real dollars at target_year (default 2026).

    Parameters
    ----------
    values:     Series indexed by datetime (quarter-end dates)
    cpi_df:     DataFrame with 'cpi' column, monthly index from FRED
    target_year: Reference year; uses latest available CPI within that year.

    Returns
    -------
    Series of inflation-adjusted values (same index as input).
    """
    cpi = cpi_df["cpi"].dropna()

    # Base CPI: latest observation in target_year (or the latest available)
    base_mask = cpi.index.year == target_year
    if base_mask.any():
        base_cpi = cpi[base_mask].iloc[-1]
    else:
        base_cpi = cpi.iloc[-1]
        console.print(
            f"[yellow]  CPI: no data for {target_year}, using latest: "
            f"{cpi.index[-1].date()} = {base_cpi:.2f}[/]"
        )

    adjusted = []
    for date, val in values.items():
        if pd.isna(val):
            adjusted.append(np.nan)
            continue
        nearest_idx = cpi.index.get_indexer([date], method="nearest")[0]
        period_cpi = cpi.iloc[nearest_idx]
        adjusted.append(val * (base_cpi / period_cpi))

    return pd.Series(adjusted, index=values.index, name=f"{values.name}_real")


# ── cycle alignment ───────────────────────────────────────────────────────────

def assign_cycle_quarter(df: pd.DataFrame, breakout_date: pd.Timestamp) -> pd.DataFrame:
    """Add cycle_quarter column: integer quarters since breakout_date (0 = breakout)."""
    df = df.copy()
    df["cycle_quarter"] = df.index.map(
        lambda d: round((d - breakout_date).days / 91.25)
    ).astype(int)
    return df


# ── analysis functions ────────────────────────────────────────────────────────

def compute_pe_trajectory(
    financials_df: pd.DataFrame,
    market_cap_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Trailing P/E = market cap / net income TTM.

    Parameters
    ----------
    financials_df:      DataFrame with 'net_income_ttm' column, indexed by date.
    market_cap_series:  Series indexed by date.

    Returns
    -------
    (pe_ratio, log_pe_ratio)
      pe_ratio   — raw trailing P/E, NaN for non-positive earnings.
      log_pe_ratio — log(P/E) for statistical tests; NaN where P/E is NaN.
    """
    mc = market_cap_series.reindex(financials_df.index)
    ni_ttm = financials_df.get("net_income_ttm", pd.Series(dtype=float))

    pe = mc / ni_ttm
    pe[ni_ttm <= 0] = np.nan
    pe[mc <= 0] = np.nan
    pe[pe <= 0] = np.nan

    log_pe = np.log(pe.replace(0, np.nan))
    pe.name = "pe_ratio"
    log_pe.name = "log_pe_ratio"
    return pe, log_pe


def compute_peg_ratio(
    pe_series: pd.Series,
    earnings_growth_series: pd.Series,
) -> pd.Series:
    """PEG = P/E / (YoY EPS growth rate expressed as whole number, e.g. 100 for 100%).

    This is the strongest 'this time is different' metric.
    PEG < 1.0 => cheap relative to growth.
    PEG > 2.0 => expensive even accounting for growth.
    NaN when growth is zero or negative (PEG is undefined).
    """
    growth = earnings_growth_series.copy()
    growth[growth <= 0] = np.nan  # PEG undefined for non-positive growth

    peg = pe_series / growth
    peg[peg <= 0] = np.nan
    peg.name = "peg_ratio"
    return peg


def compute_revenue_growth(financials_df: pd.DataFrame) -> pd.DataFrame:
    """Add revenue_yoy and revenue_qoq growth rate columns (fractional, not %).

    Requires 'revenue' column in financials_df, sorted ascending by date.
    """
    df = financials_df.sort_index().copy()
    if "revenue" not in df.columns:
        df["revenue_yoy"] = np.nan
        df["revenue_qoq"] = np.nan
        return df

    df["revenue_yoy"] = df["revenue"].pct_change(periods=4)
    df["revenue_qoq"] = df["revenue"].pct_change(periods=1)
    return df


def compute_fcf_yield(
    fcf_ttm_series: pd.Series,
    market_cap_series: pd.Series,
) -> pd.Series:
    """FCF yield = FCF TTM / market cap, expressed as a fraction (not %)."""
    mc = market_cap_series.reindex(fcf_ttm_series.index)
    yield_ = fcf_ttm_series / mc
    yield_.name = "fcf_yield"
    return yield_


def compute_rd_ratio(financials_df: pd.DataFrame) -> pd.Series:
    """R&D / revenue ratio (fractional)."""
    if "research_and_development" not in financials_df.columns or \
       "revenue" not in financials_df.columns:
        return pd.Series(dtype=float, name="rd_ratio")

    ratio = financials_df["research_and_development"] / financials_df["revenue"]
    ratio.name = "rd_ratio"
    return ratio


def compute_gross_margin(financials_df: pd.DataFrame) -> pd.Series:
    """Gross margin = gross profit / revenue (fractional)."""
    if "gross_profit" not in financials_df.columns or \
       "revenue" not in financials_df.columns:
        return pd.Series(dtype=float, name="gross_margin")

    margin = financials_df["gross_profit"] / financials_df["revenue"]
    margin.name = "gross_margin"
    return margin


# ── statistical tests ─────────────────────────────────────────────────────────

def _bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict:
    """Bootstrap test: are the means of a and b different?

    Returns observed difference, 95% CI from bootstrap, and p-value.
    """
    rng = np.random.default_rng(seed)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) < 2 or len(b) < 2:
        return {"error": "insufficient data", "p_value": np.nan}

    obs_diff = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        perm = rng.permutation(combined)
        diffs[i] = perm[: len(a)].mean() - perm[len(a):].mean()

    p_value = float(np.mean(np.abs(diffs) >= np.abs(obs_diff)))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    # Cohen's d
    pooled_std = np.sqrt(
        (np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2
    )
    cohens_d = float(obs_diff / pooled_std) if pooled_std > 0 else np.nan

    return {
        "observed_diff": float(obs_diff),
        "bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "p_value": p_value,
        "cohens_d": cohens_d,
        "n_a": len(a),
        "n_b": len(b),
        "significant": p_value < 0.05,
    }


def run_statistical_tests(
    nvda_fin: pd.DataFrame,
    csco_fin: pd.DataFrame,
) -> dict:
    """Run all required statistical tests and return results dict.

    Tests:
      1. Bootstrap test on log(P/E) distributions between eras (with Cohen's d).
      2. Shapiro-Wilk normality test on revenue growth rates.
      3. Durbin-Watson autocorrelation check on revenue growth series.
    """
    results: dict = {}

    # 1. Bootstrap on log(P/E)
    nvda_lpe = nvda_fin.get("log_pe_ratio", pd.Series(dtype=float)).dropna().values
    csco_lpe = csco_fin.get("log_pe_ratio", pd.Series(dtype=float)).dropna().values

    results["bootstrap_log_pe"] = _bootstrap_mean_diff(nvda_lpe, csco_lpe)
    results["bootstrap_log_pe"]["interpretation"] = (
        "log(P/E) mean comparison: NVDA vs CSCO dot-com era. "
        "Significant difference would indicate structurally different valuations."
    )

    # 2. Shapiro-Wilk on revenue growth
    for label, df in [("NVDA", nvda_fin), ("CSCO", csco_fin)]:
        col = "revenue_yoy"
        if col in df.columns:
            vals = df[col].dropna().values
            if len(vals) >= 3:
                stat, p = stats.shapiro(vals)
                results[f"shapiro_wilk_{label.lower()}_rev_growth"] = {
                    "statistic": float(stat),
                    "p_value": float(p),
                    "normal": p > 0.05,
                    "n": len(vals),
                    "interpretation": (
                        f"{label} revenue YoY growth is "
                        f"{'consistent with normal distribution' if p > 0.05 else 'non-normally distributed'} "
                        f"(W={stat:.4f}, p={p:.4f})"
                    ),
                }
            else:
                results[f"shapiro_wilk_{label.lower()}_rev_growth"] = {
                    "error": "insufficient data", "n": len(vals)
                }

    # 3. Durbin-Watson on revenue growth (autocorrelation)
    for label, df in [("NVDA", nvda_fin), ("CSCO", csco_fin)]:
        col = "revenue_yoy"
        if col in df.columns:
            vals = df[col].dropna().values
            if len(vals) >= 4:
                dw_stat = float(durbin_watson(vals))
                # DW ~ 2 => no autocorrelation; < 1.5 => positive; > 2.5 => negative
                interpretation = (
                    "positive autocorrelation" if dw_stat < 1.5
                    else "negative autocorrelation" if dw_stat > 2.5
                    else "no significant autocorrelation"
                )
                results[f"durbin_watson_{label.lower()}_rev_growth"] = {
                    "statistic": dw_stat,
                    "interpretation": f"{label}: {interpretation} (DW={dw_stat:.3f})",
                }
            else:
                results[f"durbin_watson_{label.lower()}_rev_growth"] = {
                    "error": "insufficient data"
                }

    return results


# ── chart helpers ─────────────────────────────────────────────────────────────

def _save_chart(fig: go.Figure, name: str) -> dict[str, str]:
    """Save chart as JSON + PNG. Returns paths dict."""
    json_path = CHART_DIR / f"{name}.json"
    png_path = CHART_DIR / f"{name}.png"

    fig.write_json(str(json_path))
    try:
        fig.write_image(str(png_path), width=1600, height=900, scale=2)
        console.print(f"[green]  Saved:[/] {png_path.name}")
    except Exception as exc:
        console.print(f"[yellow]  PNG export failed ({exc}) — JSON saved only[/]")
        png_path = None

    return {
        "json": str(json_path),
        "png": str(png_path) if png_path else None,
    }


def _dark_layout(title: str, xaxis_title: str, yaxis_title: str, **kwargs) -> dict:
    """Return common dark-theme layout kwargs."""
    return dict(
        title=dict(text=title, font=dict(size=18)),
        template=PLOTLY_TEMPLATE,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#555", borderwidth=1),
        font=dict(family="Inter, Arial, sans-serif", size=13),
        margin=dict(l=70, r=40, t=80, b=60),
        **kwargs,
    )


# ── chart functions ───────────────────────────────────────────────────────────

def chart_2_1_pe_trajectory(
    nvda_fin: pd.DataFrame,
    csco_fin: pd.DataFrame,
) -> dict[str, str]:
    """P/E ratio over time, aligned by cycle phase (cycle_quarter)."""
    fig = go.Figure()

    for df, ticker, color, dash in [
        (nvda_fin, "NVDA", NVDA_COLOR, "solid"),
        (csco_fin, "CSCO", CSCO_COLOR, "dash"),
    ]:
        if "pe_ratio" not in df.columns or "cycle_quarter" not in df.columns:
            continue
        valid = df[["cycle_quarter", "pe_ratio"]].dropna()
        if valid.empty:
            continue
        fig.add_trace(go.Scatter(
            x=valid["cycle_quarter"],
            y=valid["pe_ratio"],
            mode="lines+markers",
            name=f"{ticker} ({('2023-2026' if ticker == 'NVDA' else '1997-2003')})",
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=7),
        ))

    # Reasonable valuation band P/E 20-40
    fig.add_hrect(
        y0=20, y1=40,
        fillcolor="rgba(255,255,0,0.07)",
        line_width=0,
        annotation_text="Reasonable zone (P/E 20-40)",
        annotation_position="top left",
        annotation_font_color="#aaa",
    )
    # Historical tech average
    fig.add_hline(
        y=40, line_dash="dot", line_color="orange", line_width=1.5,
        annotation_text="Historical tech avg P/E ~40",
        annotation_position="bottom right",
        annotation_font_color="orange",
    )

    fig.update_layout(**_dark_layout(
        title="P/E Ratio Trajectory: Is NVDA's Valuation as Stretched as CSCO's?",
        xaxis_title="Quarters Since Cycle Breakout",
        yaxis_title="Trailing P/E Ratio",
    ))
    fig.update_yaxes(type="log", tickformat=".0f")

    return _save_chart(fig, "chart_2_1_pe_trajectory")


def chart_2_2_revenue_growth(
    nvda_fin: pd.DataFrame,
    csco_fin: pd.DataFrame,
) -> dict[str, str]:
    """Grouped bar chart of YoY revenue growth aligned by cycle quarter."""
    fig = go.Figure()

    for df, ticker, color in [
        (nvda_fin, "NVDA", NVDA_COLOR),
        (csco_fin, "CSCO", CSCO_COLOR),
    ]:
        if "revenue_yoy" not in df.columns or "cycle_quarter" not in df.columns:
            continue
        valid = df[["cycle_quarter", "revenue_yoy"]].dropna()
        if valid.empty:
            continue
        yoy_pct = valid["revenue_yoy"] * 100
        fig.add_trace(go.Bar(
            x=valid["cycle_quarter"],
            y=yoy_pct,
            name=f"{ticker}",
            marker_color=color,
            text=[f"{v:.0f}%" for v in yoy_pct],
            textposition="outside",
            textfont=dict(size=10),
        ))

    fig.add_hline(y=0, line_color="#888", line_width=1)

    fig.update_layout(
        **_dark_layout(
            title="Revenue Growth: NVDA's Explosive Growth vs CSCO's Steady Climb",
            xaxis_title="Quarters Since Cycle Breakout",
            yaxis_title="YoY Revenue Growth (%)",
        ),
        barmode="group",
    )

    return _save_chart(fig, "chart_2_2_revenue_growth")


def chart_2_3_mcap_vs_revenue(
    nvda_fin: pd.DataFrame,
    csco_fin: pd.DataFrame,
) -> dict[str, str]:
    """Scatter: x=revenue (real), y=market cap (real), bubble size=P/E ratio."""
    fig = go.Figure()

    for df, ticker, color in [
        (nvda_fin, "NVDA", NVDA_COLOR),
        (csco_fin, "CSCO", CSCO_COLOR),
    ]:
        rev_col = "revenue_ttm_real" if "revenue_ttm_real" in df.columns else "revenue_real"
        mc_col = "market_cap_real" if "market_cap_real" in df.columns else "market_cap"
        pe_col = "pe_ratio"

        needed = [rev_col, mc_col]
        if not all(c in df.columns for c in needed):
            continue

        plot_df = df[needed + ([pe_col] if pe_col in df.columns else [])].dropna(
            subset=needed
        )
        if plot_df.empty:
            continue

        pe_vals = plot_df[pe_col].clip(upper=200).fillna(30) if pe_col in plot_df.columns else 30
        bubble_size = (pe_vals / pe_vals.max() * 50).clip(lower=6)

        hover_text = [
            f"{ticker} {idx.date()}<br>"
            f"Revenue: ${row[rev_col]/1e9:.1f}B<br>"
            f"Mkt Cap: ${row[mc_col]/1e9:.0f}B<br>"
            f"P/E: {row[pe_col]:.1f}" if pe_col in row.index else ""
            for idx, row in plot_df.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=plot_df[rev_col] / 1e9,
            y=plot_df[mc_col] / 1e9,
            mode="markers+lines",
            name=ticker,
            marker=dict(
                size=bubble_size,
                color=color,
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            line=dict(color=color, width=1, dash="dot"),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ))

    # Reference lines P/S = 10x and 30x
    rev_range = np.logspace(0, 3, 200)
    for ps, clr, label in [(10, "#888", "P/S = 10x"), (30, "#aa4444", "P/S = 30x")]:
        fig.add_trace(go.Scatter(
            x=rev_range,
            y=rev_range * ps,
            mode="lines",
            name=label,
            line=dict(color=clr, width=1, dash="dot"),
            opacity=0.5,
        ))

    fig.update_layout(
        **_dark_layout(
            title="Valuation vs Revenue: Are AI Stocks More Grounded Than Dot-Com?",
            xaxis_title="TTM Revenue (2026 USD, Billions)",
            yaxis_title="Market Cap (2026 USD, Billions)",
        ),
        xaxis_type="log",
        yaxis_type="log",
    )

    return _save_chart(fig, "chart_2_3_mcap_vs_revenue")


def chart_2_4_fcf_yield(
    nvda_fin: pd.DataFrame,
    csco_fin: pd.DataFrame,
) -> dict[str, str]:
    """FCF yield timeline for NVDA and CSCO."""
    fig = go.Figure()

    for df, ticker, color in [
        (nvda_fin, "NVDA", NVDA_COLOR),
        (csco_fin, "CSCO", CSCO_COLOR),
    ]:
        if "fcf_yield" not in df.columns or "cycle_quarter" not in df.columns:
            continue
        valid = df[["cycle_quarter", "fcf_yield"]].dropna()
        if valid.empty:
            continue
        yield_pct = valid["fcf_yield"] * 100

        fig.add_trace(go.Scatter(
            x=valid["cycle_quarter"],
            y=yield_pct,
            mode="lines+markers",
            name=ticker,
            line=dict(color=color, width=2.5),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba")
            if color.startswith("rgb") else f"rgba({int(color[1:3],16)},"
            f"{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
        ))

    # S&P 500 average FCF yield reference
    fig.add_hline(
        y=2.0, line_dash="dash", line_color="#aaa", line_width=1.5,
        annotation_text="S&P 500 avg FCF yield ~2%",
        annotation_position="top right",
        annotation_font_color="#aaa",
    )
    fig.add_hline(y=0, line_color="#555", line_width=1)

    fig.update_layout(**_dark_layout(
        title="Free Cash Flow Yield: Real Returns to Shareholders",
        xaxis_title="Quarters Since Cycle Breakout",
        yaxis_title="FCF Yield (%)",
    ))

    return _save_chart(fig, "chart_2_4_fcf_yield")


def chart_2_5_rd_investment(
    nvda_fin: pd.DataFrame,
    csco_fin: pd.DataFrame,
) -> dict[str, str]:
    """R&D-to-revenue ratio comparison: dual line chart."""
    fig = go.Figure()

    for df, ticker, color, dash in [
        (nvda_fin, "NVDA", NVDA_COLOR, "solid"),
        (csco_fin, "CSCO", CSCO_COLOR, "dash"),
    ]:
        if "rd_ratio" not in df.columns or "cycle_quarter" not in df.columns:
            continue
        valid = df[["cycle_quarter", "rd_ratio"]].dropna()
        if valid.empty:
            continue
        rd_pct = valid["rd_ratio"] * 100

        fig.add_trace(go.Scatter(
            x=valid["cycle_quarter"],
            y=rd_pct,
            mode="lines+markers",
            name=f"{ticker} R&D/Revenue",
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=6),
        ))

    # Healthy range band for semiconductor companies
    fig.add_hrect(
        y0=15, y1=30,
        fillcolor="rgba(100,200,100,0.06)",
        line_width=0,
        annotation_text="Healthy semiconductor R&D range (15-30%)",
        annotation_position="top left",
        annotation_font_color="#aaa",
    )

    fig.update_layout(**_dark_layout(
        title="R&D Investment Intensity: Building the Moat",
        xaxis_title="Quarters Since Cycle Breakout",
        yaxis_title="R&D / Revenue (%)",
    ))

    return _save_chart(fig, "chart_2_5_rd_investment")


def chart_2_6_peg_ratio(
    nvda_fin: pd.DataFrame,
    csco_fin: pd.DataFrame,
) -> dict[str, str]:
    """PEG ratio comparison — the key differentiator chart.

    Shaded zones:
      Green  (0-1):  cheap relative to growth
      Yellow (1-2):  fair
      Red    (2+):   expensive even for growth
    """
    fig = go.Figure()

    # Shaded zones
    zone_specs = [
        (0, 1, "rgba(0,200,100,0.12)", "Cheap (PEG < 1)"),
        (1, 2, "rgba(255,200,0,0.10)", "Fair (PEG 1-2)"),
        (2, 6, "rgba(255,60,60,0.10)", "Expensive (PEG > 2)"),
    ]
    for y0, y1, color, label in zone_specs:
        fig.add_hrect(
            y0=y0, y1=y1,
            fillcolor=color,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font_color="#bbb",
            annotation_font_size=11,
        )

    # Lines
    for df, ticker, color, dash, width in [
        (nvda_fin, "NVDA", NVDA_COLOR, "solid", 2.5),
        (csco_fin, "CSCO", CSCO_COLOR, "dash", 2.5),
    ]:
        if "peg_ratio" not in df.columns or "cycle_quarter" not in df.columns:
            continue
        valid = df[["cycle_quarter", "peg_ratio"]].dropna()
        if valid.empty:
            continue

        fig.add_trace(go.Scatter(
            x=valid["cycle_quarter"],
            y=valid["peg_ratio"],
            mode="lines+markers",
            name=f"{ticker}",
            line=dict(color=color, width=width, dash=dash),
            marker=dict(size=7),
        ))

        # Annotation at PEG peak for CSCO, current for NVDA
        if ticker == "CSCO" and not valid.empty:
            peak_row = valid.loc[valid["peg_ratio"].idxmax()]
            fig.add_annotation(
                x=peak_row["cycle_quarter"],
                y=peak_row["peg_ratio"],
                text=f"CSCO peak PEG ~{peak_row['peg_ratio']:.1f}",
                showarrow=True,
                arrowhead=2,
                font=dict(color=CSCO_COLOR, size=12),
                arrowcolor=CSCO_COLOR,
            )
        if ticker == "NVDA" and not valid.empty:
            last_row = valid.iloc[-1]
            fig.add_annotation(
                x=last_row["cycle_quarter"],
                y=last_row["peg_ratio"],
                text=f"NVDA current PEG ~{last_row['peg_ratio']:.2f}",
                showarrow=True,
                arrowhead=2,
                font=dict(color=NVDA_COLOR, size=12),
                arrowcolor=NVDA_COLOR,
                xshift=10,
            )

    fig.update_layout(
        **_dark_layout(
            title="PEG Ratio: The Strongest 'This Time Is Different' Argument",
            xaxis_title="Quarters Since Cycle Breakout",
            yaxis_title="PEG Ratio",
        ),
        yaxis_range=[0, 6.5],
    )

    return _save_chart(fig, "chart_2_6_peg_ratio")


# ── data completeness check ───────────────────────────────────────────────────

def check_data_completeness(
    df: pd.DataFrame,
    ticker: str,
    expected_quarters: int,
) -> dict:
    """Report data completeness for a ticker."""
    actual = len(df)
    completeness = actual / expected_quarters * 100 if expected_quarters > 0 else 0
    missing_fields = df.isnull().sum()
    return {
        "ticker": ticker,
        "expected_quarters": expected_quarters,
        "actual_quarters": actual,
        "completeness_pct": round(completeness, 1),
        "missing_by_field": missing_fields.to_dict(),
        "flagged": completeness < 60,
    }


# ── main runner ───────────────────────────────────────────────────────────────

def run_layer2(force_refresh: bool = False) -> dict:
    """Execute Layer 2: Fundamental Comparison.

    Parameters
    ----------
    force_refresh: If True, bypass all caches and re-fetch from APIs.

    Returns
    -------
    dict with keys:
        data    — {"nvda": DataFrame, "csco": DataFrame, "cpi": DataFrame}
        stats   — statistical test results
        charts  — paths to generated chart files
        findings — list of string insight summaries
    """
    console.rule("[bold cyan]Layer 2: Fundamental Comparison[/]")
    findings: list[str] = []
    chart_paths: dict[str, dict] = {}

    # ── 1. Fetch raw quarterly financials ────────────────────────────────────
    console.print("\n[bold]1. Fetching quarterly financials...[/]")

    nvda_raw = cache_or_call(
        "nvda_quarterly_financials",
        lambda: _fetch_yf_financials("NVDA"),
        force_refresh=force_refresh,
    )
    csco_raw = cache_or_call(
        "csco_quarterly_financials",
        lambda: _fetch_yf_financials("CSCO"),
        force_refresh=force_refresh,
    )

    # ── 2. Fetch price history for market cap ────────────────────────────────
    console.print("\n[bold]2. Fetching price history...[/]")

    nvda_prices = cache_or_call(
        "nvda_price_history",
        lambda: _fetch_yf_prices("NVDA", "2022-01-01", "2026-03-28"),
        force_refresh=force_refresh,
    )
    csco_prices = cache_or_call(
        "csco_price_history",
        lambda: _fetch_yf_prices("CSCO", "1997-01-01", "2003-12-31"),
        force_refresh=force_refresh,
    )

    # ── 3. Fetch CPI ─────────────────────────────────────────────────────────
    console.print("\n[bold]3. Fetching CPI from FRED...[/]")

    try:
        cpi_df = cache_or_call(
            "cpi_aucsl",
            _fetch_cpi,
            force_refresh=force_refresh,
        )
    except EnvironmentError as exc:
        console.print(f"[red]  FRED unavailable: {exc}[/]")
        console.print("[yellow]  Proceeding without inflation adjustment.[/]")
        cpi_df = pd.DataFrame()
        findings.append(
            "FRED_API_KEY not set — inflation adjustment skipped. "
            "Set FRED_API_KEY in .env to enable real-dollar comparisons."
        )

    # ── 4. Standardize and filter date ranges ────────────────────────────────
    console.print("\n[bold]4. Standardizing columns and filtering date ranges...[/]")

    def _build(raw: pd.DataFrame, ticker: str, start: str, end: str) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()
        df = _standardize_financials(raw, ticker)
        # Filter to relevant era
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        df = df[mask].copy()
        return df

    nvda_fin = _build(nvda_raw, "NVDA", "2022-10-01", "2026-03-31")
    csco_fin = _build(csco_raw, "CSCO", "1997-07-01", "2003-07-31")

    if nvda_fin.empty:
        findings.append("WARNING: NVDA quarterly financials are empty — check yfinance.")
    if csco_fin.empty:
        findings.append(
            "WARNING: CSCO historical (1997-2003) quarterly data is sparse/empty via yfinance. "
            "This is a known limitation — yfinance has poor coverage for pre-2005 data. "
            "Consider FMP or manual EDGAR data for full dot-com era coverage."
        )

    # ── 5. Compute TTM rolling metrics ───────────────────────────────────────
    console.print("\n[bold]5. Computing TTM metrics...[/]")
    TTM_COLS = ["revenue", "net_income", "free_cash_flow", "operating_cash_flow",
                "research_and_development"]

    if not nvda_fin.empty:
        nvda_fin = compute_ttm(nvda_fin, TTM_COLS)
    if not csco_fin.empty:
        csco_fin = compute_ttm(csco_fin, TTM_COLS)

    # ── 6. Build market cap series ───────────────────────────────────────────
    console.print("\n[bold]6. Building market cap series...[/]")

    nvda_mcap = pd.Series(dtype=float, name="market_cap")
    csco_mcap = pd.Series(dtype=float, name="market_cap")

    if not nvda_fin.empty and not nvda_prices.empty:
        nvda_mcap = _build_market_cap_series(nvda_prices, nvda_fin, "NVDA")
    if not csco_fin.empty and not csco_prices.empty:
        csco_mcap = _build_market_cap_series(csco_prices, csco_fin, "CSCO")

    # Attach market cap to financials
    if not nvda_fin.empty:
        nvda_fin["market_cap"] = nvda_mcap.reindex(nvda_fin.index)
    if not csco_fin.empty:
        csco_fin["market_cap"] = csco_mcap.reindex(csco_fin.index)

    # ── 7. Inflation adjustment ──────────────────────────────────────────────
    console.print("\n[bold]7. Applying inflation adjustment...[/]")
    NOMINAL_COLS = ["revenue", "net_income", "free_cash_flow", "market_cap",
                    "research_and_development", "revenue_ttm", "net_income_ttm",
                    "free_cash_flow_ttm"]

    if not cpi_df.empty:
        for df in [nvda_fin, csco_fin]:
            if df.empty:
                continue
            for col in NOMINAL_COLS:
                if col in df.columns:
                    df[f"{col}_real"] = adjust_for_inflation(
                        df[col], cpi_df, target_year=2026
                    ).values

    # ── 8. Derived metrics ───────────────────────────────────────────────────
    console.print("\n[bold]8. Computing derived metrics...[/]")

    for df, mcap in [(nvda_fin, nvda_mcap), (csco_fin, csco_mcap)]:
        if df.empty:
            continue

        # Revenue growth
        df[["revenue_yoy", "revenue_qoq"]] = compute_revenue_growth(df)[
            ["revenue_yoy", "revenue_qoq"]
        ]

        # P/E trajectory
        pe, log_pe = compute_pe_trajectory(df, df.get("market_cap", pd.Series()))
        df["pe_ratio"] = pe.values
        df["log_pe_ratio"] = log_pe.values

        # EPS YoY growth for PEG
        if "eps_diluted" in df.columns:
            df["eps_yoy_growth_pct"] = df["eps_diluted"].pct_change(periods=4) * 100

        # PEG ratio
        if "eps_yoy_growth_pct" in df.columns:
            peg = compute_peg_ratio(
                df["pe_ratio"],
                df["eps_yoy_growth_pct"],
            )
            df["peg_ratio"] = peg.values

        # FCF yield
        if "free_cash_flow_ttm" in df.columns and "market_cap" in df.columns:
            df["fcf_yield"] = compute_fcf_yield(
                df["free_cash_flow_ttm"], df["market_cap"]
            ).values

        # R&D ratio
        df["rd_ratio"] = compute_rd_ratio(df).reindex(df.index).values

        # Gross margin
        df["gross_margin"] = compute_gross_margin(df).reindex(df.index).values

    # ── 9. Assign cycle quarters ─────────────────────────────────────────────
    console.print("\n[bold]9. Assigning cycle quarters...[/]")

    if not nvda_fin.empty:
        nvda_fin = assign_cycle_quarter(nvda_fin, NVDA_BREAKOUT)
    if not csco_fin.empty:
        csco_fin = assign_cycle_quarter(csco_fin, CSCO_BREAKOUT)

    # ── 10. Data completeness check ──────────────────────────────────────────
    console.print("\n[bold]10. Checking data completeness...[/]")

    nvda_completeness = check_data_completeness(nvda_fin, "NVDA", expected_quarters=17)
    csco_completeness = check_data_completeness(csco_fin, "CSCO", expected_quarters=24)

    console.print(
        f"  NVDA: {nvda_completeness['actual_quarters']}/{nvda_completeness['expected_quarters']} "
        f"quarters ({nvda_completeness['completeness_pct']}%)"
    )
    console.print(
        f"  CSCO: {csco_completeness['actual_quarters']}/{csco_completeness['expected_quarters']} "
        f"quarters ({csco_completeness['completeness_pct']}%)"
    )

    if csco_completeness["flagged"]:
        findings.append(
            f"DATA QUALITY: CSCO historical data completeness is "
            f"{csco_completeness['completeness_pct']}% "
            f"({csco_completeness['actual_quarters']}/{csco_completeness['expected_quarters']} quarters). "
            "yfinance is unreliable for pre-2005 data. "
            "Cross-reference against SEC EDGAR (CIK 858877) for FY1999-Q4, "
            "FY2000-Q2, FY2000-Q4, FY2001-Q2 is strongly recommended."
        )

    # ── 11. Statistical tests ────────────────────────────────────────────────
    console.print("\n[bold]11. Running statistical tests...[/]")
    stats_results = run_statistical_tests(nvda_fin, csco_fin)

    # Interpret bootstrap result
    boot = stats_results.get("bootstrap_log_pe", {})
    if "p_value" in boot and not np.isnan(boot["p_value"]):
        sig = "SIGNIFICANT" if boot["significant"] else "not significant"
        findings.append(
            f"Bootstrap test on log(P/E): NVDA vs CSCO difference is {sig} "
            f"(p={boot['p_value']:.3f}, Cohen's d={boot.get('cohens_d', float('nan')):.2f}). "
            f"95% CI for mean diff: {boot['bootstrap_ci_95']}."
        )

    # ── 12. Generate charts ───────────────────────────────────────────────────
    console.print("\n[bold]12. Generating charts...[/]")

    chart_fns = [
        ("chart_2_1", chart_2_1_pe_trajectory),
        ("chart_2_2", chart_2_2_revenue_growth),
        ("chart_2_3", chart_2_3_mcap_vs_revenue),
        ("chart_2_4", chart_2_4_fcf_yield),
        ("chart_2_5", chart_2_5_rd_investment),
        ("chart_2_6", chart_2_6_peg_ratio),
    ]
    for chart_key, fn in chart_fns:
        try:
            paths = fn(nvda_fin, csco_fin)
            chart_paths[chart_key] = paths
        except Exception as exc:
            console.print(f"[red]  Chart {chart_key} failed: {exc}[/]")
            chart_paths[chart_key] = {"error": str(exc)}

    # ── 13. Synthesize findings ───────────────────────────────────────────────
    console.print("\n[bold]13. Synthesizing key findings...[/]")

    # Revenue growth comparison
    if "revenue_yoy" in nvda_fin.columns and "revenue_yoy" in csco_fin.columns:
        nvda_peak_growth = nvda_fin["revenue_yoy"].max()
        csco_peak_growth = csco_fin["revenue_yoy"].max()
        if not pd.isna(nvda_peak_growth) and not pd.isna(csco_peak_growth):
            findings.append(
                f"REVENUE GROWTH: NVDA peak YoY growth was {nvda_peak_growth*100:.0f}% "
                f"vs CSCO peak of {csco_peak_growth*100:.0f}%. "
                f"NVDA's growth rate is {nvda_peak_growth/csco_peak_growth:.1f}x higher — "
                "the single strongest 'this time is different' argument."
            )

    # PEG comparison at common cycle quarter
    if "peg_ratio" in nvda_fin.columns and "peg_ratio" in csco_fin.columns:
        nvda_peg_median = nvda_fin["peg_ratio"].median()
        csco_peg_median = csco_fin["peg_ratio"].median()
        if not pd.isna(nvda_peg_median) and not pd.isna(csco_peg_median):
            findings.append(
                f"PEG RATIO: NVDA median PEG = {nvda_peg_median:.2f} "
                f"vs CSCO median PEG = {csco_peg_median:.2f}. "
                f"NVDA is {'cheap' if nvda_peg_median < 1 else 'fair' if nvda_peg_median < 2 else 'expensive'} "
                "relative to its earnings growth rate."
            )

    # Gross margin comparison
    if "gross_margin" in nvda_fin.columns and "gross_margin" in csco_fin.columns:
        nvda_gm = nvda_fin["gross_margin"].dropna()
        csco_gm = csco_fin["gross_margin"].dropna()
        if not nvda_gm.empty and not csco_gm.empty:
            findings.append(
                f"GROSS MARGINS: NVDA avg {nvda_gm.mean()*100:.1f}% "
                f"(latest {nvda_gm.iloc[-1]*100:.1f}%) "
                f"vs CSCO avg {csco_gm.mean()*100:.1f}%. "
                f"NVDA margin {'expansion' if nvda_gm.iloc[-1] > nvda_gm.iloc[0] else 'compression'} "
                "indicates real pricing power."
            )

    # FCF yield
    if "fcf_yield" in nvda_fin.columns:
        nvda_fcf = nvda_fin["fcf_yield"].dropna()
        if not nvda_fcf.empty:
            findings.append(
                f"FCF YIELD: NVDA latest FCF yield = {nvda_fcf.iloc[-1]*100:.2f}%, "
                f"avg = {nvda_fcf.mean()*100:.2f}%. "
                "Positive FCF yield confirms cash-generative business (unlike early dot-com era)."
            )

    # ── 14. Cache processed results ─────────────────────────────────────────
    summary = {
        "nvda_completeness": nvda_completeness,
        "csco_completeness": csco_completeness,
        "stats": stats_results,
        "findings": findings,
        "chart_paths": chart_paths,
    }
    cache_json("layer2_summary", summary, subdir="processed")

    console.rule("[bold green]Layer 2 Complete[/]")
    for i, finding in enumerate(findings, 1):
        console.print(f"  [dim]{i}.[/] {finding}")

    return {
        "data": {
            "nvda": nvda_fin,
            "csco": csco_fin,
            "cpi": cpi_df,
        },
        "stats": stats_results,
        "charts": chart_paths,
        "findings": findings,
        "completeness": {
            "nvda": nvda_completeness,
            "csco": csco_completeness,
        },
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Layer 2: Fundamental Comparison")
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Bypass cache and re-fetch all data from APIs"
    )
    args = parser.parse_args()

    result = run_layer2(force_refresh=args.force_refresh)

    console.print("\n[bold]Charts saved:[/]")
    for key, paths in result["charts"].items():
        if "json" in paths:
            console.print(f"  {key}: {paths['json']}")
        if paths.get("png"):
            console.print(f"       {paths['png']}")
