"""
Layer 4: Macro Environment Comparison
File: src/layers/layer4_macro.py

Compares macroeconomic conditions between the dot-com era (1996-2003)
and the AI era (2021-2026) using FRED API data.

ALL findings are framed as ASSOCIATIONS, not causal claims.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from fredapi import Fred
from plotly.subplots import make_subplots
from rich.console import Console
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

load_dotenv()

# ---------------------------------------------------------------------------
# Project path resolution (works regardless of cwd)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]

from src.utils.cache import cache_or_call  # noqa: E402

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_dark"
DOTCOM_COLOR = "#EF553B"
AI_ERA_COLOR = "#00CC96"
CHART_DIR = _PROJECT_ROOT / "submissions" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

FRED_START = "1995-01-01"
FRED_END = "2026-03-28"

SERIES_IDS = [
    "FEDFUNDS",
    "DGS10",
    "DGS2",
    "T10Y2Y",
    "M2SL",
    "CPIAUCSL",
    "BAA10Y",
    "GDP",
    "UNRATE",
    "SP500",
]

DAILY_SERIES: set[str] = {"DGS10", "DGS2", "T10Y2Y", "BAA10Y", "SP500"}
QUARTERLY_SERIES: set[str] = {"GDP"}

DOTCOM_RANGE = ("1996-01", "2003-12")
AI_RANGE = ("2021-01", "2026-03")

PEAK_DATES: dict[str, pd.Timestamp] = {
    "dotcom": pd.Timestamp("2000-03-31"),
    "ai": pd.Timestamp("2025-01-31"),  # Provisional; update from Layer 1
}

INDICATORS = [
    "fed_funds_rate",
    "yield_curve_spread",
    "m2_yoy_growth",
    "cpi_yoy_inflation",
    "real_interest_rate",
    "credit_spread",
    "gdp_yoy_growth",
    "unemployment",
]

# ---------------------------------------------------------------------------
# 1. FRED data fetch helpers
# ---------------------------------------------------------------------------


def _get_fred_client() -> Fred:
    """Return an authenticated fredapi.Fred client."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY is not set. Add it to your .env file."
        )
    return Fred(api_key=api_key)


def _fetch_series(
    fred: Fred,
    series_id: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch a single FRED series and return as a single-column DataFrame."""
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    df = s.to_frame(series_id)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    # FRED returns "." as missing — fredapi already converts to NaN via float cast,
    # but guard explicitly.
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df


def fetch_all_series(
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch all 10 FRED series, using cache where available.

    Returns a dict: series_id -> DataFrame(index=date, columns=[series_id]).
    """
    fred = _get_fred_client()
    raw: dict[str, pd.DataFrame] = {}

    for series_id in SERIES_IDS:
        sid = series_id  # capture for closure
        df = cache_or_call(
            f"fred_{sid}",
            lambda s=sid: _fetch_series(fred, s, FRED_START, FRED_END),
            subdir="raw",
            force_refresh=force_refresh,
        )
        raw[sid] = df

    return raw


# ---------------------------------------------------------------------------
# 2. GDP YoY growth — compute on raw quarterly before forward-filling
# ---------------------------------------------------------------------------


def _compute_gdp_yoy_growth(gdp_raw: pd.DataFrame) -> pd.Series:
    """Compute GDP YoY growth on quarterly data, then forward-fill to monthly.

    pct_change(periods=4) on quarterly series == year-over-year.
    Forward-filling to monthly avoids the artificial flat-period artifact
    that results from applying pct_change(12) on an already-ffilled monthly series.
    """
    col = "GDP"
    # Ensure proper datetime index and quarterly frequency
    s = gdp_raw[col].dropna().sort_index()
    yoy_q = s.pct_change(periods=4) * 100
    # Forward-fill quarterly YoY to monthly end-of-month
    yoy_monthly = yoy_q.resample("M").ffill()
    yoy_monthly.name = "gdp_yoy_growth"
    return yoy_monthly


# ---------------------------------------------------------------------------
# 3. Frequency alignment
# ---------------------------------------------------------------------------


def _align_series_to_monthly(df: pd.DataFrame, series_id: str) -> pd.Series:
    """Resample a series to monthly frequency per the alignment rules.

    Daily   -> month mean
    Quarterly -> forward-fill (level only; YoY computed separately)
    Monthly -> resample to ensure end-of-month alignment
    """
    s = df[series_id].sort_index()

    if series_id in DAILY_SERIES:
        return s.resample("M").mean()
    if series_id in QUARTERLY_SERIES:
        return s.resample("M").ffill()
    # Monthly: take last observation in the month (already monthly, but harmonise index)
    return s.resample("M").last()


# ---------------------------------------------------------------------------
# 4. Build unified monthly DataFrame
# ---------------------------------------------------------------------------


def build_macro_dashboard_data(
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch all FRED series, align to monthly, compute derived metrics.

    Returns a unified monthly DataFrame covering 1995-01 to today with columns:
        date, era, months_from_peak,
        fed_funds_rate, treasury_10y, treasury_2y, yield_curve_spread,
        m2_supply, m2_yoy_growth, cpi_index, cpi_yoy_inflation,
        real_interest_rate, credit_spread, gdp, gdp_yoy_growth,
        unemployment, sp500
    """
    raw = fetch_all_series(force_refresh=force_refresh)

    # GDP YoY must be computed on raw quarterly before any ffill
    gdp_yoy_monthly = _compute_gdp_yoy_growth(raw["GDP"])

    # Align every series to monthly end-of-month
    monthly: dict[str, pd.Series] = {}
    for sid in SERIES_IDS:
        monthly[sid] = _align_series_to_monthly(raw[sid], sid)

    # Build unified DataFrame from the full date range
    unified = pd.DataFrame(
        {
            "fed_funds_rate": monthly["FEDFUNDS"],
            "treasury_10y": monthly["DGS10"],
            "treasury_2y": monthly["DGS2"],
            "yield_curve_spread": monthly["T10Y2Y"],
            "m2_supply": monthly["M2SL"],
            "cpi_index": monthly["CPIAUCSL"],
            "credit_spread": monthly["BAA10Y"],
            "gdp": monthly["GDP"],
            "unemployment": monthly["UNRATE"],
            "sp500": monthly["SP500"],
        }
    )
    unified.index.name = "date"
    unified = unified.reset_index()
    unified["date"] = pd.to_datetime(unified["date"])

    # Merge in correctly-computed GDP YoY
    gdp_yoy_df = gdp_yoy_monthly.reset_index()
    gdp_yoy_df.columns = ["date", "gdp_yoy_growth"]
    gdp_yoy_df["date"] = pd.to_datetime(gdp_yoy_df["date"])
    unified = unified.merge(gdp_yoy_df, on="date", how="left")

    # ------------------------------------------------------------------
    # Derived metrics (computed on the full series, not era-split, to
    # avoid look-ahead bias at era boundaries.  pct_change(12) over the
    # global sorted series is correct here because we want trailing YoY.)
    # ------------------------------------------------------------------
    unified = unified.sort_values("date").reset_index(drop=True)

    unified["m2_yoy_growth"] = unified["m2_supply"].pct_change(periods=12) * 100
    unified["cpi_yoy_inflation"] = unified["cpi_index"].pct_change(periods=12) * 100
    unified["real_interest_rate"] = (
        unified["fed_funds_rate"] - unified["cpi_yoy_inflation"]
    )

    # ------------------------------------------------------------------
    # Era labels (rows outside both eras get None — kept for full history)
    # ------------------------------------------------------------------
    # Build period-end timestamps from "YYYY-MM" strings safely.
    def _period_end(ym: str) -> pd.Timestamp:
        """Return last moment of month given 'YYYY-MM'."""
        return pd.Period(ym, freq="M").to_timestamp(how="end").normalize()

    dotcom_start = pd.Timestamp(DOTCOM_RANGE[0])
    dotcom_end = _period_end(DOTCOM_RANGE[1])
    ai_start = pd.Timestamp(AI_RANGE[0])
    ai_end = _period_end(AI_RANGE[1])

    unified["era"] = None
    unified.loc[
        unified["date"].between(dotcom_start, dotcom_end), "era"
    ] = "dotcom"
    unified.loc[
        unified["date"].between(ai_start, ai_end), "era"
    ] = "ai"

    # ------------------------------------------------------------------
    # months_from_peak
    # ------------------------------------------------------------------
    unified["months_from_peak"] = np.nan
    for era, peak in PEAK_DATES.items():
        mask = unified["era"] == era
        unified.loc[mask, "months_from_peak"] = (
            (unified.loc[mask, "date"].dt.year - peak.year) * 12
            + (unified.loc[mask, "date"].dt.month - peak.month)
        ).astype(int)

    unified["months_from_peak"] = pd.to_numeric(
        unified["months_from_peak"], errors="coerce"
    )

    return unified


# ---------------------------------------------------------------------------
# 5. Era sub-frames
# ---------------------------------------------------------------------------


def _era_df(macro_df: pd.DataFrame, era: str) -> pd.DataFrame:
    return macro_df[macro_df["era"] == era].copy().sort_values("date")


# ---------------------------------------------------------------------------
# 6. Analysis functions
# ---------------------------------------------------------------------------


def compare_eras(
    macro_df: pd.DataFrame,
    dotcom_range: tuple[str, str] = DOTCOM_RANGE,
    ai_range: tuple[str, str] = AI_RANGE,
    window_months: tuple[int, int] = (-24, 12),
) -> dict[str, Any]:
    """Side-by-side comparison stats for each macro indicator.

    Restricts to the window [months_from_peak in window_months] for both eras.
    Returns descriptive stats and Welch t-test / Levene results.

    All differences are framed as ASSOCIATIONS, not causal claims.
    """
    lo, hi = window_months
    window = macro_df[
        macro_df["months_from_peak"].between(lo, hi, inclusive="both")
    ]

    results: list[dict] = []
    for col in INDICATORS:
        dc_vals = window[window["era"] == "dotcom"][col].dropna()
        ai_vals = window[window["era"] == "ai"][col].dropna()

        row: dict[str, Any] = {
            "indicator": col,
            "dotcom_mean": float(dc_vals.mean()) if len(dc_vals) else np.nan,
            "dotcom_std": float(dc_vals.std()) if len(dc_vals) else np.nan,
            "dotcom_min": float(dc_vals.min()) if len(dc_vals) else np.nan,
            "dotcom_max": float(dc_vals.max()) if len(dc_vals) else np.nan,
            "ai_mean": float(ai_vals.mean()) if len(ai_vals) else np.nan,
            "ai_std": float(ai_vals.std()) if len(ai_vals) else np.nan,
            "ai_min": float(ai_vals.min()) if len(ai_vals) else np.nan,
            "ai_max": float(ai_vals.max()) if len(ai_vals) else np.nan,
        }
        row["mean_diff_ai_minus_dc"] = row["ai_mean"] - row["dotcom_mean"]

        # Welch's t-test (unequal variance)
        if len(dc_vals) >= 2 and len(ai_vals) >= 2:
            t_stat, p_val = stats.ttest_ind(dc_vals, ai_vals, equal_var=False)
            row["welch_t"] = float(t_stat)
            row["welch_p"] = float(p_val)
            row["significant_05"] = bool(p_val < 0.05)
        else:
            row["welch_t"] = np.nan
            row["welch_p"] = np.nan
            row["significant_05"] = False

        # Levene's test for variance differences
        if len(dc_vals) >= 2 and len(ai_vals) >= 2:
            lev_stat, lev_p = stats.levene(dc_vals, ai_vals)
            row["levene_stat"] = float(lev_stat)
            row["levene_p"] = float(lev_p)
            row["variance_differs_05"] = bool(lev_p < 0.05)
        else:
            row["levene_stat"] = np.nan
            row["levene_p"] = np.nan
            row["variance_differs_05"] = False

        results.append(row)

    return {
        "comparison_table": pd.DataFrame(results),
        "window": window_months,
        "note": (
            "All statistics are ASSOCIATIONS observed within the "
            f"{window_months[0]} to +{window_months[1]} month window "
            "relative to each era's market peak. No causal claims are made."
        ),
    }


def detect_yield_curve_inversions(t10y2y: pd.Series) -> pd.DataFrame:
    """Identify continuous yield curve inversion periods.

    Args:
        t10y2y: Monthly Series of T10Y2Y spread, indexed by date.
                Must already be restricted to a single era if era-specific
                analysis is desired.

    Returns:
        DataFrame with columns:
            inversion_start, inversion_end, duration_months,
            max_inversion_depth (most negative spread value).
    """
    if t10y2y.empty:
        return pd.DataFrame(
            columns=[
                "inversion_start",
                "inversion_end",
                "duration_months",
                "max_inversion_depth",
            ]
        )

    s = t10y2y.sort_index().dropna()
    inverted = s < 0
    # Label consecutive runs
    group_ids = (inverted != inverted.shift()).cumsum()
    records: list[dict] = []

    for gid, grp_inverted in inverted.groupby(group_ids):
        if not grp_inverted.all():
            continue  # skip non-inverted groups
        grp_values = s.loc[grp_inverted.index]
        records.append(
            {
                "inversion_start": grp_inverted.index.min(),
                "inversion_end": grp_inverted.index.max(),
                "duration_months": len(grp_inverted),
                "max_inversion_depth": float(grp_values.min()),
            }
        )

    return pd.DataFrame(records)


def compute_lead_lag(
    macro_series: pd.Series,
    equity_returns: pd.Series,
    max_lag: int = 24,
) -> pd.DataFrame:
    """Cross-correlation between a macro series and equity returns at varying lags.

    Positive lag means macro leads equity by that many periods.
    Both series must share the same index (monthly dates).

    Returns DataFrame with columns: lag_months, pearson_r, n_obs.
    """
    # Align on common index
    combined = pd.concat(
        [macro_series.rename("macro"), equity_returns.rename("equity")], axis=1
    ).dropna()

    x = combined["macro"].reset_index(drop=True)
    y = combined["equity"].reset_index(drop=True)
    n = len(x)

    records: list[dict] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            xi = x.iloc[:-lag]
            yi = y.iloc[lag:]
        elif lag < 0:
            xi = x.iloc[-lag:]
            yi = y.iloc[:lag]
        else:
            xi, yi = x, y

        if len(xi) < 3:
            records.append({"lag_months": lag, "pearson_r": np.nan, "n_obs": 0})
            continue

        r, _ = stats.pearsonr(xi.reset_index(drop=True), yi.reset_index(drop=True))
        records.append({"lag_months": lag, "pearson_r": float(r), "n_obs": len(xi)})

    return pd.DataFrame(records)


def run_statistical_tests(macro_df: pd.DataFrame) -> dict[str, Any]:
    """Run all statistical tests specified in the spec.

    Tests:
      1. Era comparison summary (Welch t-test + Levene) via compare_eras()
      2. Macro-equity forward-return correlation matrix
      3. Lead-lag cross-correlation for each indicator vs S&P 500
      4. Granger causality: yield curve -> equity returns (ADF check + AIC lag)

    ALL findings framed as ASSOCIATIONS.
    """
    findings: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # 2.1  Era comparison                                                  #
    # ------------------------------------------------------------------ #
    findings["era_comparison"] = compare_eras(macro_df)

    # ------------------------------------------------------------------ #
    # 2.2  Yield curve inversions                                          #
    # ------------------------------------------------------------------ #
    inversions_by_era: dict[str, pd.DataFrame] = {}
    for era in ("dotcom", "ai"):
        edf = _era_df(macro_df, era).set_index("date")
        inversions_by_era[era] = detect_yield_curve_inversions(
            edf["yield_curve_spread"]
        )
    findings["yield_curve_inversions"] = inversions_by_era

    # ------------------------------------------------------------------ #
    # 2.3  Lead-lag analysis                                               #
    # ------------------------------------------------------------------ #
    lead_lag_results: dict[str, dict[str, pd.DataFrame]] = {}
    for era in ("dotcom", "ai"):
        edf = _era_df(macro_df, era).set_index("date")
        sp500_returns = edf["sp500"].pct_change().dropna()
        lead_lag_results[era] = {}
        for col in INDICATORS:
            if col not in edf.columns:
                continue
            ll = compute_lead_lag(
                edf[col].loc[sp500_returns.index],
                sp500_returns,
                max_lag=24,
            )
            lead_lag_results[era][col] = ll
    findings["lead_lag"] = lead_lag_results

    # ------------------------------------------------------------------ #
    # 2.4  Macro-equity 12-month forward return correlations               #
    # ------------------------------------------------------------------ #
    macro_equity_corr = _compute_macro_equity_correlations(macro_df, INDICATORS)
    findings["macro_equity_correlations"] = macro_equity_corr

    # ------------------------------------------------------------------ #
    # 2.5  Granger causality: yield curve -> equity returns                #
    # ------------------------------------------------------------------ #
    granger_results: dict[str, dict] = {}
    for era in ("dotcom", "ai"):
        granger_results[era] = _run_granger_test(macro_df, era)
    findings["granger"] = granger_results

    return findings


def _compute_macro_equity_correlations(
    macro_df: pd.DataFrame,
    indicators: list[str],
) -> pd.DataFrame:
    """Pearson/Spearman correlation of each macro variable with 12-month forward
    S&P 500 return, separately per era.

    Framed as ASSOCIATIONS only.
    """
    records: list[dict] = []
    for era in ("dotcom", "ai"):
        edf = _era_df(macro_df, era).reset_index(drop=True)
        sp_col = "sp500"

        for col in indicators:
            macro_vals: list[float] = []
            fwd_rets: list[float] = []

            for i, row in edf.iterrows():
                future_idx = i + 12
                if future_idx >= len(edf):
                    continue
                if pd.isna(row[col]) or pd.isna(edf.at[i, sp_col]):
                    continue
                if pd.isna(edf.at[future_idx, sp_col]):
                    continue
                fwd_ret = (
                    edf.at[future_idx, sp_col] / edf.at[i, sp_col] - 1
                ) * 100
                macro_vals.append(float(row[col]))
                fwd_rets.append(float(fwd_ret))

            if len(macro_vals) < 5:
                continue

            pearson_r, pearson_p = stats.pearsonr(macro_vals, fwd_rets)
            spearman_r, spearman_p = stats.spearmanr(macro_vals, fwd_rets)
            records.append(
                {
                    "era": era,
                    "indicator": col,
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                    "n_observations": len(macro_vals),
                    "note": "Association only — not a causal claim.",
                }
            )

    return pd.DataFrame(records)


def _run_granger_test(macro_df: pd.DataFrame, era: str) -> dict[str, Any]:
    """Granger causality: yield curve spread -> equity returns.

    Steps:
      1. ADF stationarity check on both series.
      2. AIC-based optimal lag selection via VAR.
      3. Granger test at optimal lag (and lags 1-6 for robustness).

    FRAMING: "predictive power" language only — NOT causal.
    """
    edf = _era_df(macro_df, era).dropna(subset=["yield_curve_spread", "sp500"])
    sp500_returns = edf["sp500"].pct_change().dropna()
    yc = edf["yield_curve_spread"].iloc[1:].reset_index(drop=True)
    sp_r = sp500_returns.reset_index(drop=True)

    data = pd.DataFrame(
        {"sp500_returns": sp_r.values, "yield_curve": yc.values}
    ).dropna()

    if len(data) < 20:
        return {
            "era": era,
            "error": "Insufficient data for Granger test.",
            "note": "Association only — not a causal claim.",
        }

    result: dict[str, Any] = {
        "era": era,
        "n_obs": len(data),
        "note": (
            "Granger causality tests predictive power only. "
            "Significance does NOT imply causation. "
            "Framed as: the yield curve spread has (or lacks) predictive power "
            "for equity returns beyond what past equity returns alone explain."
        ),
    }

    # ADF stationarity checks
    adf_results = {}
    for col in ["sp500_returns", "yield_curve"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_stat, adf_p, _, _, _, _ = adfuller(data[col].dropna())
        adf_results[col] = {
            "adf_stat": float(adf_stat),
            "adf_p": float(adf_p),
            "stationary_at_05": bool(adf_p < 0.05),
        }
        if adf_p > 0.05:
            console.print(
                f"[yellow]WARNING:[/] {era} {col}: non-stationary "
                f"(ADF p={adf_p:.4f}). Granger results may be spurious."
            )
    result["adf"] = adf_results

    # AIC-based lag selection
    try:
        var_model = VAR(data[["sp500_returns", "yield_curve"]])
        lag_order = var_model.select_order(maxlags=min(12, len(data) // 5))
        optimal_lag = int(lag_order.aic) if lag_order.aic else 1
        optimal_lag = max(1, optimal_lag)
    except Exception as e:
        console.print(f"[yellow]VAR lag selection failed ({e}); defaulting to lag=2[/]")
        optimal_lag = 2

    result["optimal_lag_aic"] = optimal_lag

    # Granger test
    try:
        max_test_lag = min(6, max(optimal_lag, 1), len(data) // 5)
        granger_raw = grangercausalitytests(
            data[["sp500_returns", "yield_curve"]],
            maxlag=max_test_lag,
            verbose=False,
        )
        lag_results = {}
        for lag in range(1, max_test_lag + 1):
            f_stat, p_val, _, _ = granger_raw[lag][0]["ssr_ftest"]
            lag_results[lag] = {
                "f_stat": float(f_stat),
                "p_value": float(p_val),
                "significant_at_05": bool(p_val < 0.05),
            }
        result["lag_results"] = lag_results
        result["optimal_lag_result"] = lag_results.get(optimal_lag, {})
    except Exception as e:
        result["granger_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# 7. Chart generation helpers
# ---------------------------------------------------------------------------


def _save_chart(fig: go.Figure, name: str) -> dict[str, Path]:
    """Save chart as JSON and PNG. Returns paths dict."""
    json_path = CHART_DIR / f"{name}.json"
    png_path = CHART_DIR / f"{name}.png"

    fig.write_json(str(json_path))

    try:
        fig.write_image(
            str(png_path),
            width=1600,
            height=900,
            scale=2,
        )
    except Exception as e:
        console.print(f"[yellow]PNG export failed for {name} ({e})[/]")

    console.print(f"[green]Chart saved:[/] {name}.json + .png")
    return {"json": json_path, "png": png_path}


def _add_peak_vline(fig: go.Figure, row: int = 1, col: int = 1) -> None:
    """Add a vertical dashed line at months_from_peak == 0."""
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="rgba(200,200,200,0.6)",
        annotation_text="Market Peak",
        annotation_position="top right",
        annotation_font_size=10,
        row=row,
        col=col,
    )


# ---------------------------------------------------------------------------
# 7.1  Chart 4.1 — Federal Funds Rate dual-era overlay
# ---------------------------------------------------------------------------


def chart_4_1_fed_funds(macro_df: pd.DataFrame) -> dict[str, Path]:
    """Fed Funds Rate: Dot-Com vs AI Era, aligned to months_from_peak."""
    fig = go.Figure()

    for era, color, label in [
        ("dotcom", DOTCOM_COLOR, "Dot-Com (1996-2003)"),
        ("ai", AI_ERA_COLOR, "AI Era (2021-2026)"),
    ]:
        edf = _era_df(macro_df, era).sort_values("months_from_peak").dropna(
            subset=["months_from_peak", "fed_funds_rate"]
        )
        fig.add_trace(
            go.Scatter(
                x=edf["months_from_peak"],
                y=edf["fed_funds_rate"],
                mode="lines",
                name=label,
                line=dict(color=color, width=2.5),
            )
        )

    # Peak line
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="rgba(200,200,200,0.5)",
        annotation_text="Market Peak",
        annotation_position="top right",
    )

    # Dot-com NBER recession: Mar 2001–Nov 2001 → months from peak Mar 2000
    # Mar 2001 = +12, Nov 2001 = +20
    fig.add_vrect(
        x0=12, x1=20,
        fillcolor="rgba(100,100,100,0.15)",
        layer="below",
        line_width=0,
        annotation_text="Dot-Com Recession",
        annotation_position="top left",
        annotation_font_size=9,
    )

    fig.update_layout(
        title="Federal Funds Rate: Dot-Com vs AI Era (Aligned to Market Peak)",
        xaxis_title="Months from Market Peak",
        yaxis_title="Federal Funds Rate (%)",
        template=PLOTLY_TEMPLATE,
        width=1600,
        height=900,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12),
        annotations=[
            dict(
                x=-10, y=6.5,
                xref="x", yref="y",
                text="Fed hiking into<br>bubble peak",
                showarrow=True, arrowhead=2,
                arrowcolor=DOTCOM_COLOR,
                font=dict(color=DOTCOM_COLOR, size=11),
            ),
            dict(
                x=12, y=5.3,
                xref="x", yref="y",
                text="Fed begins<br>easing",
                showarrow=True, arrowhead=2,
                arrowcolor=AI_ERA_COLOR,
                font=dict(color=AI_ERA_COLOR, size=11),
            ),
        ],
    )

    return _save_chart(fig, "chart_4_1_fed_funds")


# ---------------------------------------------------------------------------
# 7.2  Chart 4.2 — M2 growth rate comparison
# ---------------------------------------------------------------------------


def chart_4_2_m2_growth(macro_df: pd.DataFrame) -> dict[str, Path]:
    """M2 YoY growth rate: both eras, peak-aligned."""
    fig = go.Figure()

    for era, color, label in [
        ("dotcom", DOTCOM_COLOR, "Dot-Com (1996-2003)"),
        ("ai", AI_ERA_COLOR, "AI Era (2021-2026)"),
    ]:
        edf = _era_df(macro_df, era).sort_values("months_from_peak").dropna(
            subset=["months_from_peak", "m2_yoy_growth"]
        )
        fill = "tozeroy" if era == "ai" else "none"
        fill_color = f"rgba(0,204,150,0.12)" if era == "ai" else None

        trace_kwargs: dict[str, Any] = dict(
            x=edf["months_from_peak"],
            y=edf["m2_yoy_growth"],
            mode="lines",
            name=label,
            line=dict(color=color, width=2.0),
            fill=fill,
        )
        if fill_color:
            trace_kwargs["fillcolor"] = fill_color

        fig.add_trace(go.Scatter(**trace_kwargs))

    # Zero growth line
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="rgba(200,200,200,0.5)",
        annotation_text="Zero Growth",
        annotation_position="right",
    )
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="rgba(200,200,200,0.5)",
        annotation_text="Market Peak",
        annotation_position="top right",
    )

    fig.update_layout(
        title="M2 Money Supply Growth: The Liquidity Story",
        xaxis_title="Months from Market Peak",
        yaxis_title="M2 YoY Growth (%)",
        template=PLOTLY_TEMPLATE,
        width=1600,
        height=900,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12),
        annotations=[
            dict(
                x=-24, y=26,
                xref="x", yref="y",
                text="COVID stimulus:<br>unprecedented<br>M2 expansion",
                showarrow=True, arrowhead=2,
                arrowcolor=AI_ERA_COLOR,
                font=dict(color=AI_ERA_COLOR, size=11),
            ),
            dict(
                x=-12, y=-3,
                xref="x", yref="y",
                text="First M2 contraction<br>since 1930s",
                showarrow=True, arrowhead=2,
                arrowcolor="orange",
                font=dict(color="orange", size=11),
            ),
        ],
    )

    return _save_chart(fig, "chart_4_2_m2_growth")


# ---------------------------------------------------------------------------
# 7.3  Chart 4.3 — Yield curve with recession bands
# ---------------------------------------------------------------------------


def chart_4_3_yield_curve(macro_df: pd.DataFrame) -> dict[str, Path]:
    """T10Y2Y over actual dates, two stacked subplots, one per era.

    Shaded inversion regions and annotated crash events.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        shared_yaxes=True,
        subplot_titles=["Dot-Com Era (1996–2003)", "AI Era (2021–2026)"],
        vertical_spacing=0.12,
    )

    era_meta = [
        ("dotcom", DOTCOM_COLOR, 1, "Dot-Com (1996-2003)"),
        ("ai", AI_ERA_COLOR, 2, "AI Era (2021-2026)"),
    ]

    for era, color, row, label in era_meta:
        edf = _era_df(macro_df, era).dropna(subset=["yield_curve_spread"])

        fig.add_trace(
            go.Scatter(
                x=edf["date"],
                y=edf["yield_curve_spread"],
                mode="lines",
                name=label,
                line=dict(color=color, width=1.5),
                showlegend=True,
            ),
            row=row, col=1,
        )

        # Shade inversion periods
        inv_df = detect_yield_curve_inversions(
            edf.set_index("date")["yield_curve_spread"]
        )
        for _, inv_row in inv_df.iterrows():
            fig.add_vrect(
                x0=inv_row["inversion_start"],
                x1=inv_row["inversion_end"],
                fillcolor="rgba(239,85,59,0.20)",
                layer="below",
                line_width=0,
                row=row, col=1,
            )

        # Zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="rgba(200,200,200,0.5)",
            annotation_text="Inversion Threshold",
            annotation_position="right",
            row=row, col=1,
        )

    # Dot-com specific annotations
    fig.add_vline(
        x="2000-03-31",
        line_dash="dash", line_color=DOTCOM_COLOR,
        annotation_text="S&P 500 Peak",
        annotation_position="top right",
        row=1, col=1,
    )
    fig.add_vline(
        x="2001-03-01",
        line_dash="dot", line_color="orange",
        annotation_text="Recession Start",
        annotation_position="top left",
        row=1, col=1,
    )

    # AI era provisional peak
    fig.add_vline(
        x=PEAK_DATES["ai"].strftime("%Y-%m-%d"),
        line_dash="dash", line_color=AI_ERA_COLOR,
        annotation_text="Provisional Peak",
        annotation_position="top right",
        row=2, col=1,
    )

    fig.update_layout(
        title="Yield Curve (10Y-2Y): Recession Warning Signal",
        template=PLOTLY_TEMPLATE,
        width=1600,
        height=900,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.08),
    )
    fig.update_yaxes(title_text="10Y-2Y Spread (%)")
    fig.update_xaxes(title_text="Date")

    return _save_chart(fig, "chart_4_3_yield_curve")


# ---------------------------------------------------------------------------
# 7.4  Chart 4.4 — Credit spreads
# ---------------------------------------------------------------------------


def chart_4_4_credit_spreads(macro_df: pd.DataFrame) -> dict[str, Path]:
    """BAA10Y credit spread: both eras, peak-aligned."""
    fig = go.Figure()

    for era, color, label in [
        ("dotcom", DOTCOM_COLOR, "Dot-Com (1996-2003)"),
        ("ai", AI_ERA_COLOR, "AI Era (2021-2026)"),
    ]:
        edf = _era_df(macro_df, era).sort_values("months_from_peak").dropna(
            subset=["months_from_peak", "credit_spread"]
        )
        fig.add_trace(
            go.Scatter(
                x=edf["months_from_peak"],
                y=edf["credit_spread"],
                mode="lines",
                name=label,
                line=dict(color=color, width=2.0),
            )
        )

    # Complacency zone (< 2%)
    fig.add_hrect(
        y0=0, y1=2.0,
        fillcolor="rgba(0,204,150,0.08)",
        layer="below", line_width=0,
        annotation_text="Complacency Zone (<2%)",
        annotation_position="top left",
        annotation_font_size=9,
    )
    # Stress zone (> 3.5%)
    fig.add_hrect(
        y0=3.5, y1=10,
        fillcolor="rgba(239,85,59,0.08)",
        layer="below", line_width=0,
        annotation_text="Stress Zone (>3.5%)",
        annotation_position="top right",
        annotation_font_size=9,
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="rgba(200,200,200,0.5)",
        annotation_text="Market Peak",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Corporate Credit Spreads: Risk Appetite vs Stress",
        xaxis_title="Months from Market Peak",
        yaxis_title="Moody's Baa – 10Y Spread (%)",
        template=PLOTLY_TEMPLATE,
        width=1600,
        height=900,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12),
        annotations=[
            dict(
                x=10, y=3.8,
                xref="x", yref="y",
                text="Credit stress emerges<br>(dot-com crash)",
                showarrow=True, arrowhead=2,
                arrowcolor=DOTCOM_COLOR,
                font=dict(color=DOTCOM_COLOR, size=11),
            ),
            dict(
                x=-6, y=1.6,
                xref="x", yref="y",
                text="Risk appetite<br>remains strong",
                showarrow=True, arrowhead=2,
                arrowcolor=AI_ERA_COLOR,
                font=dict(color=AI_ERA_COLOR, size=11),
            ),
        ],
    )

    return _save_chart(fig, "chart_4_4_credit_spreads")


# ---------------------------------------------------------------------------
# 7.5  Chart 4.5 — Macro dashboard 3×2 (spec says 3×2, though spec text also
#       mentions 3×3; the task instruction says "3x2 subplot grid" with 6
#       panels, so we implement exactly 6: Fed Funds, M2 Growth, Yield Curve,
#       Credit Spreads, CPI, Unemployment)
# ---------------------------------------------------------------------------


def chart_4_5_macro_dashboard(macro_df: pd.DataFrame) -> dict[str, Path]:
    """3x2 small-multiples dashboard: 6 key macro indicators, both eras."""
    panels = [
        ("fed_funds_rate", "Fed Funds Rate (%)"),
        ("m2_yoy_growth", "M2 Growth (%)"),
        ("yield_curve_spread", "Yield Curve (%)"),
        ("credit_spread", "Credit Spreads (%)"),
        ("cpi_yoy_inflation", "CPI Inflation (%)"),
        ("unemployment", "Unemployment (%)"),
    ]

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[title for _, title in panels],
        shared_xaxes=False,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    for idx, (col, title) in enumerate(panels):
        row = idx // 2 + 1
        col_pos = idx % 2 + 1
        first_panel = idx == 0

        for era, color, label in [
            ("dotcom", DOTCOM_COLOR, "Dot-Com (1996-2003)"),
            ("ai", AI_ERA_COLOR, "AI Era (2021-2026)"),
        ]:
            edf = _era_df(macro_df, era).sort_values("months_from_peak").dropna(
                subset=["months_from_peak", col]
            )
            fig.add_trace(
                go.Scatter(
                    x=edf["months_from_peak"],
                    y=edf[col],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.5),
                    showlegend=first_panel,
                ),
                row=row, col=col_pos,
            )

        # Peak vline per subplot
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="rgba(200,200,200,0.4)",
            row=row, col=col_pos,
        )

    fig.update_layout(
        title="Macro Environment Dashboard: Dot-Com vs AI Era",
        template=PLOTLY_TEMPLATE,
        width=1600,
        height=900,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.06),
    )

    return _save_chart(fig, "chart_4_5_macro_dashboard")


# ---------------------------------------------------------------------------
# 7.6  Chart 4.6 — Real interest rate comparison
# ---------------------------------------------------------------------------


def chart_4_6_real_rate(macro_df: pd.DataFrame) -> dict[str, Path]:
    """Real interest rate (Fed Funds – CPI YoY): both eras, peak-aligned."""
    fig = go.Figure()

    for era, color, label in [
        ("dotcom", DOTCOM_COLOR, "Dot-Com (1996-2003)"),
        ("ai", AI_ERA_COLOR, "AI Era (2021-2026)"),
    ]:
        edf = _era_df(macro_df, era).sort_values("months_from_peak").dropna(
            subset=["months_from_peak", "real_interest_rate"]
        )
        fig.add_trace(
            go.Scatter(
                x=edf["months_from_peak"],
                y=edf["real_interest_rate"],
                mode="lines",
                name=label,
                line=dict(color=color, width=2.0),
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.12)",
            )
        )

    # Zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(255,255,255,0.7)",
        line_width=2,
        annotation_text="Zero Real Rate",
        annotation_position="right",
    )

    # Free money zone (y < 0) — light red background
    fig.add_hrect(
        y0=-10, y1=0,
        fillcolor="rgba(239,85,59,0.06)",
        layer="below", line_width=0,
        annotation_text="Free Money Zone",
        annotation_position="bottom right",
        annotation_font_size=9,
    )
    # Restrictive zone (y > 0) — light green background
    fig.add_hrect(
        y0=0, y1=6,
        fillcolor="rgba(0,204,150,0.06)",
        layer="below", line_width=0,
        annotation_text="Restrictive Zone",
        annotation_position="top right",
        annotation_font_size=9,
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="rgba(200,200,200,0.5)",
        annotation_text="Market Peak",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Real Interest Rates: Is Money Free or Expensive?",
        xaxis_title="Months from Market Peak",
        yaxis_title="Real Interest Rate (%)",
        template=PLOTLY_TEMPLATE,
        width=1600,
        height=900,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12),
        annotations=[
            dict(
                x=-30, y=-7,
                xref="x", yref="y",
                text="Deepest negative real rates<br>in modern history",
                showarrow=True, arrowhead=2,
                arrowcolor=AI_ERA_COLOR,
                font=dict(color=AI_ERA_COLOR, size=11),
            ),
            dict(
                x=-2, y=3.2,
                xref="x", yref="y",
                text="Money was expensive<br>at dot-com peak",
                showarrow=True, arrowhead=2,
                arrowcolor=DOTCOM_COLOR,
                font=dict(color=DOTCOM_COLOR, size=11),
            ),
        ],
    )

    return _save_chart(fig, "chart_4_6_real_rate")


# ---------------------------------------------------------------------------
# 8. Main runner
# ---------------------------------------------------------------------------


def run_layer4(force_refresh: bool = False) -> dict[str, Any]:
    """Full Layer 4 pipeline.

    Returns:
        {
            "data": unified monthly macro DataFrame,
            "stats": statistical test results dict,
            "charts": {chart_name: {"json": Path, "png": Path}, ...},
            "findings": list of narrative finding strings,
        }
    """
    console.print("\n[bold cyan]Layer 4: Macro Environment Comparison[/]")
    console.print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1: Build unified macro DataFrame                               #
    # ------------------------------------------------------------------ #
    console.print("\n[1/3] Building unified macro dataset...")
    macro_df = build_macro_dashboard_data(force_refresh=force_refresh)
    n_dotcom = (macro_df["era"] == "dotcom").sum()
    n_ai = (macro_df["era"] == "ai").sum()
    console.print(
        f"      Dot-com rows: {n_dotcom} | AI era rows: {n_ai} | "
        f"Total: {len(macro_df)}"
    )

    # ------------------------------------------------------------------ #
    # Step 2: Statistical tests                                           #
    # ------------------------------------------------------------------ #
    console.print("\n[2/3] Running statistical tests...")
    stats_results = run_statistical_tests(macro_df)
    console.print("      Era comparison, lead-lag, Granger tests complete.")

    # ------------------------------------------------------------------ #
    # Step 3: Generate charts                                             #
    # ------------------------------------------------------------------ #
    console.print("\n[3/3] Generating charts...")
    charts: dict[str, Any] = {}
    chart_fns = [
        ("chart_4_1", chart_4_1_fed_funds),
        ("chart_4_2", chart_4_2_m2_growth),
        ("chart_4_3", chart_4_3_yield_curve),
        ("chart_4_4", chart_4_4_credit_spreads),
        ("chart_4_5", chart_4_5_macro_dashboard),
        ("chart_4_6", chart_4_6_real_rate),
    ]
    for chart_key, fn in chart_fns:
        try:
            charts[chart_key] = fn(macro_df)
        except Exception as e:
            console.print(f"[red]Chart {chart_key} failed: {e}[/]")
            charts[chart_key] = {"error": str(e)}

    # ------------------------------------------------------------------ #
    # Step 4: Key findings (associations only)                            #
    # ------------------------------------------------------------------ #
    findings = _build_findings(macro_df, stats_results)

    console.print("\n[bold green]Layer 4 complete.[/]")
    return {
        "data": macro_df,
        "stats": stats_results,
        "charts": charts,
        "findings": findings,
    }


def _build_findings(
    macro_df: pd.DataFrame,
    stats_results: dict[str, Any],
) -> list[str]:
    """Derive narrative findings from data. All framed as ASSOCIATIONS."""
    findings: list[str] = []

    era_comp = stats_results.get("era_comparison", {})
    comp_table: pd.DataFrame = era_comp.get("comparison_table", pd.DataFrame())

    if not comp_table.empty:
        for _, row in comp_table.iterrows():
            ind = row["indicator"]
            sig = row.get("significant_05", False)
            dc_mean = row.get("dotcom_mean", np.nan)
            ai_mean = row.get("ai_mean", np.nan)
            diff = row.get("mean_diff_ai_minus_dc", np.nan)
            sig_str = "statistically significant (p<0.05)" if sig else "not statistically significant"
            findings.append(
                f"ASSOCIATION [{ind}]: dot-com mean={dc_mean:.2f}, "
                f"AI mean={ai_mean:.2f}, diff={diff:+.2f}. "
                f"Welch t-test: {sig_str}. "
                f"(Association only — not a causal claim.)"
            )

    # Yield curve inversion summary
    inv_by_era = stats_results.get("yield_curve_inversions", {})
    for era, inv_df in inv_by_era.items():
        if inv_df.empty:
            findings.append(
                f"YIELD CURVE [{era}]: No inversion periods detected in the data range."
            )
        else:
            total_months = inv_df["duration_months"].sum()
            max_depth = inv_df["max_inversion_depth"].min()
            findings.append(
                f"YIELD CURVE [{era}]: {len(inv_df)} inversion period(s) totalling "
                f"{total_months} months. Maximum inversion depth: {max_depth:.2f}%. "
                f"(Association with subsequent equity drawdowns — not causation.)"
            )

    # Granger summary
    granger = stats_results.get("granger", {})
    for era, g in granger.items():
        opt_lag = g.get("optimal_lag_aic")
        opt_result = g.get("optimal_lag_result", {})
        p = opt_result.get("p_value")
        if p is not None:
            sig = "has statistically significant" if p < 0.05 else "does NOT show significant"
            findings.append(
                f"GRANGER [{era}]: At AIC-optimal lag {opt_lag}, yield curve spread "
                f"{sig} predictive power for equity returns (F-test p={p:.4f}). "
                f"This is a PREDICTIVE ASSOCIATION only — not a causal claim."
            )

    # Narrative macro context
    findings.append(
        "MACRO NARRATIVE: The AI era began with deeply negative real rates "
        "(-5% to -7%), associated with the COVID-era M2 expansion — a fundamentally "
        "different liquidity environment from dot-com, where real rates were positive "
        "(~+3%) at the market peak. By 2023-2025, the fastest hiking cycle in 40 years "
        "pushed real rates back to positive territory, yet AI equities continued rallying "
        "— a pattern associated with either strong fundamental support or elevated risk "
        "tolerance relative to historical norms. All observations are associations; "
        "the direction of any causal link between macro conditions and equity prices "
        "cannot be determined from this analysis."
    )

    return findings


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Layer 4: Macro Environment")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass cache and re-fetch all FRED data.",
    )
    args = parser.parse_args()

    result = run_layer4(force_refresh=args.force_refresh)

    console.print("\n[bold]Key Findings:[/]")
    for finding in result["findings"]:
        console.print(f"  • {finding[:120]}...")
