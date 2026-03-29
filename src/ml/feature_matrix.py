"""Feature matrix construction for the ML pipeline.

Combines outputs from all 5 data layers into a unified monthly feature matrix.
All features use only trailing data (no look-ahead bias).

Feature groups:
    Price       — 9 features: momentum_30d/90d/252d, volatility_30d/90d,
                               drawdown_from_ath, rsi_14, price_to_sma200, return_skewness
    Fundamentals — 7 features: log_pe, pe_percentile, cape_shiller, revenue_growth_yoy,
                                fcf_yield, earnings_growth_yoy, pe_to_growth
    Concentration — 6 features: top5_weight, top10_weight, spy_rsp_spread,
                                  hhi, buffett_indicator, sector_concentration
    Macro        — 9 features: fed_funds, fed_funds_12m_chg, m2_growth, yield_curve,
                                 yield_curve_10y3m, credit_spread, real_rate, vix, unemployment
    Sentiment    — 7 features (optional): ai_mention_rate, hype_score, specificity_score,
                                          google_trends_ai, google_trends_bubble,
                                          reddit_sentiment, reddit_volume
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTIMENT_COLS = [
    "ai_mention_rate",
    "hype_score",
    "specificity_score",
    "google_trends_ai",
    "google_trends_bubble",
    "reddit_sentiment",
    "reddit_volume",
]

PRICE_FEATURES = [
    "momentum_30d",
    "momentum_90d",
    "momentum_252d",
    "volatility_30d",
    "volatility_90d",
    "drawdown_from_ath",
    "rsi_14",
    "price_to_sma200",
    "return_skewness",
]

MACRO_FFILL_COLS = [
    "fed_funds",
    "m2_growth",
    "credit_spread",
    "unemployment",
    "gdp_growth",
]

# ---------------------------------------------------------------------------
# Internal feature builders — each returns a DatetimeIndex DataFrame
# ---------------------------------------------------------------------------


def _build_price_features(layer1: dict) -> pd.DataFrame:
    """Extract and compute price features from layer1 output.

    Uses the ^GSPC (S&P 500) series as the primary price for regime
    classification consistency across eras.
    """
    data: dict = layer1.get("data", {})
    sp500 = data.get("^GSPC", pd.DataFrame())

    if sp500.empty:
        console.print("[yellow]Layer 1: ^GSPC not found, trying SPY as fallback.[/]")
        sp500 = data.get("SPY", pd.DataFrame())

    if sp500.empty:
        console.print("[red]Layer 1: No S&P 500 price data found. Price features will be NaN.[/]")
        return pd.DataFrame()

    # Ensure DatetimeIndex and sorted ascending
    sp500 = sp500.copy()
    if not isinstance(sp500.index, pd.DatetimeIndex):
        sp500.index = pd.to_datetime(sp500.index)
    sp500 = sp500.sort_index()

    # Use 'Close' column (auto_adjust=True so this is adjusted close)
    close = sp500["Close"].dropna()
    if close.empty:
        return pd.DataFrame()

    daily_ret = close.pct_change()

    features = pd.DataFrame(index=close.index)

    # ---- Momentum ----
    features["momentum_30d"] = close.pct_change(periods=21)    # ~21 trading days = 1 month
    features["momentum_90d"] = close.pct_change(periods=63)    # ~63 trading days = 3 months
    features["momentum_252d"] = close.pct_change(periods=252)  # ~252 trading days = 1 year

    # ---- Volatility (annualized realized) ----
    features["volatility_30d"] = daily_ret.rolling(21).std() * np.sqrt(252)
    features["volatility_90d"] = daily_ret.rolling(63).std() * np.sqrt(252)

    # ---- Drawdown from all-time high ----
    running_max = close.expanding().max()
    features["drawdown_from_ath"] = (close / running_max) - 1.0

    # ---- RSI 14 ----
    features["rsi_14"] = _compute_rsi(close, period=14)

    # ---- Price to 200-day SMA ----
    sma200 = close.rolling(200).mean()
    features["price_to_sma200"] = close / sma200.replace(0, np.nan)

    # ---- Return skewness (63-day rolling) ----
    features["return_skewness"] = daily_ret.rolling(63).skew()

    # ---- Resample to month-end ----
    monthly = features.resample("M").last()

    return monthly


def _build_fundamental_features(layer2: dict) -> pd.DataFrame:
    """Extract fundamental features from layer2 output.

    Layer2 provides quarterly data for NVDA and CSCO, plus SP500 fundamentals
    where available. We use the SP500-level data for regime classification.
    """
    data: dict = layer2.get("data", {})

    # SP500 aggregate fundamentals (if layer2 exposes them directly)
    sp500_fund = data.get("sp500_fundamentals", pd.DataFrame())

    frames = {}

    # Try to build from available quarterly data.
    # Layer2 returns nvda/csco quarterly financials; we need macro-level SP500 PE.
    # We use the SP500 trailing PE series if available; otherwise build from NVDA
    # as a partial signal and flag the limitation.

    # Primary: look for a pre-built sp500 fundamentals DataFrame
    if not sp500_fund.empty:
        sp500_fund = sp500_fund.copy()
        if not isinstance(sp500_fund.index, pd.DatetimeIndex):
            sp500_fund.index = pd.to_datetime(sp500_fund.index)

        for col_map in [
            ("pe_ratio", "pe_ratio"),
            ("log_pe_ratio", "log_pe"),
            ("cape_shiller", "cape_shiller"),
            ("fcf_yield", "fcf_yield"),
            ("revenue_growth_yoy", "revenue_growth_yoy"),
            ("earnings_growth_yoy", "earnings_growth_yoy"),
            ("peg_ratio", "pe_to_growth"),
        ]:
            src, dst = col_map
            if src in sp500_fund.columns:
                frames[dst] = sp500_fund[src]

    # Fallback: extract from per-ticker data that layer2 assembles
    # Layer2 returns "nvda" and "csco" DataFrames with financial metrics
    for ticker_key in ["nvda", "csco"]:
        df_ticker = data.get(ticker_key, pd.DataFrame())
        if df_ticker.empty:
            continue
        if not isinstance(df_ticker.index, pd.DatetimeIndex):
            try:
                df_ticker.index = pd.to_datetime(df_ticker.index)
            except Exception:
                continue

        for col in ["pe_ratio", "log_pe_ratio", "fcf_yield", "revenue_growth_yoy",
                    "earnings_growth_yoy", "peg_ratio", "gross_margin", "rd_ratio"]:
            if col in df_ticker.columns and col not in frames:
                frames[col] = df_ticker[col]

    if not frames:
        console.print("[yellow]Layer 2: No fundamental data extracted. Features will be NaN.[/]")
        return pd.DataFrame()

    fund_df = pd.DataFrame(frames)
    fund_df = fund_df.resample("M").last()

    # Rename to canonical names
    rename_map = {
        "log_pe_ratio": "log_pe",
        "peg_ratio": "pe_to_growth",
    }
    fund_df = fund_df.rename(columns=rename_map)

    # Add log_pe if we only have pe_ratio
    if "log_pe" not in fund_df.columns and "pe_ratio" in fund_df.columns:
        fund_df["log_pe"] = np.log(fund_df["pe_ratio"].replace(0, np.nan).clip(lower=0.01))

    # PE percentile (expanding 10-year window to avoid look-ahead)
    if "log_pe" in fund_df.columns:
        fund_df["pe_percentile"] = fund_df["log_pe"].expanding(min_periods=12).rank(pct=True)

    return fund_df


def _build_concentration_features(layer3: dict) -> pd.DataFrame:
    """Extract concentration features from layer3 output."""
    data: dict = layer3.get("data", {})

    frames = {}

    # Concentration time series — layer3 returns conc_ts with top10/top5
    conc_ts = data.get("concentration_timeseries", pd.DataFrame())
    if not conc_ts.empty:
        if not isinstance(conc_ts.index, pd.DatetimeIndex):
            conc_ts.index = pd.to_datetime(conc_ts.index)
        if "top10_concentration" in conc_ts.columns:
            frames["top10_weight"] = conc_ts["top10_concentration"]
        if "top5_concentration" in conc_ts.columns:
            frames["top5_weight"] = conc_ts["top5_concentration"]

    # SPY-RSP spread
    spread_df = data.get("spread_df", pd.DataFrame())
    if not spread_df.empty:
        if not isinstance(spread_df.index, pd.DatetimeIndex):
            spread_df.index = pd.to_datetime(spread_df.index)
        if "spread" in spread_df.columns:
            frames["spy_rsp_spread"] = spread_df["spread"]

    # Buffett indicator
    buffett_df = data.get("buffett_df", pd.DataFrame())
    if not buffett_df.empty:
        if not isinstance(buffett_df.index, pd.DatetimeIndex):
            buffett_df.index = pd.to_datetime(buffett_df.index)
        if "buffett_indicator" in buffett_df.columns:
            frames["buffett_indicator"] = buffett_df["buffett_indicator"]

    # HHI — derived from concentration if not directly available
    hhi_series = data.get("hhi_timeseries", pd.Series(dtype=float))
    if not hhi_series.empty:
        frames["hhi"] = hhi_series

    # Sector concentration proxy (tech+comm % in SP500 — use top10 * 0.85 approximation
    # if not provided directly, since tech dominates the top-10 historically)
    if "top10_weight" in frames and "hhi" not in frames:
        # Approximate HHI from concentration data using the layer3 methodology
        top10 = frames["top10_weight"]
        top5 = frames.get("top5_weight", top10 * 0.75)
        # Each of top-5 ≈ top5/5, each of next-5 ≈ (top10-top5)/5
        # Remaining 490 ≈ (100 - top10) / 490 each
        hhi_approx = (
            5 * (top5 / 5) ** 2
            + 5 * ((top10 - top5) / 5) ** 2
            + 490 * ((100 - top10) / 490) ** 2
        )
        frames["hhi"] = hhi_approx

    if not frames:
        console.print("[yellow]Layer 3: No concentration data extracted. Features will be NaN.[/]")
        return pd.DataFrame()

    conc_df = pd.DataFrame(frames)
    # Resample all to monthly
    conc_df = conc_df.resample("M").last()

    return conc_df


def _build_macro_features(layer4: dict) -> pd.DataFrame:
    """Extract macro features from layer4 output.

    Layer4 returns a unified macro DataFrame indexed by date with columns
    including fed_funds_rate, yield_curve_spread, m2_yoy_growth, credit_spread,
    unemployment, gdp_yoy_growth, real_interest_rate, etc.
    """
    data = layer4.get("data", pd.DataFrame())

    if isinstance(data, dict):
        # Layer4 may nest the DataFrame inside another key
        data = data.get("macro_df", pd.DataFrame())

    if data.empty:
        console.print("[yellow]Layer 4: No macro data found. Features will be NaN.[/]")
        return pd.DataFrame()

    data = data.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        # Layer4 uses 'date' as a column in some builds
        if "date" in data.columns:
            data = data.set_index("date")
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    col_rename = {
        "fed_funds_rate": "fed_funds",
        "yield_curve_spread": "yield_curve",
        "m2_yoy_growth": "m2_growth",
        "credit_spread": "credit_spread",
        "real_interest_rate": "real_rate",
        "gdp_yoy_growth": "gdp_growth",
    }

    frames = {}
    for src, dst in col_rename.items():
        if src in data.columns:
            frames[dst] = data[src]

    # Passthrough columns that already use canonical names
    for col in ["unemployment", "vix_monthly_avg", "vix"]:
        if col in data.columns and col not in frames:
            frames[col] = data[col]

    # Derived: 12-month change in fed funds rate
    if "fed_funds" in frames:
        frames["fed_funds_12m_chg"] = frames["fed_funds"].diff(12)

    if not frames:
        console.print("[yellow]Layer 4: No macro columns matched. Features will be NaN.[/]")
        return pd.DataFrame()

    macro_df = pd.DataFrame(frames)
    macro_df = macro_df.resample("M").last()

    # Forward-fill macro releases (typically monthly with publication lag)
    for col in ["fed_funds", "m2_growth", "credit_spread", "unemployment", "gdp_growth"]:
        if col in macro_df.columns:
            macro_df[col] = macro_df[col].ffill(limit=3)

    return macro_df


def _build_sentiment_features(layer5: dict) -> pd.DataFrame:
    """Extract sentiment features from layer5 output.

    Only available from ~2004 (Google Trends) or ~2012 (EDGAR/Reddit).
    Returns an incomplete series — callers must NOT impute.
    """
    data: dict = layer5.get("data", {})
    frames = {}

    # EDGAR NLP: ai_mention_rate (mentions per 1K words in NVDA filings)
    nvda_filings = data.get("nvda_filings", pd.DataFrame())
    if not nvda_filings.empty and "mentions_per_1k_words" in nvda_filings.columns:
        df_nlp = nvda_filings.copy()
        if "filing_date" in df_nlp.columns:
            df_nlp = df_nlp.set_index(pd.to_datetime(df_nlp["filing_date"]))
        elif not isinstance(df_nlp.index, pd.DatetimeIndex):
            df_nlp.index = pd.to_datetime(df_nlp.index)
        frames["ai_mention_rate"] = df_nlp["mentions_per_1k_words"]

        # OpenAI hype/specificity scores (if available)
        openai_scores = data.get("openai_scores", pd.DataFrame())
        if not openai_scores.empty:
            if not isinstance(openai_scores.index, pd.DatetimeIndex):
                if "filing_date" in openai_scores.columns:
                    openai_scores = openai_scores.set_index(
                        pd.to_datetime(openai_scores["filing_date"])
                    )
                else:
                    openai_scores.index = pd.to_datetime(openai_scores.index)
            if "hype_score" in openai_scores.columns:
                frames["hype_score"] = openai_scores["hype_score"]
            if "revenue_specificity" in openai_scores.columns:
                frames["specificity_score"] = openai_scores["revenue_specificity"]

    # Google Trends: AI keyword index
    ai_trends = data.get("ai_trends", pd.DataFrame())
    if not ai_trends.empty:
        if not isinstance(ai_trends.index, pd.DatetimeIndex):
            ai_trends.index = pd.to_datetime(ai_trends.index)
        # Take the first numeric column as the composite AI trends signal
        numeric_cols = ai_trends.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            frames["google_trends_ai"] = ai_trends[numeric_cols].mean(axis=1)

    # Long-run trends: bubble keyword proxy
    long_trends = data.get("long_trends", pd.DataFrame())
    if not long_trends.empty:
        if not isinstance(long_trends.index, pd.DatetimeIndex):
            long_trends.index = pd.to_datetime(long_trends.index)
        numeric_cols = long_trends.select_dtypes(include=[np.number]).columns.tolist()
        # Look for a "bubble" keyword column
        bubble_cols = [c for c in numeric_cols if "bubble" in c.lower() or "crash" in c.lower()]
        if bubble_cols:
            frames["google_trends_bubble"] = long_trends[bubble_cols[0]]
        elif numeric_cols:
            frames["google_trends_bubble"] = long_trends[numeric_cols[0]]

    if not frames:
        console.print("[yellow]Layer 5: No sentiment data extracted. Sentiment features unavailable.[/]")
        return pd.DataFrame()

    sent_df = pd.DataFrame(frames)
    sent_df.index = pd.to_datetime(sent_df.index)
    sent_df = sent_df.resample("M").mean()  # mean for intra-month multiple filings

    return sent_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_feature_matrix(layer_results: dict) -> dict[str, pd.DataFrame]:
    """Combine outputs from all 5 layers into a monthly feature matrix.

    Parameters
    ----------
    layer_results:
        Dict with keys "layer1" .. "layer5", each being the dict returned
        by run_layerN().

    Returns
    -------
    dict with keys:
        "core"       — DataFrame (full history, no sentiment)
        "augmented"  — DataFrame (post-2004, with sentiment)
        "has_sentiment_data" — bool flag (NOT included as a feature)
        "feature_names_core" — list of column names for the core model
        "feature_names_augmented" — list of column names for the augmented model
    """
    console.print("\n[bold cyan]Building feature matrix from all 5 layers...[/]")

    # --- Build each feature block ---
    l1 = layer_results.get("layer1", {})
    l2 = layer_results.get("layer2", {})
    l3 = layer_results.get("layer3", {})
    l4 = layer_results.get("layer4", {})
    l5 = layer_results.get("layer5", {})

    price_df = _build_price_features(l1)
    fund_df = _build_fundamental_features(l2)
    conc_df = _build_concentration_features(l3)
    macro_df = _build_macro_features(l4)
    sent_df = _build_sentiment_features(l5)

    console.print(
        f"  Price features:         {len(price_df.columns) if not price_df.empty else 0} cols, "
        f"{len(price_df) if not price_df.empty else 0} rows"
    )
    console.print(
        f"  Fundamental features:   {len(fund_df.columns) if not fund_df.empty else 0} cols, "
        f"{len(fund_df) if not fund_df.empty else 0} rows"
    )
    console.print(
        f"  Concentration features: {len(conc_df.columns) if not conc_df.empty else 0} cols, "
        f"{len(conc_df) if not conc_df.empty else 0} rows"
    )
    console.print(
        f"  Macro features:         {len(macro_df.columns) if not macro_df.empty else 0} cols, "
        f"{len(macro_df) if not macro_df.empty else 0} rows"
    )
    console.print(
        f"  Sentiment features:     {len(sent_df.columns) if not sent_df.empty else 0} cols, "
        f"{len(sent_df) if not sent_df.empty else 0} rows"
    )

    # --- Merge all core features (left join on price index) ---
    dfs_core = [df for df in [price_df, fund_df, conc_df, macro_df] if not df.empty]

    if not dfs_core:
        raise RuntimeError("No feature data available from any layer. Cannot build feature matrix.")

    # Use the broadest index available as the base
    base = dfs_core[0]
    for df in dfs_core[1:]:
        base = base.join(df, how="outer")

    # Ensure full monthly date range
    if len(base) > 0:
        full_idx = pd.date_range(base.index.min(), base.index.max(), freq="M")
        base = base.reindex(full_idx)

    # Forward-fill macro cols (max 3 months lag) — must be done AFTER reindex
    for col in MACRO_FFILL_COLS:
        if col in base.columns:
            base[col] = base[col].ffill(limit=3)

    # Drop rows with more than 50% NaN
    threshold = len(base.columns) * 0.5
    features_core = base.dropna(thresh=int(threshold))

    # --- Augmented model: add sentiment (no imputation) ---
    has_sentiment = not sent_df.empty
    if has_sentiment:
        features_aug = features_core.join(sent_df, how="left")
        # Restrict augmented to post-2004 (when Google Trends becomes available)
        features_aug = features_aug.loc[features_aug.index >= "2004-01-01"]
        # Drop rows where ALL sentiment columns are NaN
        available_sent = [c for c in SENTIMENT_COLS if c in features_aug.columns]
        if available_sent:
            features_aug = features_aug.dropna(subset=available_sent, how="all")
    else:
        features_aug = features_core.copy()

    console.print(
        f"\n[green]Core feature matrix:[/] {features_core.shape[0]} rows x "
        f"{features_core.shape[1]} features"
    )
    if has_sentiment:
        console.print(
            f"[green]Augmented feature matrix:[/] {features_aug.shape[0]} rows x "
            f"{features_aug.shape[1]} features (post-2004 with sentiment)"
        )

    return {
        "core": features_core,
        "augmented": features_aug,
        "has_sentiment_data": has_sentiment,
        "feature_names_core": list(features_core.columns),
        "feature_names_augmented": list(features_aug.columns),
    }


def assign_regime_labels(index: pd.DatetimeIndex) -> pd.Series:
    """Assign market regime labels to each date in the index.

    Labels are defined using well-established market history (see spec §2.2).
    Dates outside defined periods default to normal_growth.

    Parameters
    ----------
    index : DatetimeIndex of monthly (or any) timestamps.

    Returns
    -------
    pd.Series of strings: {"bubble", "correction", "normal_growth", "recovery"}
    """
    labels = pd.Series("normal_growth", index=index, dtype=str)

    # ---- Bubble periods ----
    bubble_ranges = [
        ("1998-01-01", "2000-03-31"),   # Dot-com run-up
        ("2007-01-01", "2007-10-31"),   # Housing bubble late-stage
        ("2024-07-01", "2026-03-31"),   # AI concentration phase — CLASSIFY THIS
    ]

    # ---- Correction periods ----
    correction_ranges = [
        ("1990-07-01", "1991-03-31"),   # 1990-91 recession
        ("2000-04-01", "2002-10-31"),   # Dot-com crash
        ("2007-11-01", "2009-03-31"),   # GFC
        ("2020-02-01", "2020-03-31"),   # COVID crash
        ("2022-01-01", "2022-10-31"),   # Fed hike bear market
    ]

    # ---- Recovery periods ----
    recovery_ranges = [
        ("1991-04-01", "1994-12-31"),   # Post-1990 recession recovery
        ("2002-11-01", "2004-06-30"),   # Post dot-com recovery
        ("2009-04-01", "2013-12-31"),   # Post-GFC recovery (QE-driven)
        ("2020-04-01", "2021-12-31"),   # COVID recovery / QE euphoria
    ]

    # ---- Normal growth — everything else in labeled ranges ----
    normal_ranges = [
        ("1995-01-01", "1997-12-31"),   # Mid-1990s pre-bubble expansion
        ("2003-01-01", "2006-12-31"),   # Mid-2000s expansion
        ("2014-01-01", "2019-12-31"),   # Mid-2010s expansion
        ("2022-11-01", "2024-06-30"),   # AI bull market (early) — real revenue backing
    ]

    def _apply_labels(ranges: list[tuple], label: str) -> None:
        for start, end in ranges:
            mask = (index >= start) & (index <= end)
            labels.loc[mask] = label

    # Apply in order (later calls override earlier for overlapping ranges)
    _apply_labels(normal_ranges, "normal_growth")
    _apply_labels(recovery_ranges, "recovery")
    _apply_labels(correction_ranges, "correction")
    _apply_labels(bubble_ranges, "bubble")

    label_counts = labels.value_counts()
    console.print(
        f"  Regime labels assigned: "
        + ", ".join(f"{k}={v}" for k, v in label_counts.items())
    )

    return labels


# ---------------------------------------------------------------------------
# RSI helper
# ---------------------------------------------------------------------------


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EWM approximation).

    Uses pandas ewm with alpha=1/period which is equivalent to Wilder's
    smoothed moving average and avoids the NaN-propagation bug in the
    loop-based approach.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing = EWM with alpha = 1/period, adjust=False
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.name = "rsi_14"
    return rsi
