"""Layer 5: Sentiment & NLP Analysis.

Sub-layers:
  5A — Earnings Call / SEC Filing NLP (NVDA AI-era, CSCO dot-com era)
  5B — Google Trends (pytrends)

Reddit (sub-layer 5C) is explicitly SKIPPED per project instructions.

Charts produced (saved to submissions/charts/):
  chart_5_1 — NVDA AI keyword mentions per filing over time
  chart_5_2 — Hype score vs revenue specificity scatter (or keyword density fallback)
  chart_5_3 — Google Trends lines for AI-related search terms
  chart_5_4 — Dual-axis: NVDA AI mentions (bars) + NVDA stock price (line)
  chart_5_6 — Side-by-side: NVDA AI-era keywords vs CSCO dot-com-era keywords

Statistical tests:
  - Pearson correlation: keyword mention growth rate vs NVDA stock price growth rate
  - Mann-Whitney U: quarterly keyword density, AI era vs dot-com era
"""

from __future__ import annotations

import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from rich.console import Console
from scipy import stats

import plotly.graph_objects as go

load_dotenv()

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHART_DIR = PROJECT_ROOT / "submissions" / "charts"

for _d in (RAW_DIR, PROCESSED_DIR, CHART_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PLOTLY_TEMPLATE = "plotly_dark"
NVDA_COLOR = "#00CC96"
CSCO_COLOR = "#EF553B"

console = Console()

# ---------------------------------------------------------------------------
# SEC EDGAR constants
# ---------------------------------------------------------------------------

SEC_HEADERS = {
    "User-Agent": "2Kim Datathon team@example.com",
    "Accept-Encoding": "gzip, deflate",
}
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILINGS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_FILING_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"

# CIKs (zero-padded to 10 digits for filing URLs, bare for submissions API)
NVDA_CIK = "0001045810"
CSCO_CIK = "0000858877"

NVDA_AI_KEYWORDS: list[str] = [
    "artificial intelligence",
    "AI",
    "machine learning",
    "generative AI",
    "gen AI",
    "GenAI",
    "LLM",
    "GPU",
    "data center",
    "deep learning",
    "neural network",
    "large language model",
    "foundation model",
    "transformer",
    "AI infrastructure",
    "AI accelerator",
]

CSCO_DOTCOM_KEYWORDS: list[str] = [
    "internet",
    "e-commerce",
    "e commerce",
    "electronic commerce",
    "World Wide Web",
    "web",
    "broadband",
    "networking",
    "dot-com",
    "dot com",
    "online",
]

# Pre-compiled patterns (case-insensitive, word-boundary aware where possible)
_AI_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in NVDA_AI_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
_DOTCOM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CSCO_DOTCOM_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpers: caching
# ---------------------------------------------------------------------------


def _json_cache_path(key: str) -> Path:
    return RAW_DIR / f"{key}.json"


def _load_json_cache(key: str) -> dict | None:
    p = _json_cache_path(key)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_json_cache(key: str, data: Any) -> None:
    p = _json_cache_path(key)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _parquet_cache_path(key: str) -> Path:
    return RAW_DIR / f"{key}.parquet"


def _load_df_cache(key: str) -> pd.DataFrame | None:
    p = _parquet_cache_path(key)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None


def _save_df_cache(key: str, df: pd.DataFrame) -> None:
    p = _parquet_cache_path(key)
    df.to_parquet(p, index=True)


# ---------------------------------------------------------------------------
# Helpers: EDGAR
# ---------------------------------------------------------------------------

_LAST_EDGAR_REQUEST: float = 0.0
_EDGAR_MIN_INTERVAL = 0.11  # 10 req/s max


def _edgar_get(url: str, params: dict | None = None) -> requests.Response:
    """Rate-limited GET for SEC EDGAR."""
    global _LAST_EDGAR_REQUEST
    elapsed = time.monotonic() - _LAST_EDGAR_REQUEST
    if elapsed < _EDGAR_MIN_INTERVAL:
        time.sleep(_EDGAR_MIN_INTERVAL - elapsed)
    resp = requests.get(url, params=params, headers=SEC_HEADERS, timeout=30)
    _LAST_EDGAR_REQUEST = time.monotonic()
    return resp


def _fetch_company_filings_index(cik_padded: str, form_type: str) -> list[dict]:
    """Return list of filings of given form type from EDGAR submissions JSON.

    Args:
        cik_padded: Zero-padded 10-digit CIK string (e.g. "0001045810").
        form_type: e.g. "10-Q" or "10-K".

    Returns:
        List of dicts with keys: accessionNumber, filingDate, form, primaryDocument.
    """
    url = EDGAR_SUBMISSIONS_URL.format(cik=cik_padded)
    resp = _edgar_get(url)
    resp.raise_for_status()
    data = resp.json()

    filings_raw = data.get("filings", {}).get("recent", {})
    forms = filings_raw.get("form", [])
    dates = filings_raw.get("filingDate", [])
    accessions = filings_raw.get("accessionNumber", [])
    docs = filings_raw.get("primaryDocument", [])

    results = []
    for form, date, acc, doc in zip(forms, dates, accessions, docs):
        if form == form_type:
            results.append(
                {
                    "form": form,
                    "filingDate": date,
                    "accessionNumber": acc,
                    "primaryDocument": doc,
                }
            )
    return results


def _fetch_filing_text(cik_bare: str, accession: str, primary_doc: str) -> str:
    """Download the primary document text of a filing.

    Tries the primary document first; falls back to the index page if that
    returns non-text content.
    """
    acc_nodash = accession.replace("-", "")
    url = EDGAR_FILING_DOC_URL.format(
        cik=cik_bare, accession=acc_nodash, filename=primary_doc
    )
    resp = _edgar_get(url)
    if resp.status_code != 200:
        return ""
    content_type = resp.headers.get("Content-Type", "")
    # Accept text/html, text/plain, and application/octet-stream (older EDGAR filings).
    # Reject binary formats like PDF or ZIP.
    is_binary = any(t in content_type for t in ("pdf", "zip", "image", "audio", "video"))
    if is_binary:
        return ""
    text = resp.text
    # Strip HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _count_keywords(text: str, pattern: re.Pattern) -> dict:
    """Return keyword mention statistics for a filing text."""
    words = text.split()
    word_count = len(words)
    matches = pattern.findall(text)
    mention_count = len(matches)
    mentions_per_1k = (mention_count / word_count * 1000) if word_count > 0 else 0.0
    # Per-term breakdown
    term_counts: dict[str, int] = {}
    for m in matches:
        key = m.lower()
        term_counts[key] = term_counts.get(key, 0) + 1
    return {
        "word_count": word_count,
        "mention_count": mention_count,
        "mentions_per_1k_words": mentions_per_1k,
        "term_breakdown": term_counts,
    }


# ---------------------------------------------------------------------------
# Sub-layer 5A — EDGAR filing NLP
# ---------------------------------------------------------------------------


def _fetch_nvda_filings(
    force_refresh: bool = False,
    start_year: int = 2019,
    end_year: int = 2026,
) -> pd.DataFrame:
    """Fetch NVDA 10-Q/10-K filings and count AI keywords.

    Returns DataFrame with one row per filing.
    """
    cache_key = "layer5_nvda_filings"
    if not force_refresh:
        cached = _load_df_cache(cache_key)
        if cached is not None:
            console.print("[dim]Cache hit:[/] layer5_nvda_filings.parquet")
            return cached

    console.print("[cyan]Fetching NVDA 10-Q/10-K filings from EDGAR...[/]")

    cik_padded = NVDA_CIK
    cik_bare = cik_padded.lstrip("0")

    rows: list[dict] = []

    for form in ("10-Q", "10-K"):
        try:
            filings = _fetch_company_filings_index(cik_padded, form)
        except Exception as exc:
            console.print(f"[yellow]Failed to list NVDA {form} filings: {exc}[/]")
            continue

        for filing in filings:
            filing_date = filing["filingDate"]
            year = int(filing_date[:4])
            if year < start_year or year > end_year:
                continue

            acc = filing["accessionNumber"]
            doc = filing["primaryDocument"]

            console.print(f"  [dim]{form} {filing_date} ({acc})[/]")

            try:
                text = _fetch_filing_text(cik_bare, acc, doc)
            except Exception as exc:
                console.print(f"  [yellow]Failed to fetch text: {exc}[/]")
                text = ""

            if not text:
                console.print("  [yellow]Empty document, skipping.[/]")
                continue

            stats_dict = _count_keywords(text, _AI_PATTERN)

            rows.append(
                {
                    "ticker": "NVDA",
                    "form": form,
                    "filing_date": pd.Timestamp(filing_date),
                    "accession": acc,
                    "word_count": stats_dict["word_count"],
                    "mention_count": stats_dict["mention_count"],
                    "mentions_per_1k_words": stats_dict["mentions_per_1k_words"],
                    "term_breakdown": json.dumps(stats_dict["term_breakdown"]),
                }
            )

    if not rows:
        console.print("[yellow]No NVDA filing rows collected.[/]")
        df = pd.DataFrame(
            columns=[
                "ticker", "form", "filing_date", "accession",
                "word_count", "mention_count", "mentions_per_1k_words", "term_breakdown",
            ]
        )
    else:
        df = pd.DataFrame(rows)
        df = df.sort_values("filing_date").reset_index(drop=True)

    _save_df_cache(cache_key, df)
    console.print(f"[green]Cached NVDA filings:[/] {len(df)} rows")
    return df


def _fetch_csco_filings(
    force_refresh: bool = False,
    start_year: int = 1997,
    end_year: int = 2002,
) -> pd.DataFrame:
    """Fetch CSCO 10-Q/10-K filings (dot-com era) and count dot-com keywords."""
    cache_key = "layer5_csco_filings"
    if not force_refresh:
        cached = _load_df_cache(cache_key)
        if cached is not None:
            console.print("[dim]Cache hit:[/] layer5_csco_filings.parquet")
            return cached

    console.print("[cyan]Fetching CSCO 10-Q/10-K filings from EDGAR...[/]")

    cik_padded = CSCO_CIK
    cik_bare = cik_padded.lstrip("0")

    rows: list[dict] = []

    for form in ("10-Q", "10-K"):
        try:
            filings = _fetch_company_filings_index(cik_padded, form)
        except Exception as exc:
            console.print(f"[yellow]Failed to list CSCO {form} filings: {exc}[/]")
            continue

        for filing in filings:
            filing_date = filing["filingDate"]
            year = int(filing_date[:4])
            if year < start_year or year > end_year:
                continue

            acc = filing["accessionNumber"]
            doc = filing["primaryDocument"]

            console.print(f"  [dim]{form} {filing_date} ({acc})[/]")

            try:
                text = _fetch_filing_text(cik_bare, acc, doc)
            except Exception as exc:
                console.print(f"  [yellow]Failed to fetch text: {exc}[/]")
                text = ""

            if not text:
                console.print("  [yellow]Empty document, skipping.[/]")
                continue

            stats_dict = _count_keywords(text, _DOTCOM_PATTERN)

            rows.append(
                {
                    "ticker": "CSCO",
                    "form": form,
                    "filing_date": pd.Timestamp(filing_date),
                    "accession": acc,
                    "word_count": stats_dict["word_count"],
                    "mention_count": stats_dict["mention_count"],
                    "mentions_per_1k_words": stats_dict["mentions_per_1k_words"],
                    "term_breakdown": json.dumps(stats_dict["term_breakdown"]),
                }
            )

    if not rows:
        console.print("[yellow]No CSCO filing rows collected.[/]")
        df = pd.DataFrame(
            columns=[
                "ticker", "form", "filing_date", "accession",
                "word_count", "mention_count", "mentions_per_1k_words", "term_breakdown",
            ]
        )
    else:
        df = pd.DataFrame(rows)
        df = df.sort_values("filing_date").reset_index(drop=True)

    _save_df_cache(cache_key, df)
    console.print(f"[green]Cached CSCO filings:[/] {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Sub-layer 5A — Optional OpenAI scoring
# ---------------------------------------------------------------------------

_OPENAI_SYSTEM_PROMPT = """You are a financial analyst specializing in AI/technology sector analysis.
Analyze the provided SEC filing excerpt to measure AI-related hype intensity.

You MUST respond with ONLY a valid JSON object matching this schema (no markdown, no preamble):
{
  "hype_score": <int 1-10>,
  "revenue_specificity": <int 1-10>,
  "sentiment": "<bullish|neutral|bearish>",
  "key_ai_claims": [<string>, ...],
  "revenue_figures_mentioned": [<string>, ...]
}

hype_score guide:
1-2: AI barely mentioned
3-4: AI is one of several growth areas
5-6: AI is a major strategic pillar
7-8: AI framed as transformative for the entire business
9-10: Company pivoting entirely to AI narrative

revenue_specificity guide:
1-2: Vague claims only ("AI will be big")
3-4: General sizing ("billions in AI opportunity")
5-6: Directional figures ("AI revenue doubled YoY")
7-8: Specific dollar figures ($X billion from AI this quarter)
9-10: Detailed segment breakdown with margins"""

_OPENAI_USER_TEMPLATE = """Analyze this SEC filing excerpt for {ticker} (filed {filing_date}):

--- EXCERPT START ---
{excerpt}
--- EXCERPT END ---

Respond with ONLY the JSON object."""


def _score_filing_openai(
    ticker: str,
    filing_date: str,
    text: str,
    max_chars: int = 8_000,
) -> dict | None:
    """Score a filing excerpt using GPT-4o-mini.

    Returns a dict with hype_score, revenue_specificity, sentiment, etc.
    Returns None if OpenAI is unavailable.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        return None

    try:
        from openai import OpenAI  # local import to avoid hard dep if not installed

        client = OpenAI(api_key=openai_key)

        # Use first max_chars chars (intro + business description — most hype-laden section)
        excerpt = text[:max_chars]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _OPENAI_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _OPENAI_USER_TEMPLATE.format(
                        ticker=ticker,
                        filing_date=filing_date,
                        excerpt=excerpt,
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=600,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        result["input_tokens"] = response.usage.prompt_tokens
        result["output_tokens"] = response.usage.completion_tokens
        return result

    except Exception as exc:
        console.print(f"[yellow]OpenAI scoring failed: {exc}[/]")
        return None


def _run_openai_scoring(
    nvda_df: pd.DataFrame,
    force_refresh: bool = False,
    top_n: int = 8,
) -> pd.DataFrame:
    """Score top N NVDA filings (by mention_count) using GPT-4o-mini.

    Only scores filings for which we have text cached. Returns a DataFrame
    with columns: accession, hype_score, revenue_specificity, sentiment,
    key_ai_claims, revenue_figures_mentioned.

    Falls back gracefully if OpenAI is unavailable.
    """
    cache_key = "layer5_nvda_openai_scores"
    if not force_refresh:
        cached = _load_df_cache(cache_key)
        if cached is not None:
            console.print("[dim]Cache hit:[/] layer5_nvda_openai_scores.parquet")
            return cached

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        console.print("[yellow]OPENAI_API_KEY not set — skipping GPT scoring.[/]")
        return pd.DataFrame()

    # Select top N filings by mention_count
    target_df = nvda_df.nlargest(top_n, "mention_count").copy()

    scores: list[dict] = []
    total_in_tokens = 0
    total_out_tokens = 0

    for _, row in target_df.iterrows():
        cik_bare = NVDA_CIK.lstrip("0")
        acc = row["accession"]
        acc_nodash = acc.replace("-", "")

        # Try to re-fetch text (small excerpt only, economical)
        # We re-use the primary document URL pattern from the accession
        # The primaryDocument column is not in nvda_df; we fetch index to find it.
        try:
            index_url = (
                f"https://www.sec.gov/cgi-bin/browse-edgar?"
                f"action=getcompany&CIK={cik_bare}&type=10-Q&dateb=&owner=include&count=40"
            )
            # Use the stored accession to build doc URL with a common filename pattern
            # Rather than re-fetching the document (expensive), use the index page
            # to locate the .htm or .txt file.
            doc_index_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/{acc_nodash}/"
            )
            resp = _edgar_get(doc_index_url)
            # Parse the filing index to find the .htm document
            htm_match = re.search(
                r'href="(/Archives/edgar/data/[^"]+\.htm)"',
                resp.text,
                re.IGNORECASE,
            )
            if htm_match:
                doc_url = "https://www.sec.gov" + htm_match.group(1)
                doc_resp = _edgar_get(doc_url)
                text = re.sub(r"<[^>]+>", " ", doc_resp.text)
                text = re.sub(r"\s+", " ", text).strip()
            else:
                text = ""
        except Exception as exc:
            console.print(f"  [yellow]Re-fetch failed for {acc}: {exc}[/]")
            text = ""

        if not text:
            console.print(f"  [yellow]No text for {acc}, skipping OpenAI scoring.[/]")
            continue

        filing_date_str = str(row["filing_date"])[:10]
        result = _score_filing_openai("NVDA", filing_date_str, text)

        if result is None:
            break  # OpenAI unavailable — stop trying

        total_in_tokens += result.pop("input_tokens", 0)
        total_out_tokens += result.pop("output_tokens", 0)

        result["accession"] = acc
        result["filing_date"] = row["filing_date"]
        result["ticker"] = "NVDA"
        scores.append(result)

        time.sleep(0.5)  # polite rate limiting

    if total_in_tokens:
        est_cost = total_in_tokens * 0.15 / 1_000_000 + total_out_tokens * 0.60 / 1_000_000
        console.print(
            f"[green]OpenAI scoring:[/] {len(scores)} filings, "
            f"{total_in_tokens:,} in / {total_out_tokens:,} out tokens, "
            f"~${est_cost:.3f}"
        )

    if not scores:
        return pd.DataFrame()

    result_df = pd.DataFrame(scores)
    _save_df_cache(cache_key, result_df)
    return result_df


# ---------------------------------------------------------------------------
# Sub-layer 5A — NVDA stock price helper (used in charts 5.4)
# ---------------------------------------------------------------------------


def _fetch_nvda_price_quarterly(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch NVDA quarterly closing prices via yfinance.

    Returns DataFrame indexed by quarter end date with columns: close, quarter_label.
    Falls back gracefully if yfinance is unavailable.
    """
    cache_key = "layer5_nvda_price_quarterly"
    if not force_refresh:
        cached = _load_df_cache(cache_key)
        if cached is not None:
            console.print("[dim]Cache hit:[/] layer5_nvda_price_quarterly.parquet")
            return cached

    try:
        import yfinance as yf  # optional dep

        ticker = yf.Ticker("NVDA")
        hist = ticker.history(start="2019-01-01", end="2026-04-01", interval="1mo")
        if hist.empty:
            raise ValueError("yfinance returned empty history")

        hist.index = pd.to_datetime(hist.index)
        # Resample to quarter
        quarterly = hist["Close"].resample("QE").last().dropna()
        df = quarterly.reset_index()
        df.columns = ["date", "close"]
        df["quarter_label"] = (
            df["date"].dt.to_period("Q").astype(str)
        )
        _save_df_cache(cache_key, df)
        console.print(f"[green]Cached NVDA prices:[/] {len(df)} quarters")
        return df

    except Exception as exc:
        console.print(f"[yellow]yfinance unavailable ({exc}) — NVDA price overlay skipped.[/]")
        return pd.DataFrame(columns=["date", "close", "quarter_label"])


# ---------------------------------------------------------------------------
# Sub-layer 5B — Google Trends
# ---------------------------------------------------------------------------

_AI_TREND_TERMS = ["NVIDIA stock", "AI stocks", "AI bubble", "artificial intelligence"]
_LONG_TREND_TERMS = ["NVIDIA stock", "Cisco stock", "AI stocks", "tech bubble"]


def _fetch_google_trends(
    keywords: list[str],
    timeframe: str,
    geo: str = "US",
    category: int = 0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch Google Trends interest over time, with caching.

    Returns DataFrame with columns: date, <keyword1>, <keyword2>, ...
    Returns empty DataFrame if pytrends fails.
    """
    safe_kws = "_".join(k.replace(" ", "_")[:15] for k in keywords[:3])
    tf_safe = timeframe.replace(" ", "_").replace(":", "")[:30]
    cache_key = f"layer5_gtrends_{safe_kws}_{tf_safe}"

    if not force_refresh:
        cached = _load_df_cache(cache_key)
        if cached is not None:
            console.print(f"[dim]Cache hit:[/] {cache_key}.parquet")
            return cached

    try:
        from pytrends.request import TrendReq
        from pytrends.exceptions import TooManyRequestsError
    except ImportError:
        console.print("[yellow]pytrends not installed — Google Trends skipped.[/]")
        return pd.DataFrame()

    console.print(f"[cyan]Fetching Google Trends:[/] {keywords} ({timeframe})")

    pytrends = TrendReq(
        hl="en-US",
        tz=360,
        timeout=(10, 25),
        retries=3,
        backoff_factor=1.5,
    )

    all_batches: list[pd.DataFrame] = []

    for i in range(0, len(keywords), 5):
        batch = keywords[i : i + 5]
        for attempt in range(3):
            try:
                pytrends.build_payload(
                    kw_list=batch,
                    cat=category,
                    timeframe=timeframe,
                    geo=geo,
                    gprop="",
                )
                df_batch = pytrends.interest_over_time()
                if "isPartial" in df_batch.columns:
                    df_batch = df_batch.drop("isPartial", axis=1)
                all_batches.append(df_batch)
                time.sleep(2.5 + attempt)
                break
            except TooManyRequestsError:
                wait = 30 * (attempt + 1)
                console.print(f"[yellow]Google Trends rate limited. Waiting {wait}s...[/]")
                time.sleep(wait)
            except Exception as exc:
                console.print(f"[yellow]pytrends error: {exc}[/]")
                time.sleep(5)

    if not all_batches:
        console.print("[yellow]No Google Trends data retrieved.[/]")
        return pd.DataFrame()

    combined = all_batches[0]
    for extra in all_batches[1:]:
        combined = combined.join(extra, how="outer")

    combined = combined.reset_index().rename(columns={"date": "date"})
    combined["date"] = pd.to_datetime(combined["date"])

    _save_df_cache(cache_key, combined)
    console.print(f"[green]Cached Google Trends:[/] {len(combined)} rows")
    return combined


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def _run_statistical_tests(
    nvda_df: pd.DataFrame, csco_df: pd.DataFrame, nvda_price_df: pd.DataFrame
) -> dict:
    """Run required statistical tests.

    1. Pearson correlation: NVDA quarterly keyword mention growth rate vs
       NVDA quarterly stock price growth rate.
    2. Mann-Whitney U: quarterly keyword density, AI-era (NVDA) vs dot-com era (CSCO).

    Returns dict with test results.
    """
    results: dict = {}

    # -----------------------------------------------------------------------
    # Test 1: Keyword growth rate vs price growth rate
    # -----------------------------------------------------------------------
    try:
        if nvda_df.empty or nvda_price_df.empty:
            raise ValueError("Insufficient data for correlation test")

        # Aggregate NVDA keyword density to quarterly
        nvda_q = nvda_df.copy()
        nvda_q["quarter"] = nvda_q["filing_date"].dt.to_period("Q")
        nvda_quarterly = (
            nvda_q.groupby("quarter")["mentions_per_1k_words"]
            .mean()
            .sort_index()
        )

        # Compute growth rates (pct change)
        mention_growth = nvda_quarterly.pct_change().dropna()

        # Map price to quarters
        price_df = nvda_price_df.copy()
        price_df["quarter"] = pd.to_datetime(price_df["date"]).dt.to_period("Q")
        price_quarterly = (
            price_df.groupby("quarter")["close"].last().sort_index()
        )
        price_growth = price_quarterly.pct_change().dropna()

        # Align on common quarters
        common_idx = mention_growth.index.intersection(price_growth.index)
        if len(common_idx) < 4:
            raise ValueError(f"Only {len(common_idx)} common quarters — too few for correlation")

        r, p = stats.pearsonr(
            mention_growth.loc[common_idx].values,
            price_growth.loc[common_idx].values,
        )
        results["pearson_keyword_growth_vs_price_growth"] = {
            "r": round(float(r), 4),
            "p_value": round(float(p), 4),
            "n": len(common_idx),
            "significant": bool(p < 0.05),
            "interpretation": (
                "Keyword mention growth is significantly correlated with stock price growth."
                if p < 0.05
                else "No statistically significant correlation between keyword growth and price growth."
            ),
        }
        console.print(
            f"[green]Pearson r=[/]{r:.4f} (p={p:.4f}, n={len(common_idx)}) — "
            f"keyword growth vs price growth"
        )
    except Exception as exc:
        console.print(f"[yellow]Pearson test skipped: {exc}[/]")
        results["pearson_keyword_growth_vs_price_growth"] = {"error": str(exc)}

    # -----------------------------------------------------------------------
    # Test 2: Mann-Whitney U — quarterly density, NVDA (AI era) vs CSCO (dot-com)
    # -----------------------------------------------------------------------
    try:
        if nvda_df.empty or csco_df.empty:
            raise ValueError("Insufficient data for Mann-Whitney test")

        nvda_q2 = nvda_df.copy()
        nvda_q2["quarter"] = nvda_q2["filing_date"].dt.to_period("Q")
        nvda_quarterly_density = (
            nvda_q2.groupby("quarter")["mentions_per_1k_words"].mean().values
        )

        csco_q = csco_df.copy()
        csco_q["quarter"] = csco_q["filing_date"].dt.to_period("Q")
        csco_quarterly_density = (
            csco_q.groupby("quarter")["mentions_per_1k_words"].mean().values
        )

        if len(nvda_quarterly_density) < 3 or len(csco_quarterly_density) < 3:
            raise ValueError("Not enough quarterly observations for Mann-Whitney U")

        u_stat, p_mw = stats.mannwhitneyu(
            nvda_quarterly_density,
            csco_quarterly_density,
            alternative="two-sided",
        )
        results["mann_whitney_ai_vs_dotcom_density"] = {
            "u_statistic": round(float(u_stat), 4),
            "p_value": round(float(p_mw), 4),
            "n_nvda_quarters": int(len(nvda_quarterly_density)),
            "n_csco_quarters": int(len(csco_quarterly_density)),
            "median_nvda": round(float(np.median(nvda_quarterly_density)), 4),
            "median_csco": round(float(np.median(csco_quarterly_density)), 4),
            "significant": bool(p_mw < 0.05),
            "interpretation": (
                "Keyword density distributions are significantly different between AI era and dot-com era."
                if p_mw < 0.05
                else "No statistically significant difference in keyword density distributions."
            ),
        }
        console.print(
            f"[green]Mann-Whitney U=[/]{u_stat:.1f} (p={p_mw:.4f}) — "
            f"NVDA median={np.median(nvda_quarterly_density):.2f}, "
            f"CSCO median={np.median(csco_quarterly_density):.2f}"
        )
    except Exception as exc:
        console.print(f"[yellow]Mann-Whitney test skipped: {exc}[/]")
        results["mann_whitney_ai_vs_dotcom_density"] = {"error": str(exc)}

    return results


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _save_chart(fig: go.Figure, name: str) -> dict:
    """Save a Plotly figure as both JSON and PNG. Returns paths dict."""
    json_path = CHART_DIR / f"{name}.json"
    png_path = CHART_DIR / f"{name}.png"

    with open(json_path, "w") as f:
        f.write(fig.to_json())

    try:
        fig.write_image(str(png_path), width=1600, height=900, scale=2)
    except Exception as exc:
        console.print(f"[yellow]PNG export failed for {name}: {exc}[/]")
        png_path = None

    console.print(f"[green]Saved chart:[/] {name}")
    return {"json": str(json_path), "png": str(png_path) if png_path else None}


def chart_5_1_ai_mentions(nvda_df: pd.DataFrame) -> go.Figure:
    """Chart 5.1 — NVDA AI keyword mentions per filing over time.

    Bar chart: mentions_per_1k_words per filing, sorted by date.
    Color encodes intensity (green gradient, hockey-stick expected).
    """
    if nvda_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Chart 5.1 — NVDA AI Mentions (No Data Available)",
            template=PLOTLY_TEMPLATE,
            width=1600, height=900,
        )
        return fig

    df = nvda_df.copy().sort_values("filing_date")
    df["label"] = df["filing_date"].dt.strftime("%Y-%m") + " (" + df["form"] + ")"

    # Color intensity proportional to mentions_per_1k_words
    max_val = df["mentions_per_1k_words"].max() or 1.0

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df["label"],
            y=df["mentions_per_1k_words"],
            name="AI Mentions per 1K Words",
            marker=dict(
                color=df["mentions_per_1k_words"],
                colorscale=[[0, "#1a4a3a"], [0.5, "#00a077"], [1.0, NVDA_COLOR]],
                cmin=0,
                cmax=max_val,
                colorbar=dict(title="Mentions/1K Words", x=1.02),
            ),
            opacity=0.85,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "AI Mentions/1K Words: %{y:.2f}<br>"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # Trend line (rolling mean) overlaid
    if len(df) >= 3:
        rolling = df["mentions_per_1k_words"].rolling(3, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=df["label"],
                y=rolling,
                name="3-Filing Rolling Average",
                mode="lines+markers",
                line=dict(color="#FFD700", width=2.5, dash="dot"),
                marker=dict(size=6),
            ),
            secondary_y=False,
        )

    # Annotation: ChatGPT launch approximate period
    chatgpt_date = pd.Timestamp("2023-01-01")
    nearest_idx = (df["filing_date"] - chatgpt_date).abs().idxmin()
    if nearest_idx is not None:
        label_val = df.loc[nearest_idx, "label"]
        fig.add_annotation(
            x=label_val,
            y=df.loc[nearest_idx, "mentions_per_1k_words"],
            text="ChatGPT era begins",
            showarrow=True,
            arrowhead=2,
            arrowcolor=NVDA_COLOR,
            font=dict(color=NVDA_COLOR, size=11),
            bgcolor="rgba(0,0,0,0.6)",
        )

    fig.update_layout(
        title=dict(
            text="NVDA AI Keyword Mentions in SEC Filings: The Hockey Stick",
            font=dict(size=20),
        ),
        xaxis_title="Filing (Date + Form Type)",
        xaxis=dict(tickangle=-45),
        template=PLOTLY_TEMPLATE,
        width=1600,
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(b=140),
    )
    fig.update_yaxes(title_text="AI Mentions per 1,000 Words", secondary_y=False)

    return fig


def chart_5_2_hype_vs_specificity(
    nvda_df: pd.DataFrame, openai_scores_df: pd.DataFrame
) -> go.Figure:
    """Chart 5.2 — Hype score vs revenue specificity (if OpenAI available).

    Falls back to keyword density histogram + cumulative line if OpenAI scores
    are unavailable.
    """
    has_scores = (
        openai_scores_df is not None
        and not openai_scores_df.empty
        and "hype_score" in openai_scores_df.columns
        and "revenue_specificity" in openai_scores_df.columns
    )

    if has_scores:
        df = openai_scores_df.copy()
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        df["year"] = df["filing_date"].dt.year

        # Merge mention_count from nvda_df if available
        if not nvda_df.empty and "accession" in nvda_df.columns:
            df = df.merge(
                nvda_df[["accession", "mention_count"]],
                on="accession",
                how="left",
            )
        if "mention_count" not in df.columns:
            df["mention_count"] = 10

        df["mention_count"] = df["mention_count"].fillna(5).clip(lower=3)
        size = (df["mention_count"] / df["mention_count"].max() * 40 + 8).clip(8, 50)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["hype_score"],
                y=df["revenue_specificity"],
                mode="markers",
                marker=dict(
                    size=size,
                    color=df["year"],
                    colorscale="Viridis",
                    colorbar=dict(title="Year"),
                    opacity=0.8,
                    line=dict(width=1, color="white"),
                ),
                text=df.apply(
                    lambda r: (
                        f"{r.get('ticker', 'NVDA')} {str(r['filing_date'])[:10]}"
                    ),
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Hype: %{x}<br>"
                    "Specificity: %{y}<br>"
                    "<extra></extra>"
                ),
            )
        )

        # Quadrant lines
        fig.add_vline(x=5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)

        # Quadrant labels
        annotations = [
            (8.5, 8.5, "Backed by Numbers", "green"),
            (1.5, 8.5, "Quiet Executors", "#4499FF"),
            (8.5, 1.5, "ALL HYPE,\nNO NUMBERS", "#FF4444"),
            (1.5, 1.5, "Not Playing\nthe AI Game", "gray"),
        ]
        for ax, ay, text, color in annotations:
            fig.add_annotation(
                x=ax,
                y=ay,
                text=text,
                showarrow=False,
                font=dict(color=color, size=13, family="Arial Black"),
                opacity=0.7,
            )

        title = "NVDA Filings: AI Hype Score vs Revenue Specificity"
        subtitle = "(point size = keyword mention count; color = year)"

    else:
        # Fallback: keyword density distribution
        if nvda_df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Chart 5.2 — No Data Available",
                template=PLOTLY_TEMPLATE,
                width=1600,
                height=900,
            )
            return fig

        df = nvda_df.copy().sort_values("filing_date")
        df["year"] = df["filing_date"].dt.year

        year_colors = {
            2019: "#4499FF", 2020: "#44DDFF", 2021: "#AAFFAA",
            2022: "#FFFF44", 2023: "#FFAA00", 2024: "#FF5500",
            2025: "#FF0000", 2026: "#FF00FF",
        }

        fig = go.Figure()

        for year, grp in df.groupby("year"):
            color = year_colors.get(int(year), "#CCCCCC")
            fig.add_trace(
                go.Bar(
                    x=grp["filing_date"].dt.strftime("%Y-%m"),
                    y=grp["mentions_per_1k_words"],
                    name=str(year),
                    marker_color=color,
                    opacity=0.8,
                )
            )

        title = "NVDA AI Keyword Density by Year (OpenAI Scores Unavailable)"
        subtitle = "Keyword density (mentions per 1,000 words) by filing"

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>", font=dict(size=18)),
        xaxis=dict(
            **({"range": [0.5, 10.5]} if has_scores else {}),
            title="Hype Score" if has_scores else "Filing Date",
        ),
        yaxis=dict(
            **({"range": [0.5, 10.5]} if has_scores else {}),
            title="Revenue Specificity" if has_scores else "Mentions per 1K Words",
        ),
        template=PLOTLY_TEMPLATE,
        width=1600,
        height=900,
    )

    return fig


def chart_5_3_google_trends(
    ai_trends_df: pd.DataFrame,
    long_trends_df: pd.DataFrame,
) -> go.Figure:
    """Chart 5.3 — Google Trends lines for AI-related search terms."""
    has_ai = ai_trends_df is not None and not ai_trends_df.empty
    has_long = long_trends_df is not None and not long_trends_df.empty

    if not has_ai and not has_long:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=(
                "Google Trends data unavailable.<br>"
                "pytrends may be rate-limited or not installed.<br>"
                "Re-run with force_refresh=True when API access is restored."
            ),
            showarrow=False,
            font=dict(size=16, color="#AAAAAA"),
        )
        fig.update_layout(
            title="Chart 5.3 — Google Trends: AI Search Interest (Data Unavailable)",
            template=PLOTLY_TEMPLATE,
            width=1600,
            height=900,
        )
        return fig

    fig = make_subplots(
        rows=2 if (has_ai and has_long) else 1,
        cols=1,
        subplot_titles=(
            ["Long-Term View (2004-2026, Monthly)", "AI Era Detail (2020-2026, Weekly)"]
            if (has_ai and has_long)
            else ["AI Era Google Trends (2020-2026)"]
        ),
        vertical_spacing=0.12,
    )
    n_rows = 2 if (has_ai and has_long) else 1

    long_series = [
        ("NVIDIA stock", "#EF553B", "solid"),
        ("Cisco stock", CSCO_COLOR, "solid"),
        ("AI stocks", "#FF7F0E", "dash"),
        ("tech bubble", "#AAAAAA", "dot"),
    ]
    ai_series = [
        ("NVIDIA stock", NVDA_COLOR, "solid"),
        ("AI stocks", "#FF7F0E", "solid"),
        ("AI bubble", "#8B0000", "dash"),
        ("artificial intelligence", "#B0AAFF", "dot"),
    ]

    if has_long:
        row = 1
        for col_name, color, dash in long_series:
            if col_name in long_trends_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=long_trends_df["date"],
                        y=long_trends_df[col_name],
                        mode="lines",
                        name=col_name,
                        line=dict(color=color, dash=dash, width=2),
                    ),
                    row=row, col=1,
                )
        # Shade AI era
        fig.add_vrect(
            x0="2020-01-01", x1="2026-04-01",
            fillcolor="rgba(255, 255, 100, 0.07)",
            line_width=0,
            annotation_text="AI Era",
            annotation_position="top left",
            row=row, col=1,
        )
        # Note about pre-2004 gap
        fig.add_annotation(
            x=0.01, y=-0.12,
            xref="x domain", yref="y domain",
            text="Note: Google Trends data unavailable before 2004. Dot-com peak (March 2000) cannot be shown.",
            showarrow=False,
            font=dict(color="#AAAAAA", size=10),
            row=row, col=1,
        )

    if has_ai:
        row = 2 if has_long else 1
        for col_name, color, dash in ai_series:
            if col_name in ai_trends_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=ai_trends_df["date"],
                        y=ai_trends_df[col_name],
                        mode="lines",
                        name=col_name,
                        showlegend=(not has_long),
                        line=dict(color=color, dash=dash, width=2),
                    ),
                    row=row, col=1,
                )

    fig.update_layout(
        title=dict(
            text="Google Search Interest: AI Stocks vs Historical Tech Bubble Terms",
            font=dict(size=20),
        ),
        template=PLOTLY_TEMPLATE,
        width=1600,
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Interest (0-100 relative)", col=1)

    return fig


def chart_5_4_sentiment_price_overlay(
    nvda_df: pd.DataFrame, nvda_price_df: pd.DataFrame
) -> go.Figure:
    """Chart 5.4 — AI mention count (bars) + NVDA stock price (line), dual axis."""
    if nvda_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Chart 5.4 — No Filing Data Available",
            template=PLOTLY_TEMPLATE,
            width=1600,
            height=900,
        )
        return fig

    df = nvda_df.copy().sort_values("filing_date")
    df["label"] = df["filing_date"].dt.strftime("%Y-%m")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bars: AI mention count per filing
    fig.add_trace(
        go.Bar(
            x=df["label"],
            y=df["mentions_per_1k_words"],
            name="AI Mentions per 1K Words",
            marker_color=NVDA_COLOR,
            opacity=0.7,
            hovertemplate="<b>%{x}</b><br>Mentions/1K Words: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Line: NVDA stock price
    if not nvda_price_df.empty:
        price_df = nvda_price_df.copy()
        price_df["date"] = pd.to_datetime(price_df["date"])
        fig.add_trace(
            go.Scatter(
                x=price_df["date"].dt.strftime("%Y-%m"),
                y=price_df["close"],
                name="NVDA Quarterly Close ($)",
                mode="lines+markers",
                line=dict(color="#FFD700", width=2.5),
                marker=dict(symbol="diamond", size=8),
                hovertemplate="<b>%{x}</b><br>NVDA: $%{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

    # Annotation: inflection point
    if len(df) > 3:
        peak_idx = df["mentions_per_1k_words"].idxmax()
        peak_label = df.loc[peak_idx, "label"]
        peak_val = df.loc[peak_idx, "mentions_per_1k_words"]
        fig.add_annotation(
            x=peak_label,
            y=peak_val,
            text="Peak AI mentions",
            showarrow=True,
            arrowhead=2,
            arrowcolor=NVDA_COLOR,
            font=dict(color=NVDA_COLOR, size=12),
            bgcolor="rgba(0,0,0,0.6)",
        )

    fig.update_layout(
        title=dict(
            text="NVDA AI Keyword Density vs Stock Price",
            font=dict(size=20),
        ),
        xaxis_title="Filing Date (Year-Month)",
        xaxis=dict(tickangle=-45),
        template=PLOTLY_TEMPLATE,
        width=1600,
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(b=120),
    )
    fig.update_yaxes(
        title_text="AI Mentions per 1,000 Words", secondary_y=False, color=NVDA_COLOR
    )
    fig.update_yaxes(
        title_text="NVDA Quarterly Close Price ($)", secondary_y=True, color="#FFD700"
    )

    return fig


def chart_5_6_keyword_comparison(
    nvda_df: pd.DataFrame, csco_df: pd.DataFrame
) -> go.Figure:
    """Chart 5.6 — Side-by-side: AI-era NVDA keywords vs dot-com CSCO keywords.

    Normalized per 1,000 words. Shows per-keyword breakdown from term_breakdown.
    """
    def _aggregate_terms(df: pd.DataFrame, keywords: list[str]) -> dict[str, float]:
        """Sum term counts across all filings, normalize by total words."""
        if df.empty:
            return {k: 0.0 for k in keywords}

        total_words = df["word_count"].sum()
        if total_words == 0:
            return {k: 0.0 for k in keywords}

        agg: dict[str, int] = {}
        for _, row in df.iterrows():
            try:
                breakdown = json.loads(row.get("term_breakdown") or "{}")
            except (json.JSONDecodeError, TypeError):
                breakdown = {}
            for term, count in breakdown.items():
                agg[term] = agg.get(term, 0) + count

        # Map back to canonical keyword labels
        result: dict[str, float] = {}
        for kw in keywords:
            kw_lower = kw.lower()
            count = agg.get(kw_lower, 0)
            result[kw] = count / total_words * 1000
        return result

    nvda_kw_density = _aggregate_terms(nvda_df, NVDA_AI_KEYWORDS)
    csco_kw_density = _aggregate_terms(csco_df, CSCO_DOTCOM_KEYWORDS)

    # Sort by density descending for cleaner chart
    nvda_sorted = sorted(nvda_kw_density.items(), key=lambda x: x[1], reverse=True)
    csco_sorted = sorted(csco_kw_density.items(), key=lambda x: x[1], reverse=True)

    nvda_labels = [item[0] for item in nvda_sorted]
    nvda_values = [item[1] for item in nvda_sorted]
    csco_labels = [item[0] for item in csco_sorted]
    csco_values = [item[1] for item in csco_sorted]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "NVDA AI-Era Keywords (2019-2026)",
            "CSCO Dot-Com Era Keywords (1997-2002)",
        ],
        horizontal_spacing=0.1,
    )

    fig.add_trace(
        go.Bar(
            x=nvda_labels,
            y=nvda_values,
            name="NVDA AI Keywords",
            marker_color=NVDA_COLOR,
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>%{y:.3f} per 1K words<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=csco_labels,
            y=csco_values,
            name="CSCO Dot-Com Keywords",
            marker_color=CSCO_COLOR,
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>%{y:.3f} per 1K words<extra></extra>",
        ),
        row=1, col=2,
    )

    # Summary stats annotation
    nvda_total = sum(nvda_values)
    csco_total = sum(csco_values)

    fig.add_annotation(
        x=0.25, y=1.05, xref="paper", yref="paper",
        text=f"Total density: {nvda_total:.2f} / 1K words",
        showarrow=False,
        font=dict(color=NVDA_COLOR, size=12),
    )
    fig.add_annotation(
        x=0.75, y=1.05, xref="paper", yref="paper",
        text=f"Total density: {csco_total:.2f} / 1K words",
        showarrow=False,
        font=dict(color=CSCO_COLOR, size=12),
    )

    fig.add_annotation(
        x=0.5, y=-0.12, xref="paper", yref="paper",
        text=(
            "Caveat: NVDA data are SEC 10-Q/10-K filings vs CSCO 10-Q/10-K filings. "
            "Formal filing language differs from earnings call transcripts — comparison is illustrative."
        ),
        showarrow=False,
        font=dict(color="#AAAAAA", size=10),
    )

    fig.update_layout(
        title=dict(
            text="AI-Era Hype Keywords (NVDA) vs Dot-Com Era Keywords (CSCO)<br>"
                 "<sup>Normalized per 1,000 words across all SEC filings in each period</sup>",
            font=dict(size=18),
        ),
        template=PLOTLY_TEMPLATE,
        width=1600,
        height=900,
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-40)
    fig.update_yaxes(title_text="Mentions per 1,000 Words", col=1)
    fig.update_yaxes(title_text="Mentions per 1,000 Words", col=2)

    return fig


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_layer5(force_refresh: bool = False) -> dict:
    """Run all Layer 5 sub-layers and return structured results.

    Args:
        force_refresh: If True, bypass all caches and re-fetch from APIs.

    Returns:
        dict with keys: data, stats, charts, findings, warnings.
    """
    warnings_list: list[str] = []
    chart_paths: dict[str, dict] = {}
    findings: list[str] = []

    console.rule("[bold cyan]Layer 5: Sentiment & NLP Analysis[/]")

    # -----------------------------------------------------------------------
    # 5A: EDGAR filing NLP
    # -----------------------------------------------------------------------
    console.rule("[cyan]5A: EDGAR Filing NLP[/]")

    nvda_df = pd.DataFrame()
    csco_df = pd.DataFrame()

    try:
        nvda_df = _fetch_nvda_filings(force_refresh=force_refresh)
        if nvda_df.empty:
            warnings_list.append("NVDA: no filings retrieved from EDGAR.")
        else:
            console.print(
                f"[green]NVDA filings:[/] {len(nvda_df)} total, "
                f"avg {nvda_df['mentions_per_1k_words'].mean():.2f} AI mentions/1K words"
            )
    except Exception as exc:
        warnings_list.append(f"NVDA EDGAR fetch failed: {exc}")
        console.print(f"[red]NVDA EDGAR error:[/] {exc}")

    try:
        csco_df = _fetch_csco_filings(force_refresh=force_refresh)
        if csco_df.empty:
            warnings_list.append("CSCO: no dot-com era filings retrieved from EDGAR.")
        else:
            console.print(
                f"[green]CSCO filings:[/] {len(csco_df)} total, "
                f"avg {csco_df['mentions_per_1k_words'].mean():.2f} dot-com mentions/1K words"
            )
    except Exception as exc:
        warnings_list.append(f"CSCO EDGAR fetch failed: {exc}")
        console.print(f"[red]CSCO EDGAR error:[/] {exc}")

    # Optional OpenAI scoring
    openai_scores_df = pd.DataFrame()
    try:
        openai_scores_df = _run_openai_scoring(nvda_df, force_refresh=force_refresh)
        if openai_scores_df.empty:
            warnings_list.append(
                "OpenAI scoring skipped (API key missing or unavailable). "
                "Chart 5.2 uses keyword density fallback."
            )
    except Exception as exc:
        warnings_list.append(f"OpenAI scoring error: {exc}")

    # -----------------------------------------------------------------------
    # 5B: Google Trends
    # -----------------------------------------------------------------------
    console.rule("[cyan]5B: Google Trends[/]")

    ai_trends_df = pd.DataFrame()
    long_trends_df = pd.DataFrame()

    try:
        ai_trends_df = _fetch_google_trends(
            keywords=_AI_TREND_TERMS,
            timeframe="2020-01-01 2026-03-28",
            force_refresh=force_refresh,
        )
    except Exception as exc:
        warnings_list.append(f"Google Trends (AI era) fetch failed: {exc}")
        console.print(f"[yellow]Google Trends AI fetch error:[/] {exc}")

    try:
        long_trends_df = _fetch_google_trends(
            keywords=_LONG_TREND_TERMS,
            timeframe="all",
            force_refresh=force_refresh,
        )
    except Exception as exc:
        warnings_list.append(f"Google Trends (long-term) fetch failed: {exc}")
        console.print(f"[yellow]Google Trends long fetch error:[/] {exc}")

    if ai_trends_df.empty and long_trends_df.empty:
        warnings_list.append(
            "All Google Trends requests failed. Chart 5.3 will show placeholder. "
            "Run with force_refresh=True when API access is restored, or fetch CSVs manually."
        )

    # -----------------------------------------------------------------------
    # NVDA price data
    # -----------------------------------------------------------------------
    nvda_price_df = pd.DataFrame()
    try:
        nvda_price_df = _fetch_nvda_price_quarterly(force_refresh=force_refresh)
    except Exception as exc:
        warnings_list.append(f"NVDA price fetch failed: {exc}")

    # -----------------------------------------------------------------------
    # Statistical tests
    # -----------------------------------------------------------------------
    console.rule("[cyan]Statistical Tests[/]")
    stat_results = _run_statistical_tests(nvda_df, csco_df, nvda_price_df)

    # Derive key findings from stats
    pearson = stat_results.get("pearson_keyword_growth_vs_price_growth", {})
    if pearson.get("significant"):
        r_val = pearson.get("r", 0)
        direction = "positive" if r_val > 0 else "negative"
        findings.append(
            f"Significant {direction} Pearson correlation (r={r_val:.3f}) between "
            f"NVDA AI keyword growth rate and stock price growth rate (p={pearson.get('p_value', '?')})."
        )
    else:
        findings.append(
            "No statistically significant linear correlation between NVDA keyword growth "
            "rate and stock price growth rate — narrative hype and price may diverge in timing."
        )

    mw = stat_results.get("mann_whitney_ai_vs_dotcom_density", {})
    if mw.get("significant"):
        findings.append(
            f"Mann-Whitney U test confirms AI-era keyword density (NVDA median={mw.get('median_nvda', '?'):.2f}/1K) "
            f"is statistically distinct from dot-com era density (CSCO median={mw.get('median_csco', '?'):.2f}/1K) "
            f"(U={mw.get('u_statistic', '?')}, p={mw.get('p_value', '?')})."
        )
    else:
        findings.append(
            "Mann-Whitney U test does not distinguish AI-era vs dot-com-era keyword densities — "
            "insufficient data or distributions overlap."
        )

    if not nvda_df.empty:
        peak_row = nvda_df.loc[nvda_df["mentions_per_1k_words"].idxmax()]
        findings.append(
            f"Peak AI keyword density in NVDA filings: "
            f"{peak_row['mentions_per_1k_words']:.2f} mentions/1K words "
            f"({str(peak_row['filing_date'])[:10]}, {peak_row['form']})."
        )

    if not csco_df.empty:
        peak_csco = csco_df.loc[csco_df["mentions_per_1k_words"].idxmax()]
        findings.append(
            f"Peak dot-com keyword density in CSCO filings: "
            f"{peak_csco['mentions_per_1k_words']:.2f} mentions/1K words "
            f"({str(peak_csco['filing_date'])[:10]}, {peak_csco['form']})."
        )

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    console.rule("[cyan]Generating Charts[/]")

    try:
        fig_5_1 = chart_5_1_ai_mentions(nvda_df)
        chart_paths["chart_5_1"] = _save_chart(fig_5_1, "chart_5_1_ai_mentions")
    except Exception as exc:
        warnings_list.append(f"Chart 5.1 generation failed: {exc}")
        console.print(f"[red]Chart 5.1 error:[/] {exc}")

    try:
        fig_5_2 = chart_5_2_hype_vs_specificity(nvda_df, openai_scores_df)
        chart_paths["chart_5_2"] = _save_chart(fig_5_2, "chart_5_2_hype_vs_specificity")
    except Exception as exc:
        warnings_list.append(f"Chart 5.2 generation failed: {exc}")
        console.print(f"[red]Chart 5.2 error:[/] {exc}")

    try:
        fig_5_3 = chart_5_3_google_trends(ai_trends_df, long_trends_df)
        chart_paths["chart_5_3"] = _save_chart(fig_5_3, "chart_5_3_google_trends")
    except Exception as exc:
        warnings_list.append(f"Chart 5.3 generation failed: {exc}")
        console.print(f"[red]Chart 5.3 error:[/] {exc}")

    try:
        fig_5_4 = chart_5_4_sentiment_price_overlay(nvda_df, nvda_price_df)
        chart_paths["chart_5_4"] = _save_chart(fig_5_4, "chart_5_4_sentiment_price_overlay")
    except Exception as exc:
        warnings_list.append(f"Chart 5.4 generation failed: {exc}")
        console.print(f"[red]Chart 5.4 error:[/] {exc}")

    try:
        fig_5_6 = chart_5_6_keyword_comparison(nvda_df, csco_df)
        chart_paths["chart_5_6"] = _save_chart(fig_5_6, "chart_5_6_keyword_comparison")
    except Exception as exc:
        warnings_list.append(f"Chart 5.6 generation failed: {exc}")
        console.print(f"[red]Chart 5.6 error:[/] {exc}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    console.rule("[bold green]Layer 5 Complete[/]")
    console.print(f"  Charts generated: {len(chart_paths)}")
    console.print(f"  Findings: {len(findings)}")
    if warnings_list:
        console.print(f"  [yellow]Warnings ({len(warnings_list)}):[/]")
        for w in warnings_list:
            console.print(f"    [yellow]- {w}[/]")

    return {
        "data": {
            "nvda_filings": nvda_df,
            "csco_filings": csco_df,
            "openai_scores": openai_scores_df,
            "ai_trends": ai_trends_df,
            "long_trends": long_trends_df,
            "nvda_price_quarterly": nvda_price_df,
        },
        "stats": stat_results,
        "charts": chart_paths,
        "findings": findings,
        "warnings": warnings_list,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Layer 5: Sentiment & NLP")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass all caches and re-fetch from APIs",
    )
    args = parser.parse_args()

    results = run_layer5(force_refresh=args.force_refresh)

    print("\n=== FINDINGS ===")
    for i, finding in enumerate(results["findings"], 1):
        print(f"  {i}. {finding}")

    print("\n=== STATS ===")
    print(json.dumps(results["stats"], indent=2, default=str))

    if results["warnings"]:
        print("\n=== WARNINGS ===")
        for w in results["warnings"]:
            print(f"  - {w}")
