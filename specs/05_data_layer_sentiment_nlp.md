# Layer 5: Sentiment & NLP Analysis

> **Status:** Spec Complete | **Owner:** Team 2Kim
> **Scoring targets:** Analysis & Evidence (25pts), Data Quality (20pts), Technical Rigor (15pts)
> **Upstream deps:** Layer 1 (price data for correlation), Layer 3 (fundamental data for revenue specificity validation)
> **Downstream:** Layer 7 (ML features), Layer 8 (presentation narrative)

---

## 1. Objective

Quantify the **hype narrative** around AI using natural language processing and search trend data. The core question: **Are we at "peak euphoria," and can we measure it?**

This layer provides the qualitative-turned-quantitative dimension. Price and fundamentals (Layers 1-3) tell us what is happening. Macro conditions (Layer 4) tell us the backdrop. Sentiment and NLP tell us what people *believe* is happening -- and whether that belief is running ahead of reality.

### Sub-questions this layer answers:
1. How rapidly are corporations adopting AI language in earnings calls? (Is everyone claiming to be an "AI company"?)
2. Is the AI hype backed by specific revenue claims, or is it vague hand-waving?
3. How does public search interest in "AI stocks" compare to "dot-com stocks" at equivalent stages?
4. What is retail investor sentiment (Reddit) telling us about NVDA specifically?
5. Is there a measurable divergence between hype intensity and fundamental delivery?

### Sub-layers:
- **5A:** Earnings Call NLP (OpenAI API) -- corporate hype measurement
- **5B:** Google Trends (pytrends) -- public interest measurement
- **5C:** Reddit Sentiment (praw + OpenAI) -- retail investor hype measurement

---

## 2. Sub-layer 5A: Earnings Call NLP (OpenAI API)

### 2.1 Data Source: Earnings Call Transcripts

**Primary source:** SEC EDGAR Full-Text Search API (EFTS)

```
Base URL: https://efts.sec.gov/LATEST/search-index
Endpoint: https://efts.sec.gov/LATEST/search-index?q="artificial intelligence"&dateRange=custom&startdt=2023-01-01&enddt=2026-03-28&forms=10-K,10-Q,8-K
```

**Alternative sources (ranked by preference):**
1. **SEC EDGAR EFTS** -- free, authoritative, but transcripts are embedded in 8-K filings (not all companies file transcripts as 8-Ks)
2. **Financial Modeling Prep API** (https://financialmodelingprep.com/) -- has `/earning_call_transcript/{symbol}` endpoint, free tier: 250 calls/day
3. **Alpha Vantage** -- `EARNINGS` function provides earnings dates but NOT transcripts
4. **Manual collection** -- For the ~10 most important companies, scrape from investor relations pages

**Target companies (S&P 500 tech sector, AI-relevant):**

| Tier | Companies | Rationale |
|------|-----------|-----------|
| Tier 1 (must-have) | NVDA, MSFT, GOOG/GOOGL, AMZN, META, AAPL | Magnificent 7 minus TSLA (less AI-focused) |
| Tier 2 (high-value) | AMD, AVGO, CRM, ORCL, ADBE, SNOW, PLTR | Major AI infrastructure and software |
| Tier 3 (breadth) | INTC, IBM, NOW, PANW, CRWD, MDB, AI (C3.ai) | Additional AI exposure names |

**Total:** ~20 companies x ~12 quarters (Q1 2023 through Q4 2025) = ~240 transcripts

**Date range:** Q1 2023 (ChatGPT launched Nov 2022; first full quarter of AI hype) through Q4 2025 (most recent complete quarter)

### 2.2 Transcript Acquisition Pipeline

```python
# pseudocode: fetch_transcripts.py

import requests
import time
from pathlib import Path

TRANSCRIPT_DIR = Path("data/raw/transcripts")
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# SEC EDGAR headers (required -- SEC blocks requests without proper User-Agent)
SEC_HEADERS = {
    "User-Agent": "2Kim Datathon Project team2kim@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Option A: SEC EDGAR Full-Text Search
def search_edgar_transcripts(company_cik: str, start_date: str, end_date: str) -> list[dict]:
    """Search EDGAR for earnings-related filings by CIK."""
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": '"earnings" OR "quarterly results"',
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "entityName": company_cik,
        "forms": "8-K",
    }
    resp = requests.get(url, params=params, headers=SEC_HEADERS)
    resp.raise_for_status()
    time.sleep(0.11)  # SEC rate limit: 10 requests/second
    return resp.json().get("hits", {}).get("hits", [])

# Option B: Financial Modeling Prep API (simpler, structured)
FMP_BASE = "https://financialmodelingprep.com/api/v3"

def fetch_fmp_transcript(symbol: str, year: int, quarter: int, api_key: str) -> str | None:
    """Fetch a single earnings call transcript from FMP."""
    url = f"{FMP_BASE}/earning_call_transcript/{symbol}"
    params = {"year": year, "quarter": quarter, "apikey": api_key}
    resp = requests.get(url, params=params)
    if resp.status_code == 200 and resp.json():
        return resp.json()[0].get("content", "")
    return None

# Fetch all transcripts
COMPANIES = ["NVDA", "MSFT", "GOOG", "AMZN", "META", "AAPL",
             "AMD", "AVGO", "CRM", "ORCL", "ADBE", "SNOW", "PLTR",
             "INTC", "IBM", "NOW", "PANW", "CRWD", "MDB", "AI"]
QUARTERS = [(y, q) for y in range(2023, 2026) for q in range(1, 5)]

transcripts = []
for symbol in COMPANIES:
    for year, quarter in QUARTERS:
        text = fetch_fmp_transcript(symbol, year, quarter, FMP_API_KEY)
        if text:
            transcripts.append({
                "symbol": symbol,
                "year": year,
                "quarter": quarter,
                "quarter_label": f"Q{quarter} {year}",
                "transcript_text": text,
                "char_count": len(text),
                "word_count": len(text.split()),
            })
            # Cache to disk
            path = TRANSCRIPT_DIR / f"{symbol}_{year}_Q{quarter}.txt"
            path.write_text(text)
        time.sleep(0.5)  # Rate limiting
```

### 2.3 OpenAI API Integration

**Model:** `gpt-4o-mini`
**Reason:** Cost-efficient for structured extraction tasks. At $0.15/1M input tokens and $0.60/1M output tokens (as of early 2026), processing ~240 transcripts is affordable.

#### 2.3.1 Prompt Template: AI Mention Extraction + Scoring

```python
SYSTEM_PROMPT = """You are a financial analyst specializing in AI/technology sector analysis.
You will analyze earnings call transcripts to measure the degree of AI-related hype.

You MUST respond with ONLY a valid JSON object matching the schema below.
No markdown, no explanation, no preamble -- just the JSON.

Schema:
{
  "ai_mention_count": <int>,       // Count of AI-related terms (see list below)
  "hype_score": <int 1-10>,        // 1=AI barely mentioned, 10=entire call is AI-focused transformation narrative
  "revenue_specificity": <int 1-10>, // 1=vague ("AI will drive growth"), 10=specific ("$X billion from AI products")
  "sentiment": "<string>",          // "bullish", "neutral", or "bearish" regarding AI's impact on the company
  "key_ai_claims": [<string>, ...], // Up to 5 most significant AI-related claims (direct quotes or paraphrases)
  "revenue_figures_mentioned": [<string>, ...], // Any specific AI revenue numbers mentioned
  "ai_products_named": [<string>, ...]  // Specific AI products or services named
}

AI-related terms to count (case-insensitive):
"artificial intelligence", "AI" (standalone, not part of other words like "said"),
"machine learning", "ML", "deep learning", "generative AI", "GenAI", "gen AI",
"large language model", "LLM", "GPT", "neural network", "natural language processing", "NLP",
"computer vision", "AI infrastructure", "AI accelerator", "AI training", "AI inference",
"foundation model", "transformer", "copilot", "AI agent", "agentic"

Scoring guide for hype_score:
1-2: AI mentioned in passing, not a strategic focus
3-4: AI discussed as one of several growth areas
5-6: AI positioned as a major strategic pillar
7-8: AI framed as transformative for the entire business
9-10: AI positioned as a once-in-a-generation opportunity, company pivoting entirely

Scoring guide for revenue_specificity:
1-2: No specific numbers; vague claims only ("AI will be big")
3-4: General sizing ("billions in AI opportunity") without company-specific figures
5-6: Directional company figures ("AI revenue doubled year-over-year")
7-8: Specific figures ("AI-related revenue reached $X billion this quarter")
9-10: Detailed breakdown (AI revenue by segment, growth rates, margins)"""

USER_PROMPT_TEMPLATE = """Analyze this earnings call transcript for {company} ({quarter}):

--- TRANSCRIPT START ---
{transcript_text}
--- TRANSCRIPT END ---

Respond with ONLY the JSON object as specified."""
```

#### 2.3.2 OpenAI API Call Implementation

```python
import openai
import json
from tenacity import retry, stop_after_attempt, wait_exponential

client = openai.OpenAI()  # Uses OPENAI_API_KEY from env

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def analyze_transcript(
    company: str,
    quarter: str,
    transcript_text: str,
    max_input_chars: int = 60_000,  # ~15K tokens; most transcripts fit
) -> dict:
    """Send transcript to GPT-4o-mini for AI hype analysis."""
    # Truncate if too long (preserve beginning + end, which have summary/guidance)
    if len(transcript_text) > max_input_chars:
        half = max_input_chars // 2
        transcript_text = (
            transcript_text[:half]
            + "\n\n[... MIDDLE SECTION TRUNCATED FOR LENGTH ...]\n\n"
            + transcript_text[-half:]
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                company=company,
                quarter=quarter,
                transcript_text=transcript_text,
            )},
        ],
        temperature=0.1,          # Low temperature for consistent scoring
        max_tokens=1000,          # Structured output is compact
        response_format={"type": "json_object"},  # Force JSON output
    )

    raw_output = response.choices[0].message.content
    parsed = json.loads(raw_output)

    # Validate required fields
    required_fields = ["ai_mention_count", "hype_score", "revenue_specificity", "sentiment"]
    for field in required_fields:
        if field not in parsed:
            raise ValueError(f"Missing required field: {field}")

    # Attach metadata
    parsed["model"] = response.model
    parsed["input_tokens"] = response.usage.prompt_tokens
    parsed["output_tokens"] = response.usage.completion_tokens
    parsed["total_tokens"] = response.usage.total_tokens

    return parsed
```

#### 2.3.3 Batching Strategy

```python
import pandas as pd
from tqdm import tqdm
import time

def batch_analyze_transcripts(transcripts: list[dict]) -> pd.DataFrame:
    """Analyze all transcripts with rate limiting and progress tracking."""
    results = []
    total_input_tokens = 0
    total_output_tokens = 0

    for t in tqdm(transcripts, desc="Analyzing transcripts"):
        try:
            analysis = analyze_transcript(
                company=t["symbol"],
                quarter=t["quarter_label"],
                transcript_text=t["transcript_text"],
            )
            results.append({
                "symbol": t["symbol"],
                "year": t["year"],
                "quarter": t["quarter"],
                "quarter_label": t["quarter_label"],
                "word_count": t["word_count"],
                **analysis,  # Unpacks all analysis fields
            })
            total_input_tokens += analysis.get("input_tokens", 0)
            total_output_tokens += analysis.get("output_tokens", 0)
        except Exception as e:
            results.append({
                "symbol": t["symbol"],
                "year": t["year"],
                "quarter": t["quarter"],
                "quarter_label": t["quarter_label"],
                "error": str(e),
            })

        # Rate limiting: gpt-4o-mini allows 30K RPM / 150M TPM
        # At ~15K tokens per call, we are well within limits
        # But be polite: 0.5s between calls
        time.sleep(0.5)

    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Estimated cost: ${total_input_tokens * 0.15 / 1_000_000 + total_output_tokens * 0.60 / 1_000_000:.2f}")

    return pd.DataFrame(results)
```

#### 2.3.4 Cost Estimate

| Parameter | Value | Notes |
|-----------|-------|-------|
| Average transcript length | ~8,000 words (~10,000 tokens) | Typical earnings call |
| Max transcript length (after truncation) | ~15,000 tokens | truncated at 60K chars |
| Output tokens per analysis | ~300-500 tokens | Structured JSON |
| Number of transcripts | ~240 | 20 companies x 12 quarters |
| Input cost | ~240 x 12,000 tokens x $0.15/1M = **$0.43** | Very cheap |
| Output cost | ~240 x 400 tokens x $0.60/1M = **$0.06** | Negligible |
| **Total estimated cost** | **~$0.50** | Well within budget |
| Buffer (retries, prompt iteration) | 3x | **~$1.50 max** |

### 2.4 Output Schema: Earnings Call Analysis

| Column | dtype | Description |
|--------|-------|-------------|
| `symbol` | `str` | Ticker symbol |
| `year` | `int` | Fiscal year |
| `quarter` | `int` | Fiscal quarter (1-4) |
| `quarter_label` | `str` | e.g., "Q1 2023" |
| `word_count` | `int` | Transcript word count |
| `ai_mention_count` | `int` | Number of AI-related term occurrences |
| `ai_mentions_per_1k_words` | `float64` | Normalized: `ai_mention_count / (word_count / 1000)` |
| `hype_score` | `int` | 1-10 hype intensity |
| `revenue_specificity` | `int` | 1-10 specificity of AI revenue claims |
| `sentiment` | `str` | "bullish", "neutral", or "bearish" |
| `key_ai_claims` | `list[str]` | Top 5 AI claims from transcript |
| `revenue_figures_mentioned` | `list[str]` | Specific $ figures cited |
| `ai_products_named` | `list[str]` | Named AI products/services |
| `input_tokens` | `int` | OpenAI API input tokens used |
| `output_tokens` | `int` | OpenAI API output tokens used |

### 2.5 Historical Dot-Com Comparison (Stretch Goal)

For dot-com era comparison, we cannot use the same real-time approach. Instead:

**Method:** Search SEC EDGAR for 10-K/10-Q filings from 1998-2001 containing "internet", "e-commerce", "World Wide Web", "dot-com" from companies like CSCO, MSFT, INTC, DELL, SUNW, JNPR.

```python
# pseudocode: search EDGAR for dot-com era filings
DOTCOM_TERMS = '"internet" OR "e-commerce" OR "World Wide Web" OR "electronic commerce"'
DOTCOM_COMPANIES_CIK = {
    "CSCO": "858877", "MSFT": "789019", "INTC": "50863",
    "DELL": "826083", "SUNW": "709519", "JNPR": "1043604",
}

def fetch_dotcom_filings(cik: str) -> list[dict]:
    url = f"https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": DOTCOM_TERMS,
        "dateRange": "custom",
        "startdt": "1998-01-01",
        "enddt": "2001-12-31",
        "entityName": cik,
        "forms": "10-K,10-Q",
    }
    resp = requests.get(url, params=params, headers=SEC_HEADERS)
    time.sleep(0.11)
    return resp.json().get("hits", {}).get("hits", [])
```

**Caveat:** Dot-com era filings are formal SEC documents, not conversational earnings calls. The tone and language differ significantly. This comparison is illustrative, not statistically rigorous.

---

## 3. Sub-layer 5B: Google Trends (pytrends)

### 3.1 Setup

```python
from pytrends.request import TrendReq

# Initialize pytrends with rate limiting
pytrends = TrendReq(
    hl="en-US",
    tz=360,            # US Eastern offset (minutes from UTC)
    timeout=(10, 25),  # connect, read timeouts
    retries=3,
    backoff_factor=1.0,
)
```

### 3.2 Search Terms

**AI Era terms:**

| Term | Category | Rationale |
|------|----------|-----------|
| `NVIDIA stock` | Stock interest | Direct retail interest in NVDA |
| `AI stocks` | Sector interest | Broad AI investment interest |
| `artificial intelligence investing` | Investment hype | Explicit investment intent |
| `AI bubble` | Bubble awareness | Contrarian/fear indicator |
| `buy NVIDIA` | Purchase intent | Strongest retail signal |
| `is AI a bubble` | Bubble questioning | Peak-of-cycle indicator historically |

**Dot-Com comparison terms (available only from 2004):**

| Term | Category | Rationale |
|------|----------|-----------|
| `Cisco stock` | Stock interest | Direct CSCO analog |
| `dot com stocks` | Sector interest | Broad dot-com interest |
| `internet stocks` | Sector interest | Alternative phrasing |
| `tech bubble` | Bubble awareness | Historical reference |

**Critical limitation:** Google Trends data starts January 2004. The dot-com bubble peaked in March 2000. We CANNOT get direct Google Trends data for the dot-com era. We can only see the tail end (2004 onward, which is post-crash recovery).

### 3.3 Data Acquisition

```python
import time
import pandas as pd

def fetch_google_trends(
    keywords: list[str],
    timeframe: str = "2022-01-01 2026-03-28",
    geo: str = "US",
    category: int = 0,  # 0 = All categories; 7 = Finance
) -> pd.DataFrame:
    """Fetch Google Trends data for a list of keywords.

    Timeframe formats:
    - "2022-01-01 2026-03-28"  -> weekly data
    - "today 5-y"              -> weekly data for last 5 years
    - "today 12-m"             -> weekly data for last 12 months
    - "all"                    -> monthly data from 2004 to present

    Returns DataFrame with columns: date, {keyword1}, {keyword2}, ...
    Values are 0-100 relative scale (100 = peak interest in the timeframe).
    """
    # pytrends allows max 5 keywords per request
    # Split into batches if needed
    all_results = []

    for i in range(0, len(keywords), 5):
        batch = keywords[i:i+5]
        pytrends.build_payload(
            kw_list=batch,
            cat=category,
            timeframe=timeframe,
            geo=geo,
            gprop="",  # web search (not youtube, news, etc.)
        )
        df = pytrends.interest_over_time()
        if "isPartial" in df.columns:
            df = df.drop("isPartial", axis=1)
        all_results.append(df)
        time.sleep(2)  # Rate limiting for Google

    if not all_results:
        return pd.DataFrame()

    # Merge batches on date index
    combined = all_results[0]
    for df in all_results[1:]:
        combined = combined.join(df, how="outer")

    combined = combined.reset_index()
    combined = combined.rename(columns={"date": "date"})
    return combined


# Fetch AI era data (weekly granularity)
ai_trends = fetch_google_trends(
    keywords=["NVIDIA stock", "AI stocks", "artificial intelligence investing",
              "AI bubble", "buy NVIDIA"],
    timeframe="2022-01-01 2026-03-28",
    geo="US",
)

# Fetch long-term data (monthly granularity, 2004-present)
long_trends = fetch_google_trends(
    keywords=["NVIDIA stock", "Cisco stock", "AI stocks",
              "tech bubble", "internet stocks"],
    timeframe="all",
    geo="US",
)

# Fetch "AI bubble" search interest specifically (finance category)
bubble_awareness = fetch_google_trends(
    keywords=["AI bubble", "is AI a bubble", "tech bubble", "dot com bubble"],
    timeframe="2022-01-01 2026-03-28",
    geo="US",
    category=7,  # Finance category
)
```

### 3.4 Normalization Notes

Google Trends returns values on a **0-100 relative scale** where 100 = the peak interest point *within the queried timeframe*. This means:

1. **Comparing across different timeframe queries is invalid** unless normalized against a common anchor.
2. **Strategy:** Use the "all" timeframe for cross-era comparisons (even though it is monthly, not weekly).
3. **For AI-era detail:** Use the 2022-2026 timeframe for weekly granularity.

### 3.5 Workaround for Pre-2004 Gap

Since we cannot get Google Trends for 1998-2003, we use alternative proxies:

**Option 1: Media mention counts (newspaper archives)**
- Use the `newspapers.com` or `ProQuest` APIs to count mentions of "internet stocks," "dot-com," "Cisco" in major publications (WSJ, NYT) by month from 1998-2001.
- Fallback: manually compile from available free archives.

**Option 2: Google Ngram Viewer**
- Google Books Ngram Viewer tracks word frequency in published books. Not real-time, but captures cultural zeitgeist.
- API: `https://books.google.com/ngrams/json?content=artificial+intelligence,internet&year_start=1995&year_end=2025&corpus=en-2019`
- Limitation: book publishing has a lag; not useful for real-time hype measurement.

**Option 3: Academic citation counts**
- Use Semantic Scholar API to count papers mentioning "artificial intelligence" vs "internet" by year.
- This measures technical hype, not investor hype.

**Recommended approach:** Acknowledge the gap transparently. Show the 2004-present Google Trends data and note that for the dot-com era, we rely on qualitative historical accounts and media analysis rather than quantitative search data.

### 3.6 Google Trends Output Schema

| Column | dtype | Description |
|--------|-------|-------------|
| `date` | `datetime64[ns]` | Week start date (Sunday) or month start |
| `keyword` | `str` | Search term |
| `interest` | `int` | 0-100 relative interest score |
| `timeframe` | `str` | Query timeframe (for provenance) |
| `geo` | `str` | Geographic filter used |

---

## 4. Sub-layer 5C: Reddit Sentiment (praw + OpenAI)

### 4.1 Setup

```python
import praw
import os

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="2Kim Datathon 2026 v1.0 (by u/datathon2kim)",
)
```

### 4.2 Target Subreddits and Search Strategy

| Subreddit | Focus | Relevance |
|-----------|-------|-----------|
| `r/wallstreetbets` | Retail speculation, memes, YOLO trades | Highest signal for speculative euphoria |
| `r/stocks` | General stock discussion, more measured | Moderate retail sentiment |
| `r/investing` | Long-term investing | Conservative baseline |
| `r/nvidia` | NVDA-specific community | Company-specific sentiment |

**Search query:** Posts containing "NVDA" OR "NVIDIA" OR "nvidia"

**Time range:** 2022-01-01 to 2026-03-28 (aligns with AI era)

### 4.3 Data Collection

```python
import datetime
import time

def fetch_reddit_posts(
    subreddit_name: str,
    query: str,
    start_date: datetime.date,
    end_date: datetime.date,
    limit: int = 1000,
) -> list[dict]:
    """Fetch Reddit posts matching query from a subreddit.

    Note: Reddit's search API is limited. For comprehensive historical data,
    consider using Pushshift API (https://api.pushshift.io/) as a fallback,
    though its availability varies.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    # Reddit search doesn't support date ranges natively
    # Strategy: search by relevance, fetch up to limit, filter by date locally
    for submission in subreddit.search(
        query=query,
        sort="new",           # Sort by newest first
        time_filter="all",    # All time
        limit=limit,
    ):
        post_date = datetime.date.fromtimestamp(submission.created_utc)
        if start_date <= post_date <= end_date:
            posts.append({
                "subreddit": subreddit_name,
                "post_id": submission.id,
                "title": submission.title,
                "selftext": submission.selftext[:2000] if submission.selftext else "",
                "score": submission.score,
                "upvote_ratio": submission.upvote_ratio,
                "num_comments": submission.num_comments,
                "created_utc": submission.created_utc,
                "date": post_date.isoformat(),
                "url": submission.url,
                "flair": submission.link_flair_text,
                "author": str(submission.author) if submission.author else "[deleted]",
            })
        time.sleep(0.1)  # Rate limiting

    return posts


# Fetch from all target subreddits
all_posts = []
for sub in ["wallstreetbets", "stocks", "investing", "nvidia"]:
    posts = fetch_reddit_posts(
        subreddit_name=sub,
        query="NVDA OR NVIDIA",
        start_date=datetime.date(2022, 1, 1),
        end_date=datetime.date(2026, 3, 28),
        limit=1000,
    )
    all_posts.extend(posts)
    print(f"r/{sub}: {len(posts)} posts")
    time.sleep(2)  # Between subreddits

reddit_df = pd.DataFrame(all_posts)
reddit_df.to_parquet("data/raw/reddit_nvda_posts.parquet")
```

### 4.4 Reddit Sentiment Analysis (OpenAI API)

```python
REDDIT_SENTIMENT_PROMPT = """You are analyzing a Reddit post about NVIDIA (NVDA) stock.
Classify the post's sentiment and investment stance.

Respond with ONLY a valid JSON object:
{
  "sentiment": "<string>",       // "very_bullish", "bullish", "neutral", "bearish", "very_bearish"
  "sentiment_score": <float>,    // -1.0 (most bearish) to +1.0 (most bullish)
  "is_speculative": <bool>,      // true if post discusses YOLO, options gambling, moon, etc.
  "mentions_bubble": <bool>,     // true if post discusses AI bubble / overvaluation
  "mentions_fundamentals": <bool>, // true if post discusses earnings, revenue, P/E
  "key_theme": "<string>"        // One of: "buy_signal", "sell_signal", "earnings_reaction",
                                 //         "technical_analysis", "hype", "fear", "dd", "meme", "other"
}

Subreddit context: r/{subreddit} (culture: {subreddit_culture})"""

SUBREDDIT_CULTURES = {
    "wallstreetbets": "Aggressive retail trading, memes, YOLO mentality, diamond hands vs paper hands",
    "stocks": "Moderate retail investing, balanced discussion, some DD",
    "investing": "Conservative long-term investing, Bogleheads influence",
    "nvidia": "Company enthusiast community, product-focused, some investing discussion",
}

def analyze_reddit_post(post: dict) -> dict:
    """Classify a single Reddit post using GPT-4o-mini."""
    text = f"Title: {post['title']}\n\nBody: {post['selftext'][:1500]}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": REDDIT_SENTIMENT_PROMPT.format(
                subreddit=post["subreddit"],
                subreddit_culture=SUBREDDIT_CULTURES.get(post["subreddit"], "General"),
            )},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
        max_tokens=300,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    result["post_id"] = post["post_id"]
    result["input_tokens"] = response.usage.prompt_tokens
    result["output_tokens"] = response.usage.completion_tokens
    return result


def batch_analyze_reddit(posts_df: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    """Analyze a sample of Reddit posts. Full dataset may be too expensive."""
    # Stratified sample: proportional by month
    posts_df["month"] = pd.to_datetime(posts_df["date"]).dt.to_period("M")
    sampled = posts_df.groupby("month", group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(1, sample_size // posts_df["month"].nunique())),
                           random_state=42)
    )

    results = []
    for _, post in tqdm(sampled.iterrows(), total=len(sampled), desc="Reddit sentiment"):
        try:
            result = analyze_reddit_post(post.to_dict())
            results.append(result)
        except Exception as e:
            results.append({"post_id": post["post_id"], "error": str(e)})
        time.sleep(0.3)

    return sampled.merge(pd.DataFrame(results), on="post_id", how="left")
```

### 4.5 Reddit Cost Estimate

| Parameter | Value |
|-----------|-------|
| Posts to analyze | ~500 (sampled from full dataset) |
| Average input tokens per post | ~500 (title + truncated body + system prompt) |
| Output tokens per post | ~150 |
| Input cost | 500 x 500 x $0.15/1M = **$0.04** |
| Output cost | 500 x 150 x $0.60/1M = **$0.05** |
| **Total Reddit analysis cost** | **~$0.09** |

### 4.6 Reddit Output Schema

| Column | dtype | Description |
|--------|-------|-------------|
| `post_id` | `str` | Reddit post ID |
| `subreddit` | `str` | Source subreddit |
| `title` | `str` | Post title |
| `date` | `str` | Post date (ISO format) |
| `score` | `int` | Reddit score (upvotes - downvotes) |
| `num_comments` | `int` | Comment count |
| `sentiment` | `str` | very_bullish / bullish / neutral / bearish / very_bearish |
| `sentiment_score` | `float64` | -1.0 to +1.0 |
| `is_speculative` | `bool` | YOLO / options gambling indicator |
| `mentions_bubble` | `bool` | Bubble awareness indicator |
| `mentions_fundamentals` | `bool` | Fundamental analysis indicator |
| `key_theme` | `str` | Dominant theme classification |

### 4.7 Weekly Aggregation

```python
def aggregate_reddit_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Reddit sentiment to weekly level for time series analysis."""
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W").dt.start_time

    weekly = df.groupby("week").agg(
        post_count=("post_id", "count"),
        avg_sentiment=("sentiment_score", "mean"),
        median_sentiment=("sentiment_score", "median"),
        sentiment_std=("sentiment_score", "std"),
        bullish_pct=("sentiment", lambda x: (x.isin(["bullish", "very_bullish"])).mean()),
        bearish_pct=("sentiment", lambda x: (x.isin(["bearish", "very_bearish"])).mean()),
        speculative_pct=("is_speculative", "mean"),
        bubble_mention_pct=("mentions_bubble", "mean"),
        avg_score=("score", "mean"),
        avg_comments=("num_comments", "mean"),
    ).reset_index()

    return weekly
```

---

## 5. Visualizations

### Chart 5.1: AI Mentions in Earnings Calls Over Time (Hockey Stick Chart)

**Type:** Bar chart with overlaid line
**X-axis:** `quarter_label` (Q1 2023 through Q4 2025), label: "Quarter"
**Y-axis (bars):** Average `ai_mentions_per_1k_words` across all companies, label: "AI Mentions per 1K Words"
**Y-axis (line, secondary):** Average `hype_score`, label: "Average Hype Score (1-10)"

**Bar details:**
- Grouped bars by quarter, one bar per company tier (Tier 1 vs Tier 2 vs Tier 3)
- Colors: Tier 1 = `#d62728` (red), Tier 2 = `#ff7f0e` (orange), Tier 3 = `#2ca02c` (green)
- Alternative: single bar per quarter showing the cross-company average, color gradient from light to dark red as values increase

**Line overlay:**
- Average hype score per quarter, color black, linewidth 2, marker style = circle
- Plotted on secondary Y-axis (right side), range 1-10

**Annotations:**
- Arrow at Q1 2023: "ChatGPT launches (Nov 2022)"
- Arrow at the quarter where hype_score first exceeds 7: "Peak hype threshold"
- Text box with NVDA's hype score trajectory specifically highlighted

**Title:** "AI Mentions in Earnings Calls: The Hockey Stick"
**Figure size:** 14 x 7 inches / 1000 x 550 px

```python
# pseudocode
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_chart_5_1(earnings_df: pd.DataFrame) -> go.Figure:
    quarterly_avg = earnings_df.groupby("quarter_label").agg(
        avg_mentions=("ai_mentions_per_1k_words", "mean"),
        avg_hype=("hype_score", "mean"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bars: AI mention density
    fig.add_trace(go.Bar(
        x=quarterly_avg["quarter_label"],
        y=quarterly_avg["avg_mentions"],
        name="AI Mentions per 1K Words",
        marker_color="#d62728",
        opacity=0.7,
    ), secondary_y=False)

    # Line: Hype score
    fig.add_trace(go.Scatter(
        x=quarterly_avg["quarter_label"],
        y=quarterly_avg["avg_hype"],
        name="Average Hype Score",
        mode="lines+markers",
        line=dict(color="black", width=2),
        marker=dict(size=8),
    ), secondary_y=True)

    fig.update_layout(
        title="AI Mentions in Earnings Calls: The Hockey Stick",
        xaxis_title="Quarter",
        template="plotly_white",
        width=1000, height=550,
    )
    fig.update_yaxes(title_text="AI Mentions per 1K Words", secondary_y=False)
    fig.update_yaxes(title_text="Average Hype Score (1-10)", secondary_y=True, range=[0, 10])
    return fig
```

### Chart 5.2: Hype Score vs Revenue Specificity Scatter

**Type:** Scatter plot with quadrant labels
**X-axis:** `hype_score` (1-10), label: "Hype Score (AI Positioning Intensity)"
**Y-axis:** `revenue_specificity` (1-10), label: "Revenue Specificity (Concreteness of AI Revenue Claims)"

**Points:**
- One point per company-quarter observation
- Size: proportional to `ai_mention_count` (more mentions = bigger dot)
- Color: by `quarter_label` using a sequential colorscale (`Viridis` or `RdYlGn_r`)
- Hover text: `{symbol} {quarter_label}: Hype={hype_score}, Rev={revenue_specificity}`

**Quadrant labels (text annotations):**
- Top-right (high hype, high specificity): "Backed by Numbers" -- color green
- Top-left (low hype, high specificity): "Quiet Executors" -- color blue
- Bottom-right (high hype, low specificity): "ALL HYPE, NO NUMBERS" -- color red, bold
- Bottom-left (low hype, low specificity): "Not Playing the AI Game" -- color gray

**Quadrant lines:**
- Vertical dashed line at x=5: "Hype Threshold"
- Horizontal dashed line at y=5: "Specificity Threshold"

**Key companies highlighted:**
- NVDA: star marker, labeled
- MSFT, GOOG, AMZN: diamond markers, labeled

**Title:** "Are Companies Backing Up AI Hype with Revenue Numbers?"
**Figure size:** 10 x 10 inches / 800 x 800 px

```python
def create_chart_5_2(earnings_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Color by time (quarter as numeric)
    earnings_df["quarter_num"] = (
        (earnings_df["year"] - 2023) * 4 + earnings_df["quarter"]
    )

    fig.add_trace(go.Scatter(
        x=earnings_df["hype_score"],
        y=earnings_df["revenue_specificity"],
        mode="markers",
        marker=dict(
            size=earnings_df["ai_mention_count"].clip(5, 50),
            color=earnings_df["quarter_num"],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Quarter<br>(early→late)"),
            opacity=0.7,
            line=dict(width=0.5, color="white"),
        ),
        text=earnings_df.apply(
            lambda r: f"{r['symbol']} {r['quarter_label']}", axis=1
        ),
        hovertemplate="%{text}<br>Hype: %{x}<br>Specificity: %{y}<extra></extra>",
    ))

    # Quadrant lines
    fig.add_vline(x=5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)

    # Quadrant labels
    fig.add_annotation(x=8, y=8, text="Backed by<br>Numbers", showarrow=False,
                       font=dict(color="green", size=14, family="Arial Black"))
    fig.add_annotation(x=2, y=8, text="Quiet<br>Executors", showarrow=False,
                       font=dict(color="blue", size=14))
    fig.add_annotation(x=8, y=2, text="ALL HYPE<br>NO NUMBERS", showarrow=False,
                       font=dict(color="red", size=16, family="Arial Black"))
    fig.add_annotation(x=2, y=2, text="Not in the<br>AI Game", showarrow=False,
                       font=dict(color="gray", size=12))

    fig.update_layout(
        title="Are Companies Backing Up AI Hype with Revenue Numbers?",
        xaxis_title="Hype Score (AI Positioning Intensity)",
        yaxis_title="Revenue Specificity (Concreteness of Claims)",
        xaxis=dict(range=[0.5, 10.5]), yaxis=dict(range=[0.5, 10.5]),
        template="plotly_white", width=800, height=800,
    )
    return fig
```

### Chart 5.3: Google Trends -- "AI stocks" vs Historical Comparison

**Type:** Dual-panel line chart
**Panel A (top):** Long-term view (2004-2026, monthly)
**Panel B (bottom):** AI era zoom (2022-2026, weekly)

**Panel A traces:**
- "NVIDIA stock" search interest: red line, linewidth 2
- "Cisco stock" search interest: blue line, linewidth 2
- "AI stocks" search interest: orange dashed line, linewidth 1.5
- "tech bubble" search interest: gray dotted line, linewidth 1

**Panel A annotations:**
- Shaded region 2022-2026: light yellow, label "AI Era"
- Arrow at NVDA search interest spike: "AI frenzy begins"
- Note: "Google Trends data unavailable before 2004 -- dot-com peak occurred March 2000"

**Panel B traces:**
- "NVIDIA stock": red, linewidth 2
- "AI stocks": orange, linewidth 2
- "AI bubble": dark red dashed, linewidth 1.5
- "buy NVIDIA": green dotted, linewidth 1

**Panel B Y-axis:** 0-100, label: "Google Trends Interest (0-100 relative)"

**Title:** "Google Search Interest: AI Stocks vs Historical Tech Bubbles"
**Figure size:** 14 x 10 inches / 1100 x 800 px

```python
def create_chart_5_3(long_trends: pd.DataFrame, ai_trends: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["Long-Term View (2004-2026)",
                                        "AI Era Detail (2022-2026)"],
                        vertical_spacing=0.12)

    # Panel A: Long-term
    for col, color, dash in [
        ("NVIDIA stock", "#d62728", "solid"),
        ("Cisco stock", "#1f77b4", "solid"),
        ("AI stocks", "#ff7f0e", "dash"),
        ("tech bubble", "gray", "dot"),
    ]:
        if col in long_trends.columns:
            fig.add_trace(go.Scatter(
                x=long_trends["date"], y=long_trends[col],
                mode="lines", name=col,
                line=dict(color=color, dash=dash, width=2 if dash=="solid" else 1.5),
            ), row=1, col=1)

    # Panel B: AI era detail
    for col, color, dash in [
        ("NVIDIA stock", "#d62728", "solid"),
        ("AI stocks", "#ff7f0e", "solid"),
        ("AI bubble", "#8B0000", "dash"),
        ("buy NVIDIA", "#2ca02c", "dot"),
    ]:
        if col in ai_trends.columns:
            fig.add_trace(go.Scatter(
                x=ai_trends["date"], y=ai_trends[col],
                mode="lines", name=col, showlegend=False,
                line=dict(color=color, dash=dash, width=2 if dash=="solid" else 1.5),
            ), row=2, col=1)

    fig.update_layout(
        title="Google Search Interest: AI Stocks vs Historical Tech Bubbles",
        template="plotly_white", width=1100, height=800,
    )
    fig.update_yaxes(title_text="Interest (0-100)", row=1, col=1)
    fig.update_yaxes(title_text="Interest (0-100)", row=2, col=1)
    return fig
```

### Chart 5.4: Earnings Call Sentiment Timeline with Stock Price Overlay

**Type:** Dual-axis line chart
**X-axis:** `quarter_label`, label: "Quarter"
**Y-axis (left):** Proportion of bullish/neutral/bearish sentiment, label: "Sentiment Distribution"
**Y-axis (right):** NVDA stock price (quarterly close), label: "NVDA Price ($)"

**Traces:**
- Stacked area chart for sentiment distribution:
  - Bullish: green fill, alpha 0.5
  - Neutral: gray fill, alpha 0.3
  - Bearish: red fill, alpha 0.5
- NVDA price line: black, linewidth 2.5, marker = diamond, on secondary Y-axis

**Title:** "Earnings Call Sentiment vs NVDA Price"
**Figure size:** 14 x 7 inches / 1000 x 550 px

```python
def create_chart_5_4(earnings_df: pd.DataFrame, nvda_price: pd.Series) -> go.Figure:
    # Compute sentiment distribution per quarter
    sent_dist = earnings_df.groupby("quarter_label")["sentiment"].value_counts(
        normalize=True
    ).unstack(fill_value=0).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for sent, color in [("bullish", "#2ca02c"), ("neutral", "#aaaaaa"), ("bearish", "#d62728")]:
        if sent in sent_dist.columns:
            fig.add_trace(go.Scatter(
                x=sent_dist["quarter_label"], y=sent_dist[sent],
                fill="tonexty", name=sent.capitalize(),
                line=dict(color=color), fillcolor=color,
                opacity=0.5, stackgroup="sentiment",
            ), secondary_y=False)

    # NVDA price overlay
    fig.add_trace(go.Scatter(
        x=nvda_price.index, y=nvda_price.values,
        name="NVDA Price", mode="lines+markers",
        line=dict(color="black", width=2.5),
        marker=dict(symbol="diamond", size=8),
    ), secondary_y=True)

    fig.update_layout(
        title="Earnings Call Sentiment vs NVDA Price",
        template="plotly_white", width=1000, height=550,
    )
    fig.update_yaxes(title_text="Sentiment Proportion", secondary_y=False)
    fig.update_yaxes(title_text="NVDA Price ($)", secondary_y=True)
    return fig
```

### Chart 5.5: Reddit NVDA Mention Frequency vs NVDA Volume

**Type:** Dual-axis bar + line chart
**X-axis:** Week (date), label: "Week"
**Y-axis (left, bars):** Reddit NVDA post count per week, label: "Reddit Posts (count)"
**Y-axis (right, line):** NVDA weekly average trading volume (in millions), label: "NVDA Volume (M shares)"

**Bar details:**
- Color: gradient from light blue (low sentiment) to dark red (high bullish sentiment)
- Colorscale based on `avg_sentiment` of that week's posts

**Line details:**
- NVDA weekly volume: black line, linewidth 1.5, area fill to zero with alpha 0.1

**Annotations:**
- Arrows at earnings announcement weeks: "NVDA Earnings"
- Arrows at major NVDA news events (stock splits, product launches)

**Title:** "Reddit NVDA Buzz vs Trading Volume"
**Figure size:** 14 x 6 inches / 1100 x 500 px

```python
def create_chart_5_5(reddit_weekly: pd.DataFrame, nvda_volume: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=reddit_weekly["week"],
        y=reddit_weekly["post_count"],
        name="Reddit NVDA Posts",
        marker=dict(
            color=reddit_weekly["avg_sentiment"],
            colorscale="RdYlGn",
            cmin=-1, cmax=1,
            colorbar=dict(title="Sentiment", x=1.1),
        ),
        opacity=0.8,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=nvda_volume["week"],
        y=nvda_volume["volume"] / 1e6,
        name="NVDA Volume (M)",
        mode="lines",
        line=dict(color="black", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,0,0,0.05)",
    ), secondary_y=True)

    fig.update_layout(
        title="Reddit NVDA Buzz vs Trading Volume",
        template="plotly_white", width=1100, height=500,
    )
    fig.update_yaxes(title_text="Reddit Posts (count)", secondary_y=False)
    fig.update_yaxes(title_text="NVDA Volume (M shares)", secondary_y=True)
    return fig
```

### Chart 5.6: Word Cloud Comparison (Earnings Call AI Terms by Year)

**Type:** 2x2 grid of word clouds (one per year: 2023, 2024, 2025, 2026 partial)
**Library:** `wordcloud` (Python) for generation, then embed as images in Plotly

**Word source:** Extract all AI-related phrases from `key_ai_claims` field across all transcripts, grouped by year.

**Color scheme:**
- 2023: Blues (`#1f77b4` family) -- early AI narrative
- 2024: Oranges (`#ff7f0e` family) -- peak hype
- 2025: Reds (`#d62728` family) -- maturation or continuation
- 2026 (partial): Purples (`#9467bd` family) -- current state

**Font:** Minimum size 10px, maximum size 80px, proportional to term frequency
**Background:** White
**Mask shape:** Rectangular (clean, professional)

**Title:** "Evolution of AI Language in Earnings Calls (2023-2025)"
**Figure size:** 16 x 12 inches

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

def create_chart_5_6(earnings_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Evolution of AI Language in Earnings Calls", fontsize=18, fontweight="bold")

    colors = {2023: "Blues", 2024: "Oranges", 2025: "Reds", 2026: "Purples"}

    for idx, year in enumerate([2023, 2024, 2025, 2026]):
        ax = axes[idx // 2][idx % 2]
        year_data = earnings_df[earnings_df["year"] == year]

        # Collect all AI claims and product names
        all_terms = []
        for _, row in year_data.iterrows():
            if isinstance(row.get("key_ai_claims"), list):
                all_terms.extend(row["key_ai_claims"])
            if isinstance(row.get("ai_products_named"), list):
                all_terms.extend(row["ai_products_named"])

        if not all_terms:
            ax.set_title(f"{year} (No Data)", fontsize=14)
            ax.axis("off")
            continue

        # Build frequency dict from terms
        text = " ".join(all_terms).lower()
        wc = WordCloud(
            width=800, height=600,
            background_color="white",
            colormap=colors.get(year, "viridis"),
            max_words=100,
            min_font_size=10,
            max_font_size=80,
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{year}", fontsize=16, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    return fig
```

---

## 6. Statistical Tests

### 6.1 Correlation: Hype Score vs Stock Price

**Test:** Pearson and Spearman correlation between quarterly `hype_score` (cross-company average) and NVDA/SP500 quarterly returns.

```python
from scipy import stats

def test_hype_price_correlation(earnings_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    """Test correlation between AI hype intensity and stock price movements."""
    # Aggregate hype score by quarter
    quarterly_hype = earnings_df.groupby("quarter_label").agg(
        avg_hype=("hype_score", "mean"),
        avg_specificity=("revenue_specificity", "mean"),
        avg_mentions=("ai_mentions_per_1k_words", "mean"),
    ).reset_index()

    # Merge with quarterly NVDA returns
    merged = quarterly_hype.merge(price_df, on="quarter_label")

    results = {}
    for metric in ["avg_hype", "avg_specificity", "avg_mentions"]:
        pearson_r, pearson_p = stats.pearsonr(merged[metric], merged["nvda_return"])
        spearman_r, spearman_p = stats.spearmanr(merged[metric], merged["nvda_return"])
        results[metric] = {
            "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_r": spearman_r, "spearman_p": spearman_p,
        }

    return results
```

**Expected findings:**
- Positive correlation between hype_score and NVDA price in 2023-2024 (hype fuels price)
- Correlation may weaken or reverse in 2025-2026 if the market starts demanding fundamentals

### 6.2 Granger Causality: Google Trends -> Price Moves

**Question:** Does Google search interest in "NVIDIA stock" lead or lag actual NVDA price moves?

```python
from statsmodels.tsa.stattools import grangercausalitytests

def test_trends_granger(trends_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    """Test if Google Trends interest Granger-causes NVDA price moves."""
    # Merge weekly trends with weekly NVDA returns
    merged = trends_df.merge(price_df, on="week", how="inner")

    # Ensure stationarity: use first-differences
    merged["trends_diff"] = merged["NVIDIA stock"].diff()
    merged["returns"] = merged["nvda_close"].pct_change()
    merged = merged.dropna()

    # Test: does trends_diff Granger-cause returns?
    data = merged[["returns", "trends_diff"]].values
    results = grangercausalitytests(data, maxlag=4, verbose=False)

    output = {}
    for lag in range(1, 5):
        output[f"lag_{lag}"] = {
            "f_stat": results[lag][0]["ssr_ftest"][0],
            "p_value": results[lag][0]["ssr_ftest"][1],
        }

    return output
```

**Interpretation:** If Google Trends Granger-causes price, retail interest is a leading indicator. If price Granger-causes Google Trends, retail is chasing momentum (more consistent with bubble behavior).

### 6.3 Regression: Stock Return ~ f(Hype, Mentions, Sentiment)

**Model:** OLS regression of quarterly NVDA returns on NLP-derived features.

```python
import statsmodels.api as sm

def run_hype_regression(merged_df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS: NVDA quarterly return ~ hype_score + ai_mentions + sentiment + revenue_specificity."""
    # Encode sentiment as numeric
    sentiment_map = {"very_bearish": -2, "bearish": -1, "neutral": 0, "bullish": 1, "very_bullish": 2}
    merged_df["sentiment_numeric"] = merged_df["avg_sentiment"].map(sentiment_map)

    X = merged_df[["avg_hype", "avg_mentions", "sentiment_numeric", "avg_specificity"]]
    X = sm.add_constant(X)
    y = merged_df["nvda_quarterly_return"]

    model = sm.OLS(y, X).fit(cov_type="HC1")  # Heteroscedasticity-robust SEs
    return model

# Report:
# print(model.summary())
# Key: R-squared, coefficient signs, significance of each predictor
```

**Caveat:** With only ~12 quarterly observations, this regression has very low power. It is illustrative, not definitive. State this clearly in the notebook.

### 6.4 Sentiment-Volume Correlation (Reddit)

```python
def test_reddit_volume_correlation(reddit_weekly: pd.DataFrame, nvda_volume: pd.DataFrame) -> dict:
    """Test correlation between Reddit NVDA post count and NVDA trading volume."""
    merged = reddit_weekly.merge(nvda_volume, on="week", how="inner")

    # Post count vs volume
    r_count, p_count = stats.pearsonr(merged["post_count"], merged["volume"])

    # Sentiment vs next-week returns
    merged["next_week_return"] = merged["nvda_close"].pct_change().shift(-1)
    r_sent, p_sent = stats.pearsonr(
        merged["avg_sentiment"].dropna(),
        merged["next_week_return"].dropna()
    )

    return {
        "post_count_volume": {"r": r_count, "p": p_count},
        "sentiment_next_week_return": {"r": r_sent, "p": p_sent},
    }
```

---

## 7. Key Narrative

The sentiment narrative to weave into the final presentation:

> "The hype is measurable and accelerating. Since ChatGPT's launch in November 2022, AI mentions in S&P 500 tech earnings calls have followed a hockey-stick trajectory. Every quarter, more companies position themselves as 'AI companies' -- but our NLP analysis reveals a critical divergence.
>
> Early in the cycle (2023), high hype scores accompanied LOW revenue specificity -- companies were waving the AI flag without concrete numbers. This is textbook bubble behavior: narrative outrunning fundamentals.
>
> The question for 2025-2026 is whether hype has converged with specificity. If companies like NVIDIA are now backing their AI claims with specific revenue figures (data center revenue, inference chip adoption), the hype may be transitioning from narrative to reality. But if revenue specificity remains low while hype scores keep climbing, the dot-com parallel becomes more alarming.
>
> Google search interest tells a complementary story: 'NVIDIA stock' search volume has eclipsed anything in Google Trends history. Crucially, 'AI bubble' searches have also surged -- suggesting awareness of bubble risk, which paradoxically may extend the bubble (everyone thinks they'll get out before the crash).
>
> Reddit sentiment adds the retail dimension: r/wallstreetbets NVDA mentions track closely with volume spikes, suggesting retail participation is material. When retail sentiment reaches extreme bullishness simultaneous with declining revenue specificity in earnings calls, that is the signal to watch."

---

## 8. Limitations

### 8.1 OpenAI API Bias in Sentiment Scoring

- GPT-4o-mini has its own biases in sentiment interpretation. It may systematically over-score or under-score hype relative to a human analyst.
- **Mitigation:** Manually validate a random sample of 20 transcripts against GPT's scores. Report inter-rater agreement (Cohen's kappa or correlation).
- **Mitigation:** Use `temperature=0.1` for consistency. Run 3 passes on a 10-transcript subset to measure score variance.

### 8.2 Transcript Availability and Quality

- Not all companies file earnings call transcripts as 8-K documents. Some are available only through paid services.
- Transcript quality varies: some are auto-generated (with errors), some are professionally edited.
- **Mitigation:** Focus on Tier 1 companies (Magnificent 7) where transcripts are always available and high quality. Tier 2-3 are bonus.

### 8.3 Survivorship Bias

- We are analyzing companies that are successful enough to still be in the S&P 500 in 2023-2025.
- Companies that hyped AI and then failed (e.g., smaller AI startups that went bankrupt) are not in our sample.
- This biases our analysis toward "hype that worked" rather than "all hype."

### 8.4 Google Trends Limitations

- Relative scale (0-100), not absolute search volume. Cannot compare across different query windows.
- No data before 2004. Cannot directly compare to dot-com era.
- Google may suppress or normalize certain searches.
- Regional bias: US-only data may miss global sentiment shifts.

### 8.5 Reddit Representativeness

- Reddit users are NOT representative of all investors. They skew younger, more male, more tech-savvy, and more speculative.
- r/wallstreetbets culture encourages performative bullishness ("diamond hands") that may overstate true conviction.
- Bot activity and coordinated campaigns can distort sentiment.
- **Mitigation:** Weight Reddit sentiment less heavily than earnings call NLP. Present it as "retail anecdotal" not "market consensus."

### 8.6 Temporal Mismatch

- Earnings calls happen once per quarter. Google Trends is weekly. Reddit is daily.
- Comparing across these frequencies requires careful alignment.
- The quarterly granularity of earnings analysis means we may miss within-quarter shifts.

### 8.7 Dot-Com Era Comparison Weakness

- We cannot apply the same NLP pipeline to dot-com era earnings calls (different format, different terminology).
- The comparison is qualitative (historical accounts of "internet" hype) rather than quantitative.
- This is the weakest part of this layer. Acknowledge it explicitly.

---

## 9. Combined Cost Estimate

| Component | API | Estimated Cost |
|-----------|-----|---------------|
| Earnings call transcripts | FMP API (free tier) | $0.00 |
| Earnings call NLP analysis | OpenAI gpt-4o-mini (240 calls) | ~$0.50 |
| Reddit post sentiment | OpenAI gpt-4o-mini (500 calls) | ~$0.09 |
| Google Trends | pytrends (free) | $0.00 |
| Validation re-runs | OpenAI (3x subset) | ~$0.15 |
| **Total** | | **~$0.74** |
| **With 3x safety buffer** | | **~$2.25** |

---

## 10. Code Outline (Full Implementation Pseudocode)

```python
"""
Layer 5: Sentiment & NLP Analysis
File: src/layers/sentiment_nlp.py
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import json
import time
import datetime
import requests
import pandas as pd
import numpy as np
import openai
import praw
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pytrends.request import TrendReq
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
FMP_API_KEY = os.getenv("FMP_API_KEY")  # Financial Modeling Prep

RAW_DIR = Path("data/raw")
TRANSCRIPT_DIR = RAW_DIR / "transcripts"
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

client = openai.OpenAI()

# ============================================================
# CONSTANTS
# ============================================================
COMPANIES = {
    "tier1": ["NVDA", "MSFT", "GOOG", "AMZN", "META", "AAPL"],
    "tier2": ["AMD", "AVGO", "CRM", "ORCL", "ADBE", "SNOW", "PLTR"],
    "tier3": ["INTC", "IBM", "NOW", "PANW", "CRWD", "MDB", "AI"],
}
ALL_COMPANIES = [c for tier in COMPANIES.values() for c in tier]
QUARTERS = [(y, q) for y in range(2023, 2026) for q in range(1, 5)]


# ============================================================
# SUB-LAYER 5A: EARNINGS CALL NLP
# ============================================================
def fetch_transcripts() -> pd.DataFrame:
    """Fetch all earnings call transcripts from Financial Modeling Prep."""
    transcripts = []
    for symbol in ALL_COMPANIES:
        for year, quarter in QUARTERS:
            cache = TRANSCRIPT_DIR / f"{symbol}_{year}_Q{quarter}.txt"
            if cache.exists():
                text = cache.read_text()
            else:
                text = _fetch_fmp_transcript(symbol, year, quarter)
                if text:
                    cache.write_text(text)
                time.sleep(0.5)
            if text:
                tier = "tier1" if symbol in COMPANIES["tier1"] else \
                       "tier2" if symbol in COMPANIES["tier2"] else "tier3"
                transcripts.append({
                    "symbol": symbol, "tier": tier,
                    "year": year, "quarter": quarter,
                    "quarter_label": f"Q{quarter} {year}",
                    "transcript_text": text,
                    "word_count": len(text.split()),
                })
    return pd.DataFrame(transcripts)


def _fetch_fmp_transcript(symbol: str, year: int, quarter: int) -> str | None:
    url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}"
    resp = requests.get(url, params={"year": year, "quarter": quarter, "apikey": FMP_API_KEY})
    if resp.ok and resp.json():
        return resp.json()[0].get("content", "")
    return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def analyze_transcript_openai(company: str, quarter: str, text: str) -> dict:
    """Send transcript to GPT-4o-mini for AI hype scoring."""
    MAX_CHARS = 60_000
    if len(text) > MAX_CHARS:
        half = MAX_CHARS // 2
        text = text[:half] + "\n[...TRUNCATED...]\n" + text[-half:]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                company=company, quarter=quarter, transcript_text=text)},
        ],
        temperature=0.1,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)
    parsed["input_tokens"] = response.usage.prompt_tokens
    parsed["output_tokens"] = response.usage.completion_tokens
    return parsed


def run_earnings_nlp(transcripts_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze all transcripts and return enriched DataFrame."""
    results = []
    for _, row in tqdm(transcripts_df.iterrows(), total=len(transcripts_df)):
        try:
            analysis = analyze_transcript_openai(
                row["symbol"], row["quarter_label"], row["transcript_text"]
            )
            results.append({"symbol": row["symbol"], "quarter_label": row["quarter_label"],
                            **analysis})
        except Exception as e:
            results.append({"symbol": row["symbol"], "quarter_label": row["quarter_label"],
                            "error": str(e)})
        time.sleep(0.5)

    results_df = pd.DataFrame(results)
    merged = transcripts_df.merge(results_df, on=["symbol", "quarter_label"], how="left")
    merged["ai_mentions_per_1k_words"] = (
        merged["ai_mention_count"] / (merged["word_count"] / 1000)
    )
    return merged


# ============================================================
# SUB-LAYER 5B: GOOGLE TRENDS
# ============================================================
def fetch_google_trends_data() -> dict[str, pd.DataFrame]:
    """Fetch all Google Trends datasets."""
    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=3)

    # AI era weekly detail
    pytrends.build_payload(
        ["NVIDIA stock", "AI stocks", "artificial intelligence investing",
         "AI bubble", "buy NVIDIA"],
        timeframe="2022-01-01 2026-03-28", geo="US",
    )
    ai_weekly = pytrends.interest_over_time()
    time.sleep(3)

    # Long-term monthly
    pytrends.build_payload(
        ["NVIDIA stock", "Cisco stock", "AI stocks", "tech bubble"],
        timeframe="all", geo="US",
    )
    long_monthly = pytrends.interest_over_time()
    time.sleep(3)

    # Bubble awareness (finance category)
    pytrends.build_payload(
        ["AI bubble", "is AI a bubble", "tech bubble"],
        timeframe="2022-01-01 2026-03-28", geo="US", cat=7,
    )
    bubble_awareness = pytrends.interest_over_time()

    return {
        "ai_weekly": ai_weekly.reset_index(),
        "long_monthly": long_monthly.reset_index(),
        "bubble_awareness": bubble_awareness.reset_index(),
    }


# ============================================================
# SUB-LAYER 5C: REDDIT SENTIMENT
# ============================================================
def fetch_reddit_data() -> pd.DataFrame:
    """Fetch NVDA-related posts from target subreddits."""
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent="2Kim Datathon 2026 v1.0",
    )
    all_posts = []
    for sub_name in ["wallstreetbets", "stocks", "investing", "nvidia"]:
        sub = reddit.subreddit(sub_name)
        for post in sub.search("NVDA OR NVIDIA", sort="new", time_filter="all", limit=1000):
            post_date = datetime.date.fromtimestamp(post.created_utc)
            if datetime.date(2022, 1, 1) <= post_date <= datetime.date(2026, 3, 28):
                all_posts.append({
                    "subreddit": sub_name, "post_id": post.id,
                    "title": post.title,
                    "selftext": (post.selftext or "")[:2000],
                    "score": post.score, "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "created_utc": post.created_utc,
                    "date": post_date.isoformat(),
                })
            time.sleep(0.1)
        time.sleep(2)
    return pd.DataFrame(all_posts)


def run_reddit_sentiment(posts_df: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    """Analyze a stratified sample of Reddit posts with OpenAI."""
    posts_df["month"] = pd.to_datetime(posts_df["date"]).dt.to_period("M")
    n_months = posts_df["month"].nunique()
    per_month = max(1, sample_size // n_months)

    sampled = posts_df.groupby("month", group_keys=False).apply(
        lambda x: x.sample(min(len(x), per_month), random_state=42)
    )

    results = []
    for _, post in tqdm(sampled.iterrows(), total=len(sampled)):
        try:
            analysis = _analyze_reddit_post(post.to_dict())
            results.append(analysis)
        except Exception as e:
            results.append({"post_id": post["post_id"], "error": str(e)})
        time.sleep(0.3)

    return sampled.merge(pd.DataFrame(results), on="post_id", how="left")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15))
def _analyze_reddit_post(post: dict) -> dict:
    text = f"Title: {post['title']}\n\nBody: {post['selftext'][:1500]}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": REDDIT_SENTIMENT_PROMPT.format(
                subreddit=post["subreddit"],
                subreddit_culture=SUBREDDIT_CULTURES.get(post["subreddit"], "General"),
            )},
            {"role": "user", "content": text},
        ],
        temperature=0.1, max_tokens=300,
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content)
    result["post_id"] = post["post_id"]
    return result


# ============================================================
# STATISTICAL TESTS
# ============================================================
def run_all_statistical_tests(
    earnings_df: pd.DataFrame,
    trends_data: dict,
    reddit_weekly: pd.DataFrame,
    price_data: pd.DataFrame,
) -> dict:
    """Run all Layer 5 statistical tests."""
    results = {}

    # 6.1: Hype-price correlation
    quarterly_hype = earnings_df.groupby("quarter_label").agg(
        avg_hype=("hype_score", "mean"),
        avg_specificity=("revenue_specificity", "mean"),
        avg_mentions=("ai_mentions_per_1k_words", "mean"),
    ).reset_index()
    # merge with NVDA quarterly returns from price_data ...
    # results["hype_price_corr"] = ...

    # 6.2: Google Trends Granger causality
    # merge weekly trends with NVDA weekly returns ...
    # results["trends_granger"] = ...

    # 6.3: OLS regression
    # results["ols_regression"] = model.summary() ...

    # 6.4: Reddit sentiment-volume correlation
    # results["reddit_volume_corr"] = ...

    return results


# ============================================================
# VISUALIZATIONS
# ============================================================
def create_all_charts(
    earnings_df: pd.DataFrame,
    trends_data: dict,
    reddit_weekly: pd.DataFrame,
    price_data: pd.DataFrame,
) -> dict[str, go.Figure]:
    """Generate all Layer 5 charts."""
    return {
        "5.1": create_chart_5_1(earnings_df),
        "5.2": create_chart_5_2(earnings_df),
        "5.3": create_chart_5_3(trends_data["long_monthly"], trends_data["ai_weekly"]),
        "5.4": create_chart_5_4(earnings_df, price_data),
        "5.5": create_chart_5_5(reddit_weekly, price_data),
        "5.6": create_chart_5_6(earnings_df),  # Returns matplotlib Figure
    }


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_layer_5(price_data: pd.DataFrame) -> dict:
    """Execute the full Layer 5 pipeline."""
    # 5A: Earnings Call NLP
    transcripts = fetch_transcripts()
    earnings_df = run_earnings_nlp(transcripts)
    earnings_df.to_parquet("data/processed/earnings_nlp.parquet")

    # 5B: Google Trends
    trends_data = fetch_google_trends_data()
    for key, df in trends_data.items():
        df.to_parquet(f"data/processed/google_trends_{key}.parquet")

    # 5C: Reddit Sentiment
    reddit_posts = fetch_reddit_data()
    reddit_sentiment = run_reddit_sentiment(reddit_posts, sample_size=500)
    reddit_weekly = aggregate_reddit_weekly(reddit_sentiment)
    reddit_sentiment.to_parquet("data/processed/reddit_sentiment.parquet")
    reddit_weekly.to_parquet("data/processed/reddit_weekly.parquet")

    # Statistical tests
    test_results = run_all_statistical_tests(
        earnings_df, trends_data, reddit_weekly, price_data
    )

    # Charts
    charts = create_all_charts(earnings_df, trends_data, reddit_weekly, price_data)

    return {
        "earnings_df": earnings_df,
        "trends_data": trends_data,
        "reddit_weekly": reddit_weekly,
        "test_results": test_results,
        "charts": charts,
    }
```

---

## 11. Output Artifacts

| Artifact | Path | Format | Description |
|----------|------|--------|-------------|
| Raw transcripts | `data/raw/transcripts/{SYMBOL}_{YEAR}_Q{Q}.txt` | Text | Cached earnings call text |
| Earnings NLP results | `data/processed/earnings_nlp.parquet` | Parquet | All NLP scores per company-quarter |
| Google Trends (AI weekly) | `data/processed/google_trends_ai_weekly.parquet` | Parquet | Weekly search interest 2022-2026 |
| Google Trends (long-term) | `data/processed/google_trends_long_monthly.parquet` | Parquet | Monthly search interest 2004-2026 |
| Google Trends (bubble) | `data/processed/google_trends_bubble_awareness.parquet` | Parquet | Bubble-related search terms |
| Reddit raw posts | `data/raw/reddit_nvda_posts.parquet` | Parquet | All fetched Reddit posts |
| Reddit sentiment | `data/processed/reddit_sentiment.parquet` | Parquet | Posts with sentiment scores |
| Reddit weekly agg | `data/processed/reddit_weekly.parquet` | Parquet | Weekly aggregated Reddit metrics |
| Charts 5.1-5.5 | `submissions/charts/chart_5_*.html` | HTML | Interactive Plotly charts |
| Chart 5.6 (word cloud) | `submissions/charts/chart_5_6_wordcloud.png` | PNG | Matplotlib word cloud grid |
| Statistical results | `data/processed/sentiment_stat_tests.json` | JSON | All test outputs |

---

## 12. Acceptance Criteria

- [ ] At least 80% of Tier 1 company transcripts fetched successfully (minimum: NVDA, MSFT, GOOG for all 12 quarters)
- [ ] OpenAI API returns valid JSON for >95% of transcript analyses
- [ ] Hype scores show meaningful variance (not all 5s or all 10s) -- validate by spot-checking
- [ ] Manual validation of 20 random transcripts shows >0.7 correlation with GPT scores
- [ ] Google Trends data fetched without 429 rate limit errors
- [ ] Reddit data spans at least 18 months of the 2022-2026 range
- [ ] Reddit sentiment distribution is not degenerate (not >90% one category)
- [ ] All 6 charts render without errors
- [ ] Chart 5.2 scatter has visible quadrant separation (not all points in one corner)
- [ ] Statistical test results are reported with p-values and effect sizes
- [ ] Total OpenAI API cost stays under $3.00
- [ ] Limitations section is included in the final notebook with all 7 limitations listed
