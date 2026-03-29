# Competition Strategy & Storytelling Critique
## The AI Hype Cycle — Are We in a Bubble?
### Team 2Kim | SBU AI Community Datathon 2026

**Critique written:** 2026-03-28
**Based on:** specs/00–09, competition README, submission guidelines, and JUDGING-RUBRIC.md

---

## Overall Competition Readiness

**Short answer: This will very likely win — but only if the team actually executes it.** The spec is the strongest analytical plan in this competition by a significant margin. No other team is attempting a 5-layer cross-era comparison with DTW + ML regime classification. The research question is genuinely interesting to finance professors. The narrative structure is mature. The MLA citations are pre-formatted.

The danger is not the design. The danger is execution collapse — a 30-chart, 6-API, GPT-sentiment, ML-with-SHAP submission planned over 4 days by 2 people. This is a 2-week project compressed into a datathon sprint. Every spec exists in beautiful detail, but zero implementation exists yet. The gap between this spec and a working notebook is enormous.

**Verdict: Win-capable, but overengineered for the risk tolerance of a 4-day competition.**

---

## Rubric Alignment Audit

### 1. Research Question & Track Fit (15pts)

**Current estimated score: 14–15/15**

This is near-perfect for the rubric. The question is hyper-specific (NVDA vs CSCO, 5 measurable dimensions), testable (DTW distance, t-tests, regime probability outputs), and unambiguously Finance & Economics (stock prices, P/E, Fed funds, S&P concentration). The H0/H1 framing signals academic rigor that professors will respond to positively. The sub-question structure maps cleanly to deliverables.

**Gaps:**
- The phrase "and if so, at what stage of the bubble lifecycle are we today?" in the main research question is slightly too ambitious as a falsifiable sub-question — the "stage" answer from a Random Forest with N=2 training bubbles is epistemically weak. A judge could penalize this for overstating precision.
- The question is long. Judges score fast. The 08 spec's verbal framing in speaking notes is tighter than the written question in the notebook.

**Suggestions:**
- Add a one-sentence "plain English" version of the research question at the top of Section 1 before the formal version. Something like: "We ask: does the current AI rally look like the dot-com bubble — and how similar is it across five independent dimensions?" This gives judges an anchor before the formal hypothesis machinery.
- In the notebook Section 1, explicitly call out that this is the Finance & Economics track in the first 5 lines. Judges have compliance checklists and this signal should be immediate.

---

### 2. Data Quality & Preparation (20pts)

**Current estimated score: 16–19/20**

Seven-plus sources, documented merge logic, parquet caching, before/after null counts — this is substantially above average. The yfinance + FRED combination alone exceeds what most datathon teams do. The feature engineering documentation (P/E formula, FCF yield, credit spread z-score derivations) is exactly what judges scoring this category want to see.

**Gaps (real ones, not spec gaps):**

1. **yfinance quarterly financials for CSCO 1998–2001 is unreliable.** yfinance's `quarterly_financials` for tickers this old frequently returns incomplete data, wrong fiscal year alignments, or NaN-filled frames. The spec acknowledges this risk but buries it in the risk table. This is actually a HIGH probability failure point, not "Low likelihood." A manual cross-reference against SEC EDGAR is not optional — it is required for data integrity.

2. **pytrends is not a reliable API.** It rate-limits aggressively, returns inconsistent data across queries (the same query can return different index values at different times due to normalization), and frequently blocks automated requests. The spec says "cache results" but doesn't specify a fallback that provides actual dot-com era data. Google Trends only goes back to 2004, so there is structurally zero overlap with the 1998–2001 CSCO era. This means the sentiment comparison between eras is fundamentally asymmetric — you cannot compare 2004–present Google Trends to dot-com era Google Trends because the data doesn't exist.

3. **Reddit asymmetry is acknowledged but underweighted.** The spec notes Reddit didn't exist during the dot-com era, but this is listed as a low-impact limitation. In practice, it means Layer 5 sentiment is an AI-era-only signal. Judges will ask about this. The spec should elevate this to a prominent methodological caveat.

4. **Alpha Vantage at 25 requests/day free tier is a real bottleneck** for the S&P 500 constituent data. Slickcharts/Wikipedia as the backup is reasonable, but those are static annual snapshots with no granularity below yearly — which means the concentration layer is actually annual-resolution data, not daily. This affects the merge strategy claim of "daily granularity, lower-freq data forward-filled." Forward-filling annual concentration data to daily is methodologically questionable and should be disclosed explicitly.

**Suggestions:**
- Download and cache ALL data on Day 0 (before the official competition start) and commit the cached parquet files to ensure the notebook runs. Or at minimum, ship a `data/sample/` directory with small test fixtures so `Restart & Run All` doesn't fail when APIs are rate-limited.
- Move the CSCO historical financial data validation to Sprint 1, not Sprint 2. If this data is wrong, Layers 2 and the ML pipeline collapse.

---

### 3. Analysis & Evidence (25pts) — THE TIEBREAKER

**Current estimated score: 22–24/25 (if executed). Risk of 17–19/25 if execution falters.**

The analytical design is outstanding. Five statistical tests (Pearson, t-test, z-score, cosine similarity, Mann-Whitney U) plus DTW plus a regime classifier with SHAP — this is more rigorous than anything a judge will have seen in this competition. The bubble scorecard is the synthesis move that will differentiate this submission.

**Critical gap — chart count vs. judge attention:**

The spec plans 30 charts for the notebook. This is academically appropriate but creates a real presentation risk. In a datathon, judges skim notebooks quickly. Thirty charts with no clear hierarchy signal to a judge that the team didn't know what their best evidence was. The spec distinguishes PRIMARY vs SECONDARY charts, but this distinction needs to be visually enforced in the notebook with section headers like "Key Finding" vs "Supplementary Analysis."

**Statistical validity concerns:**

1. **Pearson correlation of daily returns between NVDA and CSCO is expected to be near zero.** These are returns in different decades. The meaningful correlation is between the *normalized price paths* (the overlay visual), not the daily return correlation. If the Pearson r on daily returns comes back at 0.02 with p=0.8, this looks like a negative finding for the bubble thesis. The spec should specify: report the Pearson correlation of *the detrended normalized price series* or the DTW similarity as the primary quantitative measure for Layer 1, not daily return correlation.

2. **Two-sample t-test on quarterly revenue growth rates between NVDA and CSCO:** this test should SHOW they're different (NVDA grows faster). That's the result you want. But be aware that with N=12–16 quarters per company, the t-test is underpowered and the distributions are not normal. Report effect size (Cohen's d) alongside the p-value.

3. **Cosine similarity of "macro regime vectors"** is a creative idea but judges may not recognize it as a standard statistical test. Frame this more carefully — explain what the vectors are (feature values at each time point), what cosine similarity means in plain language, and why this is appropriate. Otherwise a finance professor may mark it down as methodological novelty disguised as rigor.

4. **The regime classifier trained on N=2 bubbles** with manual labels is the biggest technical credibility risk in the entire project. The spec acknowledges this as the top limitation, which is good. But the ML results will only be credible if the presentation is extremely honest about what the model can and cannot say. If the classifier outputs "bubble: 73%" judges will want to understand what that 73% means given the training set. The answer has to be ready.

**Suggestions:**
- In the notebook, add a clearly labeled "Executive Summary" cell after Section 6 with the top 3 findings, one sentence each. This is what judges will actually read.
- For Layer 1, report both the visual similarity (the price overlay) and a DTW distance score. Don't lead with Pearson correlation of returns — it will likely be unimpressive.
- Cap the notebook at 15 "key" charts with the remaining 15 clearly marked as "supplementary" in a collapsible section or appendix. This respects judge time and signals analytical prioritization.

---

### 4. Technical Rigor & Reproducibility (15pts)

**Current estimated score: 11–14/15**

The reproducibility design is thorough: pinned requirements, random seeds, TimeSeriesSplit, parquet caching, idempotent cells. The `Restart & Run All` test protocol in the checklist is exactly right.

**Real risks that could tank this category:**

1. **Six live API dependencies in a single notebook.** yfinance, FRED, pytrends, PRAW, OpenAI, and EDGAR. If any one of these is rate-limited, blocked, or returns unexpected data during the judge's review, the notebook partially fails. The spec mentions caching but the caching must be implemented as: "if file exists, load from cache; else call API." This pattern must be enforced for every single API call. A notebook that stops running halfway through is a catastrophic failure.

2. **The kaleido dependency for Plotly static export is frequently broken** in Jupyter environments. `fig.write_image()` will fail silently or throw an error if kaleido is not correctly installed. This breaks the slide chart export pipeline. Test this explicitly on Day 1. The alternative is to use `fig.write_html()` for the notebook and only export static images for slides manually.

3. **SHAP with Random Forest** can take minutes on a large feature matrix if parallelism is not configured. With 38 features and 384 training months, it should be fast (under 60 seconds), but test this.

4. **The notebook structure in 08_presentation_narrative (spec) lists 20 slides, but 00_project_overview says 15–16 slides, and 09_submission_checklist says 16 slides.** These three specs are inconsistent. The actual slide count will confuse whoever builds the slides. This needs to be reconciled to one authoritative number.

5. **`dtw-python` vs `fastdtw` vs `tslearn`:** The spec uses `dtw-python`. Confirm this is `pip install dtw-python` (the `dtw` module). There is also a `dtaidistance` package. The import `from dtw import dtw, accelerated_dtw` maps to `dtw-python`. Verify this installs correctly and the API matches what the spec expects before writing the code.

**Suggestions:**
- On Day 1, run the full requirements install in a clean virtual environment and verify every import works before writing a single analysis cell. Requirements failures discovered on Day 4 are fatal.
- Add a `00_setup.ipynb` or a setup cell at the top of the main notebook that tests all imports and prints version numbers. This proves reproducibility to judges.
- For the API calls, implement the cache-or-call pattern as a utility function in `src/loaders.py` and import it. Don't write ad-hoc caching logic in notebook cells.

---

### 5. Presentation Clarity & Storytelling (15pts)

**Current estimated score: 13–15/15**

The storytelling spec (08) is the strongest piece of work in the entire spec set. The dialectical thesis/antithesis/synthesis arc is sophisticated and shows analytical maturity. The "bull case then bear case" structure creates genuine tension and respects the audience's intelligence. The one-sentence key takeaway on Slide 16 is excellent.

**However, there is a timing problem.**

The cumulative time in 08_presentation_narrative reaches 10:35. The competition window is 8–10 minutes. The spec itself acknowledges this and says "trim 30 seconds" — but 90 seconds over budget is not 30 seconds. The actual overage is:

- 18 content slides at the stated times = 10:35
- Target = 9:00–9:30
- Overage = 1:05–1:35

The spec suggests cutting Slide 10 (Buffett Indicator) to save 40 seconds. That still leaves 30–55 seconds of overage. The safe cut is: remove Slide 10 entirely AND compress Slides 3–4 (framework and methodology) from 30 seconds each to 15 seconds each. Methodology details belong in the notebook, not the presentation.

**Additional storytelling concerns:**

1. **Color inconsistency between specs.** The 07_visualization_plan spec uses CSCO = Coral Red (#EF553B) and NVDA = Indigo Blue (#636EFA). The 00_project_overview spec uses NVDA = NVIDIA Green (#76b900) and CSCO = Cisco Blue (#049fd9). The 09_submission_checklist references NVDA green (#76b900) and CSCO blue (#049fd9). If judges see green NVDA in slides and blue NVDA in the notebook, it creates confusion. This must be reconciled to one color scheme before building any charts.

2. **Slide 2's speaking notes say "This is NVIDIA in blue and Cisco in red"** but the visualization plan shows NVDA as indigo blue and CSCO as coral red. Meanwhile the overview spec shows NVDA as green. This is the hook slide — the most important visual. The color labeling in the speaking notes must match what's on screen.

3. **The dashboard (localhost:5173) reference on Slide 18** is a liability in a competition. Judges cannot access localhost. If this is meant to be a live demo, it needs to be hosted or cut. A localhost URL in a judged presentation signals an unfinished product.

**Suggestions:**
- Reconcile the color scheme to one definitive standard across all specs NOW, before any chart is built. The safest choice: NVDA = green (brand color #76b900), CSCO = a clearly contrasting blue (#049fd9). Update 07 and 08 to match.
- Cut the presentation to 16 slides maximum with a hard 9-minute target. Build a timed practice run into Day 4's schedule.
- Remove the localhost dashboard reference from the presentation. It's a distraction and creates a negative impression if judges notice it can't be accessed.

---

### 6. Limitations, Ethics & Practical Insight (10pts)

**Current estimated score: 9–10/10**

This is the best-handled rubric category in the spec. The limitation list is extensive, specific, and honest. The survivorship bias acknowledgment (we chose CSCO because it crashed — that's selection bias) is sophisticated and will impress finance professors. The N=2 training set caveat is front-and-center. The ethical disclaimer about financial advice is present.

**One gap:** The limitation discussion doesn't address the **lookahead bias in regime labeling**. The CSCO era labels are assigned with perfect hindsight knowledge of when the bubble peaked and crashed. This is acknowledged implicitly but not named explicitly. Finance professors will notice. Name it: "Our regime labels for the training period were assigned with hindsight knowledge of outcomes. In live deployment, these labels would be forward-looking estimates, introducing label uncertainty the model cannot quantify."

**One risk:** The spec's limitation on AI tool usage says outputs were "validated on a sample basis" with "approximately 85% agreement." If a judge asks "what was your sample size for that validation?", you need a specific answer (200 sentences is mentioned in the Q&A prep — make sure this is in the notebook as well, not just in the speaking notes).

---

## Critical Issues (Must Fix)

### Issue 1: Color scheme is inconsistent across three specs

**Problem:** 00_project_overview and 09_checklist use NVDA=green, CSCO=blue. 07_visualization_plan and 08_presentation_narrative use NVDA=indigo blue, CSCO=coral red. The hook slide speaking notes say "NVIDIA in blue and Cisco in red" — contradicting the 00 spec.

**Rubric category affected:** Presentation Clarity & Storytelling (15pts), Analysis & Evidence (25pts — chart readability)

**Fix:** Decide now: NVDA=green (#76b900 for slides, or #00CC96 for dark backgrounds), CSCO=coral red (#EF553B). Update every spec, then every chart template. Do this before writing a single line of visualization code.

---

### Issue 2: Presentation is 90 seconds over budget

**Problem:** The sum of slide times in 08 is 10:35. Competition limit is 8–10 minutes. If this runs over, judges may cut Q&A or penalize storytelling score.

**Rubric category affected:** Presentation Clarity & Storytelling (15pts)

**Fix:** Cut Slide 5 (Bull case header — 5-second title card), Slide 8 (Bear case header — 5-second title card), and Slide 10 (Buffett Indicator — 40 seconds). Add the Buffett Indicator stat verbally during Slide 9. The header "interstitial" slides are stylistically nice but cost time without adding content. Revised total: ~9:05. Also compress Slides 3 and 4 to combined 45 seconds.

---

### Issue 3: Pearson correlation of daily returns is the wrong test for Layer 1

**Problem:** Daily return correlation between stocks in different decades will be near zero. Reporting r ≈ 0.02 as evidence for or against the bubble thesis confuses the analysis. This is a category error — correlation of returns is not the same as similarity of price trajectory shapes.

**Rubric category affected:** Analysis & Evidence (25pts), Technical Rigor (15pts)

**Fix:** For Layer 1, the primary quantitative measure should be the DTW similarity score, not Pearson r of returns. If you keep Pearson correlation, apply it to the normalized price index levels (Chart 1.1 data), not daily returns — and explain explicitly why this is appropriate. Or replace it with the Spearman rank correlation of the normalized price paths, which is more robust for non-normal distributions.

---

### Issue 4: Six live API dependencies create notebook fragility

**Problem:** yfinance, FRED, pytrends, PRAW, OpenAI, EDGAR — any one of these failing during judge evaluation produces a broken notebook. The spec's caching strategy is described but not yet implemented. The `Restart & Run All` test on Day 3/4 must work in an environment without active internet or with rate-limited APIs.

**Rubric category affected:** Technical Rigor & Reproducibility (15pts) — most likely category to score below target

**Fix:** Implement cache-or-call for every single API before writing any analysis. Structure it as: `if Path("data/raw/prices_NVDA.parquet").exists(): df = pd.read_parquet(...) else: df = download_and_cache(...)`. Commit the cached data files (if under 100MB and not rate-limited artifacts) OR provide a `scripts/download_data.py` script that must be run once before the notebook. Document this clearly at the top of the notebook. Judges who can't reproduce the data won't penalize you if the caching pattern is obvious and documented.

---

### Issue 5: Slide count inconsistency across specs (16 vs 18 vs 20)

**Problem:** 00_project_overview says 15–16 slides. 08_presentation_narrative says "18 content slides + 1 title + 1 backup = 20 slides." 09_checklist slide-by-slide table lists 16 slides. The actual narrative in 08 numbers slides 1–18 plus 2 backups. These are three different numbers.

**Rubric category affected:** Submission compliance (pre-scoring check), Presentation Clarity

**Fix:** The authoritative answer should be 16 slides (1 title + 14 content + 1 references/close) + 2 backup slides not shown in main presentation. Update 08 to reflect this. The dialectical structure (thesis/antithesis/synthesis) works in 16 slides if the two header "transition" slides (Slides 5 and 8 in the current spec) are eliminated.

---

## Presentation Strategy Critique

### Is the hook strong enough?

Yes, and it is the best hook possible for this topic. Showing the NVDA/CSCO normalized price overlay without labeling which is which — then asking "does this look familiar?" — is a textbook opening for a data-driven presentation. The 3-4 second pause after showing the chart is the right instinct.

One improvement: add the CSCO crash endpoint annotation on the chart from the first moment it's shown. Don't hold it back. Judges who know their market history will already see where CSCO went. Let the chart do the work immediately.

### Is the slide count appropriate?

16 content slides is appropriate for 9 minutes. 18–20 is too many. The current spec's 18 content slides assume ~30 seconds per slide average, but several slides (Concentration at 50 seconds, ML slides at 50 seconds each, Synthesis at 60 seconds) are significantly longer. The math does not work. Cut to 16.

### Are the speaking notes effective?

Largely yes. They are specific (cite actual numbers like "200x P/E," "265% YoY revenue"), they avoid reading from the slide, and they maintain the dialectical tension. The handoff script between Jimmy and Alice is professional.

Two weaknesses:
1. Several notes are too long to be "notes" — they are full paragraphs that would take 90+ seconds to deliver. These will be ignored under presentation pressure. Condense each note to 3–5 bullet points with the key numbers.
2. The notes for Slide 4 (methodology) reference "no lookahead bias" and "trained on data through 2018 and tested on 2019-2024" — this does not match the ML spec in 06, which trains on CSCO era (1998–2001) and applies to NVDA era (2023–2026). This is a factual inconsistency in the speaking notes that a finance professor will catch.

### Will the Q&A prep hold up?

The Q&A section in 08 is exceptional — 10 prepared questions with specific, non-defensive answers. Questions Q1 (label ranges), Q3 (small sample), Q7 (structural market changes), and Q8 (passive inflows) are exactly the questions finance professors ask. Q4 (why not neural network) is well-answered: "deliberately, interpretability > marginal accuracy."

Two Q&A gaps:

1. There is no prepared answer for: "Your DTW warping path — did you normalize the series before DTW or use raw prices? Does your choice affect the result significantly?" This is a methods question with a right answer (yes, normalize first; yes, it matters). Prepare this explicitly.

2. There is no prepared answer for: "What happens to your analysis if NVIDIA stock is down 30% next month — does that change your conclusion?" This is a professor's way of asking if the analysis is durable or just current-moment narrative. The answer should be: "It would shift the ML regime probability but not invalidate the structural analysis — the concentration and macro signals would remain elevated, and a 30% drawdown in NVDA would actually be consistent with the distribution phase in the bubble lifecycle."

---

## What Competitors Will Do (and How to Differentiate)

Most teams in this competition will submit one of the following:
- Single-asset price analysis (NVDA stock or another S&P 500 name) with basic EDA: mean, std, rolling average, maybe a moving average crossover
- Macro indicator correlation study (FRED unemployment vs something) with a linear regression
- Sector ETF comparison with a simple hypothesis test

This submission is categorically different:
- Cross-era, multi-dimensional comparison (two time periods, five data layers, three ML outputs)
- The Bubble Scorecard synthesis table has no equivalent in a standard datathon submission
- The DTW analysis is technically sophisticated but explainable to a general audience
- The narrative structure (bull/bear/verdict) demonstrates analytical maturity that purely descriptive submissions lack

**The differentiation risk is overengineering.** If the notebook is partially broken, the visuals are inconsistent, or the time is exceeded during presentation, judges will actively compare this to the simpler team that ran clean EDA and presented clearly in 8 minutes. A well-executed simple analysis beats a broken complex one every time.

The rule: better to present 10 excellent charts cleanly than 30 charts with three broken API calls and inconsistent colors. The spec should be treated as a ceiling, not a floor. Sprint 1 (9 charts) is the floor. Ship Sprint 1 first, then add Sprint 2 if time permits.

---

## Recommended Priority Fixes (Top 5)

**1. Reconcile the color scheme to one standard across all specs before writing any visualization code.**

NVDA = green, CSCO = coral red, everywhere. Update 00_project_overview, 07_visualization_plan, 08_presentation_narrative, and 09_checklist to agree. This takes 30 minutes and prevents hours of rework if charts are built with mismatched colors.

**2. Fix the presentation timing: cut to 16 slides with hard 9-minute target.**

Remove Slides 5 and 8 (transition header cards — they add zero information and cost 10 seconds each). Cut Slide 10 (Buffett Indicator) and reference it verbally on Slide 9 ("by the Buffett Indicator, total market cap to GDP is at 190% — higher than dot-com peak"). Run a timed practice on Day 3, not Day 4. A presentation that runs over time is the single fastest way to lose Presentation Clarity points.

**3. Implement cache-or-call for every API call before writing any analysis code.**

The single biggest execution risk is a broken notebook during judging. Every API call must follow the pattern: check for cached parquet → return cached data; otherwise call API → cache result → return data. This takes 2–3 hours to implement upfront and eliminates the entire notebook reproducibility risk category.

**4. Replace the Layer 1 Pearson correlation of daily returns with a measure that will actually support the argument.**

The daily return Pearson correlation between 2023–2026 NVDA and 1998–2001 CSCO will likely be near zero and statistically non-significant. Reporting this creates a misleading "the returns aren't correlated" finding. Instead: (a) report the DTW distance as the primary Layer 1 quantitative measure, (b) if keeping Pearson, apply it to the normalized price index levels on the shared trading-day axis, and (c) report the visual similarity as the primary finding and the DTW score as the confirmation.

**5. Add an "Executive Summary" cell at the end of Section 6 with the top 3 findings in three sentences.**

The Bubble Scorecard table is excellent but long. Judges reviewing a notebook quickly need to find the punchline immediately. A three-sentence summary cell (e.g., "Finding 1: Price trajectories show DTW similarity of X/100. Finding 2: NVDA's P/E is 4x lower than CSCO's at equivalent stages. Finding 3: Market concentration exceeds dot-com peak levels.") gives judges exactly what they need to award points. Put this immediately before the Limitations section, not buried inside a long narrative.

---

## Summary Scorecard (Estimated vs Target)

| Rubric Category | Weight | Target (from spec) | Realistic Estimate | Gap |
|---|---|---|---|---|
| Research Question & Track Fit | 15pts | 13–15 | 14–15 | None — excellent |
| Data Quality & Preparation | 20pts | 18–20 | 16–18 | API reliability risk |
| Analysis & Evidence | 25pts | 22–25 | 21–24 | Execution + stats test selection |
| Technical Rigor & Reproducibility | 15pts | 13–15 | 11–14 | Notebook fragility risk |
| Presentation Clarity & Storytelling | 15pts | 12–15 | 12–14 | Timing + color inconsistency |
| Limitations, Ethics & Practical Insight | 10pts | 8–10 | 9–10 | Strong as written |
| **Total** | **100pts** | **86–99** | **83–95** | |

The floor (83) wins most datathons. The ceiling (95) wins this one outright. The gap between floor and ceiling is almost entirely execution quality, not design quality. The spec is excellent. Ship Sprint 1 clean, add Sprint 2 if time allows, and don't let API fragility or timing overruns cost points in categories that are otherwise locked up.

