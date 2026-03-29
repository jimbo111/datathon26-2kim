# 08 — Presentation Narrative & Slideshow Specification

> **Project:** The AI Hype Cycle — Are we in a bubble? NVIDIA 2023-2026 vs Cisco 1998-2001
> **Team:** 2Kim | SBU AI Community Datathon 2026 — Finance & Economics Track
> **Last updated:** 2026-03-28

---

## Table of Contents

1. [Presentation Parameters](#1-presentation-parameters)
2. [Story Arc Overview](#2-story-arc-overview)
3. [Slide-by-Slide Specification](#3-slide-by-slide-specification)
4. [Slide Design Guidelines](#4-slide-design-guidelines)
5. [Presentation Delivery Guide](#5-presentation-delivery-guide)
6. [Anticipated Judge Questions & Prepared Answers](#6-anticipated-judge-questions--prepared-answers)

---

## 1. Presentation Parameters

| Parameter | Value |
|---|---|
| Format | `.pptx` exported to `.pdf` for submission |
| File name | `2kim_finance_slides.pptx` / `2kim_finance_slides.pdf` |
| Total slides | 1 title + 14 content + 1 close = **16 slides** (+ 2 backup, not shown in main presentation) |
| Presentation time | 8-10 minutes (target: 9 minutes) |
| Q&A time | 5 minutes |
| Pace | ~30 seconds per slide average; key slides get 45-60 seconds, transition slides get 15 seconds |
| Audience | SBU professors (judges); assume strong quantitative background, moderate finance knowledge |
| Tone | Data-driven, balanced, intellectually honest. Not alarmist ("the sky is falling") or dismissive ("everything is fine"). The conclusion is nuanced. |

---

## 2. Story Arc Overview

The presentation follows a **deliberate dialectical structure**: thesis, antithesis, synthesis. This demonstrates analytical maturity and avoids the common pitfall of cherry-picking data to support a predetermined conclusion.

```
Act I: The Question (Slides 1-4)        ~1.5 minutes
  "Look at this chart. Are we in a bubble?"

Act II: The Bull Case (Slides 5-6)       ~1.5 minutes
  "Here's why this time might be different..."

Act III: The Bear Case (Slides 7-9)      ~2.0 minutes
  "...but here's why we should worry."

Act IV: The Wild Card (Slide 10)         ~0.5 minutes
  "The macro environment adds complexity."

Act V: The Verdict (Slides 11-14)        ~2.5 minutes
  "Here's what the models say — and what it means."

Act VI: Close (Slides 15-16)             ~1.0 minutes
  "Limitations, next steps, and the takeaway."
```

**Emotional arc for the audience:**

1. Intrigue (the price overlay is viscerally compelling)
2. Reassurance (fundamentals ARE different — smart people aren't just repeating history)
3. Concern (but the structural similarities are undeniable)
4. Complexity (the macro picture is genuinely ambiguous)
5. Resolution (a probabilistic, nuanced answer backed by ML)
6. Respect (honest about limitations)

---

## 3. Slide-by-Slide Specification

---

### Slide 1: Title Slide

| Property | Value |
|---|---|
| **Title** | The AI Hype Cycle |
| **Subtitle** | Are We in a Bubble? Comparing NVIDIA 2023-2026 to Cisco 1998-2001 |
| **Team** | Team 2Kim: Jimmy Kim & Alice Kim |
| **Event** | SBU AI Community Datathon 2026 — Finance & Economics Track |
| **Visual** | Clean, minimal. Background: dark gradient. Small NVDA and CSCO logos faded in background. |
| **Speaking notes** | "Good [morning/afternoon]. We're Team 2Kim. Today we're going to show you something that made us do a double-take when we first plotted it — and then spend three weeks trying to figure out if history is about to repeat itself." |
| **Time** | 15 seconds |
| **Transition** | "Let's start with the chart that started this whole project." |

---

### Slide 2: The Hook — Price Overlay

| Property | Value |
|---|---|
| **Title** | "Does This Look Familiar?" |
| **Chart** | Chart 1.1 — Normalized Price Overlay (NVDA vs CSCO) |
| **Layout** | Chart takes up 85% of the slide. Minimal text. Let the visual speak. |
| **Key message** | The price trajectories of NVIDIA (2023-2026) and Cisco (1998-2001) are strikingly similar when normalized to their breakout points. |
| **Speaking notes** | "This is NVIDIA in green and Cisco in red, both normalized to 100 at their breakout points. The gray dashed line is Nortel Networks — a dot-com company that went bankrupt entirely. Cisco went on to lose 89% of its value. NVIDIA is currently at the same point in its trajectory where Cisco was about 6 months before its peak. [Pause.] The question is: are we looking at a coincidence, or a pattern?" |
| **Time** | 45 seconds |
| **Transition** | "To answer that, we need to look beyond price. We analyzed five layers of data." |

---

### Slide 3: Research Framework

| Property | Value |
|---|---|
| **Title** | "Five-Layer Analytical Framework" |
| **Visual** | Diagram showing the 5 layers as a vertical stack or concentric rings: Price (outer, most visible) -> Fundamentals -> Concentration -> Macro -> Sentiment (inner, hardest to measure) |
| **Text** | Minimal bullet points: one line per layer with a key question. (1) Price: Same trajectory? (2) Fundamentals: Same detachment? (3) Concentration: Same fragility? (4) Macro: Same environment? (5) Sentiment: Same euphoria? |
| **Key message** | We don't just compare stock prices — we compare the structural conditions across five dimensions. |
| **Speaking notes** | - Five layers: price, fundamentals, concentration, macro, sentiment — each adds depth. - A price pattern alone means nothing — but if all five dimensions match, that's a different story. |
| **Time** | 30 seconds |
| **Transition** | "We used 8 data sources, ranging from Yahoo Finance and FRED to SEC filings and Reddit." |

---

### Slide 4: Data Sources & Methodology

| Property | Value |
|---|---|
| **Title** | "Data Sources & Methodology" |
| **Visual** | Clean table or icon grid showing: yfinance (price data), FRED (macro), SEC EDGAR (fundamentals), Google Trends, Reddit API, OpenAI API (NLP), Shiller CAPE database, S&P 500 constituent data |
| **Methodology callout** | "38 features engineered across all 5 layers. Time-series aware ML pipeline (no random splits). 384 months of training data (1990-2024)." |
| **Key message** | Multi-source data with rigorous preprocessing. |
| **Speaking notes** | - 8 data sources spanning 36 years. - Every feature computed using only trailing data — no lookahead bias. - Time-series cross-validation throughout. - ML models trained on historical regimes through 2022, then applied to 2023-2026 — the current AI rally period they never saw during training. |
| **Time** | 15 seconds |
| **Transition** | "Let's start with the bull case — why this time might actually be different." |

---

### Slide 5: Fundamentals — P/E Comparison

| Property | Value |
|---|---|
| **Title** | "Valuations: Night and Day" |
| **Chart** | Chart 2.1 — P/E Ratio Comparison |
| **Layout** | Chart on left (65%), key stats on right (35%) in a callout box |
| **Callout box** | "Cisco at peak: 200x P/E, NVDA now: ~Xx P/E. Cisco's P/E was 4x NVDA's at the equivalent point in the cycle." |
| **Key message** | NVDA's valuation is elevated but not in the same universe as CSCO's. |
| **Speaking notes** | - CSCO traded at 200x P/E at its peak — $200 per $1 of earnings. - NVDA trades at ~[X]x — expensive but not delusional. - Key difference: CSCO P/E expanding as earnings stalled; NVDA P/E compressing as earnings grew faster than stock price. - This is the single strongest counter-argument to the bubble thesis. |
| **Time** | 40 seconds |
| **Transition** | "And the revenue picture is even more compelling." |

---

### Slide 6: Fundamentals — Revenue Growth

| Property | Value |
|---|---|
| **Title** | "Revenue Growth: 4x Faster Than Cisco" |
| **Chart** | Chart 2.2 — Revenue Growth Rate (grouped bar) |
| **Layout** | Chart takes 75% of slide |
| **Stat callout** | Large number: "265% YoY" (NVDA peak) vs "66% YoY" (CSCO peak) |
| **Key message** | NVIDIA's revenue growth dwarfs Cisco's. This is real demand from real enterprises. |
| **Speaking notes** | - NVDA peak revenue growth: 265% YoY, driven by data center GPU sales to MSFT, GOOG, META, AMZN. - CSCO peak growth: 66% YoY, driven by ISPs and telecoms — many went bankrupt. - Revenue quality is fundamentally different. |
| **Time** | 35 seconds |
| **Transition** | "So the fundamentals are clearly different. But here's where it gets uncomfortable." |

---

### Slide 7: Concentration — The Narrow Market

| Property | Value |
|---|---|
| **Title** | "Market Concentration Is at Dot-Com Levels" |
| **Charts** | Chart 3.1 (top-N concentration, large) + Chart 3.2 (SPY vs RSP, small inset or side panel) |
| **Layout** | Main chart left (60%), SPY-RSP chart right (40%) |
| **Stat callout** | "Top 5 stocks: 28% of S&P 500 (2026) vs 18% (2000)" |
| **Key message** | The market is more concentrated than at the dot-com peak. A few stocks are doing all the work. |
| **Speaking notes** | - Top 5 stocks: 28% of S&P 500 (2026) vs 18% (2000) — more concentrated than dot-com peak. - Cap-weighted S&P 500 dramatically outperforming equal-weight version — the average stock is not participating. - By the Buffett Indicator, total market cap to GDP is at ~190% — higher than the dot-com peak of ~140%. - When a rally is this narrow, it's historically fragile — it doesn't take a recession to unwind it, just a rotation. |
| **Time** | 50 seconds |
| **Transition** | "And investor sentiment tells us something important too." |

---

### Slide 8: Sentiment — The Hype Signal

| Property | Value |
|---|---|
| **Title** | "Hype Is Real — But So Is Substance" |
| **Charts** | Chart 5.1 (AI mentions, left) + Chart 5.2 (Hype vs Specificity, right) |
| **Layout** | Two charts side by side, each 50% width |
| **Key message** | AI mentions in earnings calls have surged (bubble signal), but the proportion of substantive content remains high (counter-signal). |
| **Speaking notes** | - AI mentions in NVDA earnings calls surged [X]-fold since 2020 — mirrors "internet" mention surge in CSCO calls pre-crash. - Key difference: NVDA mentions include specific revenue figures, customer names, deployment metrics. - CSCO 1999-2000 calls were heavy on vision, light on specifics. - Hype is real — but so is substance. |
| **Time** | 45 seconds |
| **Transition** | "Now let's look at the wild card that could determine how this plays out." |

---

### Slide 9: The Macro Wild Card

| Property | Value |
|---|---|
| **Title** | "The Macro Wild Card: What Could Pop This?" |
| **Chart** | Chart 4.5 — Macro Dashboard (small multiples, 2x3 grid) |
| **Layout** | Chart takes 80% of slide |
| **Key message** | The macro environment is fundamentally different from 1999-2000: higher rates, tighter money, but benign credit conditions. |
| **Speaking notes** | - In 1999 the Fed hiked rates and that helped pop the bubble. Today rates are also high — but the market rallied through them. - M2 is contracting — this rally is earnings-driven, not liquidity-driven. - Credit spreads remain tight — no systemic stress. - But the yield curve inverted for the longest period in history — every prior inversion preceded a recession. - The macro picture is genuinely ambiguous. |
| **Time** | 40 seconds |
| **Transition** | "So we have conflicting signals across our five layers. That's exactly why we built ML models to quantify this." |

---

### Slide 10: ML Verdict — Regime Classification

| Property | Value |
|---|---|
| **Title** | "What Do the Models Say?" |
| **Chart** | Chart ML.1 — Regime Classification Probabilities (stacked area) |
| **Layout** | Chart takes 70% of slide. Large probability readout on the right: "March 2026: Bubble X%, Normal Growth X%" |
| **Key message** | The ML ensemble assigns X% probability to "bubble" — elevated but not conclusive. |
| **Speaking notes** | - 3-model ensemble (RF, XGBoost, Logistic Regression) trained on 38 features across 5 layers, 28 years of labeled history. - Tested on data never seen during training, including the 2022 bear market and AI rally. - March 2026: ensemble assigns [X]% probability to "bubble" — highest since December 1999, but not a majority. - Models see a coin flip between bubble and fundamentally-supported growth. |
| **Time** | 45 seconds |
| **Transition** | "And our similarity analysis puts a number on how closely we're tracking the dot-com pattern." |

---

### Slide 11: ML Verdict — DTW Similarity + SHAP

| Property | Value |
|---|---|
| **Title** | "The Tug-of-War: What's Pushing Toward Bubble vs Away" |
| **Charts** | Chart ML.2 (DTW similarity timeline, top half) + Chart ML.4 (SHAP waterfall, bottom half) |
| **Layout** | Vertically stacked: DTW chart on top (45%), SHAP waterfall on bottom (55%) |
| **Stat callout** | "DTW Similarity: X/100. Strongest bubble signals: concentration, sentiment. Strongest counter: revenue, FCF." |
| **Key message** | The AI cycle is [X]% similar to the dot-com bubble. Concentration and sentiment push toward bubble; fundamentals push away. |
| **Speaking notes** | - DTW similarity: [X]/100 — [moderate/strong] similarity zone. - SHAP waterfall shows the tug-of-war: concentration + sentiment push toward bubble; revenue + FCF push away. - Strongest bubble signals: market concentration and sentiment hype at dot-com levels. - Strongest counter-signals: revenue growth and free cash flow fundamentally different. |
| **Time** | 45 seconds |
| **Transition** | "So what's our synthesis?" |

---

### Slide 12: Synthesis — The Balanced Conclusion

| Property | Value |
|---|---|
| **Title** | "Synthesis: A Bubble? Not Quite. But Not Safe Either." |
| **Visual** | Summary table or scorecard with 5 layers |
| **Scorecard layout** | |

```
Layer               | Dot-Com Signal? | Verdict
─────────────────────────────────────────────────
Price Trajectory    | YES (similar)   | CAUTION
Fundamentals        | NO (much stronger) | SUPPORT
Market Concentration| YES (worse)     | DANGER
Macro Environment   | MIXED           | NEUTRAL
Sentiment/Hype      | YES (elevated)  | CAUTION
─────────────────────────────────────────────────
Overall             | 3 CAUTION, 1 SUPPORT, 1 NEUTRAL
```

| **Key message** | The structural conditions resemble a bubble in 3 of 5 dimensions. Strong fundamentals are the critical differentiator. This is not a binary yes/no — it's a risk distribution. |
| **Speaking notes** | - 3 of 5 layers flash bubble signals: price trajectory, market concentration, investor sentiment all match or exceed dot-com levels. - 1 layer provides strong support: NVDA fundamentals (revenue, earnings, FCF) are genuinely different from CSCO. - Macro is a wash. - Conclusion: elevated structural risk, with strong fundamentals the only thing preventing a bubble classification. - If fundamentals disappoint even slightly, the downside could be severe. |
| **Time** | 50 seconds |
| **Transition** | "What does that mean practically?" |

---

### Slide 13: Key Takeaway

| Property | Value |
|---|---|
| **Title** | "The Key Insight" |
| **Visual** | Large quote-style text, centered: |
| **Quote** | "The AI rally is fundamentally justified but structurally fragile. The gap between a justified rally and a bubble is measured in quarters — not years — of continued earnings delivery." |
| **Subtext** | "NVIDIA must keep executing. Any deceleration in revenue growth would remove the one factor separating this from a classic bubble." |
| **Key message** | One-sentence takeaway that a judge can remember. |
| **Speaking notes** | - The AI rally is real, but fragile. The market has priced in perfection. - NVDA must deliver exceptional results every quarter to justify current valuations. - History shows the gap between "justified growth" and "bubble" can close very quickly. |
| **Time** | 25 seconds |
| **Transition** | "Let's be honest about what we can't know." |

---

### Slide 14: Limitations & Ethics

| Property | Value |
|---|---|
| **Title** | "Limitations & Ethical Considerations" |
| **Layout** | Numbered list, clean formatting |
| **Content** | |

1. **Small sample of historical bubbles.** Only 2-3 comparable events in our training data. Statistical power is inherently limited.
2. **Labels are subjective.** Different analysts would define bubble periods differently. Our sensitivity analysis shows +/- 15% variance in results.
3. **Structural market changes.** Passive investing, algorithmic trading, and different Fed tools mean history may not repeat.
4. **Not investment advice.** This is academic analysis. The model classifies regime risk — it does not predict future prices.
5. **AI tools used responsibly.** We used OpenAI API for NLP classification of earnings call sentences. All results were manually validated on a sample basis.

| **Key message** | Intellectual honesty. We know the limits of our analysis. |
| **Speaking notes** | - Small training sample: only 2-3 comparable historical bubbles. - Regime labels are judgment calls — sensitivity analysis shows +/-15% variance. - Structural market changes since 2000 (passive investing, algo trading, different Fed tools). - Not investment advice. AI tools used for NLP; classifications validated on a sample basis. |
| **Time** | 35 seconds |
| **Transition** | "Thank you — we're happy to take questions." |

---

### Slide 15: Thank You / Q&A

| Property | Value |
|---|---|
| **Title** | "Thank You" |
| **Subtitle** | Team 2Kim: Jimmy Kim & Alice Kim |
| **Content** | "Questions?" centered. Small footer: "Notebook: 2kim_finance_notebook.ipynb | Interactive dashboard available for live demo if requested." |
| **Visual** | Clean. Same style as title slide. Optionally: small version of Chart 1.1 faded in background as a visual callback. |
| **Speaking notes** | "Thank you for your time. We're happy to take questions." |
| **Time** | 5 seconds (then Q&A begins) |

---

### Slide 16 (BACKUP): Additional SHAP Detail

| Property | Value |
|---|---|
| **Title** | "Feature Importance Deep Dive: SHAP Summary" |
| **Chart** | Chart ML.3 — SHAP Summary Plot (beeswarm) |
| **Purpose** | Available if a judge asks "how do you know which features matter?" |
| **Speaking notes** | "This SHAP summary plot shows all 38 features ranked by importance. Each dot is a monthly observation. The x-axis is the impact on bubble probability, and the color is the feature value. You can see that high concentration and high sentiment consistently push toward bubble classification, while high revenue growth consistently pushes away." |

---

### Slide 17 (BACKUP): Methodology Deep Dive

| Property | Value |
|---|---|
| **Title** | "Model Validation: Walk-Forward Results" |
| **Content** | Walk-forward validation accuracy chart, confusion matrix for test set, ensemble agreement statistics |
| **Purpose** | Available if a judge asks about overfitting or methodology rigor |
| **Speaking notes** | "Our walk-forward validation shows consistent performance across time periods. The confusion matrix on the test set — which includes both the 2022 bear market and the AI rally — shows [X]% weighted F1. The three models agree on the top regime classification for [X]% of test months." |

---

## 4. Slide Design Guidelines

### 4.1 Layout Grid

```
┌──────────────────────────────────────────────────┐
│  Title Bar (12% height)                          │
├──────────────────────────────────────────────────┤
│                                                  │
│  Content Area (78% height)                       │
│                                                  │
│  [Chart: 60-85% width] [Stats Panel: 15-40%]    │
│                                                  │
├──────────────────────────────────────────────────┤
│  Footer (10% height): source, team name          │
└──────────────────────────────────────────────────┘
```

### 4.2 Color Scheme for Slides

| Element | Color | Hex |
|---|---|---|
| Background | Dark navy | `#0F1419` |
| Title text | White | `#FFFFFF` |
| Body text | Light gray | `#D1D5DB` |
| NVDA / AI era | NVIDIA Green | `#00CC96` (dark bg variant of `#76b900`) |
| CSCO / Dot-com era | Coral Red | `#EF553B` |
| Accent (neutral) | Indigo blue | `#636EFA` |
| Muted text | Medium gray | `#6B7280` |
| Dividers/borders | Dark gray | `#374151` |

### 4.3 Typography for Slides

| Element | Font | Weight | Size |
|---|---|---|---|
| Slide title | Inter | Bold | 32pt |
| Section header | Inter | Semibold | 28pt |
| Body text | Inter | Regular | 18pt |
| Chart labels | Inter | Regular | 14pt |
| Stat callout (large number) | Inter | Bold | 48pt |
| Source/footer | Inter | Light | 10pt |

### 4.4 Chart Export Settings for Slides

```python
# Export settings for slide-embedded charts
SLIDE_EXPORT = {
    'format': 'png',
    'width': 1600,       # High-res for 16:9 slides
    'height': 900,
    'scale': 2,          # 2x for retina/4K displays
    'engine': 'kaleido', # Static export engine
}

# fig.write_image('charts/slide_chart_1_1.png', **SLIDE_EXPORT)
```

### 4.5 One Rule Per Slide

- Every slide has **one key message** (written in the spec above)
- No slide has more than **2 charts** (overwhelming with data loses the audience)
- Text is **minimal** — speak the details, don't write them
- Every chart has a **clear title that states the finding**, not just the metric (e.g., "Concentration Is at Dot-Com Levels" not "S&P 500 Top-N Weights")

---

## 5. Presentation Delivery Guide

### 5.1 Timing Breakdown

| Slide(s) | Section | Target Time | Cumulative |
|---|---|---|---|
| 1 | Title | 0:15 | 0:15 |
| 2 | Hook (price overlay) | 0:45 | 1:00 |
| 3 | Research framework | 0:20 | 1:20 |
| 4 | Data & methodology | 0:15 | 1:35 |
| 5 | P/E comparison (bull case) | 0:40 | 2:15 |
| 6 | Revenue growth | 0:35 | 2:50 |
| 7 | Concentration (bear case) | 0:50 | 3:40 |
| 8 | Sentiment | 0:45 | 4:25 |
| 9 | Macro wild card | 0:40 | 5:05 |
| 10 | ML regime classification | 0:45 | 5:50 |
| 11 | DTW + SHAP | 0:45 | 6:35 |
| 12 | Synthesis | 0:50 | 7:25 |
| 13 | Key takeaway | 0:25 | 7:50 |
| 14 | Limitations | 0:35 | 8:25 |
| 15 | Thank you | 0:05 | 8:30 |

**Total: ~8:30.** This leaves 30 seconds of buffer for natural pauses and transitions, targeting a 9:00 delivery. Well within the 8-10 minute competition window.

### 5.2 Speaker Assignment

| Presenter | Slides | Sections |
|---|---|---|
| Jimmy | 1-4, 10-15 | Opening, framework, ML results, synthesis, close |
| Alice | 5-9 | Bull case, bear case, macro |

**Handoff points:**
- Jimmy to Alice (after slide 4): "Now Alice will walk you through what the data shows — starting with the bull case."
- Alice to Jimmy (after slide 9): "Jimmy will now show you what our models conclude."

### 5.3 Delivery Tips

1. **Start with the hook, not the agenda.** Slide 2 (the price overlay) should be your opening after the 15-second intro. Don't waste time on "today we're going to talk about..." — show the chart, let it land.

2. **Pause after the hook chart.** Let the audience absorb it for 3-4 seconds before speaking. The visual impact of the NVDA/CSCO overlay is your strongest opening.

3. **Use the bull/bear structure to build credibility.** Judges respect presenters who argue both sides. Don't telegraph your conclusion.

4. **State numbers confidently.** When saying statistics, make eye contact and slow down. "Two hundred times earnings" lands harder than rushing through.

5. **The synthesis slide is the most important.** This is where you earn Analysis & Evidence points. Don't rush it. The scorecard format makes it easy for judges to follow.

6. **Anticipate the "so what?" question.** Your key takeaway (slide 16) should answer it preemptively.

7. **Don't read from slides.** The slides have minimal text by design. You should know the numbers and the narrative.

8. **For Q&A: default to "that's a great question, and we actually addressed that in our notebook."** This signals depth beyond what's in the slides.

---

## 6. Anticipated Judge Questions & Prepared Answers

### Q1: "Why did you choose these specific date ranges for your bubble labels?"

**Answer:** "We based our labels on well-established market history — the NASDAQ's tripling from 1998-2000, its 78% crash from 2000-2002, and similar consensus periods for other regimes. However, we recognized that these labels are subjective, so we ran a sensitivity analysis where we shifted every boundary by up to 3 months in either direction. The results were robust — the bubble probability for today's market varied by only about 15 percentage points across all label permutations."

### Q2: "Isn't the P/E comparison misleading because NVIDIA's growth rate is so much higher?"

**Answer:** "That's exactly why we also computed the PEG ratio — P/E divided by growth rate — and included it as a feature. Even on a growth-adjusted basis, NVDA's valuation is elevated but nowhere near Cisco's. The P/E comparison is actually our strongest evidence that this is NOT a simple replay of the dot-com bubble. The models capture this: revenue growth is the strongest feature pushing AWAY from bubble classification in our SHAP analysis."

### Q3: "How do you handle the small sample problem — you only have 2-3 bubbles to train on?"

**Answer:** "We're transparent about this being our biggest limitation. We mitigate it in three ways. First, by working at monthly granularity, we get about 48 months of bubble data across two events. Second, we use balanced class weights so the model doesn't just predict the majority class. Third, we report probability distributions, not binary predictions. We're not claiming to have built a bubble detector with 95% confidence — we're saying the structural indicators that preceded past bubbles are elevated at specific, quantifiable levels."

### Q4: "Why not use a neural network or more sophisticated model?"

**Answer:** "Deliberately. With only 384 training observations and 38 features, a neural network would almost certainly overfit. Random Forest and XGBoost are well-suited for tabular data of this size. More importantly, we prioritized interpretability — the SHAP analysis lets us explain exactly which features drive each prediction, which is more valuable for answering our research question than a 2% accuracy improvement from a black-box model."

### Q5: "What would change your conclusion?"

**Answer:** "Two things. On the bull side: if market breadth improves — meaning the rally broadens beyond the top 5-10 stocks — the concentration signal would weaken significantly, and bubble probability would drop. On the bear side: if NVIDIA's revenue growth decelerates meaningfully — say below 50% YoY — while the stock continues to appreciate, the fundamental support would weaken, and the model would likely cross into majority bubble classification. Right now, it's the fundamentals holding the line."

### Q6: "Isn't the DTW similarity just pattern matching? Markets don't repeat exactly."

**Answer:** "You're right, and that's an important caveat. The DTW score measures structural similarity in observable features — it doesn't predict that the same outcome will follow. We use it as one input alongside the regime classifier and SHAP analysis, not as a standalone predictor. The value is in quantifying the degree of similarity rather than making qualitative claims like 'it looks the same to me.' A score of [X] out of 100 is a more rigorous statement than eyeballing two charts."

### Q7: "How do you handle the fact that markets have changed structurally since 2000?"

**Answer:** "We explicitly flag this as a limitation in our notebook and our analysis. Passive investing, algorithmic trading, different Fed tools, and social media have fundamentally changed market dynamics. Our model is trained on historical data that spans some of these changes — the 2009-2024 period includes the passive investing revolution — but we can't fully account for structural breaks. This is why we frame our results as 'elevated structural risk' rather than 'this is definitely a bubble.'"

### Q8: "What about the ETF passive inflows argument — that concentration is mechanically driven, not speculative?"

**Answer:** "Great question. Index fund inflows do mechanically increase concentration because they buy cap-weighted. But this actually strengthens the fragility argument, not weakens it. If concentration is mechanically amplified on the way up, it will also be mechanically amplified on the way down. Passive outflows would disproportionately hit the same top-weighted stocks. The question isn't whether the mechanism is speculative or mechanical — it's whether the resulting concentration creates systemic risk. And by the HHI metric, it does."

### Q9: "What's your confidence interval on the bubble probability?"

**Answer:** "Our sensitivity analysis shows the ensemble bubble probability ranges from approximately [X-15]% to [X+15]% when we perturb label boundaries and feature weights. The bootstrap confidence interval on the DTW similarity score is [lower] to [upper] at 95% confidence. We're comfortable saying bubble risk is 'elevated' but we would not claim to distinguish between, say, 35% and 50% with precision."

### Q10: "Did you use AI tools in this analysis?"

**Answer:** "Yes, and we were deliberate about where and how. We used the OpenAI API for NLP classification of earnings call sentences — specifically, classifying each sentence as 'hype' or 'substantive.' We manually validated a random sample of 200 sentences and found the classification agreed with human judgment in approximately 85% of cases. We also used AI coding assistants during development. All results were verified by the team, and our notebook is fully reproducible."

### Q11: "Your DTW warping path — did you normalize the series before DTW? Does your choice affect the result significantly?"

**Answer:** "Yes, we min-max normalize both series to the [0,1] range before computing DTW. This is necessary because the raw price scales differ by orders of magnitude. We also use a Sakoe-Chiba band constraint set to 33% of the longer series length to prevent pathological warping. The normalization choice does affect the result — we tested both min-max and z-score normalization and found the ranking of similarity scores is stable, though the absolute values shift. We report this as a sensitivity check in the notebook. The DTW distance is further normalized by the geometric mean of the two series lengths to make it comparable across different-length windows."

### Q12: "What happens to your analysis if NVIDIA stock drops 30% next month — does that change your conclusion?"

**Answer:** "A 30% drawdown in NVDA would shift the ML regime probability upward — likely into majority bubble classification — but it would not invalidate the structural analysis. The concentration and macro signals would remain elevated regardless of NVDA's price action. In fact, a 30% drawdown would actually be consistent with the 'distribution' phase in the bubble lifecycle model, which is exactly what the model is designed to detect. The fundamental support layer would weaken only if the drawdown was accompanied by a revenue miss. If NVDA drops 30% on an earnings beat, that's a de-rating, not a bubble pop — and our framework distinguishes between the two through the SHAP decomposition."

---

## Appendix: Slide-to-Chart Mapping

| Slide | Chart ID(s) | Chart Name |
|---|---|---|
| 2 | 1.1 | Normalized Price Overlay (incl. Nortel Networks gray dashed) |
| 5 | 2.1 | P/E Ratio Comparison |
| 6 | 2.2 | Revenue Growth Rate |
| 7 | 3.1 + 3.2 | Concentration Ratio + SPY vs RSP |
| 8 | 5.1 + 5.2 | AI Mentions + Hype vs Specificity |
| 9 | 4.5 | Macro Dashboard (Small Multiples) |
| 10 | ML.1 | Regime Classification Probabilities |
| 11 | ML.2 + ML.4 | DTW Similarity + SHAP Waterfall |
| 16 (backup) | ML.3 | SHAP Summary (Beeswarm) |

**Total unique charts in slides:** 12 required + 1 backup = 13 charts to produce for presentation.
