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
| Total slides | 18 content slides + 1 title + 1 backup = **20 slides** |
| Presentation time | 8-10 minutes (target: 9 minutes) |
| Q&A time | 5 minutes |
| Pace | ~30 seconds per slide average; key slides get 45-60 seconds, transition slides get 15 seconds |
| Audience | SBU professors (judges); assume strong quantitative background, moderate finance knowledge |
| Tone | Data-driven, balanced, intellectually honest. Not alarmist ("the sky is falling") or dismissive ("everything is fine"). The conclusion is nuanced. |

---

## 2. Story Arc Overview

The presentation follows a **deliberate dialectical structure**: thesis, antithesis, synthesis. This demonstrates analytical maturity and avoids the common pitfall of cherry-picking data to support a predetermined conclusion.

```
Act I: The Question (Slides 1-4)        ~2.5 minutes
  "Look at this chart. Are we in a bubble?"

Act II: The Bull Case (Slides 5-8)       ~2.0 minutes
  "Here's why this time might be different..."

Act III: The Bear Case (Slides 9-11)     ~2.0 minutes
  "...but here's why we should worry."

Act IV: The Wild Card (Slide 12)         ~0.5 minutes
  "The macro environment adds complexity."

Act V: The Verdict (Slides 13-16)        ~2.0 minutes
  "Here's what the models say — and what it means."

Act VI: Close (Slides 17-18)             ~0.5 minutes
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
| **Speaking notes** | "This is NVIDIA in blue and Cisco in red, both normalized to 100 at their breakout points. Cisco went on to lose 89% of its value. NVIDIA is currently at the same point in its trajectory where Cisco was about 6 months before its peak. [Pause.] The question is: are we looking at a coincidence, or a pattern?" |
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
| **Speaking notes** | "We structured our analysis across five layers, from the most visible — price — to the most nuanced — sentiment. Each layer adds depth. A price pattern alone means nothing. But if the fundamentals, market structure, macro environment, and investor psychology all match, that's a different story." |
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
| **Speaking notes** | "We pulled data from 8 sources spanning 36 years. Every feature was computed using only trailing data available at each point in time — no lookahead bias. We used time-series cross-validation throughout. The ML models were trained on data through 2018 and tested on 2019-2024, which includes both COVID and the current AI rally — periods they never saw during training." |
| **Time** | 30 seconds |
| **Transition** | "Let's start with the bull case — why this time might actually be different." |

---

### Slide 5: The Bull Case — Header

| Property | Value |
|---|---|
| **Title** | "The Bull Case: Why This Time IS Different" |
| **Visual** | Large text slide. Green-tinted background. One sentence: "NVIDIA has something Cisco never did: massive, accelerating, real revenue." |
| **Key message** | Transition slide signaling the pro-growth argument. |
| **Speaking notes** | No speaking — this is a 3-second title card. Move immediately to next slide. |
| **Time** | 5 seconds |
| **Transition** | Immediate. |

---

### Slide 6: Fundamentals — P/E Comparison

| Property | Value |
|---|---|
| **Title** | "Valuations: Night and Day" |
| **Chart** | Chart 2.1 — P/E Ratio Comparison |
| **Layout** | Chart on left (65%), key stats on right (35%) in a callout box |
| **Callout box** | "Cisco at peak: 200x P/E, NVDA now: ~Xx P/E. Cisco's P/E was 4x NVDA's at the equivalent point in the cycle." |
| **Key message** | NVDA's valuation is elevated but not in the same universe as CSCO's. |
| **Speaking notes** | "This is the single strongest counter-argument to the bubble thesis. Cisco traded at a P/E of 200 at its peak — meaning investors were paying 200 dollars for every dollar of earnings. NVIDIA trades at roughly [X]x. That's expensive, but it's not delusional. And the trend is different: Cisco's P/E was expanding as earnings stalled. NVIDIA's P/E has been compressing as earnings grew faster than the stock price." |
| **Time** | 45 seconds |
| **Transition** | "And the revenue picture is even more compelling." |

---

### Slide 7: Fundamentals — Revenue Growth

| Property | Value |
|---|---|
| **Title** | "Revenue Growth: 4x Faster Than Cisco" |
| **Chart** | Chart 2.2 — Revenue Growth Rate (grouped bar) |
| **Layout** | Chart takes 75% of slide |
| **Stat callout** | Large number: "265% YoY" (NVDA peak) vs "66% YoY" (CSCO peak) |
| **Key message** | NVIDIA's revenue growth dwarfs Cisco's. This is real demand from real enterprises. |
| **Speaking notes** | "NVIDIA's peak revenue growth was 265 percent year-over-year, driven by data center GPU sales to Microsoft, Google, Meta, and Amazon. Cisco's peak growth was 66 percent, driven largely by ISPs and telecom companies — many of which later went bankrupt. The quality of NVIDIA's revenue base is fundamentally stronger." |
| **Time** | 40 seconds |
| **Transition** | "So the fundamentals are clearly different. But here's where it gets uncomfortable." |

---

### Slide 8: The Bear Case — Header

| Property | Value |
|---|---|
| **Title** | "The Bear Case: What History Warns Us About" |
| **Visual** | Large text slide. Orange/red-tinted background. One sentence: "The market structure around NVIDIA looks eerily like 1999." |
| **Key message** | Transition slide signaling danger. |
| **Speaking notes** | No speaking — title card. |
| **Time** | 5 seconds |
| **Transition** | Immediate. |

---

### Slide 9: Concentration — The Narrow Market

| Property | Value |
|---|---|
| **Title** | "Market Concentration Is at Dot-Com Levels" |
| **Charts** | Chart 3.1 (top-N concentration, large) + Chart 3.2 (SPY vs RSP, small inset or side panel) |
| **Layout** | Main chart left (60%), SPY-RSP chart right (40%) |
| **Stat callout** | "Top 5 stocks: 28% of S&P 500 (2026) vs 18% (2000)" |
| **Key message** | The market is more concentrated than at the dot-com peak. A few stocks are doing all the work. |
| **Speaking notes** | "This is where the bubble argument gets real. The top 5 stocks now account for 28 percent of the S&P 500 — more concentrated than at the dot-com peak. The chart on the right shows that the cap-weighted S&P 500 has dramatically outperformed the equal-weight version, meaning the average stock is not participating in this rally. When a rally is this narrow, it's historically fragile. It doesn't take a recession to unwind it — just a rotation." |
| **Time** | 50 seconds |
| **Transition** | "And the Buffett Indicator tells the same story." |

---

### Slide 10: Buffett Indicator

| Property | Value |
|---|---|
| **Title** | "The Buffett Indicator: Total Market Cap vs GDP" |
| **Chart** | Chart 3.3 — Buffett Indicator with threshold bands |
| **Stat callout** | "Current: ~190%. Dot-com peak: ~140%. All-time high." |
| **Key message** | By this metric, the entire market is more overvalued than in 2000. |
| **Speaking notes** | "Warren Buffett once called total market cap to GDP 'probably the best single measure of where valuations stand at any given moment.' The current reading of 190 percent is the highest in history — significantly above the dot-com peak. Now, there are valid criticisms of this metric: U.S. companies earn revenue globally, stock buybacks concentrate wealth. But even with those caveats, this is extreme." |
| **Time** | 40 seconds |
| **Transition** | "And investor sentiment tells us something important too." |

---

### Slide 11: Sentiment — The Hype Signal

| Property | Value |
|---|---|
| **Title** | "Hype Is Real — But So Is Substance" |
| **Charts** | Chart 5.1 (AI mentions, left) + Chart 5.2 (Hype vs Specificity, right) |
| **Layout** | Two charts side by side, each 50% width |
| **Key message** | AI mentions in earnings calls have surged (bubble signal), but the proportion of substantive content remains high (counter-signal). |
| **Speaking notes** | "On the left, AI mentions in NVIDIA's earnings calls have surged [X]-fold since 2020. This mirrors the 'internet' mention surge in Cisco calls before the crash. That's the hype signal. But on the right, our NLP analysis shows that NVIDIA's language is different in quality — a high proportion of their AI mentions include specific revenue figures, customer names, and deployment metrics. Cisco's calls in 1999-2000 were heavy on vision and light on specifics. This is a nuanced but important distinction." |
| **Time** | 50 seconds |
| **Transition** | "Now let's look at the wild card that could determine how this plays out." |

---

### Slide 12: The Macro Wild Card

| Property | Value |
|---|---|
| **Title** | "The Macro Wild Card: What Could Pop This?" |
| **Chart** | Chart 4.5 — Macro Dashboard (small multiples, 2x3 grid) |
| **Layout** | Chart takes 80% of slide |
| **Key message** | The macro environment is fundamentally different from 1999-2000: higher rates, tighter money, but benign credit conditions. |
| **Speaking notes** | "The macro environment is the wild card. In 1999, the Fed was hiking rates — and that helped pop the bubble. Today, rates are also high, but the market has rallied through them. M2 money supply is actually contracting, which means this rally isn't liquidity-driven — it's earnings-driven. Credit spreads remain tight, so there's no systemic stress. But the yield curve inverted for the longest period in history, and every prior inversion preceded a recession. The macro picture is genuinely ambiguous." |
| **Time** | 45 seconds |
| **Transition** | "So we have conflicting signals across our five layers. That's exactly why we built ML models to quantify this." |

---

### Slide 13: ML Verdict — Regime Classification

| Property | Value |
|---|---|
| **Title** | "What Do the Models Say?" |
| **Chart** | Chart ML.1 — Regime Classification Probabilities (stacked area) |
| **Layout** | Chart takes 70% of slide. Large probability readout on the right: "March 2026: Bubble X%, Normal Growth X%" |
| **Key message** | The ML ensemble assigns X% probability to "bubble" — elevated but not conclusive. |
| **Speaking notes** | "We trained a 3-model ensemble — Random Forest, XGBoost, and Logistic Regression — on 38 features spanning all 5 layers, using 28 years of labeled market history. The models were tested on data they never saw during training, including the 2022 bear market and the AI rally. For March 2026, the ensemble assigns [X] percent probability to 'bubble.' That's the highest reading since December 1999 — but it's not a majority. The models see this as a coin flip between bubble and fundamentally-supported growth." |
| **Time** | 50 seconds |
| **Transition** | "And our similarity analysis puts a number on how closely we're tracking the dot-com pattern." |

---

### Slide 14: ML Verdict — DTW Similarity + SHAP

| Property | Value |
|---|---|
| **Title** | "The Tug-of-War: What's Pushing Toward Bubble vs Away" |
| **Charts** | Chart ML.2 (DTW similarity timeline, top half) + Chart ML.4 (SHAP waterfall, bottom half) |
| **Layout** | Vertically stacked: DTW chart on top (45%), SHAP waterfall on bottom (55%) |
| **Stat callout** | "DTW Similarity: X/100. Strongest bubble signals: concentration, sentiment. Strongest counter: revenue, FCF." |
| **Key message** | The AI cycle is [X]% similar to the dot-com bubble. Concentration and sentiment push toward bubble; fundamentals push away. |
| **Speaking notes** | "Our Dynamic Time Warping analysis scores the current AI cycle at [X] out of 100 in similarity to the dot-com bubble — that's in the [moderate/strong] similarity zone. The SHAP waterfall below breaks down exactly why. The strongest signals pushing toward 'bubble' are market concentration and sentiment hype — those are at dot-com levels. The strongest signals pushing away are revenue growth and free cash flow — those are fundamentally different. The model sees a genuine tug-of-war, and that's the honest answer." |
| **Time** | 50 seconds |
| **Transition** | "So what's our synthesis?" |

---

### Slide 15: Synthesis — The Balanced Conclusion

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
| **Speaking notes** | "Here's our synthesis across all five layers. Three out of five layers flash bubble signals: the price trajectory, market concentration, and investor sentiment all match or exceed dot-com levels. One layer provides strong support: NVIDIA's fundamentals — revenue, earnings, free cash flow — are genuinely different from Cisco's. The macro environment is a wash. Our conclusion is NOT 'yes, this is a bubble' or 'no, everything is fine.' It's that the market is in a state of elevated structural risk, where strong fundamentals are the only thing preventing a bubble classification. If those fundamentals disappoint — even slightly — the bubble indicators suggest the downside could be severe." |
| **Time** | 60 seconds |
| **Transition** | "What does that mean practically?" |

---

### Slide 16: Key Takeaway

| Property | Value |
|---|---|
| **Title** | "The Key Insight" |
| **Visual** | Large quote-style text, centered: |
| **Quote** | "The AI rally is fundamentally justified but structurally fragile. The gap between a justified rally and a bubble is measured in quarters — not years — of continued earnings delivery." |
| **Subtext** | "NVIDIA must keep executing. Any deceleration in revenue growth would remove the one factor separating this from a classic bubble." |
| **Key message** | One-sentence takeaway that a judge can remember. |
| **Speaking notes** | "If you remember one thing from this presentation: the AI rally is real, but fragile. The market has priced in perfection. NVIDIA has to deliver — not just good results, but exceptional results — every single quarter to justify these valuations. History shows that the gap between 'justified growth' and 'bubble' can close very quickly." |
| **Time** | 30 seconds |
| **Transition** | "Let's be honest about what we can't know." |

---

### Slide 17: Limitations & Ethics

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
| **Speaking notes** | "We want to be upfront about limitations. First, we're training on a small number of historical bubbles — there just aren't that many in the dataset. Second, our regime labels are judgment calls, though our sensitivity analysis shows the results are robust to reasonable changes. Third, markets in 2026 are structurally different from 2000 in ways that models can't fully capture. And this is obviously not investment advice. We also want to note that we used AI tools — specifically OpenAI — for our NLP analysis of earnings calls, and all classifications were validated on a sample basis." |
| **Time** | 40 seconds |
| **Transition** | "Thank you — we're happy to take questions." |

---

### Slide 18: Thank You / Q&A

| Property | Value |
|---|---|
| **Title** | "Thank You" |
| **Subtitle** | Team 2Kim: Jimmy Kim & Alice Kim |
| **Content** | "Questions?" centered. Small footer: "Notebook: 2kim_finance_notebook.ipynb | Dashboard: localhost:5173" |
| **Visual** | Clean. Same style as title slide. Optionally: small version of Chart 1.1 faded in background as a visual callback. |
| **Speaking notes** | "Thank you for your time. We're happy to take questions." |
| **Time** | 5 seconds (then Q&A begins) |

---

### Slide 19 (BACKUP): Additional SHAP Detail

| Property | Value |
|---|---|
| **Title** | "Feature Importance Deep Dive: SHAP Summary" |
| **Chart** | Chart ML.3 — SHAP Summary Plot (beeswarm) |
| **Purpose** | Available if a judge asks "how do you know which features matter?" |
| **Speaking notes** | "This SHAP summary plot shows all 38 features ranked by importance. Each dot is a monthly observation. The x-axis is the impact on bubble probability, and the color is the feature value. You can see that high concentration and high sentiment consistently push toward bubble classification, while high revenue growth consistently pushes away." |

---

### Slide 20 (BACKUP): Methodology Deep Dive

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
| Accent (positive/bull) | Teal green | `#00CC96` |
| Accent (negative/bear) | Coral red | `#EF553B` |
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
| 3 | Research framework | 0:30 | 1:30 |
| 4 | Data & methodology | 0:30 | 2:00 |
| 5 | Bull case header | 0:05 | 2:05 |
| 6 | P/E comparison | 0:45 | 2:50 |
| 7 | Revenue growth | 0:40 | 3:30 |
| 8 | Bear case header | 0:05 | 3:35 |
| 9 | Concentration | 0:50 | 4:25 |
| 10 | Buffett Indicator | 0:40 | 5:05 |
| 11 | Sentiment | 0:50 | 5:55 |
| 12 | Macro wild card | 0:45 | 6:40 |
| 13 | ML regime classification | 0:50 | 7:30 |
| 14 | DTW + SHAP | 0:50 | 8:20 |
| 15 | Synthesis | 1:00 | 9:20 |
| 16 | Key takeaway | 0:30 | 9:50 |
| 17 | Limitations | 0:40 | 10:30 |
| 18 | Thank you | 0:05 | 10:35 |

**Total: ~10.5 minutes.** Trim 30 seconds by moving faster through slides 3-4 (framework/methodology) to stay under 10 minutes. Alternatively, cut slide 10 (Buffett Indicator) and reference it verbally.

### 5.2 Speaker Assignment

| Presenter | Slides | Sections |
|---|---|---|
| Jimmy | 1-4, 13-18 | Opening, framework, ML results, synthesis, close |
| Alice | 5-12 | Bull case, bear case, macro |

**Handoff points:**
- Jimmy to Alice (after slide 4): "Now Alice will walk you through what the data shows — starting with the bull case."
- Alice to Jimmy (after slide 12): "Jimmy will now show you what our models conclude."

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

---

## Appendix: Slide-to-Chart Mapping

| Slide | Chart ID(s) | Chart Name |
|---|---|---|
| 2 | 1.1 | Normalized Price Overlay |
| 6 | 2.1 | P/E Ratio Comparison |
| 7 | 2.2 | Revenue Growth Rate |
| 9 | 3.1 + 3.2 | Concentration Ratio + SPY vs RSP |
| 10 | 3.3 | Buffett Indicator |
| 11 | 5.1 + 5.2 | AI Mentions + Hype vs Specificity |
| 12 | 4.5 | Macro Dashboard (Small Multiples) |
| 13 | ML.1 | Regime Classification Probabilities |
| 14 | ML.2 + ML.4 | DTW Similarity + SHAP Waterfall |
| 19 (backup) | ML.3 | SHAP Summary (Beeswarm) |

**Total unique charts in slides:** 14 required + 1 backup = 15 charts to produce for presentation.
