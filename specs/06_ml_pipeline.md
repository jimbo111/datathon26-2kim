# 06 — ML Pipeline Specification

> **Project:** The AI Hype Cycle — Are we in a bubble? NVIDIA 2023-2026 vs Cisco 1998-2001
> **Team:** 2Kim | SBU AI Community Datathon 2026 — Finance & Economics Track
> **Last updated:** 2026-03-28

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [ML Model 1 — Regime Classification](#2-ml-model-1--regime-classification)
3. [ML Model 2 — Similarity Scoring (DTW)](#3-ml-model-2--similarity-scoring-dtw)
4. [ML Model 3 — Feature Importance Analysis (SHAP)](#4-ml-model-3--feature-importance-analysis-shap)
5. [Model Validation & Robustness](#5-model-validation--robustness)
6. [Pipeline Orchestration](#6-pipeline-orchestration)
7. [Limitations & Caveats](#7-limitations--caveats)
8. [Dependencies & Environment](#8-dependencies--environment)

---

## 1. Pipeline Overview

The ML pipeline serves one purpose: **answer the research question with quantitative rigor.** Three models attack the question from different angles:

| Model | Question it answers | Output |
|---|---|---|
| Regime Classifier | "What market state are we in right now?" | Probability distribution: `{bubble, correction, normal_growth, recovery}` |
| DTW Similarity | "How similar is the current AI cycle to the dot-com bubble?" | Similarity score 0-100 |
| SHAP Feature Importance | "Which indicators drive the bubble signal?" | Ranked feature list with direction |

All three models draw features from the same unified feature matrix built from all 5 data layers. None of these models predict future prices — they classify the present state against historical patterns.

### Pipeline DAG

```
Layer 1 (Price) ──┐
Layer 2 (Fundamentals) ──┤
Layer 3 (Concentration) ──┼── Feature Engineering ── Unified Feature Matrix
Layer 4 (Macro) ──┤              │
Layer 5 (Sentiment) ──┘              │
                                     ├── Model 1: Regime Classifier ── Regime Probabilities
                                     ├── Model 2: DTW Similarity ── Similarity Score
                                     └── Model 3: SHAP Analysis ── Feature Rankings
```

---

## 2. ML Model 1 — Regime Classification

### 2.1 Objective

Classify each monthly observation of the U.S. equity market into one of four regimes:

| Regime | Definition | Characteristics |
|---|---|---|
| `bubble` | Unsustainable price appreciation driven by speculation, disconnected from fundamentals | High P/E, rising concentration, elevated sentiment, momentum acceleration |
| `correction` | Post-bubble decline or bear market | Falling prices, rising credit spreads, negative momentum, capitulation sentiment |
| `normal_growth` | Fundamentally-supported appreciation | Moderate P/E, broad market participation, stable macro |
| `recovery` | Rebound from correction, early cycle | Improving fundamentals, low valuations, accommodative policy |

### 2.2 Training Label Definitions

Historical periods are labeled using well-established market history. Each label is assigned at **monthly granularity** (end-of-month snapshots).

| Period | Start Date | End Date | Label | Justification |
|---|---|---|---|---|
| Dot-com run-up | 1998-01-01 | 2000-03-31 | `bubble` | NASDAQ tripled; P/E of tech sector exceeded 60x; IPO frenzy; Alan Greenspan's "irrational exuberance" (1996) ignored for 3+ years |
| Dot-com crash | 2000-04-01 | 2002-10-31 | `correction` | NASDAQ fell 78% from peak; $5T in market cap destroyed; hundreds of dot-com bankruptcies |
| Mid-2000s expansion | 2003-01-01 | 2006-12-31 | `normal_growth` | Moderate valuations, broad earnings recovery, GDP growth 2.5-3.5% |
| Housing bubble late stage | 2007-01-01 | 2007-10-31 | `bubble` | S&P 500 hitting highs on leverage; credit conditions deteriorating; subprime cracks visible |
| Global Financial Crisis | 2007-11-01 | 2009-03-31 | `correction` | S&P 500 fell 57% from peak; Lehman collapse Sep 2008; credit freeze; VIX > 80 |
| Post-GFC recovery | 2009-04-01 | 2013-12-31 | `recovery` | QE-driven rebound; valuations normalized; slow earnings recovery |
| Mid-2010s expansion | 2014-01-01 | 2019-12-31 | `normal_growth` | Steady earnings growth, moderate P/E (16-20x), broadening participation |
| COVID crash | 2020-02-01 | 2020-03-31 | `correction` | 34% drop in 23 trading days; exogenous shock; VIX > 80 |
| COVID recovery / QE euphoria | 2020-04-01 | 2021-12-31 | `recovery` | Unprecedented fiscal/monetary stimulus; V-shaped recovery; meme stock mania was retail-driven but fundamentals improved |
| 2022 bear market | 2022-01-01 | 2022-10-31 | `correction` | Fed rate hikes; S&P 500 down 25%; tech-led decline; inflation shock |
| AI bull market (early) | 2022-11-01 | 2024-06-30 | `normal_growth` | ChatGPT catalyst (Nov 2022); NVDA earnings inflection real; revenue backing the story |
| AI concentration phase | 2024-07-01 | 2026-03-28 | **???** | **This is what we classify.** The model's answer to our research question. |

**Pre-1998 data (1990-1997):** Labeled as `normal_growth` to provide baseline. The 1990-91 recession period (Jul 1990 - Mar 1991) labeled as `correction`; subsequent recovery (Apr 1991 - Dec 1994) as `recovery`.

**Total labeled observations:** ~384 monthly observations (Jan 1990 - Jun 2024 for training) + ~21 months for live classification (Jul 2024 - Mar 2026).

### 2.3 Feature Engineering

All features are computed at **monthly frequency** (end-of-month snapshot). Features with higher frequency (daily price data) are aggregated to monthly.

#### Layer 1: Price Features

| Feature | Formula | Window | Rationale |
|---|---|---|---|
| `momentum_30d` | `(price_t / price_{t-21}) - 1` | 21 trading days | Short-term momentum; bubble phases show persistent positive momentum |
| `momentum_90d` | `(price_t / price_{t-63}) - 1` | 63 trading days | Medium-term trend strength |
| `momentum_252d` | `(price_t / price_{t-252}) - 1` | 252 trading days | Annual return; >40% in consecutive years is historically rare outside bubbles |
| `volatility_30d` | `std(daily_returns, window=21) * sqrt(252)` | 21 trading days | Annualized realized vol; late-stage bubbles show rising vol |
| `volatility_90d` | `std(daily_returns, window=63) * sqrt(252)` | 63 trading days | Smoothed vol estimate |
| `drawdown_from_ath` | `(price_t / max(price_{0:t})) - 1` | Expanding | Distance from all-time high; 0 = at ATH, -0.5 = 50% below |
| `rsi_14` | Standard RSI formula | 14 days | Overbought (>70) / oversold (<30) oscillator |
| `price_to_sma200` | `price_t / SMA(price, 200)` | 200 days | Distance from long-term trend; >1.3 is extended |
| `return_skewness` | `skew(daily_returns, window=63)` | 63 days | Negative skew increases in bubble tops |

**Applied to:** S&P 500 (primary), NASDAQ Composite, NVDA (for current period), CSCO (for dot-com period).

For the regime classifier, we use **S&P 500** as the primary price series to maintain consistency across eras.

#### Layer 2: Fundamental Features

| Feature | Source | Rationale |
|---|---|---|
| `log_pe_sp500` | log(S&P 500 trailing P/E) -- use log transform, NOT clipped P/E | Core valuation metric; log transform handles right-skew and preserves extreme values (see Layer 2 fix). P/E > 25 historically rare outside bubbles |
| `pe_percentile` | Percentile rank of current P/E within expanding 10-year window | Normalizes for secular P/E drift |
| `cape_shiller` | Cyclically-adjusted P/E (10-year avg earnings) | Smooths earnings cycle; >30 historically dangerous |
| `revenue_growth_yoy` | YoY revenue growth of S&P 500 or focus stock | Real revenue backing vs. pure speculation |
| `fcf_yield_sp500` | Free cash flow / market cap for S&P 500 | Low FCF yield = high valuations; negative = danger zone |
| `earnings_growth_yoy` | YoY EPS growth for S&P 500 | Separates earnings-driven rallies from multiple expansion |
| `pe_to_growth` | P/E ratio / earnings growth rate (PEG ratio) | PEG > 2 suggests overvaluation even with growth |

#### Layer 3: Market Concentration Features

| Feature | Source | Rationale |
|---|---|---|
| `top5_weight_sp500` | Sum of top 5 stock weights in S&P 500 | >25% historically extreme; currently ~28% (2026) |
| `top10_weight_sp500` | Sum of top 10 stock weights in S&P 500 | >35% is dot-com territory |
| `hhi_sp500` | Herfindahl-Hirschman Index of S&P 500 weights | Concentration measure; HHI > 200 is concerning |
| `spy_rsp_spread` | `return(SPY) - return(RSP)` trailing 12 months | SPY (cap-weighted) vs RSP (equal-weight); large positive spread = narrow leadership |
| `buffett_indicator` | Total market cap / GDP | >150% historically elevated; dot-com peak was ~140% |
| `sector_concentration` | Weight of top sector (currently tech+comm) | >35% in one sector raises concentration risk |

#### Layer 4: Macro Features

| Feature | Source | Rationale |
|---|---|---|
| `fed_funds_rate` | FRED (DFF) | Rate environment; 0% = accommodative, >5% = restrictive |
| `fed_funds_12m_change` | Change in fed funds rate over 12 months | Rate of change matters more than level |
| `m2_growth_yoy` | FRED (M2SL) YoY % change | Money supply growth; >20% (2020-21) historically extreme |
| `yield_curve_10y2y` | 10Y Treasury - 2Y Treasury spread | Negative = inverted = recession signal |
| `yield_curve_10y3m` | 10Y Treasury - 3M T-bill spread | Alternative inversion metric |
| `credit_spread_baa` | Moody's BAA corporate - 10Y Treasury | Widening spreads = credit stress |
| `credit_spread_hy` | ICE BofA HY spread (if available) | High-yield spreads signal risk appetite |
| `real_rate` | Fed funds rate - CPI YoY | Positive real rate = truly restrictive |
| `vix_monthly_avg` | Average VIX for the month | Fear gauge; low VIX + high prices = complacency |

#### Layer 5: Sentiment Features

| Feature | Source | Rationale |
|---|---|---|
| `ai_mention_rate` | Count of "AI"/"artificial intelligence" per 10K words in NVDA earnings calls | Hype quantification; rising mention rate = marketing narrative |
| `hype_score` | OpenAI classification: `hype / (hype + substance)` ratio from earnings call sentences | Separates visionary claims from concrete revenue discussion |
| `specificity_score` | `1 - hype_score`; proportion of substantive claims | Higher = more grounded |
| `google_trends_ai` | Google Trends index for "artificial intelligence" (normalized 0-100) | Public interest proxy; dot-com analogue = "internet" searches |
| `google_trends_bubble` | Google Trends for "stock market bubble" | Contrarian signal; peaks near actual tops |
| `reddit_sentiment` | Mean compound sentiment from r/wallstreetbets + r/investing (VADER or OpenAI) | Retail investor mood |
| `reddit_volume` | Post count mentioning NVDA/AI per week | Engagement proxy |

#### Feature Summary

| Layer | Feature Count |
|---|---|
| Price | 9 |
| Fundamentals | 7 |
| Concentration | 6 |
| Macro | 9 |
| Sentiment | 7 |
| **Total** | **38** |

**Feature matrix shape:** `(384, 38)` for training — 384 monthly observations (Jan 1990 - Jun 2024) x 38 features.

**Note on Sentiment Features (REVISED -- DO NOT IMPUTE):**

Sentiment features (Layer 5) are only available from ~2004 (Google Trends) or ~2012 (Reddit). **Do NOT impute pre-2004 sentiment with median values.** Median imputation introduces two problems: (1) the median is computed from 2012+ data, leaking future information into pre-2004 observations, and (2) a binary `has_sentiment_data` flag teaches the model that "data before 2004 = no sentiment" as a proxy for the era itself.

**Instead, train TWO models:**

| Model | Features | Training Period | Purpose |
|-------|----------|-----------------|---------|
| **Core Model** | 31 features (price + fundamentals + concentration + macro) | Jan 1990 - Jun 2024 (full timeline) | Primary regime classifier -- uses only universally available features |
| **Augmented Model** | 38 features (core + sentiment) | Jan 2004 - Jun 2024 (post-Google Trends) | Secondary model -- tests whether sentiment features change the classification |

Compare their bubble probability for March 2026 as a sensitivity check. If both models agree (within 5-10pp), the finding is robust regardless of sentiment data. If they diverge significantly, report both and discuss which features drive the divergence.

### 2.4 Preprocessing

```python
# Pseudocode: Feature matrix construction
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def build_feature_matrix(price_df, fundamental_df, concentration_df, macro_df, sentiment_df):
    """Merge all layers into a single monthly feature matrix."""

    # Resample all to month-end frequency
    features = (
        price_df.resample('ME').last()
        .join(fundamental_df.resample('ME').last())
        .join(concentration_df.resample('ME').last())
        .join(macro_df.resample('ME').last())
        .join(sentiment_df.resample('ME').last(), how='left')  # left join: sentiment has gaps
    )

    # DO NOT impute missing sentiment -- train two separate models instead.
    # Core model: drop sentiment columns entirely (available 1990-present)
    # Augmented model: keep sentiment columns, restrict to post-2004 data only
    sentiment_cols = ['ai_mention_rate', 'hype_score', 'specificity_score',
                      'google_trends_ai', 'google_trends_bubble',
                      'reddit_sentiment', 'reddit_volume']

    # Return both versions
    features_core = features.drop(columns=sentiment_cols, errors='ignore')
    features_augmented = features.dropna(subset=sentiment_cols, how='all')  # Post-2004 only

    # Forward-fill macro data (released with lag)
    macro_cols = ['fed_funds_rate', 'm2_growth_yoy', 'credit_spread_baa']
    features[macro_cols] = features[macro_cols].ffill(limit=3)

    # Drop rows with remaining NaN (early period bootstrap)
    features = features.dropna()

    return features

# Labels
labels = assign_regime_labels(features.index)  # Series of {bubble, correction, normal_growth, recovery}
```

### 2.5 Model Selection

#### Primary Model: Random Forest Classifier

**Why:** Interpretable feature importances, robust to feature scaling, handles class imbalance via class weights, no assumption of linearity.

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,                    # Prevent overfitting on small dataset
    min_samples_leaf=5,             # At least 5 observations per leaf
    min_samples_split=10,
    max_features='sqrt',            # sqrt(38) ~ 6 features per split
    class_weight='balanced',        # Handle class imbalance
    random_state=42,
    n_jobs=-1,
    oob_score=True                  # Out-of-bag estimate for free validation
)
```

**Hyperparameter grid:**

```python
rf_param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [5, 8, 12, None],
    'min_samples_leaf': [3, 5, 10],
    'min_samples_split': [5, 10, 20],
    'max_features': ['sqrt', 'log2', 0.3],
    'class_weight': ['balanced', 'balanced_subsample'],
}
```

#### Secondary Model: XGBoost Classifier

**Why:** Typically higher accuracy than RF, built-in handling of missing values, regularization to prevent overfitting.

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=5,                     # Shallower than RF — XGB builds sequentially
    learning_rate=0.05,              # Slow learning for better generalization
    subsample=0.8,                   # Row sampling per tree
    colsample_bytree=0.7,           # Feature sampling per tree
    min_child_weight=5,
    reg_alpha=0.1,                   # L1 regularization
    reg_lambda=1.0,                  # L2 regularization
    scale_pos_weight=1,              # Will be set dynamically based on class ratios
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    # NOTE: early_stopping_rounds REMOVED from constructor.
    # early_stopping_rounds requires eval_set to be passed via fit(),
    # which GridSearchCV does NOT do by default. Including it here will
    # either throw an error or silently ignore early stopping during grid search.
    # Use early stopping ONLY in the final training step (after grid search)
    # by calling fit() with eval_set explicitly:
    #   best_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)],
    #                early_stopping_rounds=50, verbose=False)
)
```

**Hyperparameter grid:**

```python
xgb_param_grid = {
    'n_estimators': [300, 500, 800],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'min_child_weight': [3, 5, 10],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0.5, 1.0, 2.0],
}
```

#### Baseline Model: Logistic Regression

**Why:** Simple baseline to verify that the tree models actually learn something beyond linear separability.

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,                          # Regularization strength
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
)
```

**Hyperparameter grid:**

```python
lr_param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'solver': ['lbfgs', 'saga'],
    'penalty': ['l2'],               # l1 only with saga
}
```

### 2.6 Price-Feature-Excluded Model Variant (REQUIRED)

**Purpose:** Address the circular reasoning critique (CRITIQUE_RIGOR Issue 2). The training labels are defined using known price outcomes. Price-derived features (`momentum_*`, `drawdown_from_ath`, `rsi_14`, `price_to_sma200`, `volatility_*`) encode the same price trajectory information used to define the labels. To test whether the model is learning something beyond this tautology, train a variant that excludes ALL price-derived features.

**Implementation:** Train two models side-by-side:

| Model | Features Included | Purpose |
|-------|-------------------|---------|
| **Full Model** | All 38 features | Primary result |
| **Price-Excluded Model** | 29 features (exclude: `momentum_30d`, `momentum_90d`, `momentum_252d`, `volatility_30d`, `volatility_90d`, `drawdown_from_ath`, `rsi_14`, `price_to_sma200`, `return_skewness`) | Tests whether bubble signal persists without price features |

```python
PRICE_FEATURES = [
    'momentum_30d', 'momentum_90d', 'momentum_252d',
    'volatility_30d', 'volatility_90d',
    'drawdown_from_ath', 'rsi_14', 'price_to_sma200', 'return_skewness',
]

# Train price-excluded model
X_train_no_price = X_train.drop(columns=PRICE_FEATURES)
X_live_no_price = X_live.drop(columns=PRICE_FEATURES)

rf_no_price = RandomForestClassifier(**best_rf.get_params())
rf_no_price.fit(scaler_no_price.fit_transform(X_train_no_price), y_train)

# Compare bubble probability with and without price features
probs_full = best_rf.predict_proba(X_live_scaled[-1:])[0]
probs_no_price = rf_no_price.predict_proba(
    scaler_no_price.transform(X_live_no_price.iloc[-1:])
)[0]

bubble_idx = list(classes).index('bubble')
print(f"Bubble probability (full model): {probs_full[bubble_idx]:.1%}")
print(f"Bubble probability (no price features): {probs_no_price[bubble_idx]:.1%}")
print(f"Difference: {abs(probs_full[bubble_idx] - probs_no_price[bubble_idx]):.1%}")
```

**Interpretation:**
- If bubble probability barely changes (within 5-10pp), the bubble signal comes from fundamentals, concentration, macro, and sentiment -- a much stronger claim.
- If bubble probability drops dramatically (>15pp), the circular reasoning is the main driver -- acknowledge this explicitly in the limitations.
- **Present both models** in the notebook and slides. This is the single most impactful robustness check for the regime classifier's credibility.

### 2.7 Train/Test Split Strategy

**CRITICAL: Never random split time series data.** Future data leaks into training with random splits.

```
Timeline:  1990 ──────────── 2018 ── 2022 ──── 2024.06 ── 2026.03
           |── Train ────────|  |─Val─|  |─Test─|  |─Live──|
```

| Split | Period | Observations | Purpose |
|---|---|---|---|
| Train | Jan 1990 - Dec 2018 | ~348 months | Model fitting |
| Validation | Jan 2019 - Dec 2021 | ~36 months | Hyperparameter tuning (includes COVID — stress test) |
| Test | Jan 2022 - Jun 2024 | ~30 months | Final evaluation (2022 bear + AI rally — model never saw these) |
| Live | Jul 2024 - Mar 2026 | ~21 months | **The answer** — classify current period |

### 2.8 Cross-Validation Strategy

Use `TimeSeriesSplit` to respect temporal ordering during hyperparameter search.

```python
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

tscv = TimeSeriesSplit(
    n_splits=5,
    gap=3,                # 3-month gap between train and validation to prevent leakage
    max_train_size=None,  # Expanding window
    test_size=24,         # 24-month validation windows
)

# Grid search with time series CV
grid_search_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    cv=tscv,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    refit=True,
)

grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
```

**Fold structure with gap=3, test_size=24:**

```
Fold 1: Train [1990-01 to 2005-12] | Gap [2006-01 to 2006-03] | Val [2006-04 to 2008-03]
Fold 2: Train [1990-01 to 2008-03] | Gap [2008-04 to 2008-06] | Val [2008-07 to 2010-06]
Fold 3: Train [1990-01 to 2010-06] | Gap [2010-07 to 2010-09] | Val [2010-10 to 2012-09]
Fold 4: Train [1990-01 to 2012-09] | Gap [2012-10 to 2012-12] | Val [2013-01 to 2014-12]
Fold 5: Train [1990-01 to 2014-12] | Gap [2015-01 to 2015-03] | Val [2015-04 to 2017-03]
```

### 2.9 Evaluation Metrics

#### Primary Metrics

| Metric | Target | Why |
|---|---|---|
| Weighted F1 | > 0.70 on test set | Handles class imbalance; weighted by class support |
| Per-class F1 | Report all 4 | We care most about `bubble` class precision (false alarm cost) |
| Accuracy | Report but don't optimize | Misleading with imbalanced classes |
| Confusion matrix | Visual | Shows which regimes are confused (e.g., `bubble` vs `normal_growth`) |

#### Classification Report Template

```
                  precision    recall  f1-score   support
       bubble        0.XX      0.XX      0.XX        N
   correction        0.XX      0.XX      0.XX        N
normal_growth        0.XX      0.XX      0.XX        N
     recovery        0.XX      0.XX      0.XX        N

     accuracy                            0.XX        N
    macro avg        0.XX      0.XX      0.XX        N
 weighted avg        0.XX      0.XX      0.XX        N
```

#### What We Expect

- `bubble` and `correction` should be the easiest to classify (extreme feature values)
- `normal_growth` vs `recovery` will likely be confused (similar features, different trajectory)
- Acceptable if `bubble` precision > 0.80 even if `recovery` recall is low

### 2.10 The Punchline: "What Does the Model Classify TODAY As?"

This is the money shot for the presentation. After training and validation:

```python
# Get features for Mar 2026 (latest observation)
X_today = feature_matrix.loc['2026-03']  # Shape: (1, 38)

# Predict with all three models
rf_probs = best_rf.predict_proba(X_today.values.reshape(1, -1))[0]
xgb_probs = best_xgb.predict_proba(X_today.values.reshape(1, -1))[0]
lr_probs = best_lr.predict_proba(X_today.values.reshape(1, -1))[0]

# Ensemble average (simple equal weight)
ensemble_probs = (rf_probs + xgb_probs + lr_probs) / 3

classes = ['bubble', 'correction', 'normal_growth', 'recovery']
for cls, prob in zip(classes, ensemble_probs):
    print(f"  {cls:>15s}: {prob:.1%}")
```

**Expected output format (hypothetical):**

```
Market Regime Classification — March 2026 (Ensemble)
         bubble:  42.3%
     correction:  12.1%
  normal_growth:  38.7%
       recovery:   6.9%

Verdict: ELEVATED BUBBLE RISK — not conclusive, but highest bubble probability
         since Dec 1999.
```

**Rolling classification:** Run the classifier on each month from Jul 2024 - Mar 2026 to show the **trajectory** of bubble probability. Is it increasing? Stabilizing? This is more informative than a single point estimate.

```python
# Rolling regime probabilities for live period
live_period = feature_matrix.loc['2024-07':'2026-03']
live_probs = pd.DataFrame(
    best_rf.predict_proba(live_period),
    index=live_period.index,
    columns=classes
)
# Plot as stacked area chart (see Chart ML.1 in 07_visualization_plan.md)
```

### 2.11 Full Training Pipeline Code Outline

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# ──────────────────────────────────────────────
# 1. Load and merge feature matrix
# ──────────────────────────────────────────────
features = build_feature_matrix(price_df, fundamental_df, concentration_df, macro_df, sentiment_df)
labels = assign_regime_labels(features.index)

# Align
common_idx = features.index.intersection(labels.index)
X = features.loc[common_idx]
y = labels.loc[common_idx]

# ──────────────────────────────────────────────
# 2. Time-based splits
# ──────────────────────────────────────────────
train_mask = X.index <= '2018-12-31'
val_mask = (X.index >= '2019-01-01') & (X.index <= '2021-12-31')
test_mask = (X.index >= '2022-01-01') & (X.index <= '2024-06-30')
live_mask = X.index >= '2024-07-01'

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]
X_live = X[live_mask]

# ──────────────────────────────────────────────
# 3. Scale features (fit on train only)
# ──────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_live_scaled = scaler.transform(X_live)

# ──────────────────────────────────────────────
# 4. Hyperparameter tuning (on train set with TimeSeriesSplit)
# ──────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5, gap=3, test_size=24)

# Random Forest
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                       rf_param_grid, cv=tscv, scoring='f1_weighted', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_

# XGBoost
grid_xgb = GridSearchCV(XGBClassifier(random_state=42, eval_metric='mlogloss',
                                       use_label_encoder=False),
                         xgb_param_grid, cv=tscv, scoring='f1_weighted', n_jobs=-1)
grid_xgb.fit(X_train_scaled, y_train)
best_xgb = grid_xgb.best_estimator_

# Logistic Regression
grid_lr = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000,
                                          class_weight='balanced'),
                        lr_param_grid, cv=tscv, scoring='f1_weighted', n_jobs=-1)
grid_lr.fit(X_train_scaled, y_train)
best_lr = grid_lr.best_estimator_

# ──────────────────────────────────────────────
# 5. Evaluate on validation set
# ──────────────────────────────────────────────
for name, model in [('RF', best_rf), ('XGB', best_xgb), ('LR', best_lr)]:
    y_pred = model.predict(X_val_scaled)
    print(f"\n=== {name} — Validation Set ===")
    print(classification_report(y_val, y_pred, target_names=classes))

# ──────────────────────────────────────────────
# 6. Final evaluation on test set (report ONCE)
# ──────────────────────────────────────────────
for name, model in [('RF', best_rf), ('XGB', best_xgb), ('LR', best_lr)]:
    y_pred = model.predict(X_test_scaled)
    print(f"\n=== {name} — Test Set ===")
    print(classification_report(y_test, y_pred, target_names=classes))
    print(confusion_matrix(y_test, y_pred))

# ──────────────────────────────────────────────
# 7. Live classification (the punchline)
# ──────────────────────────────────────────────
ensemble_probs = np.mean([
    best_rf.predict_proba(X_live_scaled),
    best_xgb.predict_proba(X_live_scaled),
    best_lr.predict_proba(X_live_scaled),
], axis=0)

live_predictions = pd.DataFrame(ensemble_probs, index=X_live.index, columns=classes)
print("\n=== LIVE REGIME CLASSIFICATION ===")
print(live_predictions.round(3))

# ──────────────────────────────────────────────
# 8. Save artifacts
# ──────────────────────────────────────────────
joblib.dump(best_rf, 'models/regime_rf.joblib')
joblib.dump(best_xgb, 'models/regime_xgb.joblib')
joblib.dump(best_lr, 'models/regime_lr.joblib')
joblib.dump(scaler, 'models/feature_scaler.joblib')
live_predictions.to_csv('results/regime_probabilities.csv')
```

---

## 3. ML Model 2 — Similarity Scoring (DTW)

### 3.1 Objective

Quantify on a **0-100 scale** how similar the current AI/NVIDIA cycle (2023-2026) is to the dot-com/Cisco cycle (1998-2001) using Dynamic Time Warping.

**Why DTW over simple correlation?** Bubbles do not progress at the same speed. DTW aligns two time series by warping the time axis, allowing phase shifts. The dot-com bubble took ~24 months from inflection to peak; the AI cycle may be faster or slower. DTW captures the shape similarity regardless of timing.

### 3.2 Input Time Series

Two multivariate time series, aligned by "months since inflection point":

| Feature | Dot-Com Era Series | AI Era Series | Normalization |
|---|---|---|---|
| Price trajectory | CSCO monthly close, Jan 1998 - Dec 2001 (48 months) | NVDA monthly close, Jan 2023 - Mar 2026 (39 months) | Min-max to [0, 1] within each series |
| P/E ratio | CSCO trailing P/E, same period | NVDA trailing P/E, same period | Min-max to [0, 1] |
| Revenue growth YoY | CSCO quarterly revenue growth | NVDA quarterly revenue growth | Min-max to [0, 1] |
| Market cap concentration | Tech sector % of S&P 500 | Tech sector % of S&P 500 | Min-max to [0, 1] |
| Fed funds rate | Monthly fed funds rate | Monthly fed funds rate | Min-max to [0, 1] |
| Sentiment/hype proxy | CSCO media mention index (if available) or sector ETF inflows | AI mention rate + Google Trends composite | Min-max to [0, 1] |

**Alignment:** Month 0 = the month the stock first broke out of its prior trading range.
- CSCO: Month 0 = Jan 1998 (start of parabolic move)
- NVDA: Month 0 = Jan 2023 (ChatGPT catalyst, earnings inflection)

### 3.3 DTW Computation

```python
from dtaidistance import dtw
import numpy as np

def compute_dtw_similarity(series_dotcom, series_ai, window=None):
    """
    Compute DTW distance between two normalized 1D time series.
    Returns similarity score in [0, 100].

    Args:
        series_dotcom: np.array, shape (T1,), normalized to [0,1]
        series_ai: np.array, shape (T2,), normalized to [0,1]
        window: Sakoe-Chiba band width (None = no constraint)

    Returns:
        similarity: float in [0, 100]
    """
    # DTW distance (lower = more similar)
    distance = dtw.distance(
        series_dotcom.astype(np.float64),
        series_ai.astype(np.float64),
        window=window,
        use_pruning=True,
    )

    # Normalize by path length (geometric mean of series lengths)
    path_length = np.sqrt(len(series_dotcom) * len(series_ai))
    normalized_distance = distance / path_length

    # Convert to similarity: 100 = identical, 0 = maximally different
    # Max possible normalized distance for [0,1] series is 1.0
    similarity = max(0, (1 - normalized_distance)) * 100

    return similarity


def compute_composite_similarity(features_dotcom, features_ai, weights):
    """
    Compute weighted composite DTW similarity across multiple features.

    Args:
        features_dotcom: dict of {feature_name: np.array}
        features_ai: dict of {feature_name: np.array}
        weights: dict of {feature_name: float} summing to 1.0

    Returns:
        composite_score: float in [0, 100]
        per_feature_scores: dict
    """
    per_feature_scores = {}
    composite_score = 0.0

    for feature_name in weights:
        score = compute_dtw_similarity(
            features_dotcom[feature_name],
            features_ai[feature_name],
            window=max(len(features_dotcom[feature_name]),
                       len(features_ai[feature_name])) // 3,  # 33% Sakoe-Chiba band
        )
        per_feature_scores[feature_name] = score
        composite_score += weights[feature_name] * score

    return composite_score, per_feature_scores
```

### 3.4 Feature Weights

| Feature Category | Weight | Justification |
|---|---|---|
| Price trajectory | 0.25 | Most visible and directly comparable |
| Fundamentals (P/E + revenue) | 0.25 | Core question: are fundamentals different? |
| Market concentration | 0.20 | Structural similarity of market dynamics |
| Macro environment | 0.15 | Context but less directly comparable (different rate regimes) |
| Sentiment/hype | 0.15 | Behavioral similarity |
| **Total** | **1.00** | |

### 3.5 Rolling Similarity Timeline

The most compelling visualization: compute similarity score for each month as the AI era accumulates more data.

```python
def rolling_dtw_similarity(features_dotcom, features_ai_full, weights, min_months=6):
    """
    Compute DTW similarity at each month as AI-era data accumulates.

    Returns:
        timeline: pd.Series indexed by date, values = similarity score
    """
    scores = {}

    for t in range(min_months, len(features_ai_full['price']) + 1):
        # Slice AI-era features up to month t
        features_ai_t = {k: v[:t] for k, v in features_ai_full.items()}

        # Compare against the SAME number of months in dot-com era
        features_dotcom_t = {k: v[:t] for k, v in features_dotcom.items()}

        score, _ = compute_composite_similarity(features_dotcom_t, features_ai_t, weights)
        month_label = ai_era_dates[t - 1]  # Map index to actual date
        scores[month_label] = score

    return pd.Series(scores)

# Compute
similarity_timeline = rolling_dtw_similarity(dotcom_features, ai_features, weights)

# Also compute against dot-com era ALIGNED TO THE FULL CYCLE (including crash)
# This shows: are we tracking the pre-crash or post-crash trajectory?
```

### 3.6 Confidence Interval via Bootstrap

```python
def bootstrap_dtw_similarity(features_dotcom, features_ai, weights,
                              n_bootstrap=1000, noise_std=0.02):
    """
    Bootstrap confidence interval by adding small noise to both series.
    """
    scores = []
    for _ in range(n_bootstrap):
        # Add Gaussian noise to both series
        noisy_dotcom = {k: v + np.random.normal(0, noise_std, len(v))
                        for k, v in features_dotcom.items()}
        noisy_ai = {k: v + np.random.normal(0, noise_std, len(v))
                    for k, v in features_ai.items()}

        # Clip to [0, 1] after noise
        noisy_dotcom = {k: np.clip(v, 0, 1) for k, v in noisy_dotcom.items()}
        noisy_ai = {k: np.clip(v, 0, 1) for k, v in noisy_ai.items()}

        score, _ = compute_composite_similarity(noisy_dotcom, noisy_ai, weights)
        scores.append(score)

    return np.percentile(scores, [2.5, 50, 97.5])  # 95% CI
```

### 3.7 Interpretation Guide

| Similarity Score | Interpretation |
|---|---|
| 0 - 20 | Minimal resemblance. Very different market dynamics. |
| 20 - 40 | Some surface similarities but fundamentally different patterns. |
| 40 - 60 | Moderate similarity. Shared characteristics but significant divergences. This is the "interesting but inconclusive" zone. |
| 60 - 80 | Strong similarity. The patterns are tracking each other meaningfully. Warrants caution. |
| 80 - 100 | Near-identical trajectory. Historical precedent strongly suggests similar outcome. |

**Key nuance for presentation:** A similarity score of 70 does NOT mean "70% chance of crash." It means the structural patterns (price, fundamentals, concentration, macro, sentiment) are following a trajectory that is 70% similar to the dot-com path *up to this point*. The paths can diverge at any time.

### 3.8 Alternative: Multivariate DTW

For robustness, also compute multivariate DTW (all features simultaneously rather than independently):

```python
from tslearn.metrics import dtw as tslearn_dtw

def multivariate_dtw_similarity(matrix_dotcom, matrix_ai):
    """
    Compute DTW on multivariate time series directly.

    Args:
        matrix_dotcom: np.array shape (T1, D) — T1 timesteps, D features
        matrix_ai: np.array shape (T2, D)
    """
    distance = tslearn_dtw(matrix_dotcom, matrix_ai)
    # Normalize
    path_length = np.sqrt(matrix_dotcom.shape[0] * matrix_ai.shape[0])
    n_features = matrix_dotcom.shape[1]
    normalized = distance / (path_length * np.sqrt(n_features))
    similarity = max(0, (1 - normalized)) * 100
    return similarity
```

### 3.9 DTW Null Distribution and Statistical Significance (REQUIRED)

**Problem:** A DTW similarity score of "72/100" is meaningless without a reference distribution. What does 72 mean? Is it statistically distinguishable from 60?

**Solution:** Establish the score's meaning through two approaches:

#### A. Reference Period Comparisons

Compute DTW similarity of the AI era against 3-5 other historical periods to provide context:

| Comparison | Expected Similarity | Purpose |
|---|---|---|
| AI era vs Dot-com (1998-2001) | **Primary analysis** | The core research question |
| AI era vs Housing bubble (2005-2008) | Lower (different sector dynamics) | Calibration baseline |
| AI era vs Post-GFC recovery (2009-2013) | Moderate (growth story) | "Normal" growth baseline |
| AI era vs Mid-1990s tech rally (1995-1998) | Moderate | Pre-bubble growth comparison |
| AI era vs COVID recovery (2020-2022) | Moderate-high (stimulus-driven tech rally) | Recent comparison |

A score of 72 vs. dot-com becomes meaningful when you can show the AI era scores only 35 vs. post-GFC and 48 vs. housing bubble. Label the score honestly: "Similarity Index (higher = more similar to dot-com pattern)" -- do not imply a probabilistic interpretation.

#### B. Permutation Test for Empirical P-Value

Randomly shuffle the AI-era time series 1,000 times and recompute DTW. The empirical p-value (fraction of random shuffles with DTW distance <= observed distance) gives statistical grounding.

```python
def dtw_permutation_test(series_dotcom: np.ndarray, series_ai: np.ndarray,
                          n_permutations: int = 1000, seed: int = 42) -> dict:
    """
    Permutation test for DTW similarity.
    H0: The observed DTW distance is no smaller than what random chance would produce.
    """
    rng = np.random.RandomState(seed)

    # Observed DTW distance
    observed_distance = dtw.distance(series_dotcom.astype(np.float64),
                                      series_ai.astype(np.float64))

    # Generate null distribution by shuffling the AI-era series
    null_distances = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(series_ai)
        null_dist = dtw.distance(series_dotcom.astype(np.float64),
                                  shuffled.astype(np.float64))
        null_distances.append(null_dist)

    # Empirical p-value: fraction of null distances <= observed
    p_value = np.mean(np.array(null_distances) <= observed_distance)

    # Convert to similarity for interpretability
    path_length = np.sqrt(len(series_dotcom) * len(series_ai))
    observed_similarity = max(0, (1 - observed_distance / path_length)) * 100
    null_similarities = [max(0, (1 - d / path_length)) * 100 for d in null_distances]

    return {
        "observed_distance": observed_distance,
        "observed_similarity": observed_similarity,
        "p_value": p_value,
        "null_mean_similarity": np.mean(null_similarities),
        "null_std_similarity": np.std(null_similarities),
        "percentile": (1 - p_value) * 100,  # "AI-vs-dotcom similarity is in the Xth percentile"
    }
```

**Report:** "The AI-era trajectory has a DTW similarity of X to the dot-com pattern, placing it in the Yth percentile of all historical pairwise comparisons (permutation test p = Z)." This transforms the headline number from an arbitrary scale into a statistically defensible claim.

---

## 4. ML Model 3 — Feature Importance Analysis (SHAP)

### 4.1 Objective

Determine which features (across all 5 layers) contribute most to the regime classifier's `bubble` prediction for the current period. This transforms the model from a black box into an interpretable argument.

### 4.2 Method: SHAP (SHapley Additive exPlanations)

SHAP values decompose each prediction into feature contributions, grounded in cooperative game theory. For our Random Forest classifier:

```python
import shap

# Use TreeExplainer for tree-based models (exact, fast)
explainer = shap.TreeExplainer(best_rf)

# SHAP values for all test + live observations
shap_values = explainer.shap_values(X_test_scaled)
# shap_values is a list of 4 arrays (one per class), each shape (N, 38)

# For XGBoost
explainer_xgb = shap.TreeExplainer(best_xgb)
shap_values_xgb = explainer_xgb.shap_values(X_test_scaled)
```

### 4.3 Outputs

#### A. Global Feature Importance (SHAP Summary Plot)

Ranked list of all 38 features by mean |SHAP value| for the `bubble` class.

```python
# SHAP summary plot — bubble class (index depends on label encoding)
bubble_class_idx = list(classes).index('bubble')

shap.summary_plot(
    shap_values[bubble_class_idx],
    X_test,  # Use unscaled for readable feature names
    feature_names=feature_names,
    plot_type='dot',       # Dot plot shows direction + magnitude
    max_display=20,        # Top 20 features
    show=False,
)
```

**Expected top features (hypothesis):**

| Rank | Feature | Direction | Interpretation |
|---|---|---|---|
| 1 | `top10_weight_sp500` | Positive | Higher concentration pushes toward bubble classification |
| 2 | `pe_percentile` | Positive | Extreme valuations signal speculation |
| 3 | `ai_mention_rate` | Positive | Hype narrative indicator |
| 4 | `momentum_252d` | Positive | Sustained momentum is a bubble characteristic |
| 5 | `spy_rsp_spread` | Positive | Narrow leadership = fragile rally |
| 6 | `revenue_growth_yoy` | **Negative** | Strong revenue growth REDUCES bubble probability — the key counterargument |
| 7 | `fcf_yield_sp500` | Negative | Positive FCF = fundamental support |
| 8 | `google_trends_ai` | Positive | Public frenzy |
| 9 | `credit_spread_baa` | Negative | Tight spreads = no credit stress (yet) |
| 10 | `vix_monthly_avg` | Depends | Low VIX = complacency (bubble), high VIX = fear (correction) |

#### B. SHAP Waterfall Plot for "Today" (Mar 2026)

Decompose the single prediction for the latest month:

```python
# SHAP values for a single observation (today)
shap_values_today = explainer.shap_values(X_live_scaled[-1:])

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_today[bubble_class_idx][0],
        base_values=explainer.expected_value[bubble_class_idx],
        data=X_live.iloc[-1],
        feature_names=feature_names,
    ),
    max_display=15,
    show=False,
)
```

This plot answers: **"What specific factors are pushing today's classification toward or away from 'bubble'?"**

Expected narrative for presentation:

> "Concentration and sentiment are the strongest bubble signals — the top-10 weight in the S&P 500 is at dot-com levels, and AI hype metrics are at historical extremes. However, strong fundamentals are the key counterargument: NVIDIA's revenue growth and free cash flow are orders of magnitude beyond what Cisco delivered. The model sees a tug-of-war between structural bubble indicators and genuine fundamental support."

#### C. SHAP Dependence Plots

For the top 3 features, show how their values interact with the bubble probability:

```python
# P/E percentile dependence, colored by concentration
shap.dependence_plot(
    'pe_percentile',
    shap_values[bubble_class_idx],
    X_test,
    interaction_index='top10_weight_sp500',
    show=False,
)
```

#### D. SHAP Force Plot (Interactive)

For the Plotly/React dashboard — interactive force plot for any selected month:

```python
shap.force_plot(
    explainer.expected_value[bubble_class_idx],
    shap_values_today[bubble_class_idx][0],
    X_live.iloc[-1],
    feature_names=feature_names,
    matplotlib=False,  # Returns JS-based interactive plot
)
```

### 4.4 Cross-Model Comparison

Compare feature importance between RF and XGBoost to verify robustness:

```python
# RF feature importance (impurity-based)
rf_importance = pd.Series(
    best_rf.feature_importances_, index=feature_names
).sort_values(ascending=False)

# XGBoost feature importance (gain-based)
xgb_importance = pd.Series(
    best_xgb.feature_importances_, index=feature_names
).sort_values(ascending=False)

# SHAP-based importance (mean |SHAP|)
shap_importance = pd.Series(
    np.abs(shap_values[bubble_class_idx]).mean(axis=0),
    index=feature_names,
).sort_values(ascending=False)

# Compare top-10 overlap
print("Top 10 overlap (RF impurity vs SHAP):",
      len(set(rf_importance.head(10).index) & set(shap_importance.head(10).index)))
```

If top-10 features are consistent across methods, the finding is robust.

---

## 5. Model Validation & Robustness

### 5.1 Walk-Forward Validation

Beyond TimeSeriesSplit, perform true walk-forward validation that simulates real-time deployment:

```python
def walk_forward_validation(X, y, model_class, model_params,
                             initial_train_months=120, step=6):
    """
    Walk-forward validation with expanding window.
    Train on [0, t], predict [t+1, t+step], expand, repeat.
    """
    results = []

    for t in range(initial_train_months, len(X) - step, step):
        X_train_wf = X.iloc[:t]
        y_train_wf = y.iloc[:t]
        X_test_wf = X.iloc[t:t+step]
        y_test_wf = y.iloc[t:t+step]

        model = model_class(**model_params)
        model.fit(X_train_wf, y_train_wf)

        y_pred = model.predict(X_test_wf)
        y_proba = model.predict_proba(X_test_wf)

        results.append({
            'test_start': X_test_wf.index[0],
            'test_end': X_test_wf.index[-1],
            'accuracy': (y_pred == y_test_wf).mean(),
            'predictions': y_pred,
            'probabilities': y_proba,
            'actuals': y_test_wf.values,
        })

    return results
```

### 5.2 Sensitivity Analysis: Label Boundary Perturbation

The regime labels depend on exact date boundaries. How sensitive are results?

```python
def sensitivity_analysis(X, y_base_labels, date_perturbations,
                          model_class, model_params):
    """
    Shift bubble/correction boundary dates by +/- N months and re-train.
    Report variance in today's bubble probability.
    """
    bubble_probs_today = []

    for delta_months in date_perturbations:  # e.g., [-3, -2, -1, 0, 1, 2, 3]
        y_perturbed = shift_regime_labels(y_base_labels, delta_months)
        model = model_class(**model_params)
        model.fit(X_train, y_perturbed[train_mask])
        probs = model.predict_proba(X_today.values.reshape(1, -1))[0]
        bubble_idx = list(classes).index('bubble')
        bubble_probs_today.append(probs[bubble_idx])

    print(f"Bubble prob range: {min(bubble_probs_today):.1%} - {max(bubble_probs_today):.1%}")
    print(f"Mean: {np.mean(bubble_probs_today):.1%}, Std: {np.std(bubble_probs_today):.1%}")

    return bubble_probs_today
```

**Expected result:** If bubble probability stays in 35-50% range across +/- 3 month label shifts, the finding is robust.

### 5.3 Overfitting Checks

| Check | Method | Red Flag |
|---|---|---|
| Train vs test gap | Compare accuracy on train vs test | Gap > 15% = overfitting |
| OOB score vs test score | RF out-of-bag accuracy vs test accuracy | OOB >> test = overfitting |
| Learning curve | Plot accuracy vs training set size | Diverging curves = overfitting |
| Feature ablation | Remove top feature, re-train | If accuracy drops > 10%, over-reliance on single feature |
| Noise features | Add 5 random noise features | If noise features appear in top-20 importance, model is fitting noise |

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    best_rf, X_train_scaled, y_train,
    train_sizes=np.linspace(0.2, 1.0, 10),
    cv=tscv,
    scoring='f1_weighted',
    n_jobs=-1,
)
```

### 5.4 Benjamini-Hochberg Multiple Comparisons Correction (REQUIRED)

**Problem:** Across all 5 data layers + ML pipeline, we run 15-20+ statistical tests. With alpha=0.05 and 20 independent tests, the probability of at least one false positive by chance alone is 64%. Without correction, every reported p-value is suspect.

**Solution:** Collect ALL p-values from ALL layers into a single table, apply Benjamini-Hochberg FDR correction, and report both raw and adjusted p-values.

```python
from statsmodels.stats.multitest import multipletests

def apply_bh_correction(all_test_results: list[dict]) -> pd.DataFrame:
    """
    Collect all p-values across all layers and apply Benjamini-Hochberg correction.

    Parameters:
        all_test_results: list of dicts with keys: 'layer', 'test_name', 'raw_p_value'

    Returns:
        DataFrame with raw and BH-adjusted p-values
    """
    df = pd.DataFrame(all_test_results)

    # Apply Benjamini-Hochberg correction
    reject, pvals_corrected, _, _ = multipletests(
        df['raw_p_value'],
        alpha=0.05,
        method='fdr_bh'
    )

    df['bh_adjusted_p'] = pvals_corrected
    df['significant_after_correction'] = reject
    df['significant_raw'] = df['raw_p_value'] < 0.05

    # Flag findings that lose significance after correction
    df['lost_significance'] = df['significant_raw'] & ~df['significant_after_correction']

    print("=== Multiple Comparisons Correction (Benjamini-Hochberg) ===")
    print(f"Total tests: {len(df)}")
    print(f"Significant at raw alpha=0.05: {df['significant_raw'].sum()}")
    print(f"Significant after BH correction: {df['significant_after_correction'].sum()}")
    print(f"Lost significance after correction: {df['lost_significance'].sum()}")
    print()
    print(df[['layer', 'test_name', 'raw_p_value', 'bh_adjusted_p',
              'significant_after_correction']].to_string(index=False))

    return df

# Collect all p-values from all layers (fill in during implementation):
all_tests = [
    {"layer": "L1-Price", "test_name": "Pearson log-returns", "raw_p_value": None},
    {"layer": "L1-Price", "test_name": "Spearman log-returns", "raw_p_value": None},
    {"layer": "L1-Price", "test_name": "KS test returns", "raw_p_value": None},
    {"layer": "L1-Price", "test_name": "DTW permutation test", "raw_p_value": None},
    {"layer": "L2-Fundamentals", "test_name": "t-test log(P/E)", "raw_p_value": None},
    {"layer": "L2-Fundamentals", "test_name": "t-test P/S ratio", "raw_p_value": None},
    {"layer": "L2-Fundamentals", "test_name": "Chow test structural break", "raw_p_value": None},
    {"layer": "L3-Concentration", "test_name": "Mann-Whitney concentration", "raw_p_value": None},
    {"layer": "L3-Concentration", "test_name": "Concentration-drawdown corr", "raw_p_value": None},
    {"layer": "L4-Macro", "test_name": "Welch t-test (per indicator)", "raw_p_value": None},
    {"layer": "L4-Macro", "test_name": "Granger causality YC->SP500", "raw_p_value": None},
    {"layer": "L5-Sentiment", "test_name": "Hype-price correlation", "raw_p_value": None},
    {"layer": "L5-Sentiment", "test_name": "Mann-Whitney sentiment", "raw_p_value": None},
    # Add more as tests are run
]
```

**Reporting:** Include a "Statistical Tests Summary" table in the notebook (Section 5 or 6) showing all tests with raw and adjusted p-values. Any finding whose adjusted p-value exceeds 0.05 should be downgraded from "statistically significant evidence" to "suggestive pattern." This single addition adds more perceived rigor than almost any other change in the spec.

### 5.5 Ensemble Agreement Check

Do all three models agree on the current regime? Disagreement is informative.

```python
def ensemble_agreement(rf_probs, xgb_probs, lr_probs, classes):
    """Check if models agree on the top prediction."""
    rf_pred = classes[np.argmax(rf_probs)]
    xgb_pred = classes[np.argmax(xgb_probs)]
    lr_pred = classes[np.argmax(lr_probs)]

    all_agree = (rf_pred == xgb_pred == lr_pred)
    majority = max(set([rf_pred, xgb_pred, lr_pred]),
                   key=[rf_pred, xgb_pred, lr_pred].count)

    # Confidence: Jensen-Shannon divergence between model probability distributions
    from scipy.spatial.distance import jensenshannon
    mean_probs = (rf_probs + xgb_probs + lr_probs) / 3
    js_div = np.mean([
        jensenshannon(rf_probs, mean_probs),
        jensenshannon(xgb_probs, mean_probs),
        jensenshannon(lr_probs, mean_probs),
    ])

    return {
        'unanimous': all_agree,
        'majority_prediction': majority,
        'jensen_shannon_divergence': js_div,  # Lower = more agreement
        'per_model': {'RF': rf_pred, 'XGB': xgb_pred, 'LR': lr_pred},
    }
```

---

## 6. Pipeline Orchestration

### 6.1 Execution Order

```
Step 1:  Data loading (all 5 layers)                    -> data/processed/
Step 2:  Feature engineering                             -> data/features/feature_matrix.parquet
Step 3:  Label assignment                                -> data/features/regime_labels.csv
Step 4:  Train/test split                                -> in-memory
Step 5:  Hyperparameter tuning (TimeSeriesSplit)         -> models/best_params_*.json
Step 6:  Final model training                            -> models/regime_*.joblib
Step 7:  Test set evaluation                             -> results/classification_reports.txt
Step 8:  Live classification                             -> results/regime_probabilities.csv
Step 9:  DTW similarity scoring                          -> results/dtw_similarity.csv
Step 10: SHAP analysis                                   -> results/shap_values.json
Step 11: Generate all ML visualizations                  -> results/figures/ml_*.png
```

### 6.2 File Artifacts

| Artifact | Path | Format |
|---|---|---|
| Feature matrix | `data/features/feature_matrix.parquet` | Parquet (fast I/O) |
| Regime labels | `data/features/regime_labels.csv` | CSV with date index |
| Trained RF model | `models/regime_rf.joblib` | joblib serialized |
| Trained XGB model | `models/regime_xgb.joblib` | joblib serialized |
| Trained LR model | `models/regime_lr.joblib` | joblib serialized |
| Feature scaler | `models/feature_scaler.joblib` | joblib serialized |
| Best hyperparameters | `models/best_params_rf.json` | JSON |
| Live predictions | `results/regime_probabilities.csv` | CSV |
| DTW similarity scores | `results/dtw_similarity.csv` | CSV |
| SHAP values | `results/shap_values.json` | JSON (feature names + values) |
| Classification report | `results/classification_reports.txt` | Plain text |
| Confusion matrices | `results/figures/confusion_matrix_*.png` | PNG |

### 6.3 Notebook Integration

The ML pipeline runs inside the submission notebook (`2kim_finance_notebook.ipynb`), Section 5 (Statistical Analysis / Modeling). Key cells:

1. Feature matrix construction (show shape, sample rows)
2. Label distribution visualization (bar chart of class counts)
3. Model training (show best hyperparameters)
4. Test set evaluation (classification reports, confusion matrices)
5. **Live classification** (the punchline — regime probabilities for Mar 2026)
6. DTW similarity score + timeline
7. SHAP summary + waterfall plots
8. Ensemble agreement summary

---

## 7. Limitations & Caveats

### 7.1 Small Sample Problem

- **N_bubble is approximately 48 months** (dot-com + housing combined). This is a tiny positive class.
- Random Forest and XGBoost can handle this with `class_weight='balanced'`, but confidence intervals will be wide.
- We mitigate by: (a) using monthly granularity to maximize observations, (b) combining multiple bubble periods, (c) reporting probabilities not hard labels, (d) bootstrap confidence intervals.

### 7.2 Feature Leakage Risks

| Risk | Mitigation |
|---|---|
| Using future data in features (e.g., full-year revenue for partial year) | All features use trailing/available data only; forward-fill macro with limit |
| Label leakage (labels defined using price data, which is also a feature) | Labels defined using external events + price thresholds; price features are continuous (momentum, vol) not threshold-based |
| Look-ahead bias in normalization | Scaler fit on train set only; percentile features use expanding window |

### 7.3 Class Imbalance

| Class | Approximate Months | % of Training Data |
|---|---|---|
| `bubble` | 48 | 14% |
| `correction` | 72 | 21% |
| `normal_growth` | 168 | 48% |
| `recovery` | 60 | 17% |

- `normal_growth` dominates. `balanced` class weights upweight minority classes.
- Also report macro-averaged F1 alongside weighted F1.

### 7.4 Structural Breaks

Markets in 1998 and 2025 are fundamentally different:

- Algorithmic trading dominance (60%+ of volume)
- Passive investing revolution (index funds now > active)
- Different Fed toolkit (QE, forward guidance)
- Different market microstructure (circuit breakers, after-hours trading)
- Different information velocity (social media, real-time data)

These structural changes mean historical patterns may not repeat. The model trained on 1990-2018 data may not generalize to the 2020s. **We acknowledge this prominently in the presentation.**

### 7.5 "This Time Is Different" Caveat

The four most dangerous words in investing. Our analysis explicitly addresses:

- **Ways it IS different:** Real revenue (NVDA $130B+ TTM revenue vs CSCO $18B at peak), real products (ChatGPT, autonomous driving), real enterprise adoption
- **Ways it is NOT different:** Concentration levels, valuation multiples of secondary AI plays, retail sentiment patterns, "new paradigm" narrative

The model does not claim predictive power. It measures structural similarity to past bubble conditions. Markets can remain "irrationally exuberant" far longer than any model predicts.

### 7.6 Reproducibility

- All random seeds set (`random_state=42`)
- Environment pinned (`requirements.txt`)
- Feature matrix and labels saved as artifacts
- Full pipeline executable from notebook cells

---

## 8. Dependencies & Environment

### 8.1 Additional Python Packages Required

Add to `requirements.txt`:

```
# ML Pipeline
scikit-learn>=1.5
xgboost>=2.0
shap>=0.45
dtaidistance>=2.3       # DTW computation
tslearn>=0.6            # Alternative DTW + multivariate
joblib>=1.4             # Model serialization
```

### 8.2 Compute Requirements

| Step | Estimated Time | Memory |
|---|---|---|
| Feature engineering | < 10s | < 500MB |
| RF grid search (5-fold, ~162 combos) | ~5 min | < 2GB |
| XGB grid search (5-fold, ~2187 combos) | ~15 min (consider RandomizedSearchCV with n_iter=100) | < 2GB |
| LR grid search | < 30s | < 500MB |
| DTW computation (6 features x rolling) | < 2 min | < 1GB |
| SHAP (TreeExplainer) | < 1 min | < 1GB |

**Total pipeline runtime:** ~25 minutes on a modern laptop. Acceptable for competition demo.

### 8.3 Fallback: RandomizedSearchCV

If grid search is too slow for XGBoost:

```python
from sklearn.model_selection import RandomizedSearchCV

random_search_xgb = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_grid,
    n_iter=100,                    # Sample 100 combinations
    cv=tscv,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
```

---

## Appendix A: Feature Correlation Pre-Check

Before training, check for multicollinearity. Drop features with correlation > 0.90 (keep the one with lower p-value against labels).

```python
corr_matrix = X_train.corr().abs()
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
high_corr_pairs = [
    (col, row) for col in upper_tri.columns for row in upper_tri.index
    if upper_tri.loc[row, col] > 0.90
]
print(f"High correlation pairs to review: {high_corr_pairs}")
```

## Appendix B: Label Assignment Function

```python
def assign_regime_labels(date_index):
    """
    Assign regime labels to a DatetimeIndex based on predefined periods.
    Returns pd.Series with values in {'bubble', 'correction', 'normal_growth', 'recovery'}.
    """
    labels = pd.Series('normal_growth', index=date_index)

    # Bubble periods
    labels[(date_index >= '1998-01-01') & (date_index <= '2000-03-31')] = 'bubble'
    labels[(date_index >= '2007-01-01') & (date_index <= '2007-10-31')] = 'bubble'

    # Correction periods
    labels[(date_index >= '1990-07-01') & (date_index <= '1991-03-31')] = 'correction'
    labels[(date_index >= '2000-04-01') & (date_index <= '2002-10-31')] = 'correction'
    labels[(date_index >= '2007-11-01') & (date_index <= '2009-03-31')] = 'correction'
    labels[(date_index >= '2020-02-01') & (date_index <= '2020-03-31')] = 'correction'
    labels[(date_index >= '2022-01-01') & (date_index <= '2022-10-31')] = 'correction'

    # Recovery periods
    labels[(date_index >= '1991-04-01') & (date_index <= '1994-12-31')] = 'recovery'
    labels[(date_index >= '2009-04-01') & (date_index <= '2013-12-31')] = 'recovery'
    labels[(date_index >= '2020-04-01') & (date_index <= '2021-12-31')] = 'recovery'

    return labels
```
