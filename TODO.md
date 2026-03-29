# Jimmy's TODO — Data Pipeline + Statistical Analysis

**Project:** "Your Zip Code is Your Health Sentence"
**Branch:** `jimmy/health-analysis`

---

## Phase 1B — Data Pipeline (start immediately)
- [ ] Download USDA Food Access Research Atlas (food desert classification by tract)
- [ ] Download CDC PLACES (diabetes, obesity, hypertension rates by tract)
- [ ] Download ACS 5-Year Estimates (income, race, education, insurance by tract)
- [ ] Download CDC Life Expectancy at Birth (by census tract)
- [ ] Download HRSA UDS Mapper (primary care access by geography)
- [ ] Write data loaders in `src/loaders.py` for each dataset
- [ ] Clean and standardize all datasets (handle nulls, outliers, type mismatches)
- [ ] Merge all datasets on FIPS census tract code → master dataframe
- [ ] Validate join quality: check for missing tracts, null rates, coverage gaps
- [ ] Save cleaned master dataset to `data/processed/master.parquet`
- [ ] Document column names and share schema with Alice

## Phase 2 — Food Access → Chronic Disease (stats)
> Unblocked — runs on master dataframe from Phase 1B

- [ ] OLS regression: `diabetes_rate ~ food_access_score + median_income + pct_uninsured`
- [ ] OLS regression: `obesity_rate ~ food_access_score + median_income + pct_uninsured`
- [ ] Compute odds ratios: food desert vs. non-food-desert diabetes rates
- [ ] Confirm food access predicts diabetes independent of income (partial correlation)
- [ ] Export regression summary table as JSON for Alice's frontend
- [ ] Write analysis up in `notebooks/02_analysis.ipynb`

## Phase 3 — The Zip Code Effect (stats)
> Unblocked — runs on master dataframe from Phase 1B

- [ ] Variance decomposition: between-tract vs. within-tract diabetes rate variance
- [ ] Multivariate regression: `life_expectancy ~ income + food_access + race + education + pct_uninsured`
- [ ] Extract standardized betas — identify which variable dominates
- [ ] Compute life expectancy gap (richest vs. poorest quartile tracts) per metro area
- [ ] Export results as JSON for Alice's bar charts

## Phase 4 — Race as a Residual Gap (stats)
> Unblocked — runs on master dataframe from Phase 1B

- [ ] Build income quintile x race → diabetes prevalence matrix
- [ ] Logistic regression with interaction term: income x race → diabetes
- [ ] Residual analysis: is race still significant after controlling for income + food access?
- [ ] Test: high-income Black tracts vs. low-income white tracts — does income close the gap?
- [ ] Export income x race matrix as JSON for Alice's heatmap
- [ ] Review framing with Alice — sensitive topic, align messaging

## Phase 5B — Health Disadvantage Index (synthesis)
> Depends on: Phase 2–4 completed for coefficient weights

- [ ] Define Health Disadvantage Index formula: weighted composite of food access + income + care access
- [ ] Justify weights from Phase 2–4 regression coefficients
- [ ] Compute index for every census tract
- [ ] Rank all tracts by index score
- [ ] Export top/bottom 10% tract lists for Alice's map
- [ ] Summary stats table: life expectancy gap, diabetes gap, obesity gap between top/bottom deciles
- [ ] Export path diagram coefficients (poverty → food desert → obesity → diabetes) for Alice

---

## File Ownership
| File | Status |
|------|--------|
| `src/loaders.py` | Owner |
| `src/stats.py` | Owner |
| `backend/services/data_service.py` | Owner |
| `backend/routes/data.py` | Owner |
| `notebooks/01_eda.ipynb` | Owner |
| `notebooks/02_analysis.ipynb` | Owner |
| `data/processed/` | Owner |

---

## Deliverables to Alice
| What | Format | When |
|------|--------|------|
| Master dataframe schema (column names + types) | Slack/doc | After Phase 1B |
| Regression tables (odds ratios, R², p-values) | JSON in `data/processed/` | After Phase 2 |
| Life expectancy stats + standardized betas | JSON in `data/processed/` | After Phase 3 |
| Income x race diabetes matrix | JSON in `data/processed/` | After Phase 4 |
| Health Disadvantage Index ranked tracts | JSON/Parquet in `data/processed/` | After Phase 5B |
| Path diagram coefficients | JSON in `data/processed/` | After Phase 5B |

---

## Sync Checkpoints with Alice
1. After Phase 1B — share master dataframe schema + column names
2. After Phase 2 — agree on key stats to call out (odds ratio, R²)
3. After Phase 4 — review race gap finding together for framing
4. Final day — end-to-end dashboard run + presentation dry run
