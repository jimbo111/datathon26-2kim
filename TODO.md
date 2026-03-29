# Alice's TODO — Visualizations + Presentation

**Project:** "Your Zip Code is Your Health Sentence"
**Branch:** `alice/health-analysis`

---

## Phase 1A — Geographic Foundation (start immediately)
- [x] Build side-by-side choropleth: food access score + diabetes rate (same geographic extent)
- [x] Build choropleth: food deserts overlaid on income quintiles
- [x] Confirm visual clustering — food deserts are not random
- [x] Wire up choropleth to backend API endpoint in `routes/charts.py`
- [x] Create `Chart.jsx` component for map rendering (reused existing + added HealthDashboard)
- [ ] Sync with Jimmy on master dataframe column names

## Phase 2 — Food Access Visuals
> Using sample data — will swap to Jimmy's real regression outputs when ready

- [x] Scatter with regression line: food access score vs. diabetes by tract, colored by majority race
- [x] Scatter: food access score vs. obesity rate
- [x] Add chart endpoints in `backend/services/chart_service.py`
- [x] Integrate into Dashboard component

## Phase 3 — Zip Code Effect Visuals
> Using sample data — will swap to Jimmy's real stats outputs when ready

- [x] Bar chart: life expectancy by income quintile, split by race
- [ ] Annotate worst vs. best tract in same metro area (needs real tract data)
- [x] Add to Dashboard layout

## Phase 4 — Race Gap Visuals
> Blocked by: Jimmy's Phase 4 income x race matrix

- [ ] 2x2 heatmap: income quintile x race → diabetes prevalence
- [ ] Annotate: high-income Black vs. low-income white comparison callout
- [ ] Review framing with Jimmy — sensitive topic, align messaging

## Phase 5A — Synthesis Visuals + Presentation
> Blocked by: Jimmy's Phase 5B Health Disadvantage Index

- [ ] Map the Health Disadvantage Index across all tracts
- [ ] Top 10 worst tracts table in dashboard
- [ ] Path diagram: poverty → food desert → obesity → diabetes (with coefficients from Jimmy)
- [ ] Write presentation opening hook
- [ ] Write "so what" policy recommendation section
- [ ] Build presentation slides
- [ ] Dry run presentation with Jimmy

---

## File Ownership
| File | Status |
|------|--------|
| `backend/services/chart_service.py` | Owner |
| `backend/routes/charts.py` | Owner |
| `frontend/src/components/Chart.jsx` | Owner |
| `frontend/src/components/Dashboard.jsx` | Owner |
| `frontend/src/api/client.js` | Owner |
| `submissions/` | Owner |

---

## Sync Checkpoints with Jimmy
1. After Phase 1 — confirm master dataframe schema + column names
2. After Phase 2 — agree on key stats to call out (odds ratio, R²)
3. After Phase 4 — review race gap finding together for framing
4. Final day — end-to-end dashboard run + presentation dry run
