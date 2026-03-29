# Alice's TODO — Visualizations + Presentation

**Project:** "Your Zip Code is Your Health Sentence"
**Branch:** `alice/health-analysis`

---

## Phase 1A — Geographic Foundation (start immediately)
- [ ] Build side-by-side choropleth: food access score + diabetes rate (same geographic extent)
- [ ] Build choropleth: food deserts overlaid on income quintiles
- [ ] Confirm visual clustering — food deserts are not random
- [ ] Wire up choropleth to backend API endpoint in `routes/charts.py`
- [ ] Create `Chart.jsx` component for map rendering
- [ ] Sync with Jimmy on master dataframe column names

## Phase 2 — Food Access Visuals
> Blocked by: Jimmy's Phase 2 regression outputs

- [ ] Scatter with regression line: food access score vs. diabetes by tract, colored by majority race
- [ ] Scatter: food access score vs. obesity rate
- [ ] Add chart endpoints in `backend/services/chart_service.py`
- [ ] Integrate into Dashboard component

## Phase 3 — Zip Code Effect Visuals
> Blocked by: Jimmy's Phase 3 stats outputs

- [ ] Bar chart: life expectancy by income quintile, split by race
- [ ] Annotate worst vs. best tract in same metro area
- [ ] Add to Dashboard layout

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
