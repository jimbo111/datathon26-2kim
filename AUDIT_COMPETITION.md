# Competition Compliance Audit — Team 2Kim
**Audited:** 2026-03-28  
**Auditor:** QA Expert (Claude Code)  
**Notebook:** `notebooks/2kim_finance_notebook.ipynb` (62 cells, 30 code / 32 markdown)  
**Slides builder:** `scripts/build_slides.py` — FILE DOES NOT EXIST  
**Slide deck (.pptx/.pdf):** NOT FOUND anywhere in the project tree  

---

## Competition Requirement Compliance

| # | Requirement | Status | Notes |
|---|---|---|---|
| **Participant Guide** | | | |
| 1 | Team chose exactly one track (Finance & Economics) | ✓ | Cell [00] header and Cell [01] explicitly state "Finance & Economics track" |
| 2 | Research question is specific and testable | ✓ | Cell [01]: "Is the NVIDIA-led AI equity rally of 2023–2026 structurally analogous to the Cisco-led dot-com bubble of 1998–2001, and if so, at what stage of the bubble lifecycle are we today?" — specific, comparative, testable with data |
| 3 | Notebook follows recommended structure: problem → data → cleaning → EDA → stats/model → results → limitations | ✓ | Sections 1–7 present and in correct order |
| 4 | Clear, evidence-based story: question → method → evidence → conclusion → limitations | ✓ | Each EDA sub-section follows question → chart → interpretation; Cell [58] provides structured conclusions; Cell [59] provides Executive Summary |
| 5 | Reproducible workflow in a notebook | ⚠ | Cache-or-call design is documented and sound. However, `src/layers/` directory is empty — all 5 layer `.py` files (layer1_price.py through layer5_sentiment.py) are missing from the filesystem. Notebook CANNOT currently run top-to-bottom. See Critical Issues. |
| **Submission Guidelines** | | | |
| 6 | Track selection is clear | ✓ | Cell [00] header, Cell [01] body, and submission filename all identify Finance & Economics |
| 7 | Final notebook is .ipynb format | ✓ | `2kim_finance_notebook.ipynb` exists |
| 8 | Final slideshow is .pptx or .pdf | ✗ | No `.pptx` or `.pdf` file found anywhere in the project. `scripts/build_slides.py` does not exist. Zero slide deck produced. |
| 9 | File naming: `teamname_track_notebook.ipynb` | ✓ | `2kim_finance_notebook.ipynb` follows the `{teamname}_{track}_notebook.ipynb` pattern |
| 10 | File naming: `teamname_track_slides.pptx` or `.pdf` | ✗ | No slides file exists to evaluate |
| 11 | Notebook section 1: Problem statement and hypothesis | ✓ | Cell [01] — research question, 6 sub-questions, H0/H1 hypothesis, motivation |
| 12 | Notebook section 2: Dataset description | ✓ | Cell [02] — 7-source table with API, tickers, time range, granularity; caching strategy explained |
| 13 | Notebook section 3: Data cleaning and preprocessing | ✓ | Cell [03] — preprocessing steps per layer (date normalization, forward-fill, ratio computation, normalization anchors); quality audit cells [06][07][08] |
| 14 | Notebook section 4: EDA (charts + interpretation) | ✓ | Cells [09]–[39] — 5 layer sub-sections plus cross-layer correlation matrix; every chart cell followed by interpretation markdown |
| 15 | Notebook section 5: Statistical testing and/or model approach | ✓ | Cells [40]–[57] — DTW, Random Forest regime classifier, SHAP, Benjamini-Hochberg FDR correction across 7 tests |
| 16 | Notebook section 6: Results and conclusions | ✓ | Cells [58][59] — Bubble Scorecard table, Key Findings, direct answer to RQ, Executive Summary |
| 17 | Notebook section 7: Limitations and future work | ✓ | Cell [60] — 6 sub-sections: data limitations, methodological limits, correlation/causation, structural changes, ethics, future work |
| 18 | LAST CELL is "Dataset Citations (MLA 8)" markdown cell | ✓ | Cell [61] is `cell_type: markdown`, titled `## Dataset Citations (MLA 8)` — correctly positioned as the final cell |
| 19 | MLA 8 citations include: Author/Org | ✓ | All 6 citations include author/organization |
| 20 | MLA 8 citations include: Dataset title | ✓ | All 6 citations include a quoted dataset/resource title |
| 21 | MLA 8 citations include: Platform/Publisher | ✓ | All 6 citations include platform and publisher |
| 22 | MLA 8 citations include: Publication date | ✓ | All 6 citations include year (2026); some lack specific publication date, which is acceptable under MLA 8 for continuously updated datasets |
| 23 | MLA 8 citations include: URL | ✓ | All 6 citations include URL |
| 24 | MLA 8 citations include: Access date | ✓ | All 6 citations include "Accessed 28 Mar. 2026" |
| 25 | All datasets cited | ⚠ | Cell [02] lists **7 distinct data sources** but Cell [61] contains only **6 citations**. Sources 1 and 2 (Yahoo Finance prices and Yahoo Finance fundamentals) are collapsed into a single citation (citation #1). This is defensible since both use the same `yfinance` library and same platform, but technically the dataset table identifies them as separate sources. Could draw a flag from judges. |
| 26 | Slideshow includes all required elements | ✗ | No slideshow exists — cannot assess |
| **Submission Checklist** | | | |
| 27 | Exactly one track selected | ✓ | Finance & Economics |
| 28 | Notebook opens and runs in order without major errors | ✗ | `src/layers/` is empty — all 5 `run_layerN()` imports in Cell [04] will raise `ModuleNotFoundError`. Every subsequent cell that references `results[...]` will fail. Additionally, `sys.path.insert(0, "..")` uses a relative path, which requires the notebook to be run from the `notebooks/` directory. |
| 29 | Bottom notebook cell contains `Dataset Citations (MLA 8)` | ✓ | Cell [61], correctly formatted |
| 30 | All datasets cited in MLA 8 format | ⚠ | 6 citations for 7 described sources (see item 25) |
| 31 | Slideshow matches notebook findings | ✗ | No slideshow exists |
| 32 | Team information complete and correct | ✓ | Cell [00]: Team 2Kim, Jimmy Kim & Alice Kim, SBU AI Community Datathon 2026 |
| 33 | Uploaded files use clear team-based file names | ⚠ | Notebook filename is correct. Slides filename cannot be verified — no slides file exists. |

---

## Rubric Category Assessment

### 1. Research Question & Track Fit (15 pts)
**Estimated score: 13–14 / 15 (Strong to Excellent)**

The research question is specific ("structurally analogous"), time-bounded (2023–2026 vs 1998–2001), comparative (NVDA vs CSCO), and decomposed into 6 independently testable sub-questions. The Finance & Economics track fit is explicit and well-motivated. The "Why It Matters" section (Cell [01]) connects to investor hedging, Fed policy, and AI ecosystem sustainability — showing domain depth.

Minor gap: The question is slightly ambitious in scope for a datathon (5-layer + ML analysis), which is a strength, but also means partial execution is more likely to show than with a narrower question. Judges may want more direct statement of the primary null hypothesis upfront before sub-questions.

**Gaps:** None material.  
**Suggestion:** Move the H0/H1 block (currently at 1.3) to immediately follow the main research question in 1.1 for faster judge scanning.

---

### 2. Data Quality & Preparation (20 pts)
**Estimated score: 16–18 / 20 (Strong to Excellent)**

Strong aspects:
- 7 distinct data sources across price, fundamentals, macro, regulatory filings, search trends, NLP sentiment, and market concentration
- Preprocessing documented explicitly per layer (Cell [03]): date normalization, forward-fill strategy, P/E TTM computation, CPI inflation adjustment, GDP YoY computation methodology
- Data quality audit cells present for Layers 1 ([06]), 2 ([07]), and 4 ([08])
- Missing-value handling stated explicitly (linear interpolation for 1–3 day gaps; gaps flagged)
- Cache-or-call pattern ensures idempotent data loading

Gaps:
- **No data quality audit cell for Layer 3 (concentration) or Layer 5 (sentiment)**. Layers 1, 2, and 4 have explicit `.shape`, `.isnull().sum()`, and `.describe()` audit cells. Layers 3 and 5 jump directly to charts. Judges reviewing Section 3 will notice this asymmetry.
- The notebook has **zero saved cell outputs** (all `execution_count: null`, `outputs: []`). This means judges see no evidence of the data quality audit results, no intermediate `print()` outputs, no chart previews. A notebook that has never been run against live data provides no evidence of actual preprocessing results.
- Yahoo Finance data for 1998–2001 CSCO acknowledged to have gaps around stock splits — handling is described but not demonstrated.

**Suggestion:** Add audit cells for Layers 3 and 5. Run the notebook and commit it with saved outputs before submission.

---

### 3. Analysis & Evidence (25 pts)
**Estimated score: 20–22 / 25 (Strong)**

Strong aspects:
- Every chart has a following markdown interpretation cell — no chart is left unexplained
- Statistical tests present at all 5 layers (Layer 1: correlation/Kolmogorov-Smirnov inferred; Layer 2: two-sample t-test on revenue growth; Layer 4: cosine similarity test; Layer 5: Mann-Whitney U and Pearson correlation)
- Benjamini-Hochberg FDR correction applied across all 7 p-values (Cells [56][57]) — this is graduate-level rigor rarely seen in datathon submissions
- DTW with permutation null distribution is a sound method for time-series comparison
- SHAP feature importance explains ML model decisions rather than treating RF as a black box
- Bubble scorecard (Cell [58]) synthesizes all evidence into a traffic-light table

Gaps:
- **No saved outputs** — judges cannot verify what the statistical tests actually returned. Cell [15] prints `layer1_stats`, but there is no output to read. If the notebook cannot run (due to missing layer files), these cells are decorative.
- Cell [21] has a multi-level fallback for chart_2_6 (PEG ratio): it tries chart_2_6, then chart_2_3, then an inline fallback. This fragility suggests the layer modules may not reliably produce all declared charts. The fallback logic is pragmatic but visible.
- Layer 3 has only 2 chart cells ([24][26]) compared to Layer 2's 4 and Layer 1's 2+stats. Layer 3 concentration is identified as the strongest "DANGER" signal but receives less visual development than fundamentals.
- The cross-layer correlation matrix (Cell [39]) is conceptually strong but constructed inline in the notebook rather than via the layer pipeline — it may not execute cleanly if layer data is missing.

**Suggestion:** Commit a run notebook with outputs. Add a Layer 3 data audit cell. Ensure chart_2_6 is produced reliably in the layer module.

---

### 4. Technical Rigor & Reproducibility (15 pts)
**Estimated score: 8–10 / 15 (Needs Improvement to Competent)**

This is the category most at risk. Critical issues:

- **`src/layers/` directory is empty.** The 5 files imported in Cell [04] (`layer1_price.py`, `layer2_fundamentals.py`, `layer3_concentration.py`, `layer4_macro.py`, `layer5_sentiment.py`) do not exist on disk. The notebook will fail at Cell [04] with `ModuleNotFoundError`. This is the single most damaging finding in this audit.
- **`src/ml/` exists** with `feature_matrix.py` and `pipeline.py`, but these are only called inline in Cells [48]–[54], not via `run_ml_pipeline()`. The ML code is partially duplicated between the notebook and the pipeline module.
- **`sys.path.insert(0, "..")` is a relative path** (Cell [04]). This only works if the notebook is launched with `notebooks/` as the working directory. If a judge opens the notebook from the project root or a different CWD, the import will fail regardless of whether the layer files exist.
- **Zero saved cell outputs** — the notebook has never been run to completion (all `execution_count: null`). A reproducible workflow must include at minimum one clean top-to-bottom run with saved outputs before submission.
- `scripts/build_slides.py` referenced in the audit task does not exist.

Positive aspects:
- `src/ml/feature_matrix.py` and `pipeline.py` are present and well-documented (no look-ahead bias enforced, time-based splits, BH correction required per module docstring)
- Requirements file exists (`requirements.txt`)
- Caching strategy is well-designed conceptually

**Suggestion:** This category cannot score above "Competent" unless the layer files are present and the notebook runs. Fix all Critical Issues before submission.

---

### 5. Presentation Clarity & Storytelling (15 pts)
**Estimated score: 12–13 / 15 (Strong)**

Strong aspects:
- Consistent narrative structure throughout: each layer section opens with a question, presents chart(s), interprets the chart, reports statistical results, and connects back to the central question
- Color convention documented and applied consistently (NVDA green, CSCO coral, Nortel gray)
- Executive Summary (Cell [59]) is a single dense paragraph that directly answers the research question with numbers — strong closing statement
- Bubble Scorecard table (Cell [58]) provides a scannable multi-layer synthesis with traffic-light verdicts
- Section 5 (ML) explains methods before presenting results — judges who are not ML specialists can follow the reasoning

Gaps:
- **No slideshow exists.** 15% of the total score is for "Presentation Clarity & Storytelling" and the slide deck is the primary presentation artifact. Judges evaluate the live presentation from slides; without them the team has no visual support during Q&A.
- No chart for Layer 3 sentiment or Layer 5 NLP comparison shows both eras on the same axis — the most compelling visual evidence (era-to-era sentiment comparison) may be buried.
- Cell [53] ("### 5.3 SHAP Feature Importance") is a markdown header with no body text — the section has no prose explanation before the code cell. Minor polish issue.

**Suggestion:** Produce the slides. This is mandatory.

---

### 6. Limitations, Ethics & Practical Insight (10 pts)
**Estimated score: 9–10 / 10 (Excellent)**

This is the strongest category. Cell [60] contains 6 named sub-sections:
- 7.1 Data Limitations — 4 specific, named limitations with technical detail (Google Trends 2004 cutoff, SPY holdings approximation, CSCO data gaps, Reddit platform non-existence in dot-com era)
- 7.2 Methodological Limitations — 4 specific limitations (small training set with quantified variance ±15%, survivorship bias by construction, subjective regime labels, DTW normalization dependency)
- 7.3 Correlation Does Not Imply Causation — explicitly addressed with a specific example (sentiment → price vs price → sentiment vs revenue → both), and correctly notes that observational data cannot distinguish causal structures. Three causal alternatives are enumerated.
- 7.4 Structural Market Changes — passive investing, algorithmic trading, social media effects discussed
- 7.5 Ethical Considerations — financial advice disclaimer, AI tool usage disclosed with manual validation noted, reputational risk of bubble-labeling acknowledged
- 7.6 Future Work — 5 specific, implementable next steps with technical detail

This section is substantially above datathon average. The explicit enumeration of causal alternatives in 7.3 is exactly what the rubric rewards.

**No material gaps.**

---

## Critical Issues (Must Fix Before Submission)

### CRIT-1: `src/layers/` directory is empty — all 5 layer module files are missing
**Impact:** Notebook fails at Cell [04] with `ModuleNotFoundError` on every import. All 30 code cells that depend on `results[...]` are non-executable. This blocks reproducibility scoring entirely.

The imports in Cell [04] reference:
- `src.layers.layer1_price` → `run_layer1`
- `src.layers.layer2_fundamentals` → `run_layer2`
- `src.layers.layer3_concentration` → `run_layer3`
- `src.layers.layer4_macro` → `run_layer4`
- `src.layers.layer5_sentiment` → `run_layer5`

None of these files exist. The `src/layers/` directory contains only a `__pycache__` subdirectory.

**Fix:** Commit all 5 layer `.py` files to `src/layers/`. Run the notebook top-to-bottom and commit it with saved outputs.

---

### CRIT-2: No slideshow file exists (`.pptx` or `.pdf`)
**Impact:** Direct submission requirement failure. The Google Form submission requires a slideshow file. Without it, the submission is incomplete and may be flagged by staff before judging begins. The live judging presentation has no visual support.

`scripts/build_slides.py` does not exist. No `.pptx` or `.pdf` file was found outside `.venv` package files.

**Fix:** Build and export a slide deck as `2kim_finance_slides.pptx` or `2kim_finance_slides.pdf`. The required slide content (per guidelines) is: team name/members/track, research question, datasets, method summary, key findings with visuals, limitations, conclusion. Slide chart assets are partially available in `submissions/charts/` (8 `.png` + `.json` files).

---

### CRIT-3: Notebook has never been run — zero saved cell outputs
**Impact:** Judges cannot see any analysis results. Print statements in data quality audit cells (Cells [06][07][08]), statistical test outputs (Cells [15][22][30][36][57]), and chart displays all produce no output in the submitted notebook. A notebook with `execution_count: null` on all code cells and empty `outputs` arrays signals an untested, possibly broken workflow.

**Fix:** After resolving CRIT-1, run the full notebook top-to-bottom (`Kernel → Restart & Run All`), verify no errors, and commit the version with saved outputs.

---

### CRIT-4: `sys.path.insert(0, "..")` uses a relative path (Cell [04])
**Impact:** Notebook only resolves `src.layers.*` imports correctly if the Jupyter kernel's working directory is `notebooks/`. If a judge opens the notebook from the project root or any other directory, all layer imports fail. This is a reproducibility hazard independent of CRIT-1.

**Fix:** Replace with an absolute path using `pathlib`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # not valid in notebooks
# Preferred in notebooks:
sys.path.insert(0, str(Path().resolve().parent))
```
Or more robustly, package the project with a `pyproject.toml` / `setup.py` and install in editable mode (`pip install -e .`), then no path manipulation is needed.

---

## Medium Issues

### MED-1: 7 datasets described, 6 citations provided (Cell [61])
Yahoo Finance prices (Source 1) and Yahoo Finance fundamentals (Source 2) in the dataset table are collapsed into a single citation in Cell [61]. While the platform is the same (`yfinance`), the competition guidelines state "all datasets used" must be cited and the dataset description table explicitly enumerates them as separate rows. Judges performing the compliance checklist check (item 4: "Dataset citations are present and appear complete") may flag this.

**Fix:** Either split into two Yahoo Finance citations (price data vs fundamental/financial data) or add a note within citation #1 explicitly covering both use cases. The note `(Price and financial data for NVDA, CSCO...)` partially addresses this but "financial data" is ambiguous about quarterly fundamentals.

---

### MED-2: Data quality audit missing for Layers 3 and 5
Layers 1, 2, and 4 each have an explicit audit cell showing `.shape`, date range, missing value counts, and `.describe()` statistics. Layers 3 (concentration) and 5 (sentiment) jump directly from the preprocessing description to chart display. This asymmetry is noticeable when reviewing Section 3 (Data Cleaning) and weakens the "Data Quality and Preparation" rubric category.

**Fix:** Add audit cells for Layer 3 (`results["layer3"]["data"]`) and Layer 5 (`results["layer5"]["data"]`) immediately after Cells [23] and [32] respectively, mirroring the pattern in Cells [06][07][08].

---

### MED-3: Section 5.3 SHAP header (Cell [53]) has no body text
Cell [53] is a single-line markdown header (`### 5.3 SHAP Feature Importance`) with no explanatory prose. The code and interpretation cells follow immediately, but the method is not introduced. Compare to Section 5.1 (DTW, Cell [42]) which has a full paragraph explaining what DTW is and why it is appropriate. For judges unfamiliar with SHAP, this makes the section harder to follow.

**Fix:** Add 2–3 sentences explaining what SHAP values are, why they are used here, and what the waterfall chart shows conceptually.

---

### MED-4: No confirmed Layer 3 statistical test output
Sections 4.1, 4.2, 4.4, and 4.5 all have explicit `layer_stats` print cells (Cells [15][22][30][36]). Section 4.3 (Layer 3, concentration) has no corresponding statistical test output cell. The Bubble Scorecard labels Layer 3 results as "DANGER" but there is no statistical test behind those claims — only visual/quantitative comparison. For the rubric category "Analysis & Evidence," examiners reward statistical tests.

**Fix:** If Layer 3 includes a statistical test (e.g., t-test comparing top-5 concentration in dot-com era vs AI era, or a test on the Buffett Indicator level), display it in the notebook. If no test exists in the layer, add a simple two-sample comparison between the 1998–2001 and 2023–2026 concentration distributions.

---

## Low Issues

### LOW-1: `submission_guidelines/10_health_topics.md` and `team_division.md` in `submissions/`
`submissions/` contains `10_health_topics.md` and `team_division.md` — artifacts from an earlier exploratory phase before the Finance & Economics track was confirmed. These are not part of the submission but could cause confusion if the submissions folder is zipped and uploaded.

**Fix:** Remove or move these files to an `archive/` directory.

---

### LOW-2: Notebooks `01_eda.ipynb` and `02_analysis.ipynb` exist alongside the final notebook
Three `.ipynb` files exist in `notebooks/`: `01_eda.ipynb`, `02_analysis.ipynb`, and `2kim_finance_notebook.ipynb`. The first two appear to be working/draft notebooks. They do not affect judging directly (only the named submission file is scored), but could cause confusion if all notebooks are submitted.

**Fix:** Confirm that only `2kim_finance_notebook.ipynb` is submitted via the Google Form.

---

### LOW-3: MLA 8 citation heading uses `##` (h2) instead of a plain heading
The submission guidelines specify the last cell be "titled `Dataset Citations (MLA 8)`". The actual heading is `## Dataset Citations (MLA 8)` with a `##` markdown prefix rendering as an h2 heading. This matches the section heading style used throughout the notebook and is not a meaningful violation — judges will see it as a proper section header. No fix required unless a literal title match is enforced.

---

### LOW-4: Cell [21] has a 3-level chart fallback suggesting layer reliability uncertainty
The PEG ratio chart cell (Cell [21]) first tries `chart_2_6`, then falls back to `chart_2_3`, then computes inline. This suggests the team was uncertain whether the layer module reliably generates `chart_2_6`. Once CRIT-1 is resolved and the layer files are committed, verify that `run_layer2()` consistently produces `chart_2_6` and remove or document the fallback.

---

## Summary Priority List

| Priority | Issue | Blocks Submission? |
|---|---|---|
| CRIT-1 | Layer module files missing from `src/layers/` | Yes — notebook cannot run |
| CRIT-2 | No slideshow (.pptx or .pdf) created | Yes — required submission file missing |
| CRIT-3 | Notebook has no saved outputs (never run) | Yes — reproducibility criterion |
| CRIT-4 | Relative `sys.path` insert breaks cross-environment reproducibility | Yes — reproducibility risk |
| MED-1 | 7 datasets described, only 6 cited | Partial — citation completeness flag |
| MED-2 | No data audit for Layers 3 and 5 | No — scoring impact only |
| MED-3 | Section 5.3 SHAP missing intro prose | No — minor clarity gap |
| MED-4 | No Layer 3 statistical test cell | No — scoring impact only |
| LOW-1 | Health track artifacts in submissions/ | No — cleanup only |
| LOW-2 | Draft notebooks alongside final notebook | No — confirm submission file only |
| LOW-3 | MLA heading level | No — cosmetic |
| LOW-4 | Chart fallback logic in Cell [21] | No — reliability signal |

