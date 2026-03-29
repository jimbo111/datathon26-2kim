# Competition Compliance Audit — `2kim_finance_notebook.ipynb`

**Audited by:** QA Expert  
**Audit date:** 2026-03-29  
**Branch audited:** `jimmy/dev`  
**Notebook path:** `/Users/jimmykim/AI_PROJECT/datathon26-2kim/notebooks/2kim_finance_notebook.ipynb`  
**Competition rules source:**  
- `/Users/jimmykim/AI_PROJECT/datathon-info/README.md`  
- `/Users/jimmykim/AI_PROJECT/datathon-info/submission-guidelines/README.md`  
- `/Users/jimmykim/AI_PROJECT/datathon-info/submission-guidelines/JUDGING-RUBRIC.md`

---

## STOP — Track Mismatch Detected

Before reading the checklist below: the project's own `CLAUDE.md` declares:

```
Track: Healthcare & Wellness
File naming: 2kim_health_notebook.ipynb, 2kim_health_slides.pptx
Research question: "Your Zip Code is Your Health Sentence"
```

A fully formed `2kim_health_notebook.ipynb` exists in the `notebooks/` directory with the Healthcare & Wellness track.

The notebook being audited here — `2kim_finance_notebook.ipynb` — is a **Finance & Economics** submission for a different track. Both notebooks exist in the same repo. This audit covers only the finance notebook as requested. The track mismatch between the project's declared intent and this notebook is flagged as the single most important finding. See Critical Issues below.

---

## Compliance Checklist

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1a | Section 1 — Problem Statement | PASS | Cell 2: `## 1. Problem Statement & Research Question` |
| 1b | Section 2 — Dataset Description | PASS | Cell 3: `## 2. Dataset Description` |
| 1c | Section 3 — Data Cleaning / Preprocessing | PASS | Cell 4: `## 3. Data Cleaning and Preprocessing` |
| 1d | Section 4 — Exploratory Data Analysis | PASS | Cell 10: `## 4. Exploratory Data Analysis` with 5 sub-layers (4.1–4.6) |
| 1e | Section 5 — Statistical Testing / Model | PASS | Cell 41: `## 5. Statistical Modeling & Machine Learning` with DTW, Random Forest, SHAP, BH correction |
| 1f | Section 6 — Results / Conclusions | PASS | Cell 59: `## 6. Results and Conclusions` with scorecard, key findings, executive summary |
| 1g | Section 7 — Limitations | PASS | Cell 61: `## 7. Limitations, Ethics & Future Work` — 6 sub-sections including explicit correlation-causation disclaimer |
| 2 | Last cell is markdown titled "Dataset Citations (MLA 8)" | PASS | Cell 62 (final cell), type=markdown, header exactly `## Dataset Citations (MLA 8)` |
| 3a | MLA 8 — Author/Org present for all sources | PASS | All 6 citations include named author/org (Yahoo Finance, Board of Governors, U.S. SEC, Google LLC, OpenAI, S&P Global) |
| 3b | MLA 8 — Dataset title for all sources | PASS | All 6 citations include italicized title |
| 3c | MLA 8 — Platform/Publisher for all sources | PASS | All 6 citations include publisher in MLA format |
| 3d | MLA 8 — Publication date (if available) | PASS | All 6 citations include year (2026 for live data, appropriate for current APIs) |
| 3e | MLA 8 — URL for all sources | PASS | All 6 citations include URLs |
| 3f | MLA 8 — Access date for all sources | PASS | All 6 citations: "Accessed 28 Mar. 2026" |
| 4a | File naming — notebook is `2kim_finance_notebook.ipynb` | PASS | File exists at `notebooks/2kim_finance_notebook.ipynb` |
| 4b | File naming — slides are `2kim_finance_slides.pptx` | FAIL | No `.pptx` or `.pdf` found anywhere under `submissions/` or project root |
| 5 | Track stated explicitly in notebook | PASS | Cell 1: `**Track:** Finance & Economics`; Cell 2: `This is a **Finance & Economics track** submission.` |
| 6 | Slideshow file exists | FAIL | `submissions/2kim_finance_slides.pptx` does not exist; `find` confirms no finance-track slide file anywhere in the project |
| 7 | Notebook runs top-to-bottom without errors | FAIL | See Technical Rigor section — 3 blocking issues identified |

---

## Rubric Assessment

### Category 1 — Research Question and Track Fit (15 pts)

**Estimated score: 13–14 / 15 (Strong to Excellent)**

The research question is specific and testable: "Is the NVIDIA-led AI equity rally of 2023–2026 structurally analogous to the Cisco-led dot-com bubble of 1998–2001, and if so, at what stage of the bubble lifecycle are we today?" Five independently scoped sub-questions (one per data layer) add precision. The Finance & Economics track is stated twice. Motivation is documented (investor risk, academic comparison). Prior literature is cited in Section 1.5.

Minor gap: the question's testability depends on the regime labels being valid, which the notebook itself flags as a limitation (Section 7.2).

---

### Category 2 — Data Quality and Preparation (20 pts)

**Estimated score: 16–18 / 20 (Strong)**

Seven distinct data sources are documented in a clear table (Cell 3) with API/library, content, tickers/series, time range, and granularity for each. Preprocessing steps are individually described per layer (Cell 4): date normalization, forward-fill strategy, P/E computation methodology, inflation adjustment via CPI, normalization anchoring, and GPT scoring determinism.

A cache-or-call pattern is documented to protect reproducibility. Assumptions are stated (linear interpolation for 1–3 day gaps, ETF holdings as proxy for historical S&P 500 weights).

Deduction risk: all 30 code cells have `execution_count: None` and zero outputs — the notebook has never been executed. Judges cannot verify whether the described preprocessing actually runs. The `data/raw/` directory contains only a `.gitkeep`; no cached parquet files exist. If the notebook cannot run offline (API calls fail), judges see blank outputs.

---

### Category 3 — Analysis and Evidence (25 pts)

**Estimated score: 18–21 / 25 (Competent to Strong)**

The analysis design is sophisticated: DTW for time-series similarity, a Random Forest regime classifier trained on CSCO era and applied to NVDA era, SHAP feature importance, and Benjamini-Hochberg multiple comparisons correction across 7 tests. A bubble scorecard table in Section 6.1 synthesizes findings across all five layers with explicit verdicts.

Critical deduction: zero code cells have been executed. Every chart cell (`fig.show()`) produces no output in the submitted notebook file. The scorecard numbers (e.g., "top-5 at ~28%", "P/E ~40–60x", "DTW similarity 1.5–2 SD above null") appear only in markdown prose — there is no executed chart or printed result anywhere in the notebook to substantiate them. Judges evaluating from the `.ipynb` file alone see a notebook with charts promised but never rendered.

The written interpretations for each chart (Cells 13, 15, 19, 21, 26, 28, 32, 35, 38, 47, 53, 56) are substantive and correctly distinguish correlation from causation, which partially compensates — but is not a substitute for visible evidence.

---

### Category 4 — Technical Rigor and Reproducibility (15 pts)

**Estimated score: 6–9 / 15 (Needs Improvement)**

Three blocking reproducibility failures:

**Failure 1 — Missing source modules (Critical).**  
Cell 5 imports:
```python
from src.layers.layer1_price import run_layer1
from src.layers.layer2_fundamentals import run_layer2
from src.layers.layer3_concentration import run_layer3
from src.layers.layer4_macro import run_layer4
from src.layers.layer5_sentiment import run_layer5
```
The directory `src/layers/` contains only a `__pycache__/` subdirectory. The `.py` source files (`layer1_price.py`, `layer2_fundamentals.py`, etc.) are absent from the repo. Only compiled `.pyc` bytecode exists in `__pycache__`. Python cannot import from `.pyc` files when the corresponding `.py` files do not exist and the cache path is not on `sys.path`. Cell 5 will raise `ModuleNotFoundError` on any fresh clone.

**Failure 2 — Missing runtime dependencies.**  
The notebook uses `yfinance`, `fredapi`, `pytrends`, `dtw-python`, `shap`, and `scikit-learn`. None of these are listed in `requirements.txt`. Any judge running `pip install -r requirements.txt` then executing the notebook will encounter `ImportError` on multiple cells.

**Failure 3 — No cached data, API keys required.**  
`data/raw/` is empty (only `.gitkeep`). The cache-or-call pattern requires API keys in `.env` for FRED (`fredapi`), OpenAI, and SEC EDGAR. No `.env.example` documents which keys are required for the finance layers (the `.env.example` present covers different services). The notebook claims to be "safe to run offline" but this only holds if cached parquet files are present — and they are not.

**Positive factors:** The notebook uses `try/except ImportError` fallbacks for optional packages (`dtw`, `shap`), and `sys.path.insert(0, "..")` is correct for notebook-relative import resolution. The overall logic structure is coherent and cells are ordered correctly.

---

### Category 5 — Presentation Clarity and Storytelling (15 pts)

**Estimated score: 11–13 / 15 (Competent to Strong)**

Structure is clear and logically ordered. Section headers are numbered. The color convention (NVIDIA green, Cisco coral, Nortel gray) is documented. The bubble scorecard provides a clear visual summary. The executive summary (Cell 60) directly answers the research question in one paragraph. Sub-questions map one-to-one to data layers, which creates a coherent narrative thread.

Deduction: all chart cells are empty in the file. A judge reading the `.ipynb` without re-running it sees no visualizations — the entire visual evidence layer is absent from the submission artifact.

The notebook has no slide file attached, which is a separate compliance failure but also means judges cannot evaluate slide-based presentation quality.

---

### Category 6 — Limitations, Ethics, and Practical Insight (10 pts)

**Estimated score: 9–10 / 10 (Excellent)**

Section 7 is the strongest section in the notebook. It addresses:
- Data limitations (6 specific gaps including Google Trends coverage, S&P constituent weight approximations, yfinance gap handling, missing Reddit era data) — Cell 61, Section 7.1
- Methodological limitations (small training set, survivorship bias in analog selection, subjective regime labels, DTW normalization sensitivity) — Section 7.2
- Explicit correlation-causation disclaimer with three named causal alternatives — Section 7.3
- Structural market changes (passive investing, algorithmic trading) — Section 7.4
- Ethics: financial advice disclaimer, AI tool disclosure, manual validation of GPT outputs, reputational risk of public bubble analysis — Section 7.5
- Future work: 5 concrete next steps — Section 7.6

This is notably thorough and would score at the top of this category.

---

## Rubric Score Summary

| Category | Weight | Estimated Score | Rationale |
|---|---|---|---|
| Research Question and Track Fit | 15 | 13–14 | Specific, testable, well-motivated |
| Data Quality and Preparation | 20 | 16–18 | Strong documentation; deducted for unexecuted notebook / missing cache |
| Analysis and Evidence | 25 | 18–21 | Sophisticated design; zero visible outputs is a major gap |
| Technical Rigor and Reproducibility | 15 | 6–9 | Missing source `.py` files, missing requirements, no cached data |
| Presentation Clarity and Storytelling | 15 | 11–13 | Good structure; no charts rendered, no slides |
| Limitations, Ethics, and Practical Insight | 10 | 9–10 | Excellent — most complete section |
| **TOTAL** | **100** | **73–85** | **Wide range depending on whether judges re-run vs. read-only** |

If judges evaluate the `.ipynb` file as-is (read-only, no re-execution): score skews toward 73.  
If judges successfully re-run the notebook after fixes: score skews toward 85.

---

## Critical Issues

### Issue 1 — TRACK MISMATCH (Highest Priority)

The project's `CLAUDE.md` declares the team's competition track as **Healthcare & Wellness**, with submission file `2kim_health_notebook.ipynb`. A separate notebook with that filename and the Healthcare & Wellness track exists in the same `notebooks/` directory with its own complete 7-section structure and citations.

The notebook being audited (`2kim_finance_notebook.ipynb`) is a Finance & Economics submission. **These are different competition tracks. Only one track may be submitted per team.** Submitting the wrong notebook, or submitting a finance notebook when the team is registered under Healthcare & Wellness, would result in track disqualification or a compliance flag by judges before scoring begins.

Confirm with the team: which track are you actually competing in? If Healthcare & Wellness, audit `2kim_health_notebook.ipynb` instead (and that notebook also has zero executed outputs and no slides file).

---

### Issue 2 — No Slides File Exists (Blocking Submission)

`submissions/2kim_finance_slides.pptx` does not exist. No `.pptx` or `.pdf` file for a finance-track slideshow was found anywhere in the project. The submission guidelines require both a notebook and a slideshow. The judging rubric compliance check explicitly lists "Slideshow file was submitted" as a Yes/No gate before scoring begins.

---

### Issue 3 — Source Layer Files Missing (Blocking Reproducibility)

`src/layers/` contains no `.py` files — only `__pycache__/`. The five `run_layerN()` functions imported in Cell 5 cannot be loaded. The notebook will fail on Cell 5 with `ModuleNotFoundError` on any machine that does not have the `.pyc` bytecode cached under identical Python version and path conditions.

The `.py` source files must be committed to the repository. Compiled bytecode in `__pycache__` should never be the only copy of source code.

---

### Issue 4 — Notebook Never Executed (No Outputs)

All 30 code cells have `execution_count: None` and empty `outputs: []`. The notebook file as it exists on this branch has never been run end-to-end. Judges see a notebook with chart cells that produce nothing. All numerical claims in the markdown interpretation cells are unverified in the submitted artifact.

The notebook must be executed cleanly from top to bottom and saved with outputs before submission.

---

### Issue 5 — Missing Dependencies in `requirements.txt`

The notebook requires packages not listed in `requirements.txt`:
- `yfinance` (price and fundamental data — used throughout)
- `fredapi` (FRED macro data — Layer 4)
- `pytrends` (Google Trends — Layer 5)
- `dtw-python` (DTW similarity — Section 5.1)
- `shap` (SHAP analysis — Section 5.3)
- `scikit-learn` (Random Forest, cross-validation — Section 5.2)

A judge following the standard setup path (`pip install -r requirements.txt`) will encounter import errors.

---

### Issue 6 — No Cached Data, `.env` Keys Undocumented for Finance Track

`data/raw/` is empty. The notebook's claimed offline-safe behavior requires pre-populated cache files. The FRED and OpenAI API calls in Layers 4 and 5 require API keys. The existing `.env.example` does not document the finance-layer key names. A judge attempting to run the notebook from scratch cannot determine which keys to configure.

---

## Recommendations

Listed in order of submission-blocking severity:

1. **Resolve track selection.** Decide definitively whether you are submitting Finance & Economics or Healthcare & Wellness. Only one notebook is submitted. Do not submit both.

2. **Create the slides file immediately.** `submissions/2kim_finance_slides.pptx` (or `.pdf`) must exist before submission. This is a hard compliance gate — judges mark it Yes/No before scoring.

3. **Commit the missing `src/layers/*.py` source files.** The compiled `.pyc` files in `__pycache__` are not a substitute. Add `layer1_price.py`, `layer2_fundamentals.py`, `layer3_concentration.py`, `layer4_macro.py`, `layer5_sentiment.py` to the repository.

4. **Add all missing packages to `requirements.txt`.** At minimum: `yfinance`, `fredapi`, `pytrends`, `dtw-python`, `shap`, `scikit-learn`. Add a `# Finance notebook` comment block to group them.

5. **Execute the notebook end-to-end and save with outputs.** Run `jupyter nbconvert --to notebook --execute notebooks/2kim_finance_notebook.ipynb --output notebooks/2kim_finance_notebook.ipynb` (or execute via Jupyter UI) and confirm all 30 code cells produce visible output before saving.

6. **Populate `data/raw/` cache or document the setup steps.** If cached parquet files cannot be committed (gitignore), add a `notebooks/SETUP.md` or notebook cell at the top that explains exactly which environment variables are needed and how to pre-populate the cache. The "safe to run offline" claim in the notebook currently cannot be verified.

7. **Update `.env.example`** to include the FRED API key and OpenAI API key variable names used by the finance layers.

8. **Minor — MLA citation for S&P Global (source 6)** notes that "current snapshot sourced via SPY ETF holdings" — this is technically a different source (Yahoo Finance / BlackRock) than S&P Global. Consider splitting or clarifying. Not a blocking issue but a potential judge question.

---

*End of audit — `/Users/jimmykim/AI_PROJECT/datathon26-2kim/AUDIT_COMPETITION_FINAL.md`*
