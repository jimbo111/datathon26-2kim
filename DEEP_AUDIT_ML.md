# Deep Audit: ML Pipeline Implementation

## VERIFIED

- **Price-excluded RF variant**: Fully implemented in `pipeline.py` L297-317. Drops all 9 price features from `PRICE_FEATURES`, trains separate RF with own scaler, reports CV F1 alongside full model. Inference at prediction time also works (L448-463).
- **BH correction**: `apply_bh_correction()` (L1149-1222) calls `multipletests(method='fdr_bh')`. `_collect_pvalues_from_layers()` harvests p-values from ALL 5 layers + ML-DTW permutation test. Correctly structured.
- **SHAP**: Uses `shap.TreeExplainer` (L1065). Generates both summary dot plot (`chart_ml_3`) and waterfall plot (`chart_ml_4`). Extracts bubble-class-specific SHAP values correctly.
- **Two-model approach**: `feature_matrix.py` returns `"core"` (no sentiment) and `"augmented"` (with sentiment, post-2004 only) DataFrames. Sentinel cols defined in `SENTIMENT_COLS`. No imputation of sentiment -- correct per spec.
- **Time-series split**: `TimeSeriesSplit(n_splits=5, gap=3, test_size=...)` used for CV (L219-229). Outer split is chronological: train<=2018, val=2019-2021, test=2022-mid2024, live=mid2024-2026. No random splitting anywhere.
- **Regime labels**: Date ranges in `feature_matrix.py` L539-568 match the spec exactly. Hardcoded (acceptable for competition). Labels: bubble, correction, normal_growth, recovery with reasonable historical justification.
- **DTW null distribution**: 500 permutations (L731, L1675). Compares AI era against 4 reference periods: dotcom, housing, post-GFC, covid_rally. Empirical p-value + bootstrap CI computed. Rolling timeline also implemented.

## ISSUES

1. **Notebook diverges from pipeline**: The notebook (cells 57-68) has its OWN inline ML code -- completely different regime labels (Accumulation/Expansion/Euphoria/Distribution/Decline), different feature matrix builder, trains only on CSCO era, uses MinMaxScaler not StandardScaler, no XGBoost, no LR baseline. **The pipeline module (`src/ml/pipeline.py`) is never called from the notebook.** Two parallel implementations exist.
2. **Notebook DTW is stock-vs-stock only**: Cell 57-59 compute DTW between NVDA and CSCO normalized prices. The pipeline's multi-layer, multi-reference-period DTW (with permutation test across 5 feature categories) is not used. Notebook DTW has no formal p-value reporting.
3. **Augmented model never trained**: `run_ml_pipeline()` builds `features_aug` but only trains on `features_core`. The spec requires training both core and augmented models and comparing bubble probabilities. The augmented model with sentiment features is dead code.
4. **No GridSearchCV**: Spec defines hyperparameter grids for RF, XGB, and LR. Implementation uses fixed hyperparams only. No grid search or tuning.
5. **Notebook BH correction is inline**: Cell 71 reimplements p-value collection instead of calling `apply_bh_correction()`. Results may differ from pipeline's collection logic.
6. **Notebook regime labels are 5-class, pipeline is 4-class**: Notebook uses {Accumulation, Expansion, Euphoria, Distribution, Decline}. Pipeline uses {bubble, correction, normal_growth, recovery}. These are fundamentally different classification schemes answering different questions.

## COMPETITION GAPS (judge-facing)

1. **Notebook is the submission** -- judges see the inline ML, not the pipeline module. The pipeline's rigor (price-excluded RF, multi-reference DTW, BH across all layers) never appears in the deliverable.
2. **No confusion matrix or classification report rendered in notebook**: Cell 63 computes CV scores but does not display a confusion matrix heatmap or per-class metrics.
3. **No augmented-vs-core comparison**: Spec calls for reporting whether sentiment changes the classification. Not present in either codebase.
4. **SHAP waterfall chart may not render**: Cell 68 tries both waterfall and summary, but the waterfall uses a deprecated `shap.force_plot` fallback path that often fails in notebooks.
5. **Rolling DTW chart in notebook**: Uses arbitrary window sizes (100, 150, 200...) on daily data, not the monthly feature-level approach from the pipeline spec.
