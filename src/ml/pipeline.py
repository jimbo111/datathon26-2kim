"""ML Pipeline — AI Hype Cycle analysis.

Three models attack the research question from different angles:
  1. Regime Classifier   — "What market state are we in right now?"
  2. DTW Similarity      — "How similar is the AI cycle to the dot-com bubble?"
  3. SHAP Analysis       — "Which indicators drive the bubble signal?"

Entry point: run_ml_pipeline(layer_results, force_refresh=False)

Critical design choices (per spec §5):
  - Time-based splits ONLY — never random split time series.
  - No look-ahead bias — all features use only trailing data.
  - Price-excluded model variant is REQUIRED (defuses circular-reasoning critique).
  - DTW null distribution with permutation test is REQUIRED.
  - Benjamini-Hochberg correction across all p-values is REQUIRED.
  - Sentiment features: NO imputation — train two separate models instead.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE = "plotly_dark"
BUBBLE_COLOR = "#EF553B"
NORMAL_COLOR = "#00CC96"
CHART_DIR = Path(__file__).resolve().parents[2] / "submissions" / "charts"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

CHART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REGIME_CLASSES = ["bubble", "correction", "normal_growth", "recovery"]

# Price-derived features to exclude in the price-excluded model variant
PRICE_FEATURES = [
    "momentum_30d",
    "momentum_90d",
    "momentum_252d",
    "volatility_30d",
    "volatility_90d",
    "drawdown_from_ath",
    "rsi_14",
    "price_to_sma200",
    "return_skewness",
]

# DTW feature weights (must sum to 1.0)
DTW_WEIGHTS = {
    "price": 0.25,
    "fundamentals": 0.25,
    "concentration": 0.20,
    "macro": 0.15,
    "sentiment": 0.15,
}

# Ensemble model weights
ENSEMBLE_WEIGHTS = {"rf": 0.4, "xgb": 0.4, "lr": 0.2}

# ---------------------------------------------------------------------------
# Lazy imports — these are large; import only when needed
# ---------------------------------------------------------------------------


def _import_sklearn():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    return (
        RandomForestClassifier, LogisticRegression, TimeSeriesSplit,
        GridSearchCV, classification_report, confusion_matrix, f1_score,
        StandardScaler, LabelEncoder,
    )


def _import_xgb():
    """Import XGBClassifier, returning None if xgboost is not installed."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        console.print("[yellow]xgboost not installed — XGB model will be skipped.[/]")
        return None


def _import_shap():
    import shap
    return shap


def _import_statsmodels():
    from statsmodels.stats.multitest import multipletests
    return multipletests


# ---------------------------------------------------------------------------
# 1. Feature Matrix Construction (delegates to feature_matrix module)
# ---------------------------------------------------------------------------


def build_feature_matrix(layer_results: dict) -> dict[str, pd.DataFrame]:
    """Build unified monthly feature matrix from all 5 layer outputs.

    Delegates to src.ml.feature_matrix for the per-layer extraction logic.

    Returns
    -------
    dict with keys: "core", "augmented", "has_sentiment_data",
                    "feature_names_core", "feature_names_augmented"
    """
    from src.ml.feature_matrix import build_feature_matrix as _build
    return _build(layer_results)


# ---------------------------------------------------------------------------
# 2. Regime Labels
# ---------------------------------------------------------------------------


def assign_regime_labels(df_or_index) -> pd.Series:
    """Assign regime labels to a DataFrame's DatetimeIndex or a DatetimeIndex directly.

    See feature_matrix.assign_regime_labels for full label definitions.

    Parameters
    ----------
    df_or_index : pd.DataFrame or pd.DatetimeIndex
    """
    from src.ml.feature_matrix import assign_regime_labels as _assign
    if isinstance(df_or_index, pd.DatetimeIndex):
        return _assign(df_or_index)
    return _assign(df_or_index.index)


# ---------------------------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------------------------


def train_regime_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Train three regime classifiers with TimeSeriesSplit cross-validation.

    Also trains a price-feature-excluded RF variant per spec §2.6.

    Parameters
    ----------
    X_train : Scaled or unscaled training features (will be scaled internally).
    y_train : Regime labels aligned with X_train.
    X_feature_names : Column names (used if X_train is a numpy array).

    Returns
    -------
    dict with keys: "models", "scalers", "cv_scores", "label_encoder",
                    "price_excluded_rf", "price_excluded_scaler", "classes"
    """
    (
        RandomForestClassifier, LogisticRegression, TimeSeriesSplit,
        GridSearchCV, classification_report, confusion_matrix, f1_score,
        StandardScaler, LabelEncoder,
    ) = _import_sklearn()
    XGBClassifier = _import_xgb()

    console.print("\n[bold cyan]Training regime classifiers...[/]")

    # Align X and y
    common_idx = X_train.index.intersection(y_train.index)
    X = X_train.loc[common_idx].copy()
    y = y_train.loc[common_idx].copy()

    if len(X) < 50:
        raise ValueError(
            f"Insufficient training data: {len(X)} observations. Need at least 50."
        )

    # Encode labels to integers for XGBoost
    le = LabelEncoder()
    le.fit(REGIME_CLASSES)
    y_enc = pd.Series(le.transform(y), index=y.index, name="regime_encoded")

    # Fill NaN with column medians BEFORE scaling.
    # This is done on the training set only (no look-ahead bias).
    # Note: we store the training medians so we can apply them at inference.
    train_medians = X.median()
    X_filled = X.fillna(train_medians)

    # Scale features (fit on training data only)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_filled),
        index=X.index,
        columns=X.columns,
    )

    # TimeSeriesSplit with gap to prevent leakage.
    # test_size=24 requires at least n_splits*(test_size+gap)+1 samples.
    # Adapt test_size downward for smaller datasets.
    n_samples = len(X)
    n_splits = 5
    gap = 3
    # minimum samples needed: n_splits * (test_size + gap) + 1
    # => test_size <= (n_samples - 1) / n_splits - gap
    max_test_size = max(1, (n_samples - 1) // n_splits - gap)
    ts_test_size = min(24, max_test_size)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=ts_test_size)

    # ---- Random Forest ----
    console.print("  [cyan]Training Random Forest...[/]")
    rf_params = {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "oob_score": True,
    }
    rf_model = RandomForestClassifier(**rf_params)

    rf_cv_scores = _time_series_cv_score(rf_model, X_scaled.values, y.values, tscv)
    rf_model.fit(X_scaled.values, y.values)
    console.print(
        f"    CV F1 (weighted): {np.mean(rf_cv_scores):.3f} ± {np.std(rf_cv_scores):.3f}"
        f"  |  OOB: {rf_model.oob_score_:.3f}"
    )

    # ---- XGBoost ----
    xgb_model = None
    xgb_cv_scores = np.array([0.0])
    if XGBClassifier is not None:
        console.print("  [cyan]Training XGBoost...[/]")
        xgb_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        xgb_model = XGBClassifier(**xgb_params)
        xgb_cv_scores = _time_series_cv_score(xgb_model, X_scaled.values, y_enc.values, tscv)
        xgb_model.fit(X_scaled.values, y_enc.values)
        console.print(
            f"    CV F1 (weighted): {np.mean(xgb_cv_scores):.3f} ± {np.std(xgb_cv_scores):.3f}"
        )
    else:
        console.print("  [yellow]XGBoost skipped (not installed).[/]")

    # ---- Logistic Regression ----
    console.print("  [cyan]Training Logistic Regression...[/]")
    lr_params = {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
        "multi_class": "multinomial",
        "solver": "lbfgs",
    }
    lr_model = LogisticRegression(**lr_params)
    lr_cv_scores = _time_series_cv_score(lr_model, X_scaled.values, y.values, tscv)
    lr_model.fit(X_scaled.values, y.values)
    console.print(
        f"    CV F1 (weighted): {np.mean(lr_cv_scores):.3f} ± {np.std(lr_cv_scores):.3f}"
    )

    # ---- Price-excluded RF variant (spec §2.6) ----
    console.print("  [cyan]Training price-excluded RF variant...[/]")
    price_cols_present = [c for c in PRICE_FEATURES if c in X.columns]
    X_no_price = X_filled.drop(columns=price_cols_present)
    no_price_medians = X_no_price.median()
    X_no_price_filled = X_no_price.fillna(no_price_medians)
    scaler_no_price = StandardScaler()
    X_no_price_scaled = pd.DataFrame(
        scaler_no_price.fit_transform(X_no_price_filled),
        index=X_no_price_filled.index,
        columns=X_no_price_filled.columns,
    )
    rf_no_price = RandomForestClassifier(**rf_params)
    rf_no_price_cv = _time_series_cv_score(
        rf_no_price, X_no_price_scaled.values, y.values, tscv
    )
    rf_no_price.fit(X_no_price_scaled.values, y.values)
    console.print(
        f"    Price-excluded CV F1: {np.mean(rf_no_price_cv):.3f} ± {np.std(rf_no_price_cv):.3f}"
        f"  |  OOB: {rf_no_price.oob_score_:.3f}"
    )

    return {
        "models": {
            "rf": rf_model,
            "xgb": xgb_model,
            "lr": lr_model,
        },
        "scalers": {
            "main": scaler,
            "no_price": scaler_no_price,
        },
        "cv_scores": {
            "rf": rf_cv_scores.tolist(),
            "xgb": xgb_cv_scores.tolist(),
            "lr": lr_cv_scores.tolist(),
            "rf_no_price": rf_no_price_cv.tolist(),
        },
        "label_encoder": le,
        "price_excluded_rf": rf_no_price,
        "price_excluded_scaler": scaler_no_price,
        "price_excluded_features": list(X_no_price.columns),
        "train_medians": train_medians,
        "no_price_medians": no_price_medians,
        "classes": list(le.classes_),
        "feature_names": list(X.columns),
        "n_train": len(X),
    }


def _time_series_cv_score(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    tscv: Any,
) -> np.ndarray:
    """Run TimeSeriesSplit CV and return per-fold weighted F1 scores."""
    from sklearn.metrics import f1_score
    from sklearn.base import clone

    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        m = clone(model)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_val)
        score = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        scores.append(score)

    return np.array(scores) if scores else np.array([0.0])


# ---------------------------------------------------------------------------
# 4. Ensemble Prediction
# ---------------------------------------------------------------------------


def predict_regime(
    trained: dict,
    X_live: pd.DataFrame,
) -> dict[str, Any]:
    """Generate regime probability predictions using weighted ensemble.

    Ensemble weights: RF=0.40, XGB=0.40, LR=0.20 (per spec §4).

    Parameters
    ----------
    trained : Output dict from train_regime_classifier.
    X_live  : Feature DataFrame for the period to classify (e.g. Jul 2024 - Mar 2026).

    Returns
    -------
    dict with keys:
        "probabilities"     — DataFrame indexed by date, cols = regime classes
        "ensemble_probs"    — same as above (full period)
        "march_2026"        — dict of regime: probability for latest observation
        "price_excl_march"  — same but price-excluded model
        "verdict"           — human-readable classification string
        "model_agreement"   — ensemble agreement check
    """
    models = trained["models"]
    scaler = trained["scalers"]["main"]
    le = trained["label_encoder"]
    classes = trained["classes"]

    # Ensure feature alignment
    feature_names = trained.get("feature_names", list(X_live.columns))
    X_aligned = X_live.reindex(columns=feature_names).copy()

    # Fill NaN with TRAINING medians (no look-ahead bias — must not use test/live stats)
    train_medians = trained.get("train_medians", pd.Series(dtype=float))
    if not train_medians.empty:
        X_aligned = X_aligned.fillna(train_medians.reindex(X_aligned.columns))
    X_aligned = X_aligned.fillna(0)  # any remaining (e.g. feature not in training)

    X_scaled = scaler.transform(X_aligned)

    # ---- Per-model probabilities ----
    rf_probs = models["rf"].predict_proba(X_scaled)     # (N, 4), classes = rf.classes_
    lr_probs = models["lr"].predict_proba(X_scaled)     # (N, 4), classes = lr.classes_

    # Ensure RF and LR class ordering matches REGIME_CLASSES
    rf_probs = _reorder_probs(rf_probs, models["rf"].classes_, classes)
    lr_probs = _reorder_probs(lr_probs, models["lr"].classes_, classes)

    # XGBoost: use integer-encoded labels — remap to class order
    # If XGBoost was not trained (module not installed), fall back to equal RF/LR split
    xgb_probs = np.zeros_like(rf_probs)
    if models.get("xgb") is not None:
        xgb_raw = models["xgb"].predict_proba(X_scaled)    # (N, 4), indices = 0..3
        for enc_idx, cls_name in enumerate(le.classes_):
            if cls_name in classes:
                dst_idx = classes.index(cls_name)
                xgb_probs[:, dst_idx] = xgb_raw[:, enc_idx]
        w_rf = ENSEMBLE_WEIGHTS["rf"]
        w_xgb = ENSEMBLE_WEIGHTS["xgb"]
        w_lr = ENSEMBLE_WEIGHTS["lr"]
    else:
        # Redistribute XGB weight equally between RF and LR
        w_rf = ENSEMBLE_WEIGHTS["rf"] + ENSEMBLE_WEIGHTS["xgb"] / 2
        w_xgb = 0.0
        w_lr = ENSEMBLE_WEIGHTS["lr"] + ENSEMBLE_WEIGHTS["xgb"] / 2

    # Weighted ensemble
    ensemble = w_rf * rf_probs + w_xgb * xgb_probs + w_lr * lr_probs
    ensemble_df = pd.DataFrame(ensemble, index=X_live.index, columns=classes)

    # ---- Price-excluded model for latest observation ----
    no_price_result = None
    try:
        scaler_np = trained["price_excluded_scaler"]
        rf_np = trained["price_excluded_rf"]
        feats_np = trained.get("price_excluded_features", [])
        no_price_medians = trained.get("no_price_medians", pd.Series(dtype=float))
        X_np_aligned = X_live.reindex(columns=feats_np)
        if not no_price_medians.empty:
            X_np_aligned = X_np_aligned.fillna(no_price_medians.reindex(X_np_aligned.columns))
        X_np_aligned = X_np_aligned.fillna(0)
        X_np_scaled = scaler_np.transform(X_np_aligned)
        np_probs_arr = rf_np.predict_proba(X_np_scaled[-1:])
        np_probs = _reorder_probs(np_probs_arr, rf_np.classes_, classes)
        no_price_result = {cls: float(p) for cls, p in zip(classes, np_probs[0])}
    except Exception as exc:
        console.print(f"[yellow]Price-excluded inference failed: {exc}[/]")

    # ---- March 2026 (latest observation) ----
    latest = ensemble_df.iloc[-1].to_dict()
    latest_date = ensemble_df.index[-1]

    # ---- Ensemble agreement ----
    xgb_available = models.get("xgb") is not None
    agreement = _check_ensemble_agreement(
        rf_probs[-1], xgb_probs[-1], lr_probs[-1], classes,
        xgb_available=xgb_available,
    )

    # ---- Verdict ----
    top_regime = max(latest, key=latest.get)
    bubble_pct = latest.get("bubble", 0.0) * 100
    verdict = _format_verdict(top_regime, bubble_pct, latest_date)

    console.print("\n[bold green]Regime Classification — March 2026 (Ensemble)[/]")
    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("Regime", style="cyan")
    t.add_column("Probability", justify="right")
    for cls, prob in sorted(latest.items(), key=lambda x: -x[1]):
        style = "bold red" if cls == "bubble" and prob > 0.35 else ""
        t.add_row(cls, f"{prob:.1%}", style=style)
    console.print(t)
    console.print(f"\n[bold]{verdict}[/]")

    if no_price_result:
        console.print(
            f"\n[dim]Price-excluded model — bubble prob: "
            f"{no_price_result.get('bubble', 0):.1%} "
            f"(diff from full model: "
            f"{abs(no_price_result.get('bubble', 0) - latest.get('bubble', 0)):.1%})[/]"
        )

    return {
        "probabilities": ensemble_df,
        "ensemble_probs": ensemble_df,
        "march_2026": latest,
        "march_2026_date": str(latest_date.date()),
        "price_excl_march": no_price_result,
        "verdict": verdict,
        "model_agreement": agreement,
        "rf_probs": pd.DataFrame(rf_probs, index=X_live.index, columns=classes),
        "xgb_probs": pd.DataFrame(xgb_probs, index=X_live.index, columns=classes),
        "lr_probs": pd.DataFrame(lr_probs, index=X_live.index, columns=classes),
    }


def _reorder_probs(
    probs: np.ndarray,
    model_classes: list | np.ndarray,
    target_classes: list[str],
) -> np.ndarray:
    """Reorder probability columns to match target_classes ordering."""
    model_cls_list = list(model_classes)
    reordered = np.zeros((probs.shape[0], len(target_classes)))
    for dst_idx, cls in enumerate(target_classes):
        if cls in model_cls_list:
            src_idx = model_cls_list.index(cls)
            reordered[:, dst_idx] = probs[:, src_idx]
    return reordered


def _check_ensemble_agreement(
    rf_p: np.ndarray,
    xgb_p: np.ndarray,
    lr_p: np.ndarray,
    classes: list[str],
    xgb_available: bool = True,
) -> dict:
    """Check model agreement using Jensen-Shannon divergence.

    When XGBoost is unavailable, compares RF and LR only.
    """
    from scipy.spatial.distance import jensenshannon

    rf_pred = classes[int(np.argmax(rf_p))]
    lr_pred = classes[int(np.argmax(lr_p))]
    eps = 1e-10

    if xgb_available and xgb_p.sum() > 0:
        xgb_pred = classes[int(np.argmax(xgb_p))]
        preds = [rf_pred, xgb_pred, lr_pred]
        mean_p = (rf_p + xgb_p + lr_p) / 3.0
        js_div = float(np.mean([
            jensenshannon(rf_p + eps, mean_p + eps),
            jensenshannon(xgb_p + eps, mean_p + eps),
            jensenshannon(lr_p + eps, mean_p + eps),
        ]))
    else:
        xgb_pred = "N/A"
        preds = [rf_pred, lr_pred]
        mean_p = (rf_p + lr_p) / 2.0
        js_div = float(np.mean([
            jensenshannon(rf_p + eps, mean_p + eps),
            jensenshannon(lr_p + eps, mean_p + eps),
        ]))

    unanimous = len(set(preds)) == 1
    majority = max(set(preds), key=preds.count)

    return {
        "unanimous": bool(unanimous),
        "majority_prediction": majority,
        "jensen_shannon_divergence": js_div,
        "per_model": {"RF": rf_pred, "XGB": xgb_pred, "LR": lr_pred},
    }


def _format_verdict(top_regime: str, bubble_pct: float, date: pd.Timestamp) -> str:
    if bubble_pct >= 60:
        level = "HIGH BUBBLE RISK"
    elif bubble_pct >= 40:
        level = "ELEVATED BUBBLE RISK"
    elif bubble_pct >= 25:
        level = "MODERATE BUBBLE RISK"
    else:
        level = f"REGIME: {top_regime.upper().replace('_', ' ')}"
    return (
        f"{level} — {bubble_pct:.1f}% bubble probability as of "
        f"{date.strftime('%b %Y')}"
    )


# ---------------------------------------------------------------------------
# 5. Model Evaluation
# ---------------------------------------------------------------------------


def evaluate_models(
    trained: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate trained models on the hold-out test set.

    Parameters
    ----------
    trained : Output from train_regime_classifier.
    X_test  : Test features (must be aligned to training feature names).
    y_test  : True regime labels for the test period.

    Returns
    -------
    dict with classification reports, confusion matrices, and F1 scores.
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
    )

    models = trained["models"]
    scaler = trained["scalers"]["main"]
    le = trained["label_encoder"]
    classes = trained["classes"]

    feature_names = trained.get("feature_names", list(X_test.columns))
    train_medians = trained.get("train_medians", pd.Series(dtype=float))

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=feature_names)
        if not train_medians.empty:
            df = df.fillna(train_medians.reindex(df.columns))
        return df.fillna(0)

    common = X_test.index.intersection(y_test.index)
    X_scaled_common = scaler.transform(_fill(X_test.loc[common]))
    y_true = y_test.loc[common]

    results = {}

    for model_name, model in models.items():
        if model is None:
            console.print(f"  [dim]{model_name.upper()} skipped (not trained).[/]")
            continue
        if model_name == "xgb":
            y_pred_enc = model.predict(X_scaled_common)
            y_pred = le.inverse_transform(y_pred_enc.astype(int))
        else:
            y_pred = model.predict(X_scaled_common)

        report = classification_report(
            y_true, y_pred, labels=classes, zero_division=0, output_dict=True
        )
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        results[model_name] = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "f1_weighted": float(f1_w),
            "classes": classes,
        }

        console.print(
            f"  [cyan]{model_name.upper()}[/] test F1 (weighted): [bold]{f1_w:.3f}[/]"
        )
        console.print(
            classification_report(y_true, y_pred, labels=classes, zero_division=0)
        )

    return results


# ---------------------------------------------------------------------------
# 6. DTW Similarity
# ---------------------------------------------------------------------------


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0, 1]. Returns zeros if constant."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def _dtw_distance_1d(a: np.ndarray, b: np.ndarray, window: int | None = None) -> float:
    """Compute DTW distance between two 1D normalized series.

    Tries dtaidistance first (fast C implementation), falls back to a pure
    numpy DP implementation if the library is not installed.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    try:
        from dtaidistance import dtw as dtai_dtw
        kwargs: dict = {"use_pruning": True}
        if window is not None:
            kwargs["window"] = window
        return float(dtai_dtw.distance(a, b, **kwargs))
    except ImportError:
        pass

    # Pure numpy DP fallback
    n, m = len(a), len(b)
    cost = np.full((n, m), np.inf)
    cost[0, 0] = abs(a[0] - b[0])
    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + abs(a[i] - b[0])
    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + abs(a[0] - b[j])
    for i in range(1, n):
        for j in range(1, m):
            if window is not None and abs(i - j) > window:
                continue
            cost[i, j] = abs(a[i] - b[j]) + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    return float(cost[-1, -1])


def _dtw_similarity(series_a: np.ndarray, series_b: np.ndarray, window: int | None = None) -> float:
    """Convert DTW distance to similarity score in [0, 100]."""
    dist = _dtw_distance_1d(series_a, series_b, window=window)
    path_len = np.sqrt(len(series_a) * len(series_b))
    normalized = dist / max(path_len, 1e-9)
    return float(max(0.0, (1.0 - normalized)) * 100.0)


def compute_dtw_similarity(
    ai_era_features: dict[str, np.ndarray],
    reference_features: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    n_permutations: int = 500,
    n_bootstrap: int = 200,
    seed: int = 42,
    label: str = "dotcom",
) -> dict[str, Any]:
    """Compute weighted composite DTW similarity + null distribution + permutation test.

    Both ai_era_features and reference_features are dicts keyed by feature category
    (e.g. "price", "fundamentals", "concentration", "macro", "sentiment").
    Values are 1D numpy arrays, already normalized to [0,1].

    Parameters
    ----------
    ai_era_features    : Dict of feature category -> normalized array (AI era).
    reference_features : Dict of feature category -> normalized array (reference period).
    weights            : Category weights (must sum to 1.0). Uses DTW_WEIGHTS if None.
    n_permutations     : Number of permutations for empirical p-value (min 500 per spec).
    n_bootstrap        : Number of bootstrap resamples for CI.
    seed               : RNG seed for reproducibility.
    label              : Human-readable name for the reference period.

    Returns
    -------
    dict with:
        "composite_score"         — weighted DTW similarity 0-100
        "per_feature_scores"      — per-category scores
        "p_value"                 — empirical p-value from permutation test
        "p_value_percentile"      — 100 * (1 - p_value)
        "bootstrap_ci"            — [2.5%, 50%, 97.5%] from bootstrap
        "null_mean"               — mean similarity under H0
        "null_std"                — std similarity under H0
        "reference_label"         — label of the comparison period
    """
    if weights is None:
        weights = DTW_WEIGHTS

    rng = np.random.RandomState(seed)

    # ---- Per-feature DTW similarity ----
    per_feature: dict[str, float] = {}
    composite = 0.0

    for cat, w in weights.items():
        if cat not in ai_era_features or cat not in reference_features:
            continue
        arr_ai = np.asarray(ai_era_features[cat], dtype=np.float64)
        arr_ref = np.asarray(reference_features[cat], dtype=np.float64)
        if len(arr_ai) < 2 or len(arr_ref) < 2:
            continue

        # Sakoe-Chiba band: 33% of the longer series
        sc_window = max(len(arr_ai), len(arr_ref)) // 3

        score = _dtw_similarity(arr_ai, arr_ref, window=sc_window)
        per_feature[cat] = score
        composite += w * score

    if not per_feature:
        console.print(f"[yellow]DTW: No common features between AI era and {label}.[/]")
        return {"composite_score": 0.0, "reference_label": label, "error": "no_common_features"}

    # ---- Permutation test using the "price" feature as the primary series ----
    primary_cat = "price" if "price" in ai_era_features else list(ai_era_features.keys())[0]
    arr_ai_perm = np.asarray(ai_era_features[primary_cat], dtype=np.float64)
    arr_ref_perm = np.asarray(reference_features[primary_cat], dtype=np.float64)
    sc_w = max(len(arr_ai_perm), len(arr_ref_perm)) // 3

    observed_dist = _dtw_distance_1d(arr_ai_perm, arr_ref_perm, window=sc_w)
    path_len = np.sqrt(len(arr_ai_perm) * len(arr_ref_perm))
    observed_sim = max(0.0, (1.0 - observed_dist / max(path_len, 1e-9))) * 100.0

    null_dists = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(arr_ai_perm)
        d = _dtw_distance_1d(arr_ref_perm, shuffled, window=sc_w)
        null_dists.append(d)

    null_dists_arr = np.array(null_dists)
    # p-value: fraction of null distances <= observed (smaller distance = more similar)
    p_value = float(np.mean(null_dists_arr <= observed_dist))
    null_sims = [max(0.0, (1.0 - d / max(path_len, 1e-9))) * 100.0 for d in null_dists]

    # ---- Bootstrap CI (add small noise to both series) ----
    boot_scores = []
    noise_std = 0.02
    for _ in range(n_bootstrap):
        noisy_ai = {
            k: np.clip(v + rng.normal(0, noise_std, len(v)), 0, 1)
            for k, v in ai_era_features.items()
        }
        noisy_ref = {
            k: np.clip(v + rng.normal(0, noise_std, len(v)), 0, 1)
            for k, v in reference_features.items()
        }
        boot_composite = 0.0
        for cat, w in weights.items():
            if cat not in noisy_ai or cat not in noisy_ref:
                continue
            sc = max(len(noisy_ai[cat]), len(noisy_ref[cat])) // 3
            boot_composite += w * _dtw_similarity(noisy_ai[cat], noisy_ref[cat], window=sc)
        boot_scores.append(boot_composite)

    ci = np.percentile(boot_scores, [2.5, 50, 97.5]).tolist() if boot_scores else [0, 0, 0]

    console.print(
        f"  DTW [{label:20s}]: score={composite:.1f}  "
        f"p={p_value:.3f}  "
        f"CI=[{ci[0]:.1f}, {ci[2]:.1f}]  "
        f"percentile={((1 - p_value) * 100):.0f}th"
    )

    return {
        "composite_score": float(composite),
        "per_feature_scores": per_feature,
        "p_value": p_value,
        "p_value_percentile": float((1.0 - p_value) * 100.0),
        "bootstrap_ci": ci,
        "null_mean": float(np.mean(null_sims)),
        "null_std": float(np.std(null_sims)),
        "reference_label": label,
        "observed_similarity": observed_sim,
    }


def _extract_dtw_features_from_matrix(
    feature_matrix: pd.DataFrame,
    start: str,
    end: str,
) -> dict[str, np.ndarray]:
    """Slice feature matrix for a given period and split into DTW categories.

    Returns dict of category -> normalized 1D numpy array.
    Columns are averaged within each category then normalized to [0,1].
    """
    from src.ml.feature_matrix import PRICE_FEATURES

    mask = (feature_matrix.index >= start) & (feature_matrix.index <= end)
    period_df = feature_matrix.loc[mask].copy()

    if period_df.empty:
        return {}

    price_cols = [c for c in PRICE_FEATURES if c in period_df.columns]
    fund_cols = [c for c in ["log_pe", "pe_percentile", "cape_shiller",
                              "revenue_growth_yoy", "fcf_yield",
                              "earnings_growth_yoy", "pe_to_growth"]
                 if c in period_df.columns]
    conc_cols = [c for c in ["top5_weight", "top10_weight", "spy_rsp_spread",
                              "hhi", "buffett_indicator"]
                 if c in period_df.columns]
    macro_cols = [c for c in ["fed_funds", "m2_growth", "yield_curve",
                               "credit_spread", "real_rate", "unemployment"]
                  if c in period_df.columns]
    sent_cols = [c for c in ["ai_mention_rate", "hype_score", "google_trends_ai"]
                 if c in period_df.columns]

    out: dict[str, np.ndarray] = {}
    cat_map = {
        "price": price_cols,
        "fundamentals": fund_cols,
        "concentration": conc_cols,
        "macro": macro_cols,
        "sentiment": sent_cols,
    }
    for cat, cols in cat_map.items():
        if not cols:
            continue
        arr = period_df[cols].dropna(how="all").mean(axis=1).values
        if len(arr) < 2:
            continue
        out[cat] = _normalize_01(arr)

    return out


def run_dtw_analysis(
    feature_matrix: pd.DataFrame,
    n_permutations: int = 500,
) -> dict[str, Any]:
    """Full DTW similarity analysis with null distribution comparisons.

    Compares the AI era (Jan 2023 - Mar 2026) against:
      1. Dot-com bubble (Jan 1998 - Dec 2001) — primary analysis
      2. Housing bubble (Jan 2005 - Dec 2008)
      3. Post-GFC recovery (Apr 2009 - Dec 2013)
      4. COVID rally (Apr 2020 - Dec 2022)

    Also computes rolling DTW similarity timeline.

    Returns
    -------
    dict with comparison results and rolling similarity timeline.
    """
    console.print("\n[bold cyan]Running DTW Similarity Analysis...[/]")

    ai_features = _extract_dtw_features_from_matrix(
        feature_matrix, "2023-01-01", "2026-03-31"
    )
    if not ai_features:
        console.print("[red]DTW: No features for AI era. Skipping.[/]")
        return {"error": "no_ai_era_features"}

    reference_periods = [
        ("dotcom",      "1998-01-01", "2001-12-31"),
        ("housing",     "2005-01-01", "2008-12-31"),
        ("post_gfc",    "2009-04-01", "2013-12-31"),
        ("covid_rally", "2020-04-01", "2022-12-31"),
    ]

    comparisons: dict[str, Any] = {}
    for label, start, end in reference_periods:
        ref_features = _extract_dtw_features_from_matrix(feature_matrix, start, end)
        if not ref_features:
            console.print(f"  [yellow]DTW: No features for {label} period. Skipping.[/]")
            continue
        result = compute_dtw_similarity(
            ai_era_features=ai_features,
            reference_features=ref_features,
            n_permutations=n_permutations,
            label=label,
        )
        comparisons[label] = result

    # ---- Rolling similarity timeline ----
    rolling_timeline = _compute_rolling_dtw_timeline(feature_matrix)

    # ---- Summary table ----
    if comparisons:
        t = Table(title="DTW Similarity — AI Era vs Reference Periods", show_header=True)
        t.add_column("Period", style="cyan")
        t.add_column("Score (0-100)", justify="right")
        t.add_column("p-value", justify="right")
        t.add_column("95% CI", justify="right")
        for lbl, res in comparisons.items():
            if "error" in res:
                continue
            ci = res.get("bootstrap_ci", [0, 0, 0])
            t.add_row(
                lbl,
                f"{res['composite_score']:.1f}",
                f"{res['p_value']:.3f}",
                f"[{ci[0]:.1f}, {ci[2]:.1f}]",
            )
        console.print(t)

    return {
        "comparisons": comparisons,
        "rolling_timeline": rolling_timeline,
        "primary": comparisons.get("dotcom", {}),
    }


def _compute_rolling_dtw_timeline(
    feature_matrix: pd.DataFrame,
    min_months: int = 6,
) -> pd.Series:
    """Compute DTW similarity vs dot-com for each month as AI era accumulates."""
    dotcom_features_full = _extract_dtw_features_from_matrix(
        feature_matrix, "1998-01-01", "2001-12-31"
    )
    if not dotcom_features_full:
        return pd.Series(dtype=float)

    ai_mask = (feature_matrix.index >= "2023-01-01") & (feature_matrix.index <= "2026-03-31")
    ai_dates = feature_matrix.index[ai_mask]

    if len(ai_dates) < min_months:
        return pd.Series(dtype=float)

    scores: dict[pd.Timestamp, float] = {}
    for t_idx in range(min_months, len(ai_dates) + 1):
        end_date = ai_dates[t_idx - 1]
        ai_t = _extract_dtw_features_from_matrix(
            feature_matrix, "2023-01-01", str(end_date.date())
        )
        if not ai_t:
            continue

        # Use the same number of months from dot-com era for fair comparison
        dotcom_dates_all = feature_matrix.loc[
            (feature_matrix.index >= "1998-01-01") & (feature_matrix.index <= "2001-12-31")
        ].index
        if t_idx > len(dotcom_dates_all):
            break
        dotcom_end = dotcom_dates_all[t_idx - 1]
        dotcom_t = _extract_dtw_features_from_matrix(
            feature_matrix, "1998-01-01", str(dotcom_end.date())
        )
        if not dotcom_t:
            continue

        result = compute_dtw_similarity(
            ai_era_features=ai_t,
            reference_features=dotcom_t,
            n_permutations=50,  # faster for rolling
            label=f"rolling_{t_idx}",
        )
        scores[end_date] = result.get("composite_score", np.nan)

    return pd.Series(scores, name="dtw_similarity")


# ---------------------------------------------------------------------------
# 7. SHAP Analysis
# ---------------------------------------------------------------------------


def compute_shap_analysis(
    rf_model: Any,
    X_test: pd.DataFrame,
    X_march_2026: pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute SHAP values for the Random Forest model.

    Parameters
    ----------
    rf_model    : Trained RandomForestClassifier (scaled input).
    X_test      : Scaled test feature matrix.
    X_march_2026 : Scaled features for the latest observation (Mar 2026).
    feature_names : Column names list.

    Returns
    -------
    dict with SHAP values, feature importance rankings, and waterfall data.
    """
    shap = _import_shap()
    console.print("\n[bold cyan]Computing SHAP values...[/]")

    if feature_names is None and hasattr(X_test, "columns"):
        feature_names = list(X_test.columns)

    X_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)

    explainer = shap.TreeExplainer(rf_model)

    # SHAP values: list of arrays, one per class, each shape (N, n_features)
    shap_values = explainer.shap_values(X_arr)

    # Handle both list (multi-output) and single array return types
    if isinstance(shap_values, list):
        classes = list(rf_model.classes_)
        bubble_idx = classes.index("bubble") if "bubble" in classes else 0
        shap_bubble = shap_values[bubble_idx]
    else:
        # XGBoost TreeExplainer may return 3D array (N, n_features, n_classes)
        if shap_values.ndim == 3:
            classes = list(rf_model.classes_)
            bubble_idx = classes.index("bubble") if "bubble" in classes else 0
            shap_bubble = shap_values[:, :, bubble_idx]
        else:
            shap_bubble = shap_values
            bubble_idx = 0

    # Global feature importance: mean |SHAP| for bubble class
    mean_abs_shap = np.abs(shap_bubble).mean(axis=0)
    importance_series = pd.Series(
        mean_abs_shap,
        index=feature_names if feature_names else range(len(mean_abs_shap)),
    ).sort_values(ascending=False)

    console.print("  Top 10 features driving bubble classification:")
    for feat, val in importance_series.head(10).items():
        console.print(f"    {feat:30s} {val:.4f}")

    # SHAP for latest observation (Mar 2026)
    shap_march = None
    march_explanation = None
    if X_march_2026 is not None:
        X_m_arr = X_march_2026.values if hasattr(X_march_2026, "values") else np.asarray(X_march_2026)
        shap_march_all = explainer.shap_values(X_m_arr)

        if isinstance(shap_march_all, list):
            shap_march = shap_march_all[bubble_idx]
        elif shap_march_all.ndim == 3:
            shap_march = shap_march_all[:, :, bubble_idx]
        else:
            shap_march = shap_march_all

        expected_val = (
            explainer.expected_value[bubble_idx]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )

        march_explanation = {
            "shap_values": shap_march[0].tolist() if len(shap_march) > 0 else [],
            "base_value": float(expected_val),
            "feature_names": feature_names or [],
            "feature_values": (
                X_march_2026.iloc[0].tolist()
                if hasattr(X_march_2026, "iloc")
                else X_m_arr[0].tolist()
            ),
        }

    return {
        "shap_values_bubble": shap_bubble,
        "shap_values_all": shap_values,
        "feature_importance": importance_series.to_dict(),
        "feature_importance_ranked": list(importance_series.index),
        "bubble_class_idx": bubble_idx,
        "march_2026_shap": shap_march,
        "march_2026_explanation": march_explanation,
        "expected_value": (
            float(explainer.expected_value[bubble_idx])
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else float(explainer.expected_value)
        ),
        "top10_features": list(importance_series.head(10).index),
    }


# ---------------------------------------------------------------------------
# 8. Benjamini-Hochberg Correction
# ---------------------------------------------------------------------------


def apply_bh_correction(all_pvalues: dict[str, float | dict]) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction across all p-values.

    Parameters
    ----------
    all_pvalues : Flat dict of test_identifier -> raw_p_value, OR
                  nested dict of layer -> {test_name -> raw_p_value}.

    Returns
    -------
    DataFrame with columns: layer, test_name, raw_p, bh_adjusted_p,
                            significant_raw, significant_bh, lost_significance
    """
    multipletests = _import_statsmodels()

    # Flatten if nested
    records = []
    for key, val in all_pvalues.items():
        if isinstance(val, dict):
            layer = key
            for test_name, p in val.items():
                if isinstance(p, (int, float)) and not np.isnan(float(p)):
                    records.append({"layer": layer, "test_name": test_name, "raw_p": float(p)})
        elif isinstance(val, (int, float)) and not np.isnan(float(val)):
            parts = key.split(".", 1)
            records.append({
                "layer": parts[0] if len(parts) > 1 else "unknown",
                "test_name": parts[1] if len(parts) > 1 else key,
                "raw_p": float(val),
            })

    if not records:
        console.print("[yellow]BH correction: No valid p-values provided.[/]")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    reject, pvals_corrected, _, _ = multipletests(
        df["raw_p"].values,
        alpha=0.05,
        method="fdr_bh",
    )

    df["bh_adjusted_p"] = pvals_corrected
    df["significant_raw"] = df["raw_p"] < 0.05
    df["significant_bh"] = reject
    df["lost_significance"] = df["significant_raw"] & ~df["significant_bh"]

    console.print("\n[bold cyan]Multiple Comparisons Correction (Benjamini-Hochberg)[/]")
    console.print(f"  Total tests:              {len(df)}")
    console.print(f"  Significant (raw α=0.05): {df['significant_raw'].sum()}")
    console.print(f"  Significant (BH adj):     {df['significant_bh'].sum()}")
    console.print(f"  Lost significance:        {df['lost_significance'].sum()}")

    # Print the table
    t = Table(title="Statistical Tests — BH Corrected", show_header=True)
    t.add_column("Layer", style="cyan")
    t.add_column("Test", max_width=35)
    t.add_column("Raw p", justify="right")
    t.add_column("BH adj p", justify="right")
    t.add_column("Sig?", justify="center")
    for _, row in df.sort_values("bh_adjusted_p").iterrows():
        sig_str = "[green]YES[/]" if row["significant_bh"] else "[red]NO[/]"
        lost = " [yellow](lost)[/]" if row["lost_significance"] else ""
        t.add_row(
            str(row["layer"]),
            str(row["test_name"])[:35],
            f"{row['raw_p']:.4f}",
            f"{row['bh_adjusted_p']:.4f}",
            sig_str + lost,
        )
    console.print(t)

    return df


def _collect_pvalues_from_layers(layer_results: dict) -> dict[str, dict]:
    """Harvest raw p-values from all 5 layer stats dicts."""
    pvalues: dict[str, dict] = {}

    # Layer 1
    l1_stats = layer_results.get("layer1", {}).get("stats", {})
    pvalues["L1-Price"] = {}
    for key in ["pearson", "spearman"]:
        if key in l1_stats and "p_value" in l1_stats[key]:
            pvalues["L1-Price"][f"{key}_log_returns"] = l1_stats[key]["p_value"]
    if "ks_test" in l1_stats and "p_value" in l1_stats.get("ks_test", {}):
        pvalues["L1-Price"]["ks_test_returns"] = l1_stats["ks_test"]["p_value"]

    # Layer 2
    l2_stats = layer_results.get("layer2", {}).get("stats", {})
    pvalues["L2-Fundamentals"] = {}
    for key in ["ttest_log_pe", "ttest_ps_ratio", "chow_test", "bootstrap_log_pe"]:
        if key in l2_stats:
            p = (
                l2_stats[key].get("p_value")
                or l2_stats[key].get("pvalue")
                or l2_stats[key].get("p")
            )
            if p is not None and not np.isnan(float(p)):
                pvalues["L2-Fundamentals"][key] = float(p)

    # Layer 3
    l3_stats = layer_results.get("layer3", {}).get("stats", {})
    pvalues["L3-Concentration"] = {}
    for key in ["ztest_pvalue", "pearson_p", "mann_whitney_p"]:
        if key in l3_stats and l3_stats[key] is not None:
            pvalues["L3-Concentration"][key] = float(l3_stats[key])
    # Nested in findings
    l3_findings = layer_results.get("layer3", {}).get("findings", {})
    st_tests = l3_findings.get("statistical_tests", {})
    if "ztest_pvalue" in st_tests and st_tests["ztest_pvalue"] is not None:
        pvalues["L3-Concentration"]["ztest_concentration"] = float(st_tests["ztest_pvalue"])

    # Layer 4
    l4_stats = layer_results.get("layer4", {}).get("stats", {})
    pvalues["L4-Macro"] = {}
    era_comp = l4_stats.get("era_comparison", {})
    for indicator, result in era_comp.items():
        if isinstance(result, dict):
            for pk in ["p_value", "pvalue", "p"]:
                if pk in result and result[pk] is not None:
                    try:
                        p_float = float(result[pk])
                        if not np.isnan(p_float):
                            pvalues["L4-Macro"][f"welch_{indicator}"] = p_float
                    except (TypeError, ValueError):
                        pass
                    break
    for granger_key in ["granger_yc_sp500", "granger_causality"]:
        if granger_key in l4_stats:
            gv = l4_stats[granger_key]
            if isinstance(gv, dict) and "p_value" in gv:
                pvalues["L4-Macro"]["granger_yc"] = float(gv["p_value"])

    # Layer 5
    l5_stats = layer_results.get("layer5", {}).get("stats", {})
    pvalues["L5-Sentiment"] = {}
    for key in ["hype_price_correlation", "mann_whitney_sentiment"]:
        if key in l5_stats and isinstance(l5_stats[key], dict):
            p = l5_stats[key].get("p_value") or l5_stats[key].get("pvalue")
            if p is not None:
                pvalues["L5-Sentiment"][key] = float(p)

    # Remove empty layers
    return {k: v for k, v in pvalues.items() if v}


# ---------------------------------------------------------------------------
# 9. Charts
# ---------------------------------------------------------------------------


def chart_ml_1_regime_probabilities(
    predictions_over_time: pd.DataFrame,
) -> dict[str, Path]:
    """Stacked area chart of regime probabilities over the AI era period.

    Parameters
    ----------
    predictions_over_time : DataFrame with columns = regime classes,
                             index = DatetimeIndex.
    """
    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        color_map = {
            "bubble": BUBBLE_COLOR,
            "correction": "#FFA15A",
            "normal_growth": NORMAL_COLOR,
            "recovery": "#636EFA",
        }

        for regime in ["recovery", "normal_growth", "correction", "bubble"]:
            if regime not in predictions_over_time.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=predictions_over_time.index,
                    y=predictions_over_time[regime] * 100,
                    name=regime.replace("_", " ").title(),
                    stackgroup="one",
                    fillcolor=color_map.get(regime, "#888"),
                    line=dict(width=0.5, color=color_map.get(regime, "#888")),
                    hovertemplate="%{y:.1f}%<extra></extra>",
                )
            )

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Market Regime Probabilities — Ensemble Classifier (Jul 2024 – Mar 2026)",
            xaxis_title="Date",
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )

        return _save_chart(fig, "chart_ml_1_regime_probabilities")

    except Exception as exc:
        console.print(f"[red]Chart ML.1 failed: {exc}[/]")
        return {}


def chart_ml_2_dtw_similarity(
    similarity_timeline: pd.Series,
    reference_scores: dict[str, float] | None = None,
) -> dict[str, Path]:
    """Line chart of rolling DTW similarity with reference period scores.

    Parameters
    ----------
    similarity_timeline : pd.Series indexed by date, values = similarity score.
    reference_scores    : Dict of period label -> similarity score (horizontal lines).
    """
    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        if not similarity_timeline.empty:
            fig.add_trace(
                go.Scatter(
                    x=similarity_timeline.index,
                    y=similarity_timeline.values,
                    name="AI Era vs Dot-com (rolling)",
                    line=dict(color=BUBBLE_COLOR, width=2.5),
                    hovertemplate="Date: %{x}<br>Similarity: %{y:.1f}<extra></extra>",
                )
            )

        if reference_scores:
            ref_colors = {
                "dotcom": BUBBLE_COLOR,
                "housing": "#FFA15A",
                "post_gfc": NORMAL_COLOR,
                "covid_rally": "#636EFA",
            }
            for period, score in reference_scores.items():
                if score is None or np.isnan(score):
                    continue
                fig.add_hline(
                    y=score,
                    line_dash="dash",
                    line_color=ref_colors.get(period, "#888"),
                    annotation_text=f"{period} ({score:.0f})",
                    annotation_position="right",
                )

        # Interpretation zones
        for y_range, label, color in [
            ([80, 100], "Near-identical", "rgba(239,85,59,0.1)"),
            ([60, 80], "Strong similarity", "rgba(239,85,59,0.07)"),
            ([40, 60], "Moderate", "rgba(255,255,255,0.04)"),
        ]:
            fig.add_hrect(
                y0=y_range[0], y1=y_range[1],
                fillcolor=color, line_width=0,
                annotation_text=label,
                annotation_position="left",
            )

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title="DTW Similarity — AI Era vs Historical Periods",
            xaxis_title="Date",
            yaxis_title="Similarity Score (0 = different, 100 = identical)",
            yaxis=dict(range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        return _save_chart(fig, "chart_ml_2_dtw_similarity")

    except Exception as exc:
        console.print(f"[red]Chart ML.2 failed: {exc}[/]")
        return {}


def chart_ml_3_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
    max_display: int = 20,
) -> dict[str, Path]:
    """SHAP summary dot plot saved to PNG.

    Uses SHAP's own matplotlib renderer — saved to disk rather than displayed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        shap = _import_shap()

        if feature_names is None and hasattr(X, "columns"):
            feature_names = list(X.columns)

        X_display = X if hasattr(X, "values") else pd.DataFrame(X, columns=feature_names)

        plt.figure()
        shap.summary_plot(
            shap_values,
            X_display,
            feature_names=feature_names,
            plot_type="dot",
            max_display=max_display,
            show=False,
            plot_size=(12, 8),
        )
        plt.title("SHAP Feature Importance — Bubble Class", fontsize=13)
        plt.tight_layout()

        path = CHART_DIR / "chart_ml_3_shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close()

        console.print(f"[green]Saved:[/] {path.name}")
        return {"png": path}

    except Exception as exc:
        console.print(f"[red]Chart ML.3 SHAP summary failed: {exc}[/]")
        return {}


def chart_ml_4_shap_waterfall(
    shap_explanation: dict[str, Any],
    max_display: int = 15,
) -> dict[str, Path]:
    """SHAP waterfall plot for the Mar 2026 prediction.

    Parameters
    ----------
    shap_explanation : Output from compute_shap_analysis()["march_2026_explanation"].
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap = _import_shap()

        if not shap_explanation or not shap_explanation.get("shap_values"):
            console.print("[yellow]Chart ML.4: No SHAP explanation for Mar 2026.[/]")
            return {}

        sv = np.array(shap_explanation["shap_values"])
        base = shap_explanation["base_value"]
        fnames = shap_explanation.get("feature_names", [])
        fvals = np.array(shap_explanation.get("feature_values", np.zeros(len(sv))))

        expl = shap.Explanation(
            values=sv,
            base_values=base,
            data=fvals,
            feature_names=fnames,
        )

        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(expl, max_display=max_display, show=False)
        plt.title("SHAP Waterfall — March 2026 Bubble Classification", fontsize=13)
        plt.tight_layout()

        path = CHART_DIR / "chart_ml_4_shap_waterfall.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close()

        console.print(f"[green]Saved:[/] {path.name}")
        return {"png": path}

    except Exception as exc:
        console.print(f"[red]Chart ML.4 SHAP waterfall failed: {exc}[/]")
        return {}


def _save_chart(fig: Any, name: str) -> dict[str, Path]:
    """Save a Plotly figure as JSON and PNG."""
    paths: dict[str, Path] = {}
    try:
        json_path = CHART_DIR / f"{name}.json"
        fig.write_json(str(json_path))
        paths["json"] = json_path
        console.print(f"[green]Saved:[/] {json_path.name}")
    except Exception as exc:
        console.print(f"[yellow]JSON save failed for {name}: {exc}[/]")

    try:
        import kaleido  # noqa: F401 — only import to check availability
        png_path = CHART_DIR / f"{name}.png"
        fig.write_image(str(png_path), width=1400, height=700, scale=2)
        paths["png"] = png_path
        console.print(f"[green]Saved:[/] {png_path.name}")
    except ImportError:
        console.print(f"[dim]kaleido not installed — PNG skipped for {name}[/]")
    except Exception as exc:
        console.print(f"[yellow]PNG save failed for {name}: {exc}[/]")

    return paths


# ---------------------------------------------------------------------------
# 10. Main Runner
# ---------------------------------------------------------------------------


def run_ml_pipeline(
    layer_results: dict,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Execute the complete ML pipeline.

    Steps:
        1. Build unified feature matrix from all 5 layers.
        2. Assign regime labels.
        3. Time-based train/val/test/live splits.
        4. Train regime classifiers (RF, XGB, LR + price-excluded RF).
        5. Evaluate on test set.
        6. Generate live regime probabilities (Jul 2024 – Mar 2026).
        7. Run DTW similarity analysis with null distribution.
        8. Run SHAP analysis.
        9. Apply Benjamini-Hochberg correction across all p-values.
        10. Generate all ML charts.
        11. Persist model artifacts.

    Parameters
    ----------
    layer_results : Dict with keys "layer1" .. "layer5" from run_layerN().
    force_refresh : If True, recompute even if cached results exist.

    Returns
    -------
    dict with keys: feature_matrix, regime_labels, trained, evaluation,
                    predictions, dtw_results, shap_results, bh_corrections, charts
    """
    console.rule("[bold green]ML Pipeline — AI Hype Cycle Analysis")

    # ------------------------------------------------------------------ #
    # Step 1: Feature Matrix                                               #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 1:[/] Building feature matrix...")
    fm_result = build_feature_matrix(layer_results)
    features_core: pd.DataFrame = fm_result["core"]
    features_aug: pd.DataFrame = fm_result["augmented"]

    if features_core.empty:
        raise RuntimeError("Feature matrix is empty. Check that layer_results is populated.")

    # ------------------------------------------------------------------ #
    # Step 2: Regime Labels                                                #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 2:[/] Assigning regime labels...")
    from src.ml.feature_matrix import assign_regime_labels
    labels_core = assign_regime_labels(features_core.index)
    labels_aug = assign_regime_labels(features_aug.index)

    # Align
    common_core = features_core.index.intersection(labels_core.index)
    X_all = features_core.loc[common_core]
    y_all = labels_core.loc[common_core]

    # ------------------------------------------------------------------ #
    # Step 3: Time-based splits                                            #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 3:[/] Creating time-based train/val/test/live splits...")

    # Per spec §2.7:
    # Train:  Jan 1990 – Dec 2018
    # Val:    Jan 2019 – Dec 2021
    # Test:   Jan 2022 – Jun 2024
    # Live:   Jul 2024 – Mar 2026
    train_mask = X_all.index <= "2018-12-31"
    val_mask = (X_all.index >= "2019-01-01") & (X_all.index <= "2021-12-31")
    test_mask = (X_all.index >= "2022-01-01") & (X_all.index <= "2024-06-30")
    live_mask = X_all.index >= "2024-07-01"

    X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
    X_val, y_val = X_all.loc[val_mask], y_all.loc[val_mask]
    X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]
    X_live = X_all.loc[live_mask]

    console.print(
        f"  Train: {len(X_train)} obs  |  Val: {len(X_val)} obs  |  "
        f"Test: {len(X_test)} obs  |  Live: {len(X_live)} obs"
    )

    if len(X_train) < 24:
        console.print(
            "[red]WARNING: Training set has fewer than 24 observations. "
            "Feature matrix may be incomplete.[/]"
        )

    # ------------------------------------------------------------------ #
    # Step 4: Train models                                                 #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 4:[/] Training regime classifiers...")
    trained = train_regime_classifier(X_train, y_train)

    # ------------------------------------------------------------------ #
    # Step 5: Evaluate on test set                                         #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 5:[/] Evaluating on test set...")
    evaluation: dict = {}
    if len(X_test) > 0:
        evaluation = evaluate_models(trained, X_test, y_test)

    # Also evaluate on validation set
    val_evaluation: dict = {}
    if len(X_val) > 0:
        console.print("  Evaluating on validation set...")
        val_evaluation = evaluate_models(trained, X_val, y_val)

    # ------------------------------------------------------------------ #
    # Step 6: Live classification                                          #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 6:[/] Classifying live period (Jul 2024 – Mar 2026)...")
    predictions: dict = {}
    if len(X_live) > 0:
        predictions = predict_regime(trained, X_live)
    else:
        console.print("[yellow]No live observations available for classification.[/]")

    # ------------------------------------------------------------------ #
    # Step 7: DTW Similarity Analysis                                      #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 7:[/] Running DTW similarity analysis...")
    dtw_results = run_dtw_analysis(features_core, n_permutations=500)

    # ------------------------------------------------------------------ #
    # Step 8: SHAP Analysis                                                #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 8:[/] Computing SHAP values...")
    shap_results: dict = {}
    try:
        scaler = trained["scalers"]["main"]
        feature_names = trained.get("feature_names", list(X_test.columns))
        rf_model = trained["models"]["rf"]
        t_medians = trained.get("train_medians", pd.Series(dtype=float))

        def _scale_for_shap(df: pd.DataFrame) -> pd.DataFrame:
            df = df.reindex(columns=feature_names)
            if not t_medians.empty:
                df = df.fillna(t_medians.reindex(df.columns))
            return pd.DataFrame(
                scaler.transform(df.fillna(0)),
                index=df.index,
                columns=feature_names,
            )

        # Use test set for global SHAP analysis
        if len(X_test) > 0:
            X_test_scaled = _scale_for_shap(X_test)
        else:
            X_test_scaled = _scale_for_shap(X_train)

        # Latest observation for waterfall
        X_march_scaled = None
        if len(X_live) > 0:
            X_march_scaled = _scale_for_shap(X_live.iloc[-1:])

        shap_results = compute_shap_analysis(
            rf_model=rf_model,
            X_test=X_test_scaled,
            X_march_2026=X_march_scaled,
            feature_names=feature_names,
        )
    except Exception as exc:
        console.print(f"[yellow]SHAP analysis failed: {exc}[/]")
        shap_results = {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # Step 9: Benjamini-Hochberg Correction                                #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 9:[/] Applying Benjamini-Hochberg correction...")
    layer_pvalues = _collect_pvalues_from_layers(layer_results)

    # Add DTW permutation test p-value
    dtw_primary = dtw_results.get("primary", {})
    if dtw_primary.get("p_value") is not None:
        layer_pvalues.setdefault("ML-DTW", {})
        layer_pvalues["ML-DTW"]["dtw_permutation_dotcom"] = dtw_primary["p_value"]

    bh_result = pd.DataFrame()
    if layer_pvalues:
        bh_result = apply_bh_correction(layer_pvalues)
    else:
        console.print("[yellow]No p-values collected for BH correction.[/]")

    # ------------------------------------------------------------------ #
    # Step 10: Charts                                                      #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 10:[/] Generating charts...")
    charts: dict[str, Any] = {}

    if predictions.get("probabilities") is not None:
        charts["ml_1"] = chart_ml_1_regime_probabilities(predictions["probabilities"])
    else:
        console.print("[yellow]Skipping Chart ML.1 (no predictions).[/]")

    rolling_tl = dtw_results.get("rolling_timeline", pd.Series(dtype=float))
    ref_scores = {
        lbl: comparisons.get("composite_score")
        for lbl, comparisons in dtw_results.get("comparisons", {}).items()
        if not comparisons.get("error")
    }
    charts["ml_2"] = chart_ml_2_dtw_similarity(rolling_tl, ref_scores)

    if (
        shap_results.get("shap_values_bubble") is not None
        and len(X_test) > 0
    ):
        X_test_display = X_test.reindex(columns=feature_names).fillna(0)
        charts["ml_3"] = chart_ml_3_shap_summary(
            shap_values=shap_results["shap_values_bubble"],
            X=X_test_display,
            feature_names=feature_names,
        )

    if shap_results.get("march_2026_explanation"):
        charts["ml_4"] = chart_ml_4_shap_waterfall(shap_results["march_2026_explanation"])

    # ------------------------------------------------------------------ #
    # Step 11: Persist artifacts                                           #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Step 11:[/] Saving model artifacts...")
    try:
        for model_name, model_obj in trained["models"].items():
            path = MODEL_DIR / f"regime_{model_name}.joblib"
            joblib.dump(model_obj, path)

        joblib.dump(trained["scalers"]["main"], MODEL_DIR / "feature_scaler.joblib")
        joblib.dump(trained["label_encoder"], MODEL_DIR / "label_encoder.joblib")

        # Save feature matrix
        feat_dir = Path(__file__).resolve().parents[2] / "data" / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        features_core.to_parquet(feat_dir / "feature_matrix.parquet", index=True)
        labels_core.to_frame(name="regime").to_csv(feat_dir / "regime_labels.csv")

        # Save live predictions
        if predictions.get("probabilities") is not None:
            predictions["probabilities"].to_csv(RESULTS_DIR / "regime_probabilities.csv")

        # Save SHAP importance
        if shap_results.get("feature_importance"):
            pd.Series(shap_results["feature_importance"]).to_csv(
                RESULTS_DIR / "shap_feature_importance.csv"
            )

        # Save BH correction table
        if not bh_result.empty:
            bh_result.to_csv(RESULTS_DIR / "bh_correction.csv", index=False)

        console.print("[green]All artifacts saved.[/]")
    except Exception as exc:
        console.print(f"[yellow]Artifact save failed (non-critical): {exc}[/]")

    # ------------------------------------------------------------------ #
    # Final summary                                                        #
    # ------------------------------------------------------------------ #
    console.rule("[bold green]ML Pipeline Complete")

    march_probs = predictions.get("march_2026", {})
    if march_probs:
        bubble_p = march_probs.get("bubble", 0.0)
        verdict = predictions.get("verdict", "")
        console.print(f"\n[bold green]RESULT: {verdict}[/]")
        console.print(
            f"  RF CV F1:  {np.mean(trained['cv_scores']['rf']):.3f}  |  "
            f"XGB CV F1: {np.mean(trained['cv_scores']['xgb']):.3f}  |  "
            f"LR CV F1:  {np.mean(trained['cv_scores']['lr']):.3f}"
        )

    return {
        "feature_matrix": {
            "core": features_core,
            "augmented": features_aug,
            "feature_names_core": fm_result["feature_names_core"],
            "feature_names_augmented": fm_result["feature_names_augmented"],
            "has_sentiment_data": fm_result["has_sentiment_data"],
        },
        "regime_labels": labels_core,
        "splits": {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
            "live": X_live,
        },
        "trained": trained,
        "evaluation": {
            "test": evaluation,
            "validation": val_evaluation,
        },
        "predictions": predictions,
        "dtw_results": dtw_results,
        "shap_results": shap_results,
        "bh_corrections": bh_result,
        "charts": charts,
        "stats": {
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_live": len(X_live),
            "n_features_core": len(fm_result["feature_names_core"]),
            "n_features_augmented": len(fm_result["feature_names_augmented"]),
        },
    }
