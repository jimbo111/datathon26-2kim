"""Phase 5B: Health Disadvantage Index.

Builds a weighted composite index from regression coefficients,
ranks all census tracts, and exports summary stats + path diagram data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from rich.console import Console

console = Console()
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def build_health_disadvantage_index(
    master: pd.DataFrame,
    phase2_results: dict | None = None,
    phase3_results: dict | None = None,
) -> dict:
    """Compute the Health Disadvantage Index (HDI) for every census tract.

    The index is a weighted composite of:
    - Food access (is_food_desert, pct_low_access_1mi)
    - Income (poverty_rate)
    - Healthcare access (uninsured_pct, hpsa_shortage)

    Weights are justified from Phase 2-4 regression coefficients when available,
    otherwise use equal weighting of standardized components.
    """
    console.rule("[bold]Phase 5B: Health Disadvantage Index")
    results = {}

    # ── Define index components ──
    components = {
        "food_access": ["is_food_desert", "pct_low_access_1mi"],
        "income": ["poverty_rate"],
        "healthcare_access": ["uninsured_pct"],
    }

    # Add HPSA if available
    if "hpsa_shortage" in master.columns and master["hpsa_shortage"].notna().sum() > 100:
        components["healthcare_access"].append("hpsa_shortage")

    # ── Determine weights from regression coefficients ──
    weights = _derive_weights(master, phase2_results)
    results["index_weights"] = weights

    # ── Compute standardized components ──
    df = master.copy()
    component_scores = {}

    for group, cols in components.items():
        available = [c for c in cols if c in df.columns and df[c].notna().sum() > 100]
        if not available:
            continue

        # Standardize each column (higher = worse)
        z_scores = pd.DataFrame()
        for col in available:
            series = pd.to_numeric(df[col], errors="coerce")
            z = (series - series.mean()) / series.std()
            z_scores[col] = z

        # Average z-scores for this component
        component_scores[group] = z_scores.mean(axis=1)

    if not component_scores:
        console.print("[red]Not enough data to build HDI[/]")
        return results

    # ── Build composite index ──
    score_df = pd.DataFrame(component_scores)
    for group in score_df.columns:
        score_df[group] *= weights.get(group, 1.0)

    df["hdi_score"] = score_df.mean(axis=1)

    # Higher HDI = more disadvantaged
    df["hdi_percentile"] = df["hdi_score"].rank(pct=True) * 100
    df["hdi_decile"] = pd.qcut(
        df["hdi_score"].rank(method="first"), 10, labels=range(1, 11)
    ).astype("Int64")

    # ── Rank tracts ──
    ranked = df.dropna(subset=["hdi_score"]).sort_values("hdi_score", ascending=False)

    top_10pct = ranked.head(int(len(ranked) * 0.10))
    bottom_10pct = ranked.tail(int(len(ranked) * 0.10))

    top_cols = ["tract_fips", "state", "county", "hdi_score", "hdi_percentile"]
    top_cols = [c for c in top_cols if c in ranked.columns]
    if "diabetes_pct" in ranked.columns:
        top_cols.append("diabetes_pct")
    if "life_expectancy" in ranked.columns:
        top_cols.append("life_expectancy")

    results["top_10pct_worst"] = top_10pct[top_cols].head(50).to_dict(orient="records")
    results["bottom_10pct_best"] = bottom_10pct[top_cols].head(50).to_dict(orient="records")
    results["n_tracts_scored"] = int(ranked.shape[0])

    # ── Summary stats: gaps between top and bottom deciles ──
    gap_metrics = {}
    for col in ["diabetes_pct", "obesity_pct", "life_expectancy", "poverty_rate"]:
        if col in df.columns:
            top_mean = top_10pct[col].mean()
            bot_mean = bottom_10pct[col].mean()
            if pd.notna(top_mean) and pd.notna(bot_mean):
                gap_metrics[col] = {
                    "most_disadvantaged": round(top_mean, 2),
                    "least_disadvantaged": round(bot_mean, 2),
                    "gap": round(top_mean - bot_mean, 2),
                }

    results["decile_gaps"] = gap_metrics
    if gap_metrics:
        console.print("[cyan]Gaps (most vs least disadvantaged decile):[/]")
        for k, v in gap_metrics.items():
            console.print(f"  {k}: {v['most_disadvantaged']} vs {v['least_disadvantaged']} (gap: {v['gap']})")

    # ── Path diagram coefficients: poverty → food desert → obesity → diabetes ──
    path_results = _compute_path_coefficients(df)
    results["path_diagram"] = path_results

    # ── Export ──
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Save HDI as parquet
    hdi_cols = ["tract_fips", "hdi_score", "hdi_percentile", "hdi_decile"]
    hdi_cols = [c for c in hdi_cols if c in df.columns]
    for extra in ["state", "county", "diabetes_pct", "obesity_pct", "life_expectancy"]:
        if extra in df.columns:
            hdi_cols.append(extra)
    df[hdi_cols].dropna(subset=["hdi_score"]).to_parquet(
        DATA_PROCESSED / "health_disadvantage_index.parquet", index=False
    )

    # Save JSON results
    with open(DATA_PROCESSED / "phase5_health_index.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"[green]HDI computed for {results['n_tracts_scored']:,} tracts[/]")
    console.print("[green]Exported → data/processed/health_disadvantage_index.parquet[/]")
    console.print("[green]Exported → data/processed/phase5_health_index.json[/]")

    return results


def _derive_weights(df: pd.DataFrame, phase2_results: dict | None) -> dict:
    """Derive component weights from regression coefficients or use defaults."""
    # Try to extract from Phase 2 OLS results
    if phase2_results and "ols_diabetes" in phase2_results:
        coefs = phase2_results["ols_diabetes"].get("coefficients", {})
        food_coef = abs(coefs.get("is_food_desert", {}).get("coef", 0))
        income_coef = abs(coefs.get("poverty_rate", {}).get("coef", 0))
        insurance_coef = abs(coefs.get("uninsured_pct", {}).get("coef", 0))

        total = food_coef + income_coef + insurance_coef
        if total > 0:
            weights = {
                "food_access": round(food_coef / total, 3),
                "income": round(income_coef / total, 3),
                "healthcare_access": round(insurance_coef / total, 3),
            }
            console.print(f"  Weights from regression: {weights}")
            return weights

    # Fallback: derive weights from a fresh regression on available data
    reg_cols = ["diabetes_pct", "is_food_desert", "poverty_rate", "uninsured_pct"]
    available = [c for c in reg_cols if c in df.columns]
    if len(available) == len(reg_cols):
        clean = df[reg_cols].dropna()
        if len(clean) > 100:
            model = smf.ols(
                "diabetes_pct ~ is_food_desert + poverty_rate + uninsured_pct",
                data=clean,
            ).fit()
            food_coef = abs(model.params.get("is_food_desert", 0))
            income_coef = abs(model.params.get("poverty_rate", 0))
            ins_coef = abs(model.params.get("uninsured_pct", 0))
            total = food_coef + income_coef + ins_coef
            if total > 0:
                weights = {
                    "food_access": round(food_coef / total, 3),
                    "income": round(income_coef / total, 3),
                    "healthcare_access": round(ins_coef / total, 3),
                }
                console.print(f"  Weights from fresh regression: {weights}")
                return weights

    # Default: equal weights
    console.print("  Using equal weights (no regression data available)")
    return {"food_access": 0.333, "income": 0.333, "healthcare_access": 0.334}


def _compute_path_coefficients(df: pd.DataFrame) -> dict:
    """Estimate path diagram coefficients: poverty → food desert → obesity → diabetes.

    Uses sequential OLS to estimate each path:
    1. poverty → food_desert
    2. food_desert → obesity
    3. obesity → diabetes
    4. Direct: poverty → diabetes (for comparison with mediated path)
    """
    paths = {}

    # Path 1: poverty_rate → is_food_desert
    cols = ["is_food_desert", "poverty_rate"]
    clean = df[[c for c in cols if c in df.columns]].dropna()
    if len(clean) > 100 and all(c in clean.columns for c in cols):
        m = smf.ols("is_food_desert ~ poverty_rate", data=clean).fit()
        paths["poverty_to_food_desert"] = {
            "coef": round(m.params.get("poverty_rate", 0), 4),
            "r_squared": round(m.rsquared, 4),
        }

    # Path 2: is_food_desert → obesity_pct
    cols = ["obesity_pct", "is_food_desert"]
    clean = df[[c for c in cols if c in df.columns]].dropna()
    if len(clean) > 100 and all(c in clean.columns for c in cols):
        m = smf.ols("obesity_pct ~ is_food_desert", data=clean).fit()
        paths["food_desert_to_obesity"] = {
            "coef": round(m.params.get("is_food_desert", 0), 4),
            "r_squared": round(m.rsquared, 4),
        }

    # Path 3: obesity_pct → diabetes_pct
    cols = ["diabetes_pct", "obesity_pct"]
    clean = df[[c for c in cols if c in df.columns]].dropna()
    if len(clean) > 100 and all(c in clean.columns for c in cols):
        m = smf.ols("diabetes_pct ~ obesity_pct", data=clean).fit()
        paths["obesity_to_diabetes"] = {
            "coef": round(m.params.get("obesity_pct", 0), 4),
            "r_squared": round(m.rsquared, 4),
        }

    # Direct path: poverty → diabetes (total effect)
    cols = ["diabetes_pct", "poverty_rate"]
    clean = df[[c for c in cols if c in df.columns]].dropna()
    if len(clean) > 100 and all(c in clean.columns for c in cols):
        m = smf.ols("diabetes_pct ~ poverty_rate", data=clean).fit()
        paths["poverty_to_diabetes_direct"] = {
            "coef": round(m.params.get("poverty_rate", 0), 4),
            "r_squared": round(m.rsquared, 4),
        }

    if paths:
        console.print("[cyan]Path diagram coefficients:[/]")
        for path, vals in paths.items():
            console.print(f"  {path}: β={vals['coef']}, R²={vals['r_squared']}")

    return paths
