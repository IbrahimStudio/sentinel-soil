#!/usr/bin/env python3
"""
soil_texture_pipeline.py

Thesis-grade pipeline runner for soil texture prediction from a pre-built feature table.

Key features
------------
1) Runs BOTH validation strategies:
   - Random KFold CV (optimistic; standard ML)
   - Spatial Block CV via GroupKFold (more realistic for geospatial generalization)

2) Runs multiple algorithms and configurations automatically for each target:
   - RandomForestRegressor (several configs)
   - HistGradientBoostingRegressor (several configs)
   - ElasticNet (several configs; scaled)

3) Saves reports for reproducibility & later inspection:
   - summary_metrics.csv (all results in one table)
   - summary_metrics.json (same data)
   - fold_scores__{target}__{model}__{cv}.csv (per-fold metrics)
   - feature_importance__{target}__rf__{cv}.csv (RF Gini importance)
   - permutation_importance__{target}__{model}__{cv}.csv (optional; slower)
   - plots:
       * model_comparison__{target}.png  (barplot of mean R2 for each model+CV)
       * cv_scatter__{target}.png        (RandomCV vs SpatialCV mean R2 scatter)

Spatial blocks
--------------
If your dataset does not provide a group column, this script can build one automatically from lat/lon:

- It converts (lat, lon) to EPSG:3857 meters (WebMercator) using pyproj (if available),
  otherwise it falls back to an approximate meters-per-degree conversion.
- Then it assigns each point to a grid cell of size --block_size_m (default 20000m = 20 km),
  producing a 'block_id' used for GroupKFold.

With ~1100 points, 20 km blocks often yield a reasonable number of groups (depends on extent).
Try 10km or 25km if needed.

Usage
-----
Basic (runs random + spatial CV, full sweep of models/configs):
    python soil_texture_pipeline.py --input features.xlsx --targets Silt Clay Sand --output_dir reports

Tune spatial block size (meters):
    python soil_texture_pipeline.py --input features.xlsx --targets Silt --block_size_m 10000 --output_dir reports

If you already have your own spatial groups:
    python soil_texture_pipeline.py --input features.xlsx --targets Silt --group_col field_id --output_dir reports

Optional permutation importance (slower but more robust interpretation):
    python soil_texture_pipeline.py --input features.xlsx --targets Silt --do_perm_importance

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from joblib import dump

# plotting (matplotlib only, no seaborn)
import matplotlib.pyplot as plt


# ---------------------------
# Core pipeline construction
# ---------------------------

def build_texture_pipeline(
    X: pd.DataFrame,
    model: RegressorMixin,
    *,
    scale_numeric: bool = False,
) -> Pipeline:
    """
    Build a preprocessing + model pipeline for soil texture prediction.

    - Numeric-only predictors
    - Median imputation
    - Optional scaling (recommended for linear models / SVR; not needed for RF/HGB)
    """
    numeric_features = list(X.select_dtypes(include=[np.number]).columns)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


# ---------------------------
# Data prep
# ---------------------------

def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError("Unsupported file type. Use .csv or .xlsx/.xls")


def _prepare_xy(
    df: pd.DataFrame,
    target: str,
    *,
    id_cols: Sequence[str] = ("POINT_ID",),
    drop_cols: Sequence[str] = (),
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare X/y: drop IDs, isolate target, keep numeric predictors, drop rows with missing target."""
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe.")

    data = df.copy()

    # drop IDs from predictors
    for c in id_cols:
        if c in data.columns:
            data.drop(columns=c, inplace=True)

    y = pd.to_numeric(data[target], errors="coerce")
    X = data.drop(columns=[target])

    if drop_cols:
        X = X.drop(columns=list(drop_cols), errors="ignore")

    # keep numeric predictors only
    X = X.select_dtypes(include=[np.number])

    # drop missing target
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    return X, y


# ---------------------------
# Spatial grouping
# ---------------------------

def _to_float_series(s: pd.Series) -> pd.Series:
    """
    Convert a column that may use comma as decimal separator to float.
    """
    if s.dtype.kind in "if":
        return s.astype(float)
    # common in EU CSV/Excel exports: "48,0932"
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def add_spatial_blocks(
    df: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    block_size_m: int = 20000,
    out_col: str = "block_id",
) -> pd.DataFrame:
    """
    Create a spatial block grouping column from lat/lon.

    Tries to convert lat/lon to EPSG:3857 meters using pyproj. If not available,
    uses approximate conversion to meters.

    block_id = floor(x_m / block_size_m) + "_" + floor(y_m / block_size_m)
    """
    out = df.copy()

    if lat_col not in out.columns or lon_col not in out.columns:
        raise ValueError(f"Need '{lat_col}' and '{lon_col}' columns to build spatial blocks.")

    lat = _to_float_series(out[lat_col])
    lon = _to_float_series(out[lon_col])

    if lat.isna().any() or lon.isna().any():
        # If parsing fails, stop early with a helpful message
        bad = out[lat.isna() | lon.isna()][[lat_col, lon_col]].head(5)
        raise ValueError(
            "Could not parse some lat/lon values to float. "
            "Example problematic rows:\n" + bad.to_string(index=False)
        )

    # Try pyproj -> EPSG:3857
    x_m = None
    y_m = None
    try:
        from pyproj import Transformer  # type: ignore
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_m, y_m = transformer.transform(lon.to_numpy(), lat.to_numpy())
        x_m = np.asarray(x_m, dtype=float)
        y_m = np.asarray(y_m, dtype=float)
    except Exception:
        # fallback: approximate meters-per-degree
        # 1 degree lat ~ 111,320 m; 1 degree lon ~ 111,320*cos(lat)
        lat_rad = np.deg2rad(lat.to_numpy(dtype=float))
        y_m = lat.to_numpy(dtype=float) * 111_320.0
        x_m = lon.to_numpy(dtype=float) * (111_320.0 * np.cos(lat_rad))

    bx = np.floor(x_m / float(block_size_m)).astype(int)
    by = np.floor(y_m / float(block_size_m)).astype(int)
    out[out_col] = bx.astype(str) + "_" + by.astype(str)
    return out


# ---------------------------
# CV evaluation
# ---------------------------

def evaluate_texture_model(
    df: pd.DataFrame,
    target: str,
    model: RegressorMixin,
    *,
    id_cols: Sequence[str] = ("POINT_ID",),
    drop_cols: Sequence[str] = (),
    group_col: Optional[str] = None,
    cv_mode: str = "random",  # "random" | "spatial"
    n_splits: int = 5,
    random_state: int = 42,
    scale_numeric: bool = False,
    n_jobs: int = -1,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Evaluate with either:
      - cv_mode="random": shuffled KFold
      - cv_mode="spatial": GroupKFold with group_col
    """
    X, y = _prepare_xy(df, target, id_cols=id_cols, drop_cols=drop_cols)
    pipe = build_texture_pipeline(X, model, scale_numeric=scale_numeric)

    if cv_mode == "random":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_kwargs = {}
        cv_type = "RandomKFold"
    elif cv_mode == "spatial":
        if group_col is None or group_col not in df.columns:
            raise ValueError("Spatial CV requested but group_col is missing or not in df.")
        groups = df.loc[X.index, group_col]
        cv = GroupKFold(n_splits=n_splits)
        cv_kwargs = {"groups": groups}
        cv_type = "SpatialGroupKFold"
    else:
        raise ValueError("cv_mode must be 'random' or 'spatial'.")

    scoring = {
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }

    cv_out = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=n_jobs,
        **cv_kwargs,
    )

    fold_df = pd.DataFrame({
        "fold": np.arange(1, n_splits + 1),
        "r2": cv_out["test_r2"],
        "rmse": -cv_out["test_neg_rmse"],
        "mae": -cv_out["test_neg_mae"],
    })

    results = {
        "target": target,
        "cv_type": cv_type,
        "n_splits": int(n_splits),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "cv_r2_mean": float(fold_df["r2"].mean()),
        "cv_r2_std": float(fold_df["r2"].std(ddof=0)),
        "cv_rmse_mean": float(fold_df["rmse"].mean()),
        "cv_rmse_std": float(fold_df["rmse"].std(ddof=0)),
        "cv_mae_mean": float(fold_df["mae"].mean()),
        "cv_mae_std": float(fold_df["mae"].std(ddof=0)),
    }

    # Fit final model on all data (for inference)
    pipe.fit(X, y)

    return pipe, results, fold_df


# ---------------------------
# Interpretation
# ---------------------------

def get_rf_feature_importance(
    fitted_pipe: Pipeline,
    df: pd.DataFrame,
    target: str,
    *,
    id_cols: Sequence[str] = ("POINT_ID",),
    drop_cols: Sequence[str] = (),
    top_k: int = 30,
) -> pd.DataFrame:
    """
    Get RF feature importances on original numeric feature names.
    Returns top_k features: feature, importance
    """
    model = fitted_pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")

    X, _ = _prepare_xy(df, target, id_cols=id_cols, drop_cols=drop_cols)
    numeric_cols = list(X.columns)

    importances = pd.Series(model.feature_importances_, index=numeric_cols)
    out = (
        importances.sort_values(ascending=False)
        .head(top_k)
        .reset_index()
        .rename(columns={"index": "feature", 0: "importance"})
    )
    out["importance"] = out["importance"].astype(float)
    return out


def permutation_importance_report(
    fitted_pipe: Pipeline,
    df: pd.DataFrame,
    target: str,
    *,
    id_cols: Sequence[str] = ("POINT_ID",),
    drop_cols: Sequence[str] = (),
    n_repeats: int = 10,
    random_state: int = 42,
    top_k: int = 30,
) -> pd.DataFrame:
    """Permutation importance on the fitted pipeline (model-agnostic)."""
    X, y = _prepare_xy(df, target, id_cols=id_cols, drop_cols=drop_cols)
    perm = permutation_importance(
        fitted_pipe, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="r2"
    )
    imp = pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).head(top_k)

    imp["importance_mean"] = imp["importance_mean"].astype(float)
    imp["importance_std"] = imp["importance_std"].astype(float)
    return imp


# ---------------------------
# Plotting helpers
# ---------------------------

def plot_model_comparison(summary_df: pd.DataFrame, target: str, out_path: Path) -> None:
    """
    Bar plot: mean R2 by model/config and CV type for a given target.
    """
    df_t = summary_df[summary_df["target"] == target].copy()
    if df_t.empty:
        return

    # Create labels
    df_t["label"] = df_t["model"] + ":" + df_t["config"] + " | " + df_t["cv_type"]
    df_t = df_t.sort_values("cv_r2_mean", ascending=False)

    plt.figure(figsize=(12, max(4, 0.35 * len(df_t))))
    y = np.arange(len(df_t))
    plt.barh(y, df_t["cv_r2_mean"].to_numpy())
    plt.yticks(y, df_t["label"].to_numpy())
    plt.xlabel("Mean R² (CV)")
    plt.title(f"Model comparison for {target}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_random_vs_spatial_scatter(summary_df: pd.DataFrame, target: str, out_path: Path) -> None:
    """
    Scatter plot: RandomCV mean R2 vs SpatialCV mean R2 for each model/config.
    """
    df_t = summary_df[summary_df["target"] == target].copy()
    if df_t.empty:
        return

    # pivot by cv_type
    piv = df_t.pivot_table(
        index=["model", "config"],
        columns="cv_type",
        values="cv_r2_mean",
        aggfunc="mean"
    ).reset_index()

    if "RandomKFold" not in piv.columns or "SpatialGroupKFold" not in piv.columns:
        # need both to draw scatter
        return

    x = piv["RandomKFold"].to_numpy()
    y = piv["SpatialGroupKFold"].to_numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    plt.xlabel("Mean R² (Random CV)")
    plt.ylabel("Mean R² (Spatial CV)")
    plt.title(f"Random vs Spatial CV for {target}")

    # 1:1 line
    min_v = np.nanmin([x.min(), y.min()])
    max_v = np.nanmax([x.max(), y.max()])
    plt.plot([min_v, max_v], [min_v, max_v])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ---------------------------
# Experiment runner
# ---------------------------

@dataclass
class ModelSpec:
    name: str
    config: str
    model: RegressorMixin
    scale_numeric: bool = False
    rf_importance: bool = False


def build_model_suite(random_state: int) -> List[ModelSpec]:
    """
    Multiple algorithms and configurations.
    Keep this compact but meaningful for 1100 samples.
    """
    suite: List[ModelSpec] = []

    # Random Forest: a few configs
    suite.append(ModelSpec(
        name="rf",
        config="n800_leaf2_sqrt",
        model=RandomForestRegressor(
            n_estimators=800,
            random_state=random_state,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2,
        ),
        rf_importance=True
    ))
    suite.append(ModelSpec(
        name="rf",
        config="n1200_leaf1_sqrt",
        model=RandomForestRegressor(
            n_estimators=1200,
            random_state=random_state,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=1,
        ),
        rf_importance=True
    ))
    suite.append(ModelSpec(
        name="rf",
        config="n800_leaf4_log2",
        model=RandomForestRegressor(
            n_estimators=800,
            random_state=random_state,
            n_jobs=-1,
            max_features="log2",
            min_samples_leaf=4,
        ),
        rf_importance=True
    ))

    # HistGradientBoosting: a couple configs
    suite.append(ModelSpec(
        name="hgb",
        config="lr0.05_iter800",
        model=HistGradientBoostingRegressor(
            random_state=random_state,
            learning_rate=0.05,
            max_iter=800,
        ),
        rf_importance=False
    ))
    suite.append(ModelSpec(
        name="hgb",
        config="lr0.03_iter1200",
        model=HistGradientBoostingRegressor(
            random_state=random_state,
            learning_rate=0.03,
            max_iter=1200,
        ),
        rf_importance=False
    ))

    # ElasticNet: sanity checks (scaled)
    suite.append(ModelSpec(
        name="enet",
        config="a0.01_l10.2",
        model=ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=random_state),
        scale_numeric=True,
        rf_importance=False
    ))
    suite.append(ModelSpec(
        name="enet",
        config="a0.05_l10.5",
        model=ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=random_state),
        scale_numeric=True,
        rf_importance=False
    ))

    return suite


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def main():
    parser = argparse.ArgumentParser(description="Soil texture prediction pipeline runner.")
    parser.add_argument("--input", required=True, help="Path to feature dataset (CSV or Excel).")
    parser.add_argument("--targets", nargs="+", required=True, help="Target columns (e.g., Silt Clay Sand).")
    parser.add_argument("--output_dir", default="reports", help="Output directory.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--top_k", type=int, default=30, help="Top K features for importance reports.")
    parser.add_argument("--do_perm_importance", action="store_true", help="Compute permutation importance (slower).")

    # spatial CV options
    parser.add_argument("--group_col", default=None, help="Existing group column for spatial CV (optional).")
    parser.add_argument("--block_size_m", type=int, default=20000, help="Block size (m) for auto spatial blocks from lat/lon.")
    parser.add_argument("--lat_col", default="lat", help="Latitude column name.")
    parser.add_argument("--lon_col", default="lon", help="Longitude column name.")

    args = parser.parse_args()

    df = load_dataset(args.input)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{run_id}"
    ensure_dir(out_dir)

    # Prepare/ensure spatial grouping column
    # If user provided group_col, use it; else build block_id from lat/lon.
    effective_group_col = args.group_col
    if effective_group_col is None:
        try:
            df = add_spatial_blocks(
                df,
                lat_col=args.lat_col,
                lon_col=args.lon_col,
                block_size_m=args.block_size_m,
                out_col="block_id",
            )
            effective_group_col = "block_id"
        except Exception as e:
            print(f"[WARN] Could not create spatial blocks automatically: {e}")
            print("       Spatial CV will be skipped unless you provide --group_col.")
            effective_group_col = None

    # Save run config
    config = {
        "input": args.input,
        "targets": args.targets,
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "top_k": args.top_k,
        "do_perm_importance": bool(args.do_perm_importance),
        "provided_group_col": args.group_col,
        "effective_group_col": effective_group_col,
        "block_size_m": args.block_size_m,
        "lat_col": args.lat_col,
        "lon_col": args.lon_col,
        "timestamp": run_id,
    }
    save_json(out_dir / "config.json", config)

    # Model suite
    suite = build_model_suite(args.random_state)

    summary_rows: List[Dict[str, Any]] = []

    # For each target, run each model config under both CV regimes
    cv_modes = ["random"]
    if effective_group_col is not None:
        cv_modes.append("spatial")

    for target in args.targets:
        for ms in suite:
            for cv_mode in cv_modes:
                print(f"[{run_id}] target='{target}' model='{ms.name}' config='{ms.config}' cv='{cv_mode}'")

                pipe, results, fold_df = evaluate_texture_model(
                    df,
                    target=target,
                    model=ms.model,
                    group_col=effective_group_col,
                    cv_mode=cv_mode,
                    n_splits=args.n_splits,
                    random_state=args.random_state,
                    scale_numeric=ms.scale_numeric,
                    n_jobs=-1,
                )

                # annotate results
                results["model"] = ms.name
                results["config"] = ms.config
                results["cv_mode"] = cv_mode

                # Save per-fold scores
                fold_path = out_dir / f"fold_scores__{target}__{ms.name}__{ms.config}__{results['cv_type']}.csv"
                fold_df.to_csv(fold_path, index=False)

                # Save fitted pipeline (for inference)
                model_path = out_dir / f"pipeline__{target}__{ms.name}__{ms.config}__{results['cv_type']}.joblib"
                dump(pipe, model_path)

                # Save metrics JSON
                save_json(out_dir / f"metrics__{target}__{ms.name}__{ms.config}__{results['cv_type']}.json", results)

                summary_rows.append(results)

                # Feature importance (RF only)
                if ms.rf_importance:
                    try:
                        imp_df = get_rf_feature_importance(
                            pipe, df, target, top_k=args.top_k
                        )
                        imp_df.to_csv(out_dir / f"feature_importance__{target}__{ms.name}__{ms.config}__{results['cv_type']}.csv", index=False)
                    except Exception as e:
                        print(f"  [WARN] RF feature importance failed: {e}")

                # Permutation importance (optional; can be slow)
                if args.do_perm_importance:
                    try:
                        pimp_df = permutation_importance_report(
                            pipe, df, target,
                            n_repeats=10,
                            random_state=args.random_state,
                            top_k=args.top_k
                        )
                        pimp_df.to_csv(out_dir / f"permutation_importance__{target}__{ms.name}__{ms.config}__{results['cv_type']}.csv", index=False)
                    except Exception as e:
                        print(f"  [WARN] Permutation importance failed: {e}")

        # After finishing all models for a target, make plots
        summary_df_for_plots = pd.DataFrame(summary_rows)
        plot_model_comparison(summary_df_for_plots, target, out_dir / f"model_comparison__{target}.png")
        plot_random_vs_spatial_scatter(summary_df_for_plots, target, out_dir / f"cv_scatter__{target}.png")

    # Save global summary
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["target", "cv_type", "cv_r2_mean"], ascending=[True, True, False]
    )
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    save_json(out_dir / "summary_metrics.json", summary_rows)

    print("\nDone.")
    print(f"Reports saved to: {out_dir.resolve()}")
    print("Key files:")
    print(f" - {out_dir/'summary_metrics.csv'}")
    print(f" - {out_dir/'config.json'}")


def run_pipeline(features_path: str, target: str, reports_dir: str) -> None:
    """Entry point called from training/main.py (Docker job)."""
    import sys
    sys.argv = [sys.argv[0], "--input", features_path, "--targets", target, "--output_dir", reports_dir]
    main()


if __name__ == "__main__":
    main()
