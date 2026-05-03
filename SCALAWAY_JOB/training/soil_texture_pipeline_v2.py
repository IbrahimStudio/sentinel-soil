#!/usr/bin/env python3
"""
soil_texture_pipeline_v2.py

Soil texture prediction pipeline runner (pre-built feature table).

What's new (v2)
---------------
✅ Prevents DATA LEAKAGE automatically:
   - If you predict one texture fraction (Clay/Silt/Sand/Coarse),
     the other fractions are dropped from predictors.

✅ Exports correlation matrices:
   - correlations/corr__{target}__predictors.csv
   - correlations/corr__{target}__with_target.csv
   - correlations/corr_heatmap__{target}.png

✅ Optional SHAP interpretation (tree models):
   - {target}__{model}__{config}__{cv}__shap_summary_bar.png
   - {target}__{model}__{config}__{cv}__shap_summary_dot.png
   - If SHAP is not installed, it will skip with a warning.

Still included
--------------
- Runs both Random CV (KFold) and Spatial CV (GroupKFold with spatial blocks from lat/lon).
- Sweeps multiple models + configs for each target.
- Saves reports and fitted pipelines.

Usage
-----
python soil_texture_pipeline_v2.py --input features.xlsx --targets Clay Silt Sand Coarse --output_dir reports
python soil_texture_pipeline_v2.py --input features.xlsx --targets Clay --block_size_m 10000 --output_dir reports
python soil_texture_pipeline_v2.py --input features.xlsx --targets Clay --do_shap --output_dir reports
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

import matplotlib.pyplot as plt


# ---------------------------
# Core pipeline construction
# ---------------------------

def build_texture_pipeline(X: pd.DataFrame, model: RegressorMixin, *, scale_numeric: bool = False) -> Pipeline:
    numeric_features = list(X.select_dtypes(include=[np.number]).columns)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline(steps=numeric_steps), numeric_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


# ---------------------------
# Data I/O
# ---------------------------

def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    raise ValueError("Unsupported file type. Use .csv, .xlsx/.xls, or .parquet")


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------
# Leakage prevention + prep
# ---------------------------

TEXTURE_FRACTIONS_DEFAULT = ("Clay", "Silt", "Sand", "Coarse")

# Lab chemistry columns measured on the same samples as the texture labels.
# Including them as predictors leaks ground-truth information — drop by default.
CHEMISTRY_COLS_DEFAULT = (
    "pH_CaCl2", "pH_H2O", "EC", "OC", "CaCO3", "P", "N", "K",
    "OC (20-30 cm)", "CaCO3 (20-30 cm)", "Ox_Al", "Ox_Fe",
)

# Geographic coordinates — including raw lat/lon as features causes spatial leakage:
# the model learns "location → texture" instead of "spectrum → texture".
# Especially harmful with spatial GroupKFold because groups are derived from these columns.
COORD_COLS_DEFAULT = (
    "lat", "lon", "TH_LAT", "TH_LONG", "Elev",
)

# Statistics API metadata — coverage quality metrics and processing parameters.
# These are not spectral features; including them biases the model toward data-quality proxies.
COVERAGE_META_COLS_DEFAULT = (
    "coverage_threshold", "n_days_total", "n_days_kept",
    "kept_ratio", "coverage_median_kept", "coverage_min_kept",
)

# Combined default drop set (excludes texture fractions — those are handled separately).
NON_SPECTRAL_COLS_DEFAULT = CHEMISTRY_COLS_DEFAULT + COORD_COLS_DEFAULT + COVERAGE_META_COLS_DEFAULT


def _to_float_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "if":
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _prepare_xy(
    df: pd.DataFrame,
    target: str,
    *,
    id_cols: Sequence[str] = ("POINT_ID",),
    drop_cols: Sequence[str] = (),
    texture_fraction_cols: Sequence[str] = TEXTURE_FRACTIONS_DEFAULT,
    auto_drop_other_texture_fractions: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe.")

    data = df.copy()

    for c in id_cols:
        if c in data.columns:
            data.drop(columns=c, inplace=True)

    y = pd.to_numeric(data[target], errors="coerce")
    X = data.drop(columns=[target])

    if drop_cols:
        X = X.drop(columns=list(drop_cols), errors="ignore")

    if auto_drop_other_texture_fractions and target in texture_fraction_cols:
        other_fracs = [c for c in texture_fraction_cols if c != target and c in X.columns]
        if other_fracs:
            X = X.drop(columns=other_fracs, errors="ignore")

    X = X.select_dtypes(include=[np.number])

    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    return X, y


# ---------------------------
# Spatial blocks
# ---------------------------

def add_spatial_blocks(
    df: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    block_size_m: int = 20000,
    out_col: str = "block_id",
) -> pd.DataFrame:
    out = df.copy()

    if lat_col not in out.columns or lon_col not in out.columns:
        raise ValueError(f"Need '{lat_col}' and '{lon_col}' columns to build spatial blocks.")

    lat = _to_float_series(out[lat_col])
    lon = _to_float_series(out[lon_col])

    if lat.isna().any() or lon.isna().any():
        bad = out[lat.isna() | lon.isna()][[lat_col, lon_col]].head(5)
        raise ValueError(
            "Could not parse some lat/lon values to float. Example rows:\n" + bad.to_string(index=False)
        )

    try:
        from pyproj import Transformer  # type: ignore
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_m, y_m = transformer.transform(lon.to_numpy(), lat.to_numpy())
        x_m = np.asarray(x_m, dtype=float)
        y_m = np.asarray(y_m, dtype=float)
    except Exception:
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
    texture_fraction_cols: Sequence[str] = TEXTURE_FRACTIONS_DEFAULT,
    auto_drop_other_texture_fractions: bool = True,
    group_col: Optional[str] = None,
    cv_mode: str = "random",
    n_splits: int = 5,
    random_state: int = 42,
    scale_numeric: bool = False,
    n_jobs: int = -1,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.Series]:
    X, y = _prepare_xy(
        df, target,
        id_cols=id_cols,
        drop_cols=drop_cols,
        texture_fraction_cols=texture_fraction_cols,
        auto_drop_other_texture_fractions=auto_drop_other_texture_fractions,
    )

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

    scoring = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error", "neg_mae": "neg_mean_absolute_error"}

    cv_out = cross_validate(
        pipe, X, y,
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

    pipe.fit(X, y)
    return pipe, results, fold_df, X, y


# ---------------------------
# Interpretation helpers
# ---------------------------

def get_rf_feature_importance(fitted_pipe: Pipeline, X: pd.DataFrame, top_k: int = 30) -> pd.DataFrame:
    model = fitted_pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")
    imp = pd.Series(model.feature_importances_, index=list(X.columns))
    out = imp.sort_values(ascending=False).head(top_k).reset_index()
    out.columns = ["feature", "importance"]
    out["importance"] = out["importance"].astype(float)
    return out


def permutation_importance_report(
    fitted_pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    top_k: int = 30,
) -> pd.DataFrame:
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


def try_shap_plots(
    fitted_pipe: Pipeline,
    X: pd.DataFrame,
    *,
    out_prefix: Path,
    max_samples: int = 500,
) -> None:
    try:
        import shap  # type: ignore
    except Exception as e:
        print(f"  [WARN] SHAP not available ({e}). Skipping SHAP plots.")
        return

    model = fitted_pipe.named_steps["model"]
    preprocess = fitted_pipe.named_steps["preprocess"]

    Xs = X.copy()
    if len(Xs) > max_samples:
        Xs = Xs.sample(n=max_samples, random_state=42)

    X_trans = preprocess.transform(Xs)

    try:
        feat_names = preprocess.get_feature_names_out()
        feat_names = [str(f) for f in feat_names]
    except Exception:
        feat_names = [f"f{i}" for i in range(X_trans.shape[1])]

    try:
        explainer = shap.Explainer(model, X_trans, feature_names=feat_names)
        shap_values = explainer(X_trans)
    except Exception as e:
        print(f"  [WARN] SHAP explainer failed: {e}")
        return

    plt.figure()
    shap.summary_plot(shap_values, features=X_trans, feature_names=feat_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "__shap_summary_bar.png"), dpi=180)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, features=X_trans, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "__shap_summary_dot.png"), dpi=180)
    plt.close()


# ---------------------------
# Correlation exports
# ---------------------------

def export_correlations(X: pd.DataFrame, y: pd.Series, *, target: str, out_dir: Path) -> None:
    ensure_dir(out_dir)

    corr_pred = X.corr(numeric_only=True, method="pearson")
    corr_pred.to_csv(out_dir / f"corr__{target}__predictors.csv")

    df_all = X.copy()
    df_all[target] = y
    corr_all = df_all.corr(numeric_only=True, method="pearson")
    corr_all.to_csv(out_dir / f"corr__{target}__with_target.csv")

    mat = corr_all.to_numpy()
    labels = list(corr_all.columns)

    plt.figure(figsize=(max(8, 0.35 * len(labels)), max(6, 0.35 * len(labels))))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title(f"Correlation matrix (Pearson) - {target}")
    plt.tight_layout()
    plt.savefig(out_dir / f"corr_heatmap__{target}.png", dpi=180)
    plt.close()


# ---------------------------
# Plotting helpers
# ---------------------------

def plot_model_comparison(summary_df: pd.DataFrame, target: str, out_path: Path) -> None:
    df_t = summary_df[summary_df["target"] == target].copy()
    if df_t.empty:
        return
    df_t["label"] = df_t["model"] + ":" + df_t["config"] + " | " + df_t["cv_type"]
    df_t = df_t.sort_values("cv_r2_mean", ascending=False)

    plt.figure(figsize=(12, max(4, 0.35 * len(df_t))))
    yy = np.arange(len(df_t))
    plt.barh(yy, df_t["cv_r2_mean"].to_numpy())
    plt.yticks(yy, df_t["label"].to_numpy())
    plt.xlabel("Mean R² (CV)")
    plt.title(f"Model comparison for {target}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_random_vs_spatial_scatter(summary_df: pd.DataFrame, target: str, out_path: Path) -> None:
    df_t = summary_df[summary_df["target"] == target].copy()
    if df_t.empty:
        return

    piv = df_t.pivot_table(
        index=["model", "config"],
        columns="cv_type",
        values="cv_r2_mean",
        aggfunc="mean"
    ).reset_index()

    if "RandomKFold" not in piv.columns or "SpatialGroupKFold" not in piv.columns:
        return

    x = piv["RandomKFold"].to_numpy()
    y = piv["SpatialGroupKFold"].to_numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    plt.xlabel("Mean R² (Random CV)")
    plt.ylabel("Mean R² (Spatial CV)")
    plt.title(f"Random vs Spatial CV for {target}")

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
    shap_ok: bool = False


def build_model_suite(random_state: int) -> List[ModelSpec]:
    suite: List[ModelSpec] = []

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
        rf_importance=True,
        shap_ok=True,
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
        rf_importance=True,
        shap_ok=True,
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
        rf_importance=True,
        shap_ok=True,
    ))

    suite.append(ModelSpec(
        name="hgb",
        config="lr0.05_iter800",
        model=HistGradientBoostingRegressor(
            random_state=random_state,
            learning_rate=0.05,
            max_iter=800,
        ),
        shap_ok=True,
    ))
    suite.append(ModelSpec(
        name="hgb",
        config="lr0.03_iter1200",
        model=HistGradientBoostingRegressor(
            random_state=random_state,
            learning_rate=0.03,
            max_iter=1200,
        ),
        shap_ok=True,
    ))

    suite.append(ModelSpec(
        name="enet",
        config="a0.01_l10.2",
        model=ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=random_state),
        scale_numeric=True,
    ))
    suite.append(ModelSpec(
        name="enet",
        config="a0.05_l10.5",
        model=ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=random_state),
        scale_numeric=True,
    ))

    return suite


def main():
    parser = argparse.ArgumentParser(description="Soil texture prediction pipeline runner.")
    parser.add_argument("--input", required=True, help="Path to feature dataset (CSV or Excel).")
    parser.add_argument("--targets", nargs="+", required=True, help="Target columns.")
    parser.add_argument("--output_dir", default="reports", help="Output directory.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--top_k", type=int, default=30, help="Top K features for importance reports.")
    parser.add_argument("--do_perm_importance", action="store_true", help="Compute permutation importance.")
    parser.add_argument("--do_shap", action="store_true", help="Compute SHAP plots for tree models (requires shap).")

    # spatial CV
    parser.add_argument("--group_col", default=None, help="Existing group column for spatial CV (optional).")
    parser.add_argument("--block_size_m", type=int, default=20000, help="Block size (m) for auto spatial blocks.")
    parser.add_argument("--lat_col", default="lat", help="Latitude column name.")
    parser.add_argument("--lon_col", default="lon", help="Longitude column name.")

    # leakage prevention
    parser.add_argument("--texture_cols", nargs="*", default=list(TEXTURE_FRACTIONS_DEFAULT),
                        help="Texture fraction columns treated as mutually exclusive predictors.")
    parser.add_argument("--no_auto_drop_texture", action="store_true",
                        help="Disable automatic dropping of other texture fractions (NOT recommended).")
    parser.add_argument("--drop_cols", nargs="*", default=list(NON_SPECTRAL_COLS_DEFAULT),
                        help="Columns to drop before training (default: lab chemistry, coordinates, and coverage metadata).")

    args = parser.parse_args()

    df = load_dataset(args.input)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{run_id}"
    ensure_dir(out_dir)

    # spatial group
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

    config = {
        "input": args.input,
        "targets": args.targets,
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "top_k": args.top_k,
        "do_perm_importance": bool(args.do_perm_importance),
        "do_shap": bool(args.do_shap),
        "provided_group_col": args.group_col,
        "effective_group_col": effective_group_col,
        "block_size_m": args.block_size_m,
        "lat_col": args.lat_col,
        "lon_col": args.lon_col,
        "texture_cols": args.texture_cols,
        "auto_drop_other_texture_fractions": not args.no_auto_drop_texture,
        "drop_cols": args.drop_cols,
        "timestamp": run_id,
    }
    save_json(out_dir / "config.json", config)

    suite = build_model_suite(args.random_state)
    summary_rows: List[Dict[str, Any]] = []

    cv_modes = ["random"]
    if effective_group_col is not None:
        cv_modes.append("spatial")

    corr_dir = out_dir / "correlations"
    ensure_dir(corr_dir)

    for target in args.targets:
        # correlation export (leakage-safe)
        try:
            Xcorr, ycorr = _prepare_xy(
                df, target,
                drop_cols=args.drop_cols,
                texture_fraction_cols=args.texture_cols,
                auto_drop_other_texture_fractions=not args.no_auto_drop_texture,
            )
            export_correlations(Xcorr, ycorr, target=target, out_dir=corr_dir)
        except Exception as e:
            print(f"[WARN] Correlation export failed for {target}: {e}")

        for ms in suite:
            for cv_mode in cv_modes:
                print(f"[{run_id}] target='{target}' model='{ms.name}' config='{ms.config}' cv='{cv_mode}'")

                pipe, results, fold_df, X_used, y_used = evaluate_texture_model(
                    df,
                    target=target,
                    model=ms.model,
                    group_col=effective_group_col,
                    cv_mode=cv_mode,
                    n_splits=args.n_splits,
                    random_state=args.random_state,
                    scale_numeric=ms.scale_numeric,
                    n_jobs=-1,
                    drop_cols=args.drop_cols,
                    texture_fraction_cols=args.texture_cols,
                    auto_drop_other_texture_fractions=not args.no_auto_drop_texture,
                )

                results["model"] = ms.name
                results["config"] = ms.config
                results["cv_mode"] = cv_mode
                summary_rows.append(results)

                fold_df.to_csv(out_dir / f"fold_scores__{target}__{ms.name}__{ms.config}__{results['cv_type']}.csv", index=False)
                dump(pipe, out_dir / f"pipeline__{target}__{ms.name}__{ms.config}__{results['cv_type']}.joblib")
                save_json(out_dir / f"metrics__{target}__{ms.name}__{ms.config}__{results['cv_type']}.json", results)

                if ms.rf_importance:
                    try:
                        imp_df = get_rf_feature_importance(pipe, X_used, top_k=args.top_k)
                        imp_df.to_csv(out_dir / f"feature_importance__{target}__{ms.name}__{ms.config}__{results['cv_type']}.csv", index=False)
                    except Exception as e:
                        print(f"  [WARN] RF feature importance failed: {e}")

                if args.do_perm_importance:
                    try:
                        pimp_df = permutation_importance_report(
                            pipe, X_used, y_used, n_repeats=10, random_state=args.random_state, top_k=args.top_k
                        )
                        pimp_df.to_csv(out_dir / f"permutation_importance__{target}__{ms.name}__{ms.config}__{results['cv_type']}.csv", index=False)
                    except Exception as e:
                        print(f"  [WARN] Permutation importance failed: {e}")

                if args.do_shap and ms.shap_ok:
                    out_prefix = out_dir / f"{target}__{ms.name}__{ms.config}__{results['cv_type']}"
                    try_shap_plots(pipe, X_used, out_prefix=out_prefix)

        tmp_summary = pd.DataFrame(summary_rows)
        plot_model_comparison(tmp_summary, target, out_dir / f"model_comparison__{target}.png")
        plot_random_vs_spatial_scatter(tmp_summary, target, out_dir / f"cv_scatter__{target}.png")

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["target", "cv_type", "cv_r2_mean"], ascending=[True, True, False])
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    save_json(out_dir / "summary_metrics.json", summary_rows)

    print("\nDone.")
    print(f"Reports saved to: {out_dir.resolve()}")
    print(f" - {out_dir/'summary_metrics.csv'}")
    print(f" - {out_dir/'config.json'}")
    print(f" - {corr_dir} (correlations)")


def run_pipeline(features_path: str, target: str, reports_dir: str) -> None:
    """Entry point called from training/main.py (Docker job)."""
    import sys
    sys.argv = [sys.argv[0], "--input", features_path, "--targets", target, "--output_dir", reports_dir]
    main()


if __name__ == "__main__":
    main()
