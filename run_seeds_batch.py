from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from sentinel_soil.config import load_config
from sentinel_soil.clients.sentinelhub_client import SentinelHubClient, SentinelHubCredentials
from sentinel_soil.clients.evalscript_builder import build_orbit_timeseries_evalscript
from sentinel_soil.utils.geometry import bbox_around_point

from sentinel_soil.utils.sentinelhub_io import (
    extract_response_tar,
    read_userdata_dates,
    split_stacked_tif_to_per_date,
)
from sentinel_soil.features.feature_engineering import (
    read_per_date_stack,
    stack_to_long_dataframe,
    add_indices,
)


def safe_point_name(lat: float, lon: float) -> str:
    return f"point_{lat:.6f}_{lon:.6f}".replace(".", "p")


def plot_grid(grid: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(grid)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_seed_point_features(feat_df: pd.DataFrame, seed_id: str, lat: float, lon: float, min_obs: int) -> pd.DataFrame:
    out = {
        "seed_id": seed_id,
        "lat": lat,
        "lon": lon,
        "n_pixels": int(len(feat_df)),
        "n_pixels_with_min_obs": int((feat_df["n_obs_baresoil"] >= min_obs).sum()),
        "mean_n_obs_baresoil": float(feat_df["n_obs_baresoil"].mean()),
        "median_n_obs_baresoil": float(feat_df["n_obs_baresoil"].median()),
    }

    valid = feat_df[feat_df["n_obs_baresoil"] >= min_obs].copy()
    exclude = {"row", "col", "x", "y", "n_obs_baresoil"}
    feature_cols = [c for c in valid.columns if c not in exclude and pd.api.types.is_numeric_dtype(valid[c])]

    for col in feature_cols:
        s = valid[col].dropna()
        if s.empty:
            out[f"{col}_space_median"] = np.nan
            out[f"{col}_space_mean"] = np.nan
            out[f"{col}_space_std"] = np.nan
        else:
            out[f"{col}_space_median"] = float(s.median())
            out[f"{col}_space_mean"] = float(s.mean())
            out[f"{col}_space_std"] = float(s.std(ddof=0))

    return pd.DataFrame([out])


def run_one_seed(
    sh: SentinelHubClient,
    base_dir: Path,
    cfg,
    seed_id: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    batch = cfg.batch
    if batch is None:
        raise ValueError("cfg.batch is None. Add a 'batch:' section to your YAML config.")

    bands = list(batch.bands)
    size = tuple(batch.size)
    start_date = batch.start_date
    end_date = batch.end_date

    # AOI bbox around point in metric CRS
    bbox, epsg = bbox_around_point(
        lat=lat, lon=lon, size_m=batch.window_m, dst_epsg=batch.dst_epsg
    )

    ts_root = (
        base_dir
        / "timeseries"
        / safe_point_name(lat, lon)
        / f"window_{int(batch.window_m)}m"
        / f"{start_date}_{end_date}"
    )
    per_date_dir = ts_root / "per_date"
    stacked_dir = ts_root / "stacked"
    tmp_dir = ts_root / "_tmp"

    # ---- Extraction stage (skip if already extracted) ----
    already = per_date_dir.exists() and any(per_date_dir.glob("*.tif")) and (per_date_dir / "bands.json").exists()
    if not already:
        evalscript = build_orbit_timeseries_evalscript(bands=bands, units="REFLECTANCE")

        tmp_dir.mkdir(parents=True, exist_ok=True)
        sh.request_tiff(
            evalscript=evalscript,
            bbox=bbox,
            crs_epsg=epsg,
            time_interval=(start_date, end_date),
            size=size,
            max_cloud_coverage=int(batch.max_cloud_coverage),
            output_folder=str(tmp_dir),
            mosaicking_order="leastCC",
        )

        extract_response_tar(tmp_dir)

        default_candidates = list(tmp_dir.glob("**/default.tif"))
        if not default_candidates:
            raise FileNotFoundError("default.tif not found after extracting response.tar")
        stacked_tif = default_candidates[0]

        dates = read_userdata_dates(tmp_dir)

        stacked_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stacked_tif, stacked_dir / "default.tif")
        userdata_candidates = list(tmp_dir.glob("**/userdata.json"))
        if userdata_candidates:
            shutil.copy2(userdata_candidates[0], stacked_dir / "userdata.json")

        split_stacked_tif_to_per_date(stacked_tif, dates, bands, per_date_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- Feature engineering stage ----
    per_date_tifs = sorted(per_date_dir.glob("*.tif"))
    if not per_date_tifs:
        raise FileNotFoundError(f"No per-date tifs in {per_date_dir}")
    dates = [p.stem for p in per_date_tifs]

    stack, transform, profile0 = read_per_date_stack(per_date_tifs)
    ts_df = stack_to_long_dataframe(stack, dates, bands, transform)
    ts_df = add_indices(ts_df)
    ts_df["is_baresoil"] = ts_df["NDVI"] < float(batch.ndvi_threshold)

    feat_root = (
        base_dir
        / "features"
        / safe_point_name(lat, lon)
        / f"window_{int(batch.window_m)}m"
        / f"{start_date}_{end_date}"
    )
    feat_root.mkdir(parents=True, exist_ok=True)

    # Save per-pixel time series
    ts_df_out = ts_df.copy()
    ts_df_out.insert(0, "seed_id", seed_id)
    ts_df_out.insert(1, "seed_lat", lat)
    ts_df_out.insert(2, "seed_lon", lon)
    ts_df_out.to_parquet(feat_root / "pixel_timeseries.parquet", index=False)

    # Aggregate per pixel on bare-soil rows
    bs = ts_df[ts_df["is_baresoil"]].copy()
    group_cols = ["row", "col", "x", "y"]
    agg_targets = bands + ["NDVI", "NDMI", "NDWI", "NBR", "MSAVI", "EVI"]
    agg_dict = {k: ["median", "mean", "std"] for k in agg_targets}

    agg = bs.groupby(group_cols).agg(agg_dict)
    agg.columns = [f"{col}_{stat}_bs" for col, stat in agg.columns]
    agg = agg.reset_index()

    n_obs = bs.groupby(group_cols).size().reset_index(name="n_obs_baresoil")
    feat_df = agg.merge(n_obs, on=group_cols, how="left")

    # Min obs enforcement
    min_obs = int(batch.min_obs_baresoil)
    feature_cols = [c for c in feat_df.columns if c not in group_cols and c != "n_obs_baresoil"]
    insufficient = feat_df["n_obs_baresoil"] < min_obs
    feat_df.loc[insufficient, feature_cols] = np.nan

    # Save per-pixel features
    feat_df_out = feat_df.copy()
    feat_df_out.insert(0, "seed_id", seed_id)
    feat_df_out.insert(1, "seed_lat", lat)
    feat_df_out.insert(2, "seed_lon", lon)
    feat_df_out.to_parquet(feat_root / "bare_soil_pixel_features.parquet", index=False)

    # Save seed-level features (one row)
    seed_df = build_seed_point_features(feat_df, seed_id, lat, lon, min_obs=min_obs)
    # store batch metadata too
    seed_df["window_m"] = float(batch.window_m)
    seed_df["start_date"] = start_date
    seed_df["end_date"] = end_date
    seed_df["ndvi_threshold"] = float(batch.ndvi_threshold)
    seed_df["max_cloud_coverage"] = int(batch.max_cloud_coverage)
    seed_df.to_parquet(feat_root / "seed_point_features.parquet", index=False)

    # Quick sanity plots
    T, _, H, W = stack.shape
    plots_dir = feat_root / "plots"
    ndvi_grid = np.full((H, W), np.nan, dtype=np.float32)
    ndmi_grid = np.full((H, W), np.nan, dtype=np.float32)
    nobs_grid = np.zeros((H, W), dtype=np.float32)

    for _, rec in feat_df.iterrows():
        r = int(rec["row"])
        c = int(rec["col"])
        ndvi_grid[r, c] = rec.get("NDVI_median_bs", np.nan)
        ndmi_grid[r, c] = rec.get("NDMI_median_bs", np.nan)
        nobs_grid[r, c] = float(rec.get("n_obs_baresoil", 0) or 0)

    plot_grid(ndvi_grid, plots_dir / "ndvi_median_bs.png", f"{seed_id} NDVI median (bare-soil)")
    plot_grid(ndmi_grid, plots_dir / "ndmi_median_bs.png", f"{seed_id} NDMI median (bare-soil)")
    plot_grid(nobs_grid, plots_dir / "n_obs_baresoil.png", f"{seed_id} n_obs bare-soil")

    return seed_df


def main(config_path: str = "configs/dev.yaml"):
    cfg = load_config(config_path)

    if cfg.batch is None:
        raise ValueError("Missing 'batch:' section in config. Add it to configs/dev.yaml.")

    base_dir = Path(cfg.data.base_dir)

    creds = SentinelHubCredentials(
        client_id=cfg.cdse.client_id,
        client_secret=cfg.cdse.client_secret,
    )
    sh = SentinelHubClient(creds)

    seeds_csv = cfg.batch.seeds_csv
    df_seeds = pd.read_csv(seeds_csv)

    if "lat" not in df_seeds.columns or "lon" not in df_seeds.columns:
        raise ValueError("seeds CSV must contain columns: lat, lon (and optionally seed_id)")

    if "seed_id" not in df_seeds.columns:
        df_seeds["seed_id"] = [f"seed_{i:04d}" for i in range(len(df_seeds))]

    batch_out_dir = base_dir / "features" / "batch" / cfg.batch.name
    batch_out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: List[pd.DataFrame] = []
    errors: List[dict] = []

    for _, row in df_seeds.iterrows():
        seed_id = str(row["seed_id"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        try:
            seed_df = run_one_seed(sh, base_dir, cfg, seed_id, lat, lon)
            seed_rows.append(seed_df)
            print(f"✅ Done: {seed_id} ({lat}, {lon})")
        except Exception as e:
            errors.append({"seed_id": seed_id, "lat": lat, "lon": lon, "error": str(e)})
            print(f"❌ Failed: {seed_id} ({lat}, {lon}) -> {e}")

    if seed_rows:
        all_seed_features = pd.concat(seed_rows, ignore_index=True)
        out_path = batch_out_dir / "seed_features.parquet"
        all_seed_features.to_parquet(out_path, index=False)
        print(f"\n✅ Consolidated seed features written to: {out_path}")

    if errors:
        err_df = pd.DataFrame(errors)
        err_path = batch_out_dir / "errors.csv"
        err_df.to_csv(err_path, index=False)
        print(f"⚠️ Errors written to: {err_path}")


if __name__ == "__main__":
    main()
