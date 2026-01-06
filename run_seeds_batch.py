from __future__ import annotations

import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
import matplotlib.pyplot as plt

from sentinel_soil.config import load_config
from sentinel_soil.clients.sentinelhub_client import SentinelHubClient, SentinelHubCredentials
from sentinel_soil.clients.evalscript_builder import build_orbit_timeseries_evalscript
from sentinel_soil.utils.geometry import bbox_around_point


# ----------------------------
# Configuration for the batch
# ----------------------------
@dataclass
class BatchParams:
    # Spatial
    window_m: float = 100.0
    dst_epsg: int = 32632  # UTM 32N for N. Italy

    # Time
    start_date: str = "2024-06-01"
    end_date: str = "2024-06-30"

    # Raster sampling grid
    size: Tuple[int, int] = (10, 10)  # 10x10 over 100m ~ 10m per pixel

    # Bands (includes 20m bands resampled to requested grid)
    bands: Tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11", "B12")

    # Filtering / aggregation
    ndvi_threshold: float = 0.2
    min_obs_baresoil: int = 2

    # Sentinel Hub dataFilter
    max_cloud_coverage: int = 80

    # Output naming
    batch_name: str = "seeds_batch"


def safe_point_name(lat: float, lon: float) -> str:
    return f"point_{lat:.6f}_{lon:.6f}".replace(".", "p")


# ----------------------------
# Sentinel Hub response helpers
# ----------------------------
def extract_response_tar(folder: Path) -> None:
    tar_path = next(folder.glob("**/response.tar"), None)
    if tar_path is None:
        raise FileNotFoundError("response.tar not found in Sentinel Hub output folder")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=folder)


def read_userdata_dates(folder: Path) -> List[str]:
    candidates = list(folder.glob("**/userdata.json"))
    if not candidates:
        raise FileNotFoundError("userdata.json not found")
    userdata_path = candidates[0]
    with open(userdata_path, "r", encoding="utf-8") as f:
        userdata = json.load(f)

    dates_str = userdata.get("acquisition_dates")
    if not dates_str:
        raise KeyError("acquisition_dates missing in userdata.json")

    dates = json.loads(dates_str)
    return [d[:10] for d in dates]  # normalize to YYYY-MM-DD


def split_stacked_tif_to_per_date(
    stacked_tif: Path,
    dates: List[str],
    bands: List[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_bands = len(bands)

    with rasterio.open(stacked_tif) as src:
        profile = src.profile.copy()
        expected_count = len(dates) * n_bands
        if src.count != expected_count:
            raise ValueError(
                f"Band count mismatch: src.count={src.count}, expected={expected_count} "
                f"(dates={len(dates)}, bands_per_date={n_bands})"
            )

        for i, date in enumerate(dates):
            start_band = i * n_bands + 1  # 1-based
            data = src.read(list(range(start_band, start_band + n_bands))).astype("float32")

            out_path = out_dir / f"{date}.tif"
            out_profile = profile.copy()
            out_profile.update(count=n_bands, dtype="float32")

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(data)

    with open(out_dir / "bands.json", "w", encoding="utf-8") as f:
        json.dump({"bands": bands}, f, indent=2)


# ----------------------------
# Timeseries -> DataFrame helpers
# ----------------------------
def read_per_date_stack(per_date_tifs: List[Path]) -> tuple[np.ndarray, rasterio.Affine, dict]:
    """
    Returns:
      stack: (T, B, H, W) float32
      transform: affine
      profile: raster profile
    """
    if not per_date_tifs:
        raise ValueError("No per-date GeoTIFFs provided")

    with rasterio.open(per_date_tifs[0]) as src0:
        profile0 = src0.profile.copy()
        transform0 = src0.transform
        B = src0.count
        H, W = src0.height, src0.width

    T = len(per_date_tifs)
    stack = np.zeros((T, B, H, W), dtype=np.float32)

    for t, tif_path in enumerate(per_date_tifs):
        with rasterio.open(tif_path) as src:
            arr = src.read().astype(np.float32)  # (B,H,W)
            if arr.shape != (B, H, W):
                raise ValueError(f"Shape mismatch in {tif_path}: {arr.shape} vs {(B, H, W)}")
            stack[t] = arr

    return stack, transform0, profile0


def stack_to_long_dataframe(
    stack: np.ndarray,
    dates: List[str],
    bands: List[str],
    transform: rasterio.Affine,
) -> pd.DataFrame:
    """
    stack: (T,B,H,W)
    Output: long df with rows = T*H*W
    Columns: date, row, col, x, y, bands...
    """
    T, B, H, W = stack.shape
    if len(dates) != T:
        raise ValueError("dates length must match stack T")

    rows = []
    for t in range(T):
        date = dates[t]
        for r in range(H):
            for c in range(W):
                x, y = xy(transform, r, c)
                rec = {"date": date, "row": r, "col": c, "x": float(x), "y": float(y)}
                for bi, bname in enumerate(bands):
                    rec[bname] = float(stack[t, bi, r, c])
                rows.append(rec)
    return pd.DataFrame(rows)


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.where(den != 0, num / den, np.nan)


def add_indices(ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds NDVI, NDMI, NDWI, NBR, MSAVI, EVI
    Requires B02,B03,B04,B08,B11,B12
    """
    nir = ts_df["B08"].to_numpy(dtype=float)
    red = ts_df["B04"].to_numpy(dtype=float)
    green = ts_df["B03"].to_numpy(dtype=float)
    blue = ts_df["B02"].to_numpy(dtype=float)
    swir1 = ts_df["B11"].to_numpy(dtype=float)
    swir2 = ts_df["B12"].to_numpy(dtype=float)

    ts_df["NDVI"] = safe_ratio(nir - red, nir + red)
    ts_df["NDMI"] = safe_ratio(nir - swir1, nir + swir1)
    ts_df["NDWI"] = safe_ratio(green - nir, green + nir)  # McFeeters
    ts_df["NBR"] = safe_ratio(nir - swir2, nir + swir2)

    term = (2 * nir + 1) ** 2 - 8 * (nir - red)
    term = np.where(term < 0, 0, term)
    ts_df["MSAVI"] = (2 * nir + 1 - np.sqrt(term)) / 2

    ts_df["EVI"] = safe_ratio(2.5 * (nir - red), (nir + 6 * red - 7.5 * blue + 1))

    return ts_df


def plot_grid(grid: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(grid)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_seed_point_features(
    feat_df: pd.DataFrame,
    seed_id: str,
    lat: float,
    lon: float,
    params: BatchParams,
) -> pd.DataFrame:
    """
    One-row feature vector per seed, aggregating per-pixel features across space.
    """
    out = {
        "seed_id": seed_id,
        "lat": lat,
        "lon": lon,
        "window_m": params.window_m,
        "start_date": params.start_date,
        "end_date": params.end_date,
        "ndvi_threshold": params.ndvi_threshold,
        "min_obs_baresoil": params.min_obs_baresoil,
        "n_pixels": int(len(feat_df)),
        "n_pixels_with_min_obs": int((feat_df["n_obs_baresoil"] >= params.min_obs_baresoil).sum()),
        "mean_n_obs_baresoil": float(feat_df["n_obs_baresoil"].mean()),
        "median_n_obs_baresoil": float(feat_df["n_obs_baresoil"].median()),
    }

    valid = feat_df[feat_df["n_obs_baresoil"] >= params.min_obs_baresoil].copy()
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


# ----------------------------
# Core: run for one seed
# ----------------------------
def run_one_seed(
    sh: SentinelHubClient,
    base_dir: Path,
    seed_id: str,
    lat: float,
    lon: float,
    params: BatchParams,
) -> pd.DataFrame:
    bands = list(params.bands)

    bbox, epsg = bbox_around_point(lat=lat, lon=lon, size_m=params.window_m, dst_epsg=params.dst_epsg)

    ts_root = (
        base_dir
        / "timeseries"
        / safe_point_name(lat, lon)
        / f"window_{int(params.window_m)}m"
        / f"{params.start_date}_{params.end_date}"
    )
    per_date_dir = ts_root / "per_date"
    stacked_dir = ts_root / "stacked"
    tmp_dir = ts_root / "_tmp"

    already = per_date_dir.exists() and any(per_date_dir.glob("*.tif")) and (per_date_dir / "bands.json").exists()
    if not already:
        evalscript = build_orbit_timeseries_evalscript(bands=bands, units="REFLECTANCE")

        tmp_dir.mkdir(parents=True, exist_ok=True)
        sh.request_tiff(
            evalscript=evalscript,
            bbox=bbox,
            crs_epsg=epsg,
            time_interval=(params.start_date, params.end_date),
            size=params.size,
            max_cloud_coverage=params.max_cloud_coverage,
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

    # ---- aggregation ----
    per_date_tifs = sorted(per_date_dir.glob("*.tif"))
    if not per_date_tifs:
        raise FileNotFoundError(f"No per-date tifs in {per_date_dir}")

    dates = [p.stem for p in per_date_tifs]

    stack, transform, profile0 = read_per_date_stack(per_date_tifs)
    ts_df = stack_to_long_dataframe(stack, dates, bands, transform)
    ts_df = add_indices(ts_df)
    ts_df["is_baresoil"] = ts_df["NDVI"] < params.ndvi_threshold

    feat_root = (
        base_dir
        / "features"
        / safe_point_name(lat, lon)
        / f"window_{int(params.window_m)}m"
        / f"{params.start_date}_{params.end_date}"
    )
    feat_root.mkdir(parents=True, exist_ok=True)

    # Save time series parquet
    ts_df_out = ts_df.copy()
    ts_df_out.insert(0, "seed_id", seed_id)
    ts_df_out.insert(1, "seed_lat", lat)
    ts_df_out.insert(2, "seed_lon", lon)
    ts_df_out.to_parquet(feat_root / "pixel_timeseries.parquet", index=False)

    # Per-pixel aggregation on bare-soil rows
    bs = ts_df[ts_df["is_baresoil"]].copy()
    group_cols = ["row", "col", "x", "y"]
    agg_targets = list(bands) + ["NDVI", "NDMI", "NDWI", "NBR", "MSAVI", "EVI"]
    agg_dict = {k: ["median", "mean", "std"] for k in agg_targets}

    agg = bs.groupby(group_cols).agg(agg_dict)
    agg.columns = [f"{col}_{stat}_bs" for col, stat in agg.columns]
    agg = agg.reset_index()

    n_obs = bs.groupby(group_cols).size().reset_index(name="n_obs_baresoil")
    feat_df = agg.merge(n_obs, on=group_cols, how="left")

    feature_cols = [c for c in feat_df.columns if c not in group_cols and c != "n_obs_baresoil"]
    insufficient = feat_df["n_obs_baresoil"] < params.min_obs_baresoil
    feat_df.loc[insufficient, feature_cols] = np.nan

    # Save per-pixel feature parquet
    feat_df_out = feat_df.copy()
    feat_df_out.insert(0, "seed_id", seed_id)
    feat_df_out.insert(1, "seed_lat", lat)
    feat_df_out.insert(2, "seed_lon", lon)
    feat_df_out.to_parquet(feat_root / "bare_soil_pixel_features.parquet", index=False)

    # Median composite raster (bands only)
    T, _, H, W = stack.shape
    composite = np.full((len(bands), H, W), np.nan, dtype=np.float32)
    for _, rec in feat_df.iterrows():
        r = int(rec["row"])
        c = int(rec["col"])
        for bi, bname in enumerate(bands):
            composite[bi, r, c] = rec.get(f"{bname}_median_bs", np.nan)

    comp_profile = profile0.copy()
    comp_profile.update(count=len(bands), dtype="float32")
    with rasterio.open(feat_root / "bare_soil_median_composite.tif", "w", **comp_profile) as dst:
        dst.write(np.nan_to_num(composite, nan=0.0).astype(np.float32))

    # Sanity plots
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

    # Seed-level features (one row)
    seed_df = build_seed_point_features(feat_df, seed_id, lat, lon, params)
    seed_df.to_parquet(feat_root / "seed_point_features.parquet", index=False)

    return seed_df


# ----------------------------
# Batch runner
# ----------------------------
def main(config_path: str = "configs/dev.yaml", seeds_csv: str = "seeds.csv"):
    """
    seeds.csv expected columns:
      - lat
      - lon
      - seed_id (optional)
    """
    cfg = load_config(config_path)
    base_dir = Path(cfg.data.base_dir)

    params = BatchParams(
        batch_name="seeds_batch",
        window_m=100.0,
        start_date="2024-06-01",
        end_date="2024-06-30",
        size=(10, 10),
        ndvi_threshold=0.2,
        min_obs_baresoil=2,
        max_cloud_coverage=80,
    )

    creds = SentinelHubCredentials(
        client_id=cfg.cdse.client_id,
        client_secret=cfg.cdse.client_secret,
    )
    sh = SentinelHubClient(creds)

    df_seeds = pd.read_csv(seeds_csv)
    if "lat" not in df_seeds.columns or "lon" not in df_seeds.columns:
        raise ValueError("seeds.csv must contain columns: lat, lon (and optionally seed_id)")

    if "seed_id" not in df_seeds.columns:
        df_seeds["seed_id"] = [f"seed_{i:04d}" for i in range(len(df_seeds))]

    batch_out_dir = base_dir / "features" / "batch" / params.batch_name
    batch_out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: List[pd.DataFrame] = []
    errors: List[dict] = []

    for _, row in df_seeds.iterrows():
        seed_id = str(row["seed_id"])
        lat = float(row["lat"])
        lon = float(row["lon"])

        try:
            seed_df = run_one_seed(sh, base_dir, seed_id, lat, lon, params)
            seed_rows.append(seed_df)
            print(f"✅ Done: {seed_id} ({lat}, {lon})")
        except Exception as e:
            errors.append({"seed_id": seed_id, "lat": lat, "lon": lon, "error": str(e)})
            print(f"❌ Failed: {seed_id} ({lat}, {lon}) -> {e}")

    if seed_rows:
        all_seed_features = pd.concat(seed_rows, ignore_index=True)
        seed_features_parquet = batch_out_dir / "seed_features.parquet"
        all_seed_features.to_parquet(seed_features_parquet, index=False)
        print(f"\n✅ Consolidated seed features written to: {seed_features_parquet}")

    if errors:
        err_df = pd.DataFrame(errors)
        err_path = batch_out_dir / "errors.csv"
        err_df.to_csv(err_path, index=False)
        print(f"⚠️ Errors written to: {err_path}")


if __name__ == "__main__":
    main()
