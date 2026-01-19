from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
import matplotlib.pyplot as plt

from sentinel_soil.utils.geometry import grid_for_n_pixels

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# ----------------------------
# Logging (console + .log file)
# ----------------------------
def setup_logger(
    *,
    log_dir: Path,
    name: str = "sentinel_soil_features",
    level: int = logging.INFO,
) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logging initialized → {log_path}")
    return logger


# ----------------------------
# Utilities
# ----------------------------
def safe_point_name(lat: float, lon: float) -> str:
    return f"point_{lat:.6f}_{lon:.6f}".replace(".", "p")


def _safe_ratio(num, den):
    return np.where(den != 0, num / den, np.nan)


def load_bands(per_date_dir: Path) -> List[str]:
    p = per_date_dir / "bands.json"
    if not p.exists():
        raise FileNotFoundError(f"bands.json not found in {per_date_dir}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)["bands"]


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
                raise ValueError(f"Shape mismatch in {tif_path}: {arr.shape} vs {(B,H,W)}")
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
        d = dates[t]
        for r in range(H):
            for c in range(W):
                x, y = xy(transform, r, c)
                rec = {"date": d, "row": r, "col": c, "x": float(x), "y": float(y)}
                for bi, bname in enumerate(bands):
                    rec[bname] = float(stack[t, bi, r, c])
                rows.append(rec)

    return pd.DataFrame(rows)


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
    *,
    seed_id: str,
    lat: float,
    lon: float,
    grid_px: int,
    res_m: float,
    survey_date: str,
    window_days: int,
    start_date: str,
    end_date: str,
    min_obs: int,
) -> pd.DataFrame:
    """
    Build a single-row DF summarizing the seed by aggregating per-pixel features across space.
    Only pixels with n_obs_baresoil >= min_obs contribute to aggregated statistics.
    """
    out = {
        "seed_id": seed_id,
        "lat": lat,
        "lon": lon,
        "grid_px": grid_px,
        "res_m": res_m,
        "survey_date": survey_date,
        "window_days": window_days,
        "start_date": start_date,
        "end_date": end_date,
        "min_obs_baresoil": min_obs,
        "n_pixels": int(len(feat_df)),
        "n_pixels_with_min_obs": int((feat_df["n_obs_baresoil"] >= min_obs).sum()),
        "mean_n_obs_baresoil": float(feat_df["n_obs_baresoil"].mean()),
        "median_n_obs_baresoil": float(feat_df["n_obs_baresoil"].median()),
    }

    valid = feat_df[feat_df["n_obs_baresoil"] >= min_obs].copy()

    exclude = {"row", "col", "x", "y", "n_obs_baresoil"}
    feature_cols = [
        c for c in valid.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(valid[c])
    ]

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
# Path resolver (NEW)
# ----------------------------
def resolve_timeseries_root(
    *,
    base_dir: Path,
    seed_id: Optional[str],
    lat: float,
    lon: float,
    N: int,
    res_m: float,
    survey_date: Union[str, date],
    window_days: int,
) -> Tuple[Path, Path]:
    """
    Matches extract_one() output structure:
      base/timeseries/<id_part>/grid_<W>x<H>_res<res>m/survey_<date>_pm<window_days>d
    Returns:
      ts_root, per_date_dir
    """
    W, H = grid_for_n_pixels(N)
    survey_str = survey_date.isoformat() if isinstance(survey_date, date) else str(survey_date)[:10]

    id_part = f"seed_{seed_id}" if seed_id else safe_point_name(lat, lon)

    ts_root = (
        base_dir
        / "timeseries"
        / id_part
        / f"grid_{W}x{H}_res{int(res_m)}m"
        / f"survey_{survey_str}_pm{window_days}d"
    )
    return ts_root, ts_root / "per_date"


# ----------------------------
# Main feature computation
# ----------------------------
def compute_baresoil_features_from_ts_root(
    *,
    ts_root: Path,
    seed_id: str,
    lat: float,
    lon: float,
    ndvi_threshold: float = 0.2,
    min_obs: int = 2,
    base_dir: Path = Path("./data"),
    logger: Optional[logging.Logger] = None,
) -> tuple[Path, pd.DataFrame]:
    """
    Reads per_date GeoTIFFs from ts_root/per_date, computes indices, filters bare soil,
    aggregates, and writes outputs under base/features/<same relative structure as ts_root>.
    Returns (feat_root, seed_summary_df).
    """
    logger = logger or logging.getLogger("sentinel_soil_features")

    per_date_dir = ts_root / "per_date"
    if not per_date_dir.exists():
        raise FileNotFoundError(f"per_date dir not found: {per_date_dir}")

    # derive a "relative" subpath under timeseries to mirror under features
    # e.g. data/timeseries/<...>  -> data/features/<...>
    # This assumes ts_root contains ".../timeseries/..."
    parts = ts_root.parts
    if "timeseries" not in parts:
        raise ValueError(f"ts_root does not look like a timeseries path: {ts_root}")
    idx = parts.index("timeseries")
    rel = Path(*parts[idx + 1 :])  # everything after 'timeseries'

    feat_root = base_dir / "features" / rel
    feat_root.mkdir(parents=True, exist_ok=True)

    bands = load_bands(per_date_dir)
    required = {"B02", "B03", "B04", "B08", "B11", "B12"}
    if not required.issubset(set(bands)):
        raise ValueError(f"Need {required} for indices. Found bands: {bands}")

    per_date_tifs = sorted(per_date_dir.glob("*.tif"))
    if not per_date_tifs:
        raise FileNotFoundError(f"No per-date tif files in {per_date_dir}")

    dates = [p.stem for p in per_date_tifs]
    stack, transform, profile0 = read_per_date_stack(per_date_tifs)

    logger.info(f"[{seed_id}] Loaded stack: T={stack.shape[0]} B={stack.shape[1]} H={stack.shape[2]} W={stack.shape[3]}")
    logger.info(f"[{seed_id}] Date span: {dates[0]} → {dates[-1]}")

    ts_df = stack_to_long_dataframe(stack, dates, bands, transform)

    # indices
    nir = ts_df["B08"].astype(float)
    red = ts_df["B04"].astype(float)
    blue = ts_df["B02"].astype(float)

    ts_df["NDVI"] = _safe_ratio(ts_df["B08"] - ts_df["B04"], ts_df["B08"] + ts_df["B04"])
    ts_df["NDMI"] = _safe_ratio(ts_df["B08"] - ts_df["B11"], ts_df["B08"] + ts_df["B11"])
    ts_df["NDWI"] = _safe_ratio(ts_df["B03"] - ts_df["B08"], ts_df["B03"] + ts_df["B08"])
    ts_df["NBR"]  = _safe_ratio(ts_df["B08"] - ts_df["B12"], ts_df["B08"] + ts_df["B12"])

    term = (2 * nir + 1) ** 2 - 8 * (nir - red)
    term = np.where(term < 0, 0, term)
    ts_df["MSAVI"] = (2 * nir + 1 - np.sqrt(term)) / 2

    ts_df["EVI"] = _safe_ratio(2.5 * (nir - red), (nir + 6 * red - 7.5 * blue + 1))

    # save full timeseries parquet
    ts_parquet = feat_root / "pixel_timeseries.parquet"
    ts_df.to_parquet(ts_parquet, index=False)
    logger.info(f"[{seed_id}] ✅ pixel_timeseries.parquet: {ts_parquet}")

    # bare soil filter
    ts_df["is_baresoil"] = ts_df["NDVI"] < ndvi_threshold
    bs = ts_df[ts_df["is_baresoil"]].copy()
    logger.info(f"[{seed_id}] Bare-soil rows: {len(bs)} / {len(ts_df)} (NDVI < {ndvi_threshold})")

    group_cols = ["row", "col", "x", "y"]
    agg_dict = {b: ["median", "mean", "std"] for b in bands}
    for idx_name in ["NDVI", "NDMI", "NDWI", "NBR", "MSAVI", "EVI"]:
        agg_dict[idx_name] = ["median", "mean", "std"]

    if bs.empty:
        feat_df = pd.DataFrame(columns=group_cols + ["n_obs_baresoil"])
    else:
        agg = bs.groupby(group_cols).agg(agg_dict)
        agg.columns = [f"{col}_{stat}_bs" for col, stat in agg.columns]
        agg = agg.reset_index()

        n_obs = bs.groupby(group_cols).size().reset_index(name="n_obs_baresoil")
        feat_df = agg.merge(n_obs, on=group_cols, how="left")

        # enforce min_obs
        feature_cols = [c for c in feat_df.columns if c not in group_cols]
        mask_insufficient = feat_df["n_obs_baresoil"] < min_obs
        feat_df.loc[mask_insufficient, feature_cols] = np.nan

    feat_parquet = feat_root / "bare_soil_pixel_features.parquet"
    feat_df.to_parquet(feat_parquet, index=False)
    logger.info(f"[{seed_id}] ✅ bare_soil_pixel_features.parquet: {feat_parquet}")

    # composite tif (band median)
    T, B, H, W = stack.shape
    composite = np.full((B, H, W), np.nan, dtype=np.float32)
    if not feat_df.empty:
        for _, rec in feat_df.iterrows():
            r = int(rec["row"])
            c = int(rec["col"])
            for bi, bname in enumerate(bands):
                composite[bi, r, c] = rec.get(f"{bname}_median_bs", np.nan)

    out_profile = profile0.copy()
    out_profile.update(count=B, dtype="float32")

    composite_path = feat_root / "bare_soil_median_composite.tif"
    with rasterio.open(composite_path, "w", **out_profile) as dst:
        dst.write(np.nan_to_num(composite, nan=0.0).astype(np.float32))
    logger.info(f"[{seed_id}] ✅ bare_soil_median_composite.tif: {composite_path}")

    # plots
    plots_dir = feat_root / "plots"
    ndvi_grid = np.full((H, W), np.nan, dtype=np.float32)
    nobs_grid = np.full((H, W), 0, dtype=np.int16)

    if not feat_df.empty and "NDVI_median_bs" in feat_df.columns:
        for _, rec in feat_df.iterrows():
            r = int(rec["row"])
            c = int(rec["col"])

            ndvi_grid[r, c] = rec.get("NDVI_median_bs", np.nan)

            val = rec.get("n_obs_baresoil")
            nobs_grid[r, c] = int(val) if pd.notna(val) else 0


    plot_grid(ndvi_grid, plots_dir / "ndvi_median_bs.png", "NDVI median (bare-soil filtered)")
    plot_grid(nobs_grid.astype(np.float32), plots_dir / "n_obs_baresoil.png", "n_obs (bare-soil observations)")
    logger.info(f"[{seed_id}] ✅ plots: {plots_dir}")

    print(feat_df)


    # seed-level summary row
    seed_summary_df = build_seed_point_features(
        feat_df,
        seed_id=seed_id,
        lat=lat,
        lon=lon,
        grid_px=-1,          # optional if you want; can be parsed from ts_root if you want
        res_m=-1.0,
        survey_date="",
        window_days=-1,
        start_date=dates[0],
        end_date=dates[-1],
        min_obs=min_obs,
    )
    # useful extra columns
    seed_summary_df["ndvi_threshold"] = ndvi_threshold
    seed_summary_df["ts_root"] = str(ts_root)
    seed_summary_df["feat_root"] = str(feat_root)

    seed_summary_path = feat_root / "seed_summary.parquet"
    seed_summary_df.to_parquet(seed_summary_path, index=False)
    logger.info(f"[{seed_id}] ✅ seed_summary.parquet: {seed_summary_path}")

    return feat_root, seed_summary_df



if __name__ == "__main__":
    logger = setup_logger(log_dir=Path("logs"))

    # Example: must match the extraction parameters you used
    feat_root = compute_baresoil_features(
        seed_id="1",  # same as SeedRecord.seed_id used in extract_one()
        lat=44.8915307,
        lon=10.01263632,
        N=2,                  # MUST match extraction N
        res_m=10.0,            # MUST match extraction res_m
        survey_date="2015-06-15",
        window_days=30,        # MUST match extraction window_days
        ndvi_threshold=0.2,
        min_obs=2,
        base_dir=Path("./data"),
        logger=logger,
    )

    logger.info(f"Done. Features written under: {feat_root}")
