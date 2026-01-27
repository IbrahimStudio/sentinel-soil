# sentinel_soil/domain/bare_soil_features.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy


# -----------------------------
# Minimal helpers (pure, local)
# -----------------------------
def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float32)
    m = den != 0
    out[m] = num[m] / den[m]
    return out


def _load_bands(per_date_dir: Path) -> List[str]:
    p = per_date_dir / "bands.json"
    if not p.exists():
        raise FileNotFoundError(f"bands.json not found in {per_date_dir}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    bands = obj.get("bands")
    if not bands or not isinstance(bands, list):
        raise ValueError(f"Invalid bands.json format: {p}")
    return bands


def _read_per_date_stack(per_date_tifs: List[Path]) -> Tuple[np.ndarray, rasterio.Affine, dict]:
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


def _stack_to_long_dataframe(
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
    if len(bands) != B:
        raise ValueError("bands length must match stack B")

    rows: list[dict] = []
    for t in range(T):
        d = dates[t]
        for r in range(H):
            for c in range(W):
                x_, y_ = xy(transform, r, c)
                rec = {"date": d, "row": r, "col": c, "x": float(x_), "y": float(y_)}
                for bi, bname in enumerate(bands):
                    rec[bname] = float(stack[t, bi, r, c])
                rows.append(rec)

    return pd.DataFrame(rows)


def _build_seed_summary(
    feat_df: pd.DataFrame,
    *,
    seed_id: str,
    lat: float,
    lon: float,
    min_obs: int,
) -> pd.DataFrame:
    """
    Minimal single-row summary:
      - counts of pixels/valid pixels
      - aggregated feature stats across space (median/mean/std) over pixels with min_obs
    """
    out: Dict[str, object] = {
        "seed_id": seed_id,
        "lat": float(lat),
        "lon": float(lon),
        "min_obs_baresoil": int(min_obs),
        "n_pixels": int(len(feat_df)),
        "n_pixels_with_min_obs": int((feat_df["n_obs_baresoil"] >= min_obs).sum()) if "n_obs_baresoil" in feat_df else 0,
    }

    if feat_df.empty or "n_obs_baresoil" not in feat_df.columns:
        return pd.DataFrame([out])

    valid = feat_df[feat_df["n_obs_baresoil"] >= min_obs].copy()

    exclude = {"row", "col", "x", "y", "n_obs_baresoil"}
    numeric_cols = [
        c for c in valid.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(valid[c])
    ]

    for col in numeric_cols:
        s = valid[col].dropna()
        out[f"{col}_space_median"] = float(s.median()) if not s.empty else np.nan
        out[f"{col}_space_mean"] = float(s.mean()) if not s.empty else np.nan
        out[f"{col}_space_std"] = float(s.std(ddof=0)) if not s.empty else np.nan

    return pd.DataFrame([out])


# -----------------------------
# Public API: job-oriented features
# -----------------------------
def compute_baresoil_features_from_ts_root(
    *,
    ts_root: Path,
    seed_id: str,
    lat: float,
    lon: float,
    ndvi_threshold: float = 0.2,
    min_obs: int = 2,
    feat_out_root: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Path, pd.DataFrame]:
    """
    Minimal bare-soil feature extraction for a single job.

    Inputs:
      ts_root/per_date/*.tif
      ts_root/per_date/bands.json

    Writes under:
      <feat_out_root>/<seed_id>/   (caller can make this job_id-based if preferred)
        pixel_timeseries.parquet
        bare_soil_pixel_features.parquet
        seed_summary.parquet

    Returns:
      (feat_root, seed_summary_df)
    """
    logger = logger or logging.getLogger(__name__)

    per_date_dir = Path(ts_root) / "per_date"
    if not per_date_dir.exists():
        raise FileNotFoundError(f"per_date dir not found: {per_date_dir}")

    feat_root = Path(feat_out_root) / str(seed_id)
    feat_root.mkdir(parents=True, exist_ok=True)

    bands = _load_bands(per_date_dir)
    required = {"B02", "B03", "B04", "B08", "B11", "B12"}
    if not required.issubset(set(bands)):
        raise ValueError(f"Need {required} for indices. Found bands: {bands}")

    per_date_tifs = sorted(per_date_dir.glob("*.tif"))
    if not per_date_tifs:
        raise FileNotFoundError(f"No per-date tif files in {per_date_dir}")

    dates = [p.stem for p in per_date_tifs]
    stack, transform, _profile0 = _read_per_date_stack(per_date_tifs)

    logger.info(f"[{seed_id}] Loaded stack: T={stack.shape[0]} B={stack.shape[1]} H={stack.shape[2]} W={stack.shape[3]}")
    logger.info(f"[{seed_id}] Date span: {dates[0]} → {dates[-1]}")

    # long format table
    ts_df = _stack_to_long_dataframe(stack, dates, bands, transform)

    # If evalscript already masked invalid acquisitions, they come as NaNs.
    # Drop them before computing indices.
    base_bands = ["B02", "B03", "B04", "B08"]
    ts_df = ts_df.dropna(subset=base_bands, how="any").copy()

    # indices (vectorized over dataframe columns)
    nir = ts_df["B08"].astype(np.float32).to_numpy()
    red = ts_df["B04"].astype(np.float32).to_numpy()
    blue = ts_df["B02"].astype(np.float32).to_numpy()
    green = ts_df["B03"].astype(np.float32).to_numpy()
    swir1 = ts_df["B11"].astype(np.float32).to_numpy()
    swir2 = ts_df["B12"].astype(np.float32).to_numpy()

    ts_df["NDVI"] = _safe_ratio(nir - red, nir + red)
    ts_df["NDMI"] = _safe_ratio(nir - swir1, nir + swir1)
    ts_df["NDWI"] = _safe_ratio(green - nir, green + nir)
    ts_df["NBR"] = _safe_ratio(nir - swir2, nir + swir2)

    term = (2 * nir + 1) ** 2 - 8 * (nir - red)
    term = np.where(term < 0, 0, term)
    ts_df["MSAVI"] = (2 * nir + 1 - np.sqrt(term)) / 2

    ts_df["EVI"] = _safe_ratio(2.5 * (nir - red), (nir + 6 * red - 7.5 * blue + 1))

    # store full pixel-time series (useful for debugging + later features)
    ts_parquet = feat_root / "pixel_timeseries.parquet"
    ts_df.to_parquet(ts_parquet, index=False)
    logger.info(f"[{seed_id}] Wrote: {ts_parquet}")

    # bare soil filter
    if not (-1.0 <= float(ndvi_threshold) <= 1.0):
        raise ValueError(f"ndvi_threshold must be in [-1,1]. Got {ndvi_threshold}")

    ts_df["is_baresoil"] = ts_df["NDVI"] < float(ndvi_threshold)
    bs = ts_df[ts_df["is_baresoil"]].copy()
    logger.info(f"[{seed_id}] Bare-soil rows: {len(bs)} / {len(ts_df)} (NDVI < {ndvi_threshold})")

    group_cols = ["row", "col", "x", "y"]

    # aggregate per pixel over time (only bare-soil observations)
    agg_dict: Dict[str, List[str]] = {b: ["median", "mean", "std"] for b in bands}
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

        # enforce min_obs: if insufficient observations, null-out numeric features
        if min_obs > 1:
            feature_cols = [c for c in feat_df.columns if c not in group_cols]
            mask_insufficient = feat_df["n_obs_baresoil"] < int(min_obs)
            feat_df.loc[mask_insufficient, feature_cols] = np.nan

    feat_parquet = feat_root / "bare_soil_pixel_features.parquet"
    feat_df.to_parquet(feat_parquet, index=False)
    logger.info(f"[{seed_id}] Wrote: {feat_parquet}")

    # seed-level summary
    seed_summary_df = _build_seed_summary(
        feat_df,
        seed_id=seed_id,
        lat=lat,
        lon=lon,
        min_obs=int(min_obs),
    )
    seed_summary_df["ndvi_threshold"] = float(ndvi_threshold)
    seed_summary_df["ts_root"] = str(ts_root)
    seed_summary_df["feat_root"] = str(feat_root)
    seed_summary_df["start_date"] = dates[0]
    seed_summary_df["end_date"] = dates[-1]

    seed_summary_path = feat_root / "seed_summary.parquet"
    seed_summary_df.to_parquet(seed_summary_path, index=False)
    logger.info(f"[{seed_id}] Wrote: {seed_summary_path}")

    return feat_root, seed_summary_df
