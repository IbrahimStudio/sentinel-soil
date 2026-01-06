from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

import matplotlib.pyplot as plt


def build_seed_point_features(
    feat_df: pd.DataFrame,
    lat: float,
    lon: float,
    window_m: float,
    start_date: str,
    end_date: str,
    min_obs: int,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame summarizing the window (seed point) by aggregating
    per-pixel features across space.
    """

    out = {
        "lat": lat,
        "lon": lon,
        "window_m": window_m,
        "start_date": start_date,
        "end_date": end_date,
        "min_obs_baresoil": min_obs,
        "n_pixels": int(len(feat_df)),
        "n_pixels_with_min_obs": int((feat_df["n_obs_baresoil"] >= min_obs).sum()),
        "mean_n_obs_baresoil": float(feat_df["n_obs_baresoil"].mean()),
        "median_n_obs_baresoil": float(feat_df["n_obs_baresoil"].median()),
    }

    # Only use pixels that have enough observations
    valid = feat_df[feat_df["n_obs_baresoil"] >= min_obs].copy()

    # Aggregate all numeric feature columns across pixels
    # Exclude identifiers
    exclude = {"row", "col", "x", "y", "n_obs_baresoil"}
    feature_cols = [c for c in valid.columns if c not in exclude and pd.api.types.is_numeric_dtype(valid[c])]

    for col in feature_cols:
        s = valid[col].dropna()
        if s.empty:
            out[f"{col}_space_median"] = np.nan
            out[f"{col}_space_mean"] = np.nan
            out[f"{col}_space_std"] = np.nan
            continue

        out[f"{col}_space_median"] = float(s.median())
        out[f"{col}_space_mean"] = float(s.mean())
        out[f"{col}_space_std"] = float(s.std(ddof=0))

    return pd.DataFrame([out])



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
        date = dates[t]
        for r in range(H):
            for c in range(W):
                x, y = xy(transform, r, c)
                rec = {"date": date, "row": r, "col": c, "x": float(x), "y": float(y)}
                for bi, bname in enumerate(bands):
                    rec[bname] = float(stack[t, bi, r, c])
                rows.append(rec)

    df = pd.DataFrame(rows)
    return df


def plot_grid(
    grid: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """
    Saves a quick sanity check plot. Uses matplotlib defaults (no custom styling).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(grid)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main(lat, lon , window_m, start_date, end_date, ndvi_threshold: float = 0.2, min_obs: int = 2, config_path: str = "configs/dev.yaml"):
    # ---- params (must match extraction) ----
    # lat = 44.8915307
    # lon = 10.01263632
    # window_m = 100.0
    # start_date = "2024-06-01"
    # end_date = "2024-06-30"

    base = Path("./data")
    ts_root = (
        base
        / "timeseries"
        / safe_point_name(lat, lon)
        / f"window_{int(window_m)}m"
        / f"{start_date}_{end_date}"
    )
    per_date_dir = ts_root / "per_date"

    bands = load_bands(per_date_dir)
    required = {"B04", "B08"}
    if not required.issubset(set(bands)):
        raise ValueError(f"Need {required} to compute NDVI. Found bands: {bands}")

    per_date_tifs = sorted(per_date_dir.glob("*.tif"))
    if not per_date_tifs:
        raise FileNotFoundError(f"No per-date tif files in {per_date_dir}")

    # Dates inferred from filenames (YYYY-MM-DD.tif)
    dates = [p.stem for p in per_date_tifs]

    stack, transform, profile0 = read_per_date_stack(per_date_tifs)

    # ---- per-pixel time series (long DF) ----
    ts_df = stack_to_long_dataframe(stack, dates, bands, transform)

    # Compute NDVI per row (per date, per pixel)
    # --- Indices (per date, per pixel) ---
    # NDVI
    den = ts_df["B08"] + ts_df["B04"]
    ts_df["NDVI"] = _safe_ratio(ts_df["B08"] - ts_df["B04"], den)

    # NDMI
    den = ts_df["B08"] + ts_df["B11"]
    ts_df["NDMI"] = _safe_ratio(ts_df["B08"] - ts_df["B11"], den)

    # NDWI (McFeeters)
    den = ts_df["B03"] + ts_df["B08"]
    ts_df["NDWI"] = _safe_ratio(ts_df["B03"] - ts_df["B08"], den)

    # NBR
    den = ts_df["B08"] + ts_df["B12"]
    ts_df["NBR"] = _safe_ratio(ts_df["B08"] - ts_df["B12"], den)

    # MSAVI
    # (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - RED))) / 2
    nir = ts_df["B08"].astype(float)
    red = ts_df["B04"].astype(float)
    term = (2 * nir + 1) ** 2 - 8 * (nir - red)
    # numerical safety: clamp negatives to 0
    term = np.where(term < 0, 0, term)
    ts_df["MSAVI"] = (2 * nir + 1 - np.sqrt(term)) / 2

    # EVI
    # 2.5*(NIR-RED)/(NIR + 6*RED - 7.5*BLUE + 1)
    blue = ts_df["B02"].astype(float)
    den = nir + 6 * red - 7.5 * blue + 1
    ts_df["EVI"] = _safe_ratio(2.5 * (nir - red), den)


    # Save time series parquet
    feat_root = (
        base
        / "features"
        / safe_point_name(lat, lon)
        / f"window_{int(window_m)}m"
        / f"{start_date}_{end_date}"
    )
    feat_root.mkdir(parents=True, exist_ok=True)

    ts_parquet = feat_root / "pixel_timeseries.parquet"
    ts_df.to_parquet(ts_parquet, index=False)
    print(f"✅ Saved per-pixel time series parquet: {ts_parquet}")

    # ---- bare-soil filtering (per row) ----
    ts_df["is_baresoil"] = ts_df["NDVI"] < ndvi_threshold

    # ---- aggregate per pixel (x,y,row,col) across time, using only bare-soil rows ----
    bs = ts_df[ts_df["is_baresoil"]].copy()

    # number of observations per pixel surviving filter
    group_cols = ["row", "col", "x", "y"]
    agg_dict = {b: ["median", "mean", "std"] for b in bands}

    index_cols = ["NDVI", "NDMI", "NDWI", "NBR", "MSAVI", "EVI"]
    for idx in index_cols:
        agg_dict[idx] = ["median", "mean", "std"]


    agg = bs.groupby(group_cols).agg(agg_dict)

    # Flatten column names
    agg.columns = [f"{col}_{stat}_bs" for col, stat in agg.columns]
    agg = agg.reset_index()

    # Add n_obs_baresoil
    n_obs = bs.groupby(group_cols).size().reset_index(name="n_obs_baresoil")
    feat_df = agg.merge(n_obs, on=group_cols, how="left")

    # Enforce min_obs: set feature columns to NaN if insufficient observations
    feature_cols = [c for c in feat_df.columns if c not in group_cols]
    mask_insufficient = feat_df["n_obs_baresoil"] < min_obs
    feat_df.loc[mask_insufficient, feature_cols] = np.nan

    # Save aggregated features parquet
    feat_parquet = feat_root / "bare_soil_pixel_features.parquet"
    feat_df.to_parquet(feat_parquet, index=False)
    print(f"✅ Saved bare-soil per-pixel features parquet: {feat_parquet}")

    # ---- also write a bare-soil median composite GeoTIFF (bands only) ----
    # Build (B,H,W) from feat_df for band_median_bs columns
    # Determine grid size from stack
    T, B, H, W = stack.shape

    # Create empty composite
    composite = np.zeros((B, H, W), dtype=np.float32)
    # Fill with NaN first to spot missing
    composite[:] = np.nan

    # Map features back to grid
    for _, rec in feat_df.iterrows():
        if pd.isna(rec["row"]) or pd.isna(rec["col"]):
            continue
        r = int(rec["row"])
        c = int(rec["col"])
        for bi, bname in enumerate(bands):
            key = f"{bname}_median_bs"
            composite[bi, r, c] = rec.get(key, np.nan)

    out_profile = profile0.copy()
    out_profile.update(count=B, dtype="float32")

    composite_path = feat_root / "bare_soil_median_composite.tif"
    with rasterio.open(composite_path, "w", **out_profile) as dst:
        dst.write(np.nan_to_num(composite, nan=0.0).astype(np.float32))
    print(f"✅ Saved bare-soil median composite GeoTIFF: {composite_path}")

    # ---- quick sanity maps ----
    # NDVI median grid
    ndvi_key = "NDVI_median_bs"
    # This exists because we aggregated NDVI as well
    ndvi_grid = np.full((H, W), np.nan, dtype=np.float32)
    nobs_grid = np.full((H, W), 0, dtype=np.int16)

    for _, rec in feat_df.iterrows():
        if rec.get("n_obs_baresoil") == np.float64('nan'):
            continue
        r = int(rec["row"])
        c = int(rec["col"])
        ndvi_grid[r, c] = rec.get(ndvi_key, np.nan)
        try:
            nobs_grid[r, c] = int(rec.get("n_obs_baresoil", 0) or 0)
        except:
            print(rec.get("n_obs_baresoil", 0))

    plots_dir = feat_root / "plots"
    plot_grid(ndvi_grid, plots_dir / "ndvi_median_bs.png", "NDVI median (bare-soil filtered)")
    plot_grid(nobs_grid.astype(np.float32), plots_dir / "n_obs_baresoil.png", "n_obs (bare-soil observations)")

    print(f"✅ Saved sanity plots in: {plots_dir}")


if __name__ == "__main__":
    lat = 44.8915307
    lon = 10.01263632
    window_m = 100.0
    start_date = "2024-06-01"
    end_date = "2024-06-30"
    main(lat, lon , window_m, start_date, end_date)
