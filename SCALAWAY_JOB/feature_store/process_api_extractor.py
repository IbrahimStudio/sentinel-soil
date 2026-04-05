"""
process_api_extractor.py — Build feature vectors from raw rasters stored on S3.

Reads .npz files written by ingestion/process_api, applies filter_config.json,
extracts the center pixel, computes temporal median + spectral indices, and
joins with ground-truth labels from the LUCAS DataFrame.

Output DataFrame columns:
  POINT_ID, TH_LAT, TH_LONG,
  B02_median … B12_median (10 bands), NDVI, NDMI,   ← feature contract
  n_valid_dates,
  Clay, Silt, Sand, Coarse  (when present in lucas_df)

Band indices in raw raster — must match evalscript_process_api.js exactly:
  0=B02, 1=B03, 2=B04, 3=B05, 4=B06, 5=B07, 6=B08, 7=B8A, 8=B11, 9=B12, 10=SCL

Index formulas (copy exactly — do not rephrase):
  NDVI = (B08 - B04) / (B08 + B04)   →  bands[6], bands[2]
  NDMI = (B08 - B11) / (B08 + B11)   →  bands[6], bands[8]
  NBR2 = (B11 - B12) / (B11 + B12)   →  bands[8], bands[9]  (filter only, not a feature)
"""

from __future__ import annotations

import io
import json
import logging
from typing import Optional

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config as BotoConfig

log = logging.getLogger(__name__)

# Band index constants — must match evalscript_process_api.js band order.
_B04 = 2
_B08 = 6
_B11 = 8
_B12 = 9
_SCL = 10

FEATURE_NAMES = [
    "B02_median", "B03_median", "B04_median", "B05_median",
    "B06_median", "B07_median", "B08_median", "B8A_median",
    "B11_median", "B12_median",
    "NDVI", "NDMI",
]

_TARGET_COLS = ["Clay", "Silt", "Sand", "Coarse"]


def _safe_nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    return np.where(np.abs(denom) < 1e-9, 0.0, (a - b) / denom)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _make_s3(endpoint: str, bucket: str, access_key: str, secret_key: str):
    cfg = BotoConfig(retries={"max_attempts": 6, "mode": "standard"}, connect_timeout=15, read_timeout=120)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="fr-par",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=cfg,
    )
    return client, bucket


def _load_npz(s3, bucket: str, key: str) -> Optional[dict]:
    try:
        data = np.load(io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read()), allow_pickle=False)
        return dict(data)
    except Exception as exc:
        log.warning("Could not load %s: %s", key, exc)
        return None


def _load_meta(s3, bucket: str, key: str) -> Optional[dict]:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode())
    except Exception as exc:
        log.warning("Could not load meta %s: %s", key, exc)
        return None


# ---------------------------------------------------------------------------
# Per-pixel filter
# ---------------------------------------------------------------------------

def _apply_pixel_filter(
    rasters: np.ndarray,   # (N_dates, H, W, 11)
    scl_keep: list[int],
    ndvi_min: float,
    ndvi_max: float,
    nbr2_max: float,
) -> np.ndarray:
    """Returns bool mask (N_dates, H, W): True = pixel passes all filters.

    Filter chain (applied in order):
      1. SCL allowlist  — reject cloud, shadow, water, snow, unclassified
      2. NDVI > ndvi_min — reject very dark pixels (water/shadow missed by SCL);
                           _safe_nd returns 0.0 for near-zero denominators so a
                           lower bound is required to catch those edge cases
      3. NDVI < ndvi_max — reject active green vegetation
      4. NBR2 < nbr2_max — reject dry crop residues and wet soils
    """
    scl = rasters[..., _SCL].astype(np.int16)

    scl_mask = np.zeros_like(scl, dtype=bool)
    for cls in scl_keep:
        scl_mask |= (scl == cls)

    ndvi = _safe_nd(rasters[..., _B08], rasters[..., _B04])
    nbr2 = _safe_nd(rasters[..., _B11], rasters[..., _B12])

    return scl_mask & (ndvi > ndvi_min) & (ndvi < ndvi_max) & (nbr2 < nbr2_max)


# ---------------------------------------------------------------------------
# Feature extraction for one point
# ---------------------------------------------------------------------------

def _extract_one(
    point_id: str,
    rasters: np.ndarray,    # (N_dates, 9, 9, 11)
    center_pixel: list[int],
    filter_config: dict,
) -> Optional[dict]:
    pf = filter_config["pixel_filter"]
    min_obs = filter_config["temporal_aggregation"].get("min_valid_observations_per_pixel", 3)

    mask = _apply_pixel_filter(
        rasters,
        pf["scl_keep_classes"],
        pf.get("ndvi_min", -0.1),
        pf["ndvi_max"],
        pf["nbr2_max"],
    )

    row, col = center_pixel
    center_valid = mask[:, row, col]
    n_valid = int(center_valid.sum())

    if n_valid < min_obs:
        log.debug("%s: %d valid dates at center pixel (need %d) — skipped.", point_id, n_valid, min_obs)
        return None

    spectral = rasters[center_valid, row, col, :10].astype(np.float64)  # (n_valid, 10)
    medians  = np.median(spectral, axis=0)                               # (10,)

    b08_med = medians[_B08]
    b04_med = medians[_B04]
    b11_med = medians[_B11]

    ndvi = float(_safe_nd(np.array([b08_med]), np.array([b04_med]))[0])
    ndmi = float(_safe_nd(np.array([b08_med]), np.array([b11_med]))[0])

    band_names = ["B02_median", "B03_median", "B04_median", "B05_median",
                  "B06_median", "B07_median", "B08_median", "B8A_median",
                  "B11_median", "B12_median"]
    row_dict = {name: float(medians[i]) for i, name in enumerate(band_names)}
    row_dict["NDVI"] = ndvi
    row_dict["NDMI"] = ndmi
    row_dict["n_valid_dates"] = n_valid
    return row_dict


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_features(
    lucas_df: pd.DataFrame,
    filter_config: dict,
    *,
    s3_endpoint: str,
    s3_bucket: str,
    s3_access_key: str,
    s3_secret_key: str,
    raster_prefix: str = "raw_rasters/",
) -> pd.DataFrame:
    """
    Load raw rasters from S3 for each LUCAS point, apply filter_config, return
    a DataFrame with feature columns, POINT_ID, coordinates, and target labels.

    Points with insufficient valid observations are dropped with a warning.
    """
    s3, bucket = _make_s3(s3_endpoint, s3_bucket, s3_access_key, s3_secret_key)

    if not raster_prefix.endswith("/"):
        raster_prefix += "/"

    rows: list[dict] = []
    n_skipped = 0

    for _, src_row in lucas_df.iterrows():
        pid = str(src_row["POINT_ID"])

        npz  = _load_npz( s3, bucket, f"{raster_prefix}{pid}.npz")
        meta = _load_meta(s3, bucket, f"{raster_prefix}{pid}_meta.json")

        if npz is None or meta is None:
            log.warning("%s: missing raster or meta on S3 — run ingestion first.", pid)
            n_skipped += 1
            continue

        rasters = npz["rasters"]
        if rasters.ndim != 4 or rasters.shape[1:] != (9, 9, 11):
            log.warning("%s: unexpected raster shape %s — skipped.", pid, rasters.shape)
            n_skipped += 1
            continue

        feat = _extract_one(pid, rasters, meta.get("center_pixel", [4, 4]), filter_config)
        if feat is None:
            n_skipped += 1
            continue

        feat["POINT_ID"] = pid
        feat["TH_LAT"]   = float(src_row.get("TH_LAT",  meta.get("lucas_lat",  float("nan"))))
        feat["TH_LONG"]  = float(src_row.get("TH_LONG", meta.get("lucas_lon",  float("nan"))))

        for col in _TARGET_COLS:
            if col in src_row.index:
                feat[col] = src_row[col]

        rows.append(feat)

    log.info("Feature extraction: %d extracted, %d skipped.", len(rows), n_skipped)

    if not rows:
        raise RuntimeError("No features extracted — check S3 rasters and filter thresholds.")

    df = pd.DataFrame(rows)
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing}")

    _sanity_check(df)
    return df


def _sanity_check(df: pd.DataFrame) -> None:
    """
    Validate the extracted feature DataFrame before it reaches training.
    Logs warnings for soft issues, raises RuntimeError for hard failures.
    """
    issues: list[str] = []
    hard:   list[str] = []

    # --- Row count ---
    if len(df) < 10:
        hard.append(f"Only {len(df)} rows — too few to train any model (need >= 10).")
    elif len(df) < 50:
        issues.append(f"Only {len(df)} rows — CV scores will be unreliable (recommend >= 50).")

    # --- Reflectance range ---
    band_cols = [c for c in FEATURE_NAMES if c.endswith("_median")]
    if band_cols:
        band_vals = df[band_cols].values
        band_max  = float(np.nanmax(band_vals))
        band_min  = float(np.nanmin(band_vals))
        if band_max > 10.0:
            hard.append(
                f"Band reflectance max = {band_max:.1f} — values are not in [0,1]. "
                "Check evalscript: divide by 10000 if returning DN instead of BOA reflectance."
            )
        elif band_max > 1.5:
            issues.append(
                f"Band reflectance P99 > 1.0 (max={band_max:.2f}) — possible scaling issue."
            )
        if band_min < -0.5:
            issues.append(f"Band reflectance min = {band_min:.3f} — unexpected negative values.")

    # --- Constant / all-NaN columns ---
    for col in FEATURE_NAMES:
        if col not in df.columns:
            continue
        if df[col].isna().all():
            hard.append(f"Feature column '{col}' is all NaN.")
        elif df[col].nunique() <= 1:
            issues.append(f"Feature column '{col}' is constant (value={df[col].iloc[0]:.4f}) — will contribute nothing to the model.")

    # --- Target sum check (Clay + Silt + Sand + Coarse ≈ 100) ---
    target_cols = [c for c in ["Clay", "Silt", "Sand", "Coarse"] if c in df.columns]
    if len(target_cols) == 4:
        total = df[target_cols].sum(axis=1)
        bad = (total < 85) | (total > 115)
        n_bad = int(bad.sum())
        if n_bad > 0:
            issues.append(
                f"{n_bad} rows have Clay+Silt+Sand+Coarse outside [85,115] "
                f"(min={total.min():.1f}, max={total.max():.1f}) — check label quality."
            )

    # --- NaN fraction ---
    for col in FEATURE_NAMES:
        if col not in df.columns:
            continue
        nan_pct = df[col].isna().mean()
        if nan_pct > 0.05:
            issues.append(f"Feature '{col}' has {nan_pct:.1%} NaN values.")

    for msg in issues:
        log.warning("Sanity check WARNING: %s", msg)
    for msg in hard:
        log.error("Sanity check ERROR: %s", msg)

    if hard:
        raise RuntimeError(f"Feature sanity check failed ({len(hard)} error(s)) — see logs above.")
