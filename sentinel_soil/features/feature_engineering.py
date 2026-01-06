from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy


def read_per_date_stack(per_date_tifs: List[Path]) -> tuple[np.ndarray, rasterio.Affine, dict]:
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
            arr = src.read().astype(np.float32)
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
    T, B, H, W = stack.shape
    if len(dates) != T:
        raise ValueError("dates length must match stack T")

    rows = []
    for t, date in enumerate(dates):
        for r in range(H):
            for c in range(W):
                x, y = xy(transform, r, c)
                rec = {"date": date, "row": r, "col": c, "x": float(x), "y": float(y)}
                for bi, bname in enumerate(bands):
                    rec[bname] = float(stack[t, bi, r, c])
                rows.append(rec)

    return pd.DataFrame(rows)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.where(den != 0, num / den, np.nan)


def add_indices(ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds NDVI, NDMI, NDWI, NBR, MSAVI, EVI.
    Requires B02,B03,B04,B08,B11,B12.
    """
    nir = ts_df["B08"].to_numpy(dtype=float)
    red = ts_df["B04"].to_numpy(dtype=float)
    green = ts_df["B03"].to_numpy(dtype=float)
    blue = ts_df["B02"].to_numpy(dtype=float)
    swir1 = ts_df["B11"].to_numpy(dtype=float)
    swir2 = ts_df["B12"].to_numpy(dtype=float)

    ts_df["NDVI"] = _safe_ratio(nir - red, nir + red)
    ts_df["NDMI"] = _safe_ratio(nir - swir1, nir + swir1)
    ts_df["NDWI"] = _safe_ratio(green - nir, green + nir)
    ts_df["NBR"] = _safe_ratio(nir - swir2, nir + swir2)

    term = (2 * nir + 1) ** 2 - 8 * (nir - red)
    term = np.where(term < 0, 0, term)
    ts_df["MSAVI"] = (2 * nir + 1 - np.sqrt(term)) / 2

    ts_df["EVI"] = _safe_ratio(2.5 * (nir - red), (nir + 6 * red - 7.5 * blue + 1))
    return ts_df
