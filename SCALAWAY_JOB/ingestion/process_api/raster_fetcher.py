"""
raster_fetcher.py — Fetch raw 9×9 rasters for each LUCAS point and store on S3.

Storage layout:
  s3://{BUCKET}/{prefix}{point_id}.npz
  s3://{BUCKET}/{prefix}{point_id}_meta.json

The .npz contains:
  rasters   float32  (N_dates, 9, 9, 11)  — band order per BAND_NAMES
  dates     str      (N_dates,)            — "YYYY-MM-DD"

One multi-temporal Process API request per point (ORBIT mosaicking) replaces
the previous per-date loop, reducing API calls by ~300×.

Re-running is safe: existing .npz files are skipped (S3 cache check).
"""

from __future__ import annotations

import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import re

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from pyproj import Transformer
from tqdm import tqdm

from sh_clients import ProcessClient, BAND_NAMES

log = logging.getLogger(__name__)


def _normalise_endpoint(endpoint: str) -> str:
    """Strip bucket prefix from Scaleway virtual-hosted endpoint URLs.
    https://<bucket>.s3.<region>.scw.cloud → https://s3.<region>.scw.cloud
    Needed for path-style boto3 access outside Scaleway's own infrastructure."""
    m = re.match(r"(https://)[^.]+\.(s3\.[^.]+\.scw\.cloud)", endpoint)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return endpoint

RASTER_SHAPE = (9, 9, 11)   # H × W × bands
RESOLUTION_M = 10
HALF_EXTENT_M = (RASTER_SHAPE[0] * RESOLUTION_M) / 2.0   # 45 m


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _make_s3(endpoint: str, bucket: str, access_key: str, secret_key: str):
    cfg = BotoConfig(
        retries={"max_attempts": 6, "mode": "standard"},
        connect_timeout=15,
        read_timeout=120,
        s3={"addressing_style": "path"},
    )
    client = boto3.client(
        "s3",
        endpoint_url=_normalise_endpoint(endpoint),
        region_name=os.environ.get("SCALEWAY_S3_REGION", "fr-par"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=cfg,
    )
    return client, bucket


def _s3_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def _s3_put_bytes(s3, bucket: str, key: str, data: bytes, content_type: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def _bbox_3857(lat: float, lon: float) -> tuple[float, float, float, float]:
    """90×90 m bbox centered on (lat, lon), returned in EPSG:3857."""
    x, y = _TO_3857.transform(lon, lat)
    return (x - HALF_EXTENT_M, y - HALF_EXTENT_M, x + HALF_EXTENT_M, y + HALF_EXTENT_M)


def _center_pixel(lat: float, lon: float, bbox_3857: tuple) -> list[int]:
    """Which pixel in the 9×9 grid is closest to (lat, lon). Returns [row, col]."""
    x_pt, y_pt = _TO_3857.transform(lon, lat)
    xmin, ymin, xmax, ymax = bbox_3857
    col = max(0, min(RASTER_SHAPE[1] - 1, int((x_pt - xmin) / RESOLUTION_M)))
    row = max(0, min(RASTER_SHAPE[0] - 1, int((ymax - y_pt) / RESOLUTION_M)))
    return [row, col]


# ---------------------------------------------------------------------------
# Per-point fetch
# ---------------------------------------------------------------------------

@dataclass
class FetchSpec:
    point_id: str
    lat: float
    lon: float
    start_date: str   # "YYYY-MM-DD"
    end_date: str
    season_months: frozenset[int] = field(default_factory=frozenset)


def _fetch_one_point(
    spec: FetchSpec,
    s3,
    bucket: str,
    raster_prefix: str,
    process: ProcessClient,
) -> str:
    """
    Fetch all orbit passes for one LUCAS point in a single multi-temporal
    Process API request and upload the resulting .npz to S3.

    Returns point_id on success. Raises on failure.
    """
    npz_key  = f"{raster_prefix}{spec.point_id}.npz"
    meta_key = f"{raster_prefix}{spec.point_id}_meta.json"

    if _s3_exists(s3, bucket, npz_key):
        log.debug("%s: already exists on S3, skipping.", spec.point_id)
        return spec.point_id

    b3857     = _bbox_3857(spec.lat, spec.lon)
    center_px = _center_pixel(spec.lat, spec.lon, b3857)

    # One multi-temporal call — all orbit passes in the window
    rasters_arr, dates_list = process.fetch_all_dates(b3857, spec.start_date, spec.end_date)

    if rasters_arr.shape[0] == 0:
        log.warning("%s: no acquisitions found (%s – %s).", spec.point_id, spec.start_date, spec.end_date)
        return spec.point_id

    # Seasonal filter applied client-side after the fetch
    if spec.season_months:
        keep = [i for i, d in enumerate(dates_list) if int(d[5:7]) in spec.season_months]
        if not keep:
            log.warning("%s: no dates remain after seasonal filter (months %s).",
                        spec.point_id, sorted(spec.season_months))
            return spec.point_id
        rasters_arr = rasters_arr[keep]
        dates_list  = [dates_list[i] for i in keep]
        log.debug("%s: %d dates after seasonal filter.", spec.point_id, len(dates_list))

    # Store .npz
    buf = io.BytesIO()
    np.savez(buf,
             rasters=rasters_arr,
             dates=np.array(dates_list, dtype="U10"))
    _s3_put_bytes(s3, bucket, npz_key, buf.getvalue(), "application/octet-stream")

    # Store _meta.json
    meta = {
        "lucas_point_id":  spec.point_id,
        "lucas_lat":       spec.lat,
        "lucas_lon":       spec.lon,
        "bbox_epsg3857":   list(b3857),
        "resolution_m":    RESOLUTION_M,
        "raster_shape":    list(RASTER_SHAPE),
        "center_pixel":    center_px,
        "bands":           BAND_NAMES,
        "time_range":      [spec.start_date, spec.end_date],
        "n_dates_fetched": len(dates_list),
    }
    _s3_put_bytes(s3, bucket, meta_key, json.dumps(meta, indent=2).encode(), "application/json")

    log.info("%s: stored %d dates → s3://%s/%s", spec.point_id, len(dates_list), bucket, npz_key)
    return spec.point_id


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_all_lucas_rasters(
    lucas_df: pd.DataFrame,
    process: ProcessClient,
    *,
    s3_endpoint: str,
    s3_bucket: str,
    s3_access_key: str,
    s3_secret_key: str,
    raster_prefix: str = "raw_rasters/",
    time_window_days: int = 365,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    workers: int = 4,
    season_months: Optional[list[int]] = None,
) -> None:
    """
    Fetch raw rasters for all LUCAS points and store on S3.

    Expected columns in lucas_df: POINT_ID, TH_LAT, TH_LONG, SURVEY_DATE.

    One multi-temporal Process API request is made per point (vs one per date
    previously), reducing API calls by ~300× and PU cost by the same factor.

    start_date / end_date: fixed window for all points.  When both are absent,
    each point uses ±time_window_days around its SURVEY_DATE.
    """
    s3, bucket = _make_s3(s3_endpoint, s3_bucket, s3_access_key, s3_secret_key)

    if not raster_prefix.endswith("/"):
        raster_prefix += "/"

    specs: list[FetchSpec] = []
    for _, row in lucas_df.iterrows():
        if start_date and end_date:
            pt_start, pt_end = start_date, end_date
        else:
            survey   = pd.to_datetime(row["SURVEY_DATE"])
            pt_start = (survey - timedelta(days=time_window_days)).strftime("%Y-%m-%d")
            pt_end   = (survey + timedelta(days=time_window_days)).strftime("%Y-%m-%d")
        specs.append(FetchSpec(
            point_id=str(row["POINT_ID"]),
            lat=float(row["TH_LAT"]),
            lon=float(row["TH_LONG"]),
            start_date=pt_start,
            end_date=pt_end,
        ))

    _season_set = frozenset(season_months) if season_months else frozenset()
    for spec in specs:
        spec.season_months = _season_set

    if start_date and end_date:
        log.info("Fetching rasters for %d LUCAS points (workers=%d, fixed window %s – %s).",
                 len(specs), workers, start_date, end_date)
    else:
        log.info("Fetching rasters for %d LUCAS points (workers=%d, window=±%d days).",
                 len(specs), workers, time_window_days)
    if _season_set:
        log.info("Seasonal filter active: only months %s will be kept after fetch.",
                 sorted(_season_set))

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one_point, spec, s3, bucket, raster_prefix, process): spec.point_id
            for spec in specs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching rasters"):
            pid = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                log.error("FAILED %s: %s", pid, exc)
                failed.append(pid)

    if failed:
        log.warning("Fetch completed with %d failures: %s", len(failed), failed)
    else:
        log.info("Fetch complete: all %d points processed.", len(specs))
