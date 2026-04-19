"""
raster_fetcher.py — Fetch raw 9×9 rasters for each LUCAS point and store on S3.

Storage layout:
  s3://{BUCKET}/raw_rasters/{point_id}.npz
  s3://{BUCKET}/raw_rasters/{point_id}_meta.json

The .npz contains:
  rasters   float32  (N_dates, 9, 9, 11)  — band order per BAND_NAMES
  dates     str      (N_dates,)            — "YYYY-MM-DD"
  cloud_pct float32  (N_dates,)            — from Catalog API

Re-running is safe: existing .npz files are skipped (S3 cache check).
"""

from __future__ import annotations

import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date as _date, timedelta
from typing import Optional

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from pyproj import Transformer
from tqdm import tqdm

from sh_clients import CatalogClient, ProcessClient, BAND_NAMES

log = logging.getLogger(__name__)

RASTER_SHAPE = (9, 9, 11)   # H × W × bands
RESOLUTION_M = 10
HALF_EXTENT_M = (RASTER_SHAPE[0] * RESOLUTION_M) / 2.0   # 45 m


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _make_s3(endpoint: str, bucket: str, access_key: str, secret_key: str):
    cfg = BotoConfig(retries={"max_attempts": 6, "mode": "standard"}, connect_timeout=15, read_timeout=120)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
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


def _bbox_4326(bbox_3857: tuple) -> tuple[float, float, float, float]:
    """Convert 3857 bbox to 4326 (for Catalog API)."""
    xmin, ymin, xmax, ymax = bbox_3857
    lon_min, lat_min = _TO_4326.transform(xmin, ymin)
    lon_max, lat_max = _TO_4326.transform(xmax, ymax)
    return (lon_min, lat_min, lon_max, lat_max)


def _center_pixel(lat: float, lon: float, bbox_3857: tuple) -> list[int]:
    """
    Identify which pixel in the 9×9 grid is closest to (lat, lon).
    Returns [row, col]. Will be [4, 4] for a correctly centered bbox.
    """
    x_pt, y_pt = _TO_3857.transform(lon, lat)
    xmin, ymin, xmax, ymax = bbox_3857

    col = int((x_pt - xmin) / RESOLUTION_M)
    row = int((ymax - y_pt) / RESOLUTION_M)

    # Clamp to valid range
    col = max(0, min(RASTER_SHAPE[1] - 1, col))
    row = max(0, min(RASTER_SHAPE[0] - 1, row))
    return [row, col]


# ---------------------------------------------------------------------------
# Per-point fetch logic
# ---------------------------------------------------------------------------

@dataclass
class FetchSpec:
    point_id: str
    lat: float
    lon: float
    start_date: str   # "YYYY-MM-DD"
    end_date: str


def _fetch_one_point(
    spec: FetchSpec,
    s3,
    bucket: str,
    raster_prefix: str,
    catalog: CatalogClient,
    process: ProcessClient,
    filter_config: dict,
) -> str:
    """
    Fetch raw raster for one LUCAS point, upload to S3.
    Returns point_id on success. Raises on failure.
    """
    npz_key  = f"{raster_prefix}{spec.point_id}.npz"
    meta_key = f"{raster_prefix}{spec.point_id}_meta.json"

    # --- Cache check ---
    if _s3_exists(s3, bucket, npz_key):
        log.debug("%s: raw raster already exists on S3, skipping.", spec.point_id)
        return spec.point_id

    # --- Geometry ---
    b3857 = _bbox_3857(spec.lat, spec.lon)
    b4326 = _bbox_4326(b3857)
    center_px = _center_pixel(spec.lat, spec.lon, b3857)

    # --- Catalog API ---
    scenes = catalog.search_acquisitions(
        bbox_4326=b4326,
        start_date=spec.start_date,
        end_date=spec.end_date,
        catalog_prefilter=filter_config["catalog_prefilter"],
    )

    if not scenes:
        log.warning("%s: no acquisitions found (%s – %s).", spec.point_id, spec.start_date, spec.end_date)
        return spec.point_id

    # --- Process API: one fetch per scene date ---
    rasters_list: list[np.ndarray] = []
    dates_list: list[str] = []
    cloud_list: list[float] = []

    for scene in scenes:
        try:
            arr = process.fetch_raster(b3857, scene.date)  # (9, 9, 11)
            rasters_list.append(arr)
            dates_list.append(scene.date)
            cloud_list.append(scene.cloud_pct)
        except Exception as exc:
            log.warning("%s: failed to fetch date %s: %s", spec.point_id, scene.date, exc)

    if not rasters_list:
        log.warning("%s: all date fetches failed.", spec.point_id)
        return spec.point_id

    rasters_arr = np.stack(rasters_list, axis=0).astype(np.float32)  # (N, 9, 9, 11)
    dates_arr   = np.array(dates_list, dtype="U10")
    cloud_arr   = np.array(cloud_list, dtype=np.float32)

    # --- Save .npz to S3 ---
    buf = io.BytesIO()
    np.savez(buf, rasters=rasters_arr, dates=dates_arr, cloud_pct=cloud_arr)
    _s3_put_bytes(s3, bucket, npz_key, buf.getvalue(), "application/octet-stream")

    # --- Save _meta.json to S3 ---
    meta = {
        "lucas_point_id":  spec.point_id,
        "lucas_lat":       spec.lat,
        "lucas_lon":       spec.lon,
        "bbox_epsg4326":   list(b4326),
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
    filter_config: dict,
    catalog: CatalogClient,
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
) -> None:
    """
    Fetch raw rasters for all LUCAS points in lucas_df and store on S3.

    Expected columns in lucas_df: POINT_ID, TH_LAT, TH_LONG, SURVEY_DATE.

    start_date / end_date: when both are provided, use a fixed window for all
    points (e.g. full Sentinel-2 archive).  Otherwise fall back to
    ±time_window_days around each point's SURVEY_DATE.
    """
    s3, bucket = _make_s3(s3_endpoint, s3_bucket, s3_access_key, s3_secret_key)

    if not raster_prefix.endswith("/"):
        raster_prefix += "/"

    specs: list[FetchSpec] = []
    for _, row in lucas_df.iterrows():
        if start_date and end_date:
            pt_start, pt_end = start_date, end_date
        else:
            survey  = pd.to_datetime(row["SURVEY_DATE"])
            pt_start = (survey - timedelta(days=time_window_days)).strftime("%Y-%m-%d")
            pt_end   = (survey + timedelta(days=time_window_days)).strftime("%Y-%m-%d")
        specs.append(FetchSpec(
            point_id=str(row["POINT_ID"]),
            lat=float(row["TH_LAT"]),
            lon=float(row["TH_LONG"]),
            start_date=pt_start,
            end_date=pt_end,
        ))

    if start_date and end_date:
        log.info("Fetching rasters for %d LUCAS points (workers=%d, fixed window %s – %s).",
                 len(specs), workers, start_date, end_date)
    else:
        log.info("Fetching rasters for %d LUCAS points (workers=%d, window=±%d days).",
                 len(specs), workers, time_window_days)

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _fetch_one_point, spec, s3, bucket, raster_prefix,
                catalog, process, filter_config
            ): spec.point_id
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
