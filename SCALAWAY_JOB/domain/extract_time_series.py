# sentinel_soil/domain/extract_time_series.py
from __future__ import annotations

import json
import shutil
import tarfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import rasterio

from utils.config import load_config
from utils.sentinelhub_client import SentinelHubClient, SentinelHubCredentials
from utils.evalscript_builder import build_orbit_timeseries_evalscript
from utils.geometry import bbox_for_grid_around_point  # must accept width/height px + res_m

import logging


import logging


# -----------------------------
# Minimal helpers (pure, local)
# -----------------------------
def _safe_id_part(point_id: Optional[str], lat: float, lon: float) -> str:
    """Stable folder name for the point."""
    if point_id:
        return f"point_{str(point_id).strip()}"
    return f"point_{lat:.6f}_{lon:.6f}".replace(".", "p")


def _compute_interval_around_survey(survey: date, window_days: int) -> Tuple[str, str]:
    start = survey - timedelta(days=window_days)
    end = survey + timedelta(days=window_days)
    return start.isoformat(), end.isoformat()


def _extract_response_tar(folder: Path) -> None:
    tar_path = next(folder.glob("**/response.tar"), None)
    if tar_path is None:
        raise FileNotFoundError("response.tar not found in Sentinel Hub output folder")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=folder)


def _read_userdata_dates(folder: Path) -> List[str]:
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
    # normalize to YYYY-MM-DD
    return [d[:10] for d in dates]


def _split_stacked_tif_to_per_date(
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

        for i, d in enumerate(dates):
            start_band = i * n_bands + 1  # 1-based
            data = src.read(list(range(start_band, start_band + n_bands))).astype("float32")

            out_path = out_dir / f"{d}.tif"
            out_profile = profile.copy()
            out_profile.update(count=n_bands, dtype="float32")

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(data)

    # tiny metadata file for downstream
    (out_dir / "bands.json").write_text(json.dumps({"bands": bands}, indent=2), encoding="utf-8")


def _parse_survey_date(value: Union[str, date, datetime, pd.Timestamp]) -> date:
    """Worker-safe date parsing (keep minimal)."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    s = str(value).strip()
    # ISO fast path: 'YYYY-MM-DD' or 'YYYY-MM-DDTHH...'
    return date.fromisoformat(s[:10])


# -----------------------------
# Public API: job-oriented extract
# -----------------------------
def extract_one(
    *,
    lat: float,
    lon: float,
    survey_date: Union[str, date, datetime, pd.Timestamp],
    window_days: int,
    pixel_window_w: int,
    pixel_window_h: int,
    res_m: float = 10.0,
    config_path: str = "configs/dev.yaml",
    point_id: Optional[str] = None,
    max_cloud_coverage: int = 80,
    mosaicking_order: str = "leastCC",
    out_root: Path,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Minimal extraction for a single job.

    Writes under:
      <out_root>/<id_part>/grid_<W>x<H>_res<res>m/survey_<date>_pm<window_days>d/
        per_date/YYYY-MM-DD.tif
        per_date/bands.json
        stacked/default.tif
        stacked/userdata.json (optional)

    Returns:
      ts_root (local path) - caller uploads it to Object Storage (intermediate/)
    """
    if pixel_window_w <= 0 or pixel_window_h <= 0:
        raise ValueError("pixel_window_w/h must be positive integers")

    logger = logger or logging.getLogger(__name__)
    cfg = load_config(config_path)

    survey = _parse_survey_date(survey_date)
    start_date, end_date = _compute_interval_around_survey(survey, window_days)

    # keep your same band set
    bands = ["B02", "B03", "B04", "B08", "B11", "B12"]

    # geometry
    bbox, epsg = bbox_for_grid_around_point(
        lat=lat,
        lon=lon,
        width_px=pixel_window_w,
        height_px=pixel_window_h,
        res_m=res_m,
    )
    size = (pixel_window_w, pixel_window_h)

    id_part = _safe_id_part(point_id, lat, lon)

    ts_root = (
        Path(out_root)
        / id_part
        / f"grid_{pixel_window_w}x{pixel_window_h}_res{int(res_m)}m"
        / f"survey_{survey.isoformat()}_pm{window_days}d"
    )

    per_date_dir = ts_root / "per_date"
    stacked_dir = ts_root / "stacked"
    tmp_dir = ts_root / "_tmp"

    # fresh dirs
    ts_root.mkdir(parents=True, exist_ok=True)
    per_date_dir.mkdir(parents=True, exist_ok=True)
    stacked_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    creds = SentinelHubCredentials(
        client_id=cfg.cdse.client_id,
        client_secret=cfg.cdse.client_secret,
    )
    sh = SentinelHubClient(creds)

    evalscript = build_orbit_timeseries_evalscript(bands=bands, units="REFLECTANCE")

    logger.info(
        f"Extracting time series: id_part={id_part} size={pixel_window_w}x{pixel_window_h} "
        f"res={res_m}m interval={start_date}..{end_date} cloud<={max_cloud_coverage}"
    )

    # Sentinel Hub request
    sh.request_tiff(
        evalscript=evalscript,
        bbox=bbox,
        crs_epsg=epsg,
        time_interval=(start_date, end_date),
        size=size,
        max_cloud_coverage=max_cloud_coverage,
        output_folder=str(tmp_dir),
        mosaicking_order=mosaicking_order,
    )

    _extract_response_tar(tmp_dir)

    default_candidates = list(tmp_dir.glob("**/default.tif"))
    if not default_candidates:
        raise FileNotFoundError("default.tif not found after extracting response.tar")
    stacked_tif = default_candidates[0]

    dates = _read_userdata_dates(tmp_dir)
    if not dates:
        raise ValueError("No acquisition dates found in userdata.json")

    # keep stacked outputs (useful for debugging, optional but cheap)
    shutil.copy2(stacked_tif, stacked_dir / "default.tif")
    userdata_candidates = list(tmp_dir.glob("**/userdata.json"))
    if userdata_candidates:
        shutil.copy2(userdata_candidates[0], stacked_dir / "userdata.json")

    # produce per-date rasters for feature step
    _split_stacked_tif_to_per_date(
        stacked_tif=stacked_tif,
        dates=dates,
        bands=bands,
        out_dir=per_date_dir,
    )

    # cleanup tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        f"Extraction complete: ts_root={ts_root} dates={len(dates)} per_date_dir={per_date_dir}"
    )
    return ts_root