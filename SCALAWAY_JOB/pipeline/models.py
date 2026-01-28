# sentinel_soil/pipeline/models.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict


@dataclass(frozen=True)
class PixelWindow:
    w: int
    h: int


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    point_id: str
    lat: float
    lon: float
    survey_date: date
    window: PixelWindow
    ndvi_threshold: float
    min_obs: int

    # optional knobs (defaults for job runner)
    window_days: int = 30
    res_m: float = 10.0
    max_cloud_coverage: int = 80
    mosaicking_order: str = "leastCC"


def _to_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "")
    # European decimal comma support
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    return float(s)


def _to_int(x: Any) -> int:
    if isinstance(x, int):
        return x
    return int(float(str(x).strip()))


def parse_job(payload: Dict[str, Any]) -> JobSpec:
    missing = [k for k in ["job_id", "point_id", "lat", "lon", "survey_date", "spatial_window"] if k not in payload]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    sw = payload["spatial_window"]
    if not isinstance(sw, dict) and not isinstance(sw, PixelWindow):
        raise ValueError("spatial_window must be {'type':'pixel_window','w':int,'h':int}")

    w = _to_int(sw.get("w"))
    h = _to_int(sw.get("h"))
    if w <= 0 or h <= 0:
        raise ValueError(f"pixel_window w/h must be > 0 (got w={w}, h={h})")

    # Your bbox_for_grid_around_point currently enforces odd dims for centering
    if (w % 2 == 0) or (h % 2 == 0):
        raise ValueError(f"pixel_window w/h must be odd to center on the point (got w={w}, h={h})")

    lat = _to_float(payload["lat"])
    lon = _to_float(payload["lon"])
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"lat out of range: {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"lon out of range: {lon}")

    survey_date = date.fromisoformat(str(payload["survey_date"])[:10])

    ndvi_threshold = float(payload.get("ndvi_threshold", 0.2))
    # NDVI is in [-1, 1]; for bare soil you usually use [0, 0.3], but keep it flexible
    if not (-1.0 <= ndvi_threshold <= 1.0):
        raise ValueError(f"ndvi_threshold out of range [-1,1]: {ndvi_threshold}")

    min_obs = _to_int(payload.get("min_obs", 2))
    if min_obs < 1:
        raise ValueError(f"min_obs must be >= 1 (got {min_obs})")

    window_days = _to_int(payload.get("window_days", 30))
    if window_days < 0:
        raise ValueError(f"window_days must be >= 0 (got {window_days})")

    res_m = float(payload.get("res_m", 10.0))
    if res_m <= 0:
        raise ValueError(f"res_m must be > 0 (got {res_m})")

    max_cloud_coverage = _to_int(payload.get("max_cloud_coverage", 80))
    if not (0 <= max_cloud_coverage <= 100):
        raise ValueError(f"max_cloud_coverage must be in [0,100] (got {max_cloud_coverage})")

    mosaicking_order = str(payload.get("mosaicking_order", "leastCC"))

    return JobSpec(
        job_id=str(payload["job_id"]).strip(),
        point_id=str(payload["point_id"]).strip(),
        lat=lat,
        lon=lon,
        survey_date=survey_date,
        window=PixelWindow(w=w, h=h),
        ndvi_threshold=ndvi_threshold,
        min_obs=min_obs,
        window_days=window_days,
        res_m=res_m,
        max_cloud_coverage=max_cloud_coverage,
        mosaicking_order=mosaicking_order,
    )
