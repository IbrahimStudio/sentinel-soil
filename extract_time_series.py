from __future__ import annotations

import json
import shutil
import tarfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
import rasterio
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

from sentinel_soil.config import load_config
from sentinel_soil.clients.sentinelhub_client import SentinelHubClient, SentinelHubCredentials
from sentinel_soil.clients.evalscript_builder import build_orbit_timeseries_evalscript
from sentinel_soil.utils.geometry import *

import logging

def setup_logger(
    *,
    log_dir: Path,
    name: str = "sentinel_soil",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Creates a logger that writes both to console and to a rotating .log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs

    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler (keeps current behavior)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    # --- File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logging initialized → {log_path}")
    return logger


log_root = Path("logs")        # or cfg.data.base_dir / "logs"
logger = setup_logger(log_dir=log_root)


def safe_point_name(lat: float, lon: float) -> str:
    return f"point_{lat:.6f}_{lon:.6f}".replace(".", "p")


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

        for i, d in enumerate(dates):
            start_band = i * n_bands + 1  # 1-based
            data = src.read(list(range(start_band, start_band + n_bands))).astype("float32")

            out_path = out_dir / f"{d}.tif"
            out_profile = profile.copy()
            out_profile.update(count=n_bands, dtype="float32")

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(data)

    with open(out_dir / "bands.json", "w", encoding="utf-8") as f:
        json.dump({"bands": bands}, f, indent=2)


def parse_survey_date(value) -> date:
    """
    Robust parser for LUCAS / EU survey dates.

    Supported:
    - YYYY-MM-DD
    - YYYY-MM-DDTHH:MM:SS
    - DD-MM-YY / DD-MM-YYYY
    - DD/MM/YY / DD/MM/YYYY
    - pandas.Timestamp
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("Empty SURVEY_DATE")

    # pandas Timestamp
    if isinstance(value, pd.Timestamp):
        return value.date()

    # datetime / date
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    s = str(value).strip()

    # --- ISO first (fast path)
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        pass

    # --- European formats
    for fmt in ("%d-%m-%y", "%d-%m-%Y", "%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue

    raise ValueError(f"Unrecognized SURVEY_DATE format: {value!r}")


def compute_interval_around_survey(survey: date, window_days: int) -> Tuple[str, str]:
    start = survey - timedelta(days=window_days)
    end = survey + timedelta(days=window_days)
    return start.isoformat(), end.isoformat()


@dataclass(frozen=True)
class SeedRecord:
    seed_id: str
    lat: float
    lon: float
    survey_date: date



def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float32)
    m = den != 0
    out[m] = num[m] / den[m]
    return out


def debug_export_timeseries_csv(
    ts_root: Path,
    *,
    max_dates: int = 60,
) -> None:
    """
    Writes small, readable debug CSVs next to Sentinel extraction output:
      - debug/per_date_stats.csv
      - debug/pixel_samples.csv

    Assumes:
      ts_root/per_date/*.tif
      ts_root/per_date/bands.json
    """
    per_date_dir = ts_root / "per_date"
    if not per_date_dir.exists():
        raise FileNotFoundError(f"per_date not found: {per_date_dir}")

    bands_path = per_date_dir / "bands.json"
    if not bands_path.exists():
        raise FileNotFoundError(f"bands.json not found: {bands_path}")

    bands = json.loads(bands_path.read_text(encoding="utf-8"))["bands"]

    tifs = sorted(per_date_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No per-date tif files in {per_date_dir}")

    # limit to keep debug fast/readable
    if len(tifs) > max_dates:
        tifs = tifs[:max_dates]

    debug_dir = ts_root / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    per_date_rows = []
    pixel_rows = []

    # will define sample pixels from the first raster shape
    with rasterio.open(tifs[0]) as src0:
        H, W = src0.height, src0.width

    samples = [
        ("center", H // 2, W // 2),
        ("top_left", 0, 0),
        ("top_right", 0, W - 1),
        ("bottom_left", H - 1, 0),
        ("bottom_right", H - 1, W - 1),
    ]

    for tif in tifs:
        d = tif.stem  # YYYY-MM-DD
        with rasterio.open(tif) as src:
            arr = src.read().astype(np.float32)  # (B,H,W)

        # Per-band stats
        for bi, bname in enumerate(bands):
            band = arr[bi]
            finite = np.isfinite(band)
            n = band.size
            n_finite = int(finite.sum())
            n_nan = int((~finite).sum())
            n_zero = int((band == 0).sum())

            per_date_rows.append({
                "date": d,
                "band": bname,
                "mean": float(np.nanmean(band)) if n_finite else np.nan,
                "median": float(np.nanmedian(band)) if n_finite else np.nan,
                "min": float(np.nanmin(band)) if n_finite else np.nan,
                "max": float(np.nanmax(band)) if n_finite else np.nan,
                "pct_nan": n_nan / n,
                "pct_zero": n_zero / n,
            })

        # Sample pixels time series + indices
        # map by name -> band array for index calc
        band_map = {bands[i]: arr[i] for i in range(len(bands))}
        # guard required bands
        has = set(band_map.keys())
        req = {"B04", "B08"}
        if not req.issubset(has):
            raise ValueError(f"Missing required bands {req} in {bands}")

        nir = band_map["B08"]
        red = band_map["B04"]
        ndvi = _safe_ratio(nir - red, nir + red)

        for label, r, c in samples:
            row = {
                "date": d,
                "sample": label,
                "row": r,
                "col": c,
                "NDVI": float(ndvi[r, c]) if np.isfinite(ndvi[r, c]) else np.nan,
            }
            # add raw band values for the pixel
            for bname in bands:
                v = band_map[bname][r, c]
                row[bname] = float(v) if np.isfinite(v) else np.nan
            pixel_rows.append(row)

    pd.DataFrame(per_date_rows).to_csv(debug_dir / "per_date_stats.csv", index=False)
    pd.DataFrame(pixel_rows).to_csv(debug_dir / "pixel_samples.csv", index=False)

    print(f"✅ Debug CSVs written to: {debug_dir}")
    print(f"   - per_date_stats.csv")
    print(f"   - pixel_samples.csv")


# ----------------------------
# NEW: read seeds from CSV/XLSX
# ----------------------------
def load_seeds_from_table(
    path: str | Path,
    *,
    # column mapping (your file uses these names)
    id_col: str = "POINT_ID",
    lat_col: str = "TH_LAT",
    lon_col: str = "TH_LONG",
    date_col: str = "SURVEY_DATE",
    depth_col: str = "Depth",
    # optional depth filter
    keep_depth_values: Optional[List[str]] = None,   # e.g. ["0-20", "0-30", "0-20 cm"]
    keep_depth_numeric_max: Optional[float] = None,  # e.g. 20.0 if Depth is numeric
) -> List[SeedRecord]:
    """
    Reads CSV or XLSX and returns seeds usable by extract_for_seeds().

    - seed_id comes from POINT_ID (kept as string to avoid float/NE issues)
    - coordinates from TH_LAT / TH_LONG
    - date from SURVEY_DATE
    - optional: filter rows by Depth
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        # Handles both dot and comma decimals safely:
        # - read as strings first, then normalize numbers ourselves
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .csv or .xlsx")

    missing = [c for c in [id_col, lat_col, lon_col, date_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    # Normalize numeric strings (CSV cases): "44,8915307" -> "44.8915307"
    def _to_float(x) -> float:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            raise ValueError("Empty numeric value")
        s = str(x).strip()
        s = s.replace(" ", "")
        # common European decimal comma
        if s.count(",") == 1 and s.count(".") == 0:
            s = s.replace(",", ".")
        return float(s)

    # Optional depth filtering TO BE CHANGED WITH OTHER FILTERS IN FUTURE
    # if depth_col in df.columns:
    #     if keep_depth_values is not None:
    #         df = df[df[depth_col].astype(str).str.strip().isin(set(keep_depth_values))]
    #     elif keep_depth_numeric_max is not None:
    #         # try interpret Depth as numeric
    #         depth_num = df[depth_col].apply(lambda v: _to_float(v) if pd.notna(v) else None)
    #         df = df[depth_num.notna() & (depth_num <= keep_depth_numeric_max)]

    seeds: List[SeedRecord] = []
    for _, row in df.iterrows():
        seed_id = str(row[id_col]).strip()

        # Skip weird IDs (e.g. "NE") if present
        if seed_id == "" or seed_id.upper() in {"NE", "N/A", "NA", "NONE"}:
            continue

        lat = _to_float(row[lat_col])
        lon = _to_float(row[lon_col])
        survey = parse_survey_date(row[date_col])

        seeds.append(SeedRecord(seed_id=seed_id, lat=lat, lon=lon, survey_date=survey))

    if not seeds:
        raise ValueError("No valid seeds found after parsing/filtering.")

    return seeds


def extract_one(
    *,
    lat: float,
    lon: float,
    N: int,  # number of pixels around lat/lon from seed
    survey_date: date,
    window_days: int,
    res_m: float = 10.0,
    config_path: str = "configs/dev.yaml",
    seed_id: Optional[str] = None,
    max_cloud_coverage: int = 80,
    mosaicking_order: str = "leastCC",
) -> Path:
    cfg = load_config(config_path)

    bands = ["B02", "B03", "B04", "B08", "B11", "B12"]

    W, H = grid_for_n_pixels(N)

    bbox, epsg = bbox_for_grid_around_point(
        lat=lat, lon=lon, width_px=W, height_px=H, res_m=res_m
    )
    size = (W, H)

    start_date, end_date = compute_interval_around_survey(survey_date, window_days)

    base = Path(cfg.data.base_dir)
    id_part = f"seed_{seed_id}" if seed_id else safe_point_name(lat, lon)

    ts_root = (
        base
        / "timeseries"
        / id_part
        / f"grid_{W}x{H}_res{int(res_m)}m"
        / f"survey_{survey_date.isoformat()}_pm{window_days}d"
    )

    per_date_dir = ts_root / "per_date"
    stacked_dir = ts_root / "stacked"
    tmp_dir = ts_root / "_tmp"

    creds = SentinelHubCredentials(
        client_id=cfg.cdse.client_id,
        client_secret=cfg.cdse.client_secret,
    )
    sh = SentinelHubClient(creds)

    evalscript = build_orbit_timeseries_evalscript(bands=bands, units="REFLECTANCE")

    tmp_dir.mkdir(parents=True, exist_ok=True)
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

    split_stacked_tif_to_per_date(
        stacked_tif=stacked_tif,
        dates=dates,
        bands=bands,
        out_dir=per_date_dir,
    )

    if cfg.debug:
        debug_export_timeseries_csv(ts_root)


    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"✅ Time series extracted: {ts_root}")
    print(f"   Grid: {W}x{H} @ {res_m}m")
    print(f"   Interval: {start_date} → {end_date} (survey={survey_date}, ±{window_days}d)")
    print(f"   Dates: {len(dates)} | Bands/date: {bands}")
    print(f"   Per-date GeoTIFFs: {per_date_dir}")
    logger.info(f"✅ Time series extracted: {ts_root}")
    logger.info(f"   Grid: {W}x{H} @ {res_m}m")
    logger.info(f"   Interval: {start_date} → {end_date} (survey={survey_date}, ±{window_days}d)")
    logger.info(f"   Dates: {len(dates)} | Bands/date: {bands}")
    logger.info(f"   Per-date GeoTIFFs: {per_date_dir}")


    return ts_root


def extract_for_seeds(
    seeds: Iterable[SeedRecord],
    *,
    N: int,
    window_days: int,
    res_m: float = 10.0,
    config_path: str = "configs/dev.yaml",
) -> List[Tuple[str, Path]]:
    outputs: List[Tuple[str, Path]] = []
    failures: List[Tuple[str, str]] = []

    for s in seeds:
        try:
            out = extract_one(
                lat=s.lat,
                lon=s.lon,
                N=N,
                survey_date=s.survey_date,
                window_days=window_days,
                res_m=res_m,
                config_path=config_path,
                seed_id=s.seed_id,
            )
            outputs.append((s.seed_id, out))
        except Exception as e:
            failures.append((s.seed_id, str(e)))
            print(f"❌ Seed {s.seed_id} failed: {e}")
            logger.info(f"❌ Seed {s.seed_id} failed: {e}")

    if failures:
        print("\n=== Failures summary ===")
        for seed_id, err in failures:
            print(f"- {seed_id}: {err}")

    return outputs


if __name__ == "__main__":
    
    # Example: load seeds from your LUCAS-enriched table and extract
    seeds = load_seeds_from_table(
        "data\seed_data\seed.xlsx",   # or .csv
        # Optional examples:
        # keep_depth_values=["0-20", "0-20 cm"],
        # keep_depth_numeric_max=20.0,
    )

    extract_for_seeds(
        seeds,
        N=2,               # CHOOSE BETWEEN 1 AND 2
        window_days=30,
        res_m=10.0,
        config_path="configs/dev.yaml",
    )
