from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import List, Tuple

import rasterio

from sentinel_soil.config import load_config
from sentinel_soil.clients.sentinelhub_client import SentinelHubClient, SentinelHubCredentials
from sentinel_soil.clients.evalscript_builder import build_orbit_timeseries_evalscript
from sentinel_soil.utils.geometry import bbox_around_point


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

        for i, date in enumerate(dates):
            start_band = i * n_bands + 1  # 1-based
            data = src.read(list(range(start_band, start_band + n_bands))).astype("float32")

            out_path = out_dir / f"{date}.tif"
            out_profile = profile.copy()
            out_profile.update(count=n_bands, dtype="float32")

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(data)

    # write band order metadata once
    with open(out_dir / "bands.json", "w", encoding="utf-8") as f:
        json.dump({"bands": bands}, f, indent=2)


def main(lat, lon, window_m, start_date, end_date, config_path: str = "configs/dev.yaml"):
    cfg = load_config(config_path)

    # # ---- parameters ----
    # lat = 44.8915307
    # lon = 10.01263632
    # window_m = 100.0
    # start_date = "2024-06-01"
    # end_date = "2024-06-30"

    # Include 10m + 20m bands (20m will be resampled to requested grid)
    bands = ["B02", "B03", "B04", "B08", "B11", "B12"]

    # Request 10x10 over 100m -> ~10m grid
    size = (10, 10)

    # Metric bbox around point (UTM 32N)
    bbox, epsg = bbox_around_point(lat=lat, lon=lon, size_m=window_m, dst_epsg=32632)

    base = Path(cfg.data.base_dir)
    ts_root = (
        base
        / "timeseries"
        / safe_point_name(lat, lon)
        / f"window_{int(window_m)}m"
        / f"{start_date}_{end_date}"
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
        max_cloud_coverage=80,
        output_folder=str(tmp_dir),
        mosaicking_order="leastCC",
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

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"✅ Time series extracted: {ts_root}")
    print(f"   Dates: {len(dates)}")
    print(f"   Bands per date: {bands}")
    print(f"   Per-date GeoTIFFs: {per_date_dir}")


if __name__ == "__main__":
    main()
