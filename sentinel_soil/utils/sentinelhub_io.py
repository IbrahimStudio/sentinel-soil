from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import List

import rasterio


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
    return [d[:10] for d in dates]  # YYYY-MM-DD


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
            start_band = i * n_bands + 1
            data = src.read(list(range(start_band, start_band + n_bands))).astype("float32")

            out_path = out_dir / f"{date}.tif"
            out_profile = profile.copy()
            out_profile.update(count=n_bands, dtype="float32")

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(data)

    # store band order for downstream steps
    import json
    with open(out_dir / "bands.json", "w", encoding="utf-8") as f:
        json.dump({"bands": bands}, f, indent=2)
