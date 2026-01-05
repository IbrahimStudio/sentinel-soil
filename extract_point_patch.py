from pathlib import Path
import shutil
import tarfile

from sentinel_soil.config import load_config
from sentinel_soil.clients.sentinelhub_client import (
    SentinelHubClient,
    SentinelHubCredentials,
)
from sentinel_soil.clients.evalscript_builder import build_raw_bands_evalscript
from sentinel_soil.io.storage import StorageManager
from sentinel_soil.utils.geometry import bbox_around_point


def main(config_path="configs/dev.yaml"):
    cfg = load_config(config_path)

    # ---- target point ----
    lat = 44.8915307
    lon = 10.01263632
    window_m = 100.0
    time_interval = ("2024-06-10", "2024-06-20")


    bbox, epsg = bbox_around_point(lat, lon, window_m)

    # ---- storage ----
    storage = StorageManager(
        cfg.data.base_dir,
        cfg.data.raw_dir,
        cfg.data.indices_dir,
        cfg.data.metadata_dir,
        aoi_name="point_44.8915_10.0126",
    )

    # ---- sentinel hub client ----
    creds = SentinelHubCredentials(
        client_id=cfg.cdse.client_id,
        client_secret=cfg.cdse.client_secret,
    )
    sh_client = SentinelHubClient(creds)

    # ---- evalscript ----
    bands = ["B02", "B03", "B04", "B08"]
    evalscript = build_raw_bands_evalscript(bands)

    tmp_dir = storage.raw_dir / "_tmp_point"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sh_client.request_tiff(
        evalscript=evalscript,
        bbox=bbox,
        crs_epsg=epsg,
        time_interval=time_interval,
        size=(10, 10),  # 10m pixels → 100m window
        max_cloud_coverage=20,
        output_folder=str(tmp_dir),
    )

    # ---- extract result ----
    tar_path = next(tmp_dir.glob("**/response.tar"))
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=tmp_dir)

    out_path = storage.raw_dir / "point_patch_100m_2024-06-15.tif"
    shutil.move(str(tmp_dir / "default.tif"), out_path)
    shutil.rmtree(tmp_dir)

    print(f"Saved 100m x 100m patch → {out_path}")


if __name__ == "__main__":
    main()
