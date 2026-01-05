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


def main(config_path="configs/dev.yaml"):
    cfg = load_config(config_path)

    storage = StorageManager(
        cfg.data.base_dir,
        cfg.data.raw_dir,
        cfg.data.indices_dir,
        cfg.data.metadata_dir,
        cfg.aoi.name,
    )

    creds = SentinelHubCredentials(
        client_id=cfg.cdse.client_id,
        client_secret=cfg.cdse.client_secret,
    )

    sh_client = SentinelHubClient(creds)

    evalscript = build_raw_bands_evalscript(cfg.sentinel2.bands)

    output_folder = storage.raw_dir / "_tmp"
    output_folder.mkdir(parents=True, exist_ok=True)

    sh_client.request_tiff(
        evalscript=evalscript,
        bbox=cfg.aoi.bbox,
        crs_epsg=cfg.aoi.crs,
        time_interval=(cfg.temporal.start_date, cfg.temporal.end_date),
        size=(512, 512),
        max_cloud_coverage=cfg.temporal.max_cloud_cover,
        output_folder=str(output_folder),
    )

    # SDK writes response.tar → extract
    tar_path = next(output_folder.glob("**/response.tar"))
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=output_folder)

    final_path = storage.raw_mosaic_path(
        collection=cfg.sentinel2.collection,
        from_date=cfg.temporal.start_date,
        to_date=cfg.temporal.end_date,
        bands=cfg.sentinel2.bands,
        width=512,
        height=512,
    )

    shutil.move(str(output_folder / "default.tif"), final_path)
    shutil.rmtree(output_folder)

    print(f"Saved raw mosaic → {final_path}")


if __name__ == "__main__":
    main()
