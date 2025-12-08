from pathlib import Path
import json
from dataclasses import asdict

from sentinel_soil.config import load_config
from sentinel_soil.clients.auth import CDSEAuthClient
from sentinel_soil.clients.process_client import (
    SentinelHubProcessClient,
    ProcessRequestParams,
)
from sentinel_soil.io.storage import StorageManager


def main(config_path: str = "configs/dev.yaml"):
    cfg = load_config(config_path)

    auth = CDSEAuthClient(cfg.cdse)
    storage = StorageManager(
        cfg.data.base_dir,
        cfg.data.raw_dir,
        cfg.data.indices_dir,
        cfg.data.metadata_dir,
        cfg.aoi.name,
    )
    process_client = SentinelHubProcessClient(cfg, auth)

    params = ProcessRequestParams(
        bbox=list(cfg.aoi.bbox),
        crs_epsg=cfg.aoi.crs,
        from_date=cfg.temporal.start_date,
        to_date=cfg.temporal.end_date,
        max_cloud=cfg.temporal.max_cloud_cover,
        bands=cfg.sentinel2.bands,
        width=512,
        height=512,
    )

    out_path = storage.raw_mosaic_path(
        collection=cfg.sentinel2.collection,
        from_date=cfg.temporal.start_date,
        to_date=cfg.temporal.end_date,
        bands=cfg.sentinel2.bands,
        width=params.width,
        height=params.height,
    )

    if out_path.exists():
        print(f"Raw mosaic already exists: {out_path}")
        return

    geotiff_bytes = process_client.download_geotiff_bytes(params)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(geotiff_bytes)

    print(f"Saved raw GeoTIFF to: {out_path}")

    # Save metadata next to it
    meta_path = storage.raw_metadata_path(
        collection=cfg.sentinel2.collection,
        from_date=cfg.temporal.start_date,
        to_date=cfg.temporal.end_date,
        bands=cfg.sentinel2.bands,
        width=params.width,
        height=params.height,
    )

    metadata = {
        "aoi": {
            "name": cfg.aoi.name,
            "crs": cfg.aoi.crs,
            "bbox": cfg.aoi.bbox,
        },
        "temporal": {
            "from": cfg.temporal.start_date,
            "to": cfg.temporal.end_date,
        },
        "bands": cfg.sentinel2.bands,
        "process_request": asdict(params),
    }
    storage.save_metadata(meta_path, metadata)
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
