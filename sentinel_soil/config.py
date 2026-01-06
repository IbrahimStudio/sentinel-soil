from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import yaml
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class BatchConfig:
    name: str
    seeds_csv: str
    window_m: float
    dst_epsg: int
    size: Tuple[int, int]
    start_date: str
    end_date: str
    bands: List[str]
    ndvi_threshold: float
    min_obs_baresoil: int
    max_cloud_coverage: int


@dataclass
class CDSEConfig:
    auth_url: str
    client_id: str
    client_secret: str | None = None


@dataclass
class SentinelHubConfig:
    process_url: str
    catalog_url: str


@dataclass
class DataPathsConfig:
    base_dir: Path
    raw_dir: Path
    indices_dir: Path
    metadata_dir: Path


@dataclass
class AOIConfig:
    name: str
    crs: int
    bbox: Tuple[float, float, float, float]


@dataclass
class TemporalConfig:
    start_date: str
    end_date: str
    max_cloud_cover: int


@dataclass
class Sentinel2Config:
    collection: str
    bands: List[str]


@dataclass
class AppConfig:
    cdse: CDSEConfig
    sentinelhub: SentinelHubConfig
    data: DataPathsConfig
    aoi: AOIConfig
    temporal: TemporalConfig
    sentinel2: Sentinel2Config


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # ---- CDSE ----
    cdse_raw = raw.get("cdse", {})
    cdse_conf = CDSEConfig(
        auth_url=cdse_raw.get("auth_url"),
        client_id=cdse_raw.get("client_id") or os.getenv("CDSE_CLIENT_ID"),
        client_secret=cdse_raw.get("client_secret") or os.getenv("CDSE_CLIENT_SECRET"),
    )

    # ---- SentinelHub ----
    sh_raw = raw.get("sentinelhub", {})
    sh_conf = SentinelHubConfig(
        process_url=sh_raw.get("process_url"),
        catalog_url=sh_raw.get("catalog_url"),
    )

    # ---- Data paths ----
    data_raw = raw.get("data", {})
    base_dir = Path(data_raw["base_dir"])
    data_conf = DataPathsConfig(
        base_dir=base_dir,
        raw_dir=base_dir / data_raw.get("raw_dir", "raw"),
        indices_dir=base_dir / data_raw.get("indices_dir", "indices"),
        metadata_dir=base_dir / data_raw.get("metadata_dir", "metadata"),
    )

    # ---- AOI (optional for batch mode) ----
    aoi_raw = raw.get("aoi")
    aoi_conf = None
    if aoi_raw:
        aoi_conf = AOIConfig(
            name=aoi_raw["name"],
            crs=aoi_raw["crs"],
            bbox=tuple(aoi_raw["bbox"]),
        )

    # ---- Temporal (global defaults) ----
    temporal_raw = raw.get("temporal", {})
    temporal_conf = TemporalConfig(
        start_date=temporal_raw.get("start_date"),
        end_date=temporal_raw.get("end_date"),
        max_cloud_cover=int(temporal_raw.get("max_cloud_cover", 80)),
    )

    # ---- Sentinel-2 ----
    s2_raw = raw.get("sentinel2", {})
    s2_conf = Sentinel2Config(
        collection=s2_raw.get("collection"),
        bands=list(s2_raw.get("bands", [])),
    )

    # ---- Batch (NEW) ----
    batch_conf = None
    batch_raw = raw.get("batch")
    if batch_raw:
        size = batch_raw.get("size", [10, 10])
        batch_conf = BatchConfig(
            name=batch_raw.get("name", "default_batch"),
            seeds_csv=batch_raw["seeds_csv"],
            window_m=float(batch_raw.get("window_m", 100.0)),
            dst_epsg=int(batch_raw.get("dst_epsg", 32632)),
            size=(int(size[0]), int(size[1])),
            start_date=batch_raw.get("start_date", temporal_conf.start_date),
            end_date=batch_raw.get("end_date", temporal_conf.end_date),
            bands=list(batch_raw.get("bands", s2_conf.bands)),
            ndvi_threshold=float(batch_raw.get("ndvi_threshold", 0.2)),
            min_obs_baresoil=int(batch_raw.get("min_obs_baresoil", 2)),
            max_cloud_coverage=int(batch_raw.get("max_cloud_coverage", temporal_conf.max_cloud_cover)),
        )

    return AppConfig(
        cdse=cdse_conf,
        sentinelhub=sh_conf,
        data=data_conf,
        aoi=aoi_conf,
        temporal=temporal_conf,
        sentinel2=s2_conf,
        batch=batch_conf,
    )

