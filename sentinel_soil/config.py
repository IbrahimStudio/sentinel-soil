from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import yaml
import os
from dotenv import load_dotenv

load_dotenv()


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

    cdse_conf = CDSEConfig(
        auth_url=raw["cdse"]["auth_url"],
        client_id=os.getenv("CDSE_CLIENT_ID"),
        client_secret=os.getenv("CDSE_CLIENT_SECRET"),
    )

    sh_conf = SentinelHubConfig(
        process_url=raw["sentinelhub"]["process_url"],
        catalog_url=raw["sentinelhub"]["catalog_url"],
    )

    base_dir = Path(raw["data"]["base_dir"])
    data_conf = DataPathsConfig(
        base_dir=base_dir,
        raw_dir=base_dir / raw["data"]["raw_dir"],
        indices_dir=base_dir / raw["data"]["indices_dir"],
        metadata_dir=base_dir / raw["data"]["metadata_dir"],
    )

    aoi_conf = AOIConfig(
        name=raw["aoi"]["name"],
        crs=raw["aoi"]["crs"],
        bbox=tuple(raw["aoi"]["bbox"]),
    )

    temporal_conf = TemporalConfig(
        start_date=raw["temporal"]["start_date"],
        end_date=raw["temporal"]["end_date"],
        max_cloud_cover=int(raw["temporal"]["max_cloud_cover"]),
    )

    s2_conf = Sentinel2Config(
        collection=raw["sentinel2"]["collection"],
        bands=list(raw["sentinel2"]["bands"]),
    )

    return AppConfig(
        cdse=cdse_conf,
        sentinelhub=sh_conf,
        data=data_conf,
        aoi=aoi_conf,
        temporal=temporal_conf,
        sentinel2=s2_conf,
    )
