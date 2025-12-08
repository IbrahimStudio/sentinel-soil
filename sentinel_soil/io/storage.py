from pathlib import Path
from typing import Any, Dict, List
import json


class StorageManager:
    def __init__(
        self,
        base_dir: Path,
        raw_dir: Path,
        indices_dir: Path,
        metadata_dir: Path,
        aoi_name: str,
    ):
        self.base_dir = base_dir
        self.raw_dir = raw_dir
        self.indices_dir = indices_dir
        self.metadata_dir = metadata_dir
        self.aoi_name = aoi_name

        # Ensure top-level dirs exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    # ---------- helpers to build filenames ----------

    @staticmethod
    def mosaic_basename(
        collection: str,
        from_date: str,
        to_date: str,
        bands: List[str] | None = None,
        width: int | None = None,
        height: int | None = None,
        extra_suffix: str | None = None,
    ) -> str:
        parts: List[str] = [collection, from_date, to_date]
        if bands:
            parts.append("-".join(bands))
        if width and height:
            parts.append(f"{width}x{height}")
        if extra_suffix:
            parts.append(extra_suffix)
        return "_".join(parts)

    # ---------- raw rasters ----------

    def raw_mosaic_path(
        self,
        collection: str,
        from_date: str,
        to_date: str,
        bands: List[str],
        width: int,
        height: int,
    ) -> Path:
        basename = self.mosaic_basename(collection, from_date, to_date, bands, width, height)
        # data/raw/<collection>/<aoi_name>/<basename>.tif
        dir_path = self.raw_dir / collection / self.aoi_name
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{basename}.tif"

    def raw_metadata_path(
        self,
        collection: str,
        from_date: str,
        to_date: str,
        bands: List[str],
        width: int,
        height: int,
    ) -> Path:
        basename = self.mosaic_basename(collection, from_date, to_date, bands, width, height)
        # data/metadata/raw/<collection>/<aoi_name>/<basename>.json
        dir_path = self.metadata_dir / "raw" / collection / self.aoi_name
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{basename}.json"

    # ---------- indices ----------

    def index_mosaic_path(
        self,
        index_name: str,
        collection: str,
        from_date: str,
        to_date: str,
        width: int,
        height: int,
    ) -> Path:
        basename = self.mosaic_basename(
            collection, from_date, to_date, bands=None, width=width, height=height, extra_suffix=index_name
        )
        # data/indices/<index_name>/<aoi_name>/<basename>.tif
        dir_path = self.indices_dir / index_name / self.aoi_name
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{basename}.tif"

    # ---------- generic metadata save ----------

    def save_metadata(self, path: Path, metadata: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
