from pathlib import Path
import rasterio
from rasterio.transform import xy

from sentinel_soil.config import load_config
from sentinel_soil.io.storage import StorageManager


def main(config_path: str = "configs/dev.yaml"):
    cfg = load_config(config_path)
    storage = StorageManager(
        cfg.data.base_dir,
        cfg.data.raw_dir,
        cfg.data.indices_dir,
        cfg.data.metadata_dir,
        cfg.aoi.name,
    )

    # now we have nested structure → recurse
    raw_files = sorted(storage.raw_dir.glob("**/*.tif"))
    if not raw_files:
        print("No raw GeoTIFFs found. Run ingest_raw.py first.")
        return

    tif_path = raw_files[0]
    print(f"Inspecting: {tif_path}")

    with rasterio.open(tif_path) as src:
        print("CRS:", src.crs)
        print("Width, height:", src.width, src.height)
        print("Count (bands):", src.count)
        print("Transform:", src.transform)

        data = src.read()
        print("Data shape:", data.shape)

        center_row = src.height // 2
        center_col = src.width // 2

        band_names = cfg.sentinel2.bands
        values = {band_names[i]: float(data[i, center_row, center_col])
                  for i in range(len(band_names))}

        x, y = xy(src.transform, center_row, center_col)
        print(f"Center pixel row/col: ({center_row}, {center_col})")
        print(f"Center pixel lon/lat approx: ({x}, {y})")
        print("Band values at center pixel:")
        for b, v in values.items():
            print(f"  {b}: {v}")


if __name__ == "__main__":
    main()
