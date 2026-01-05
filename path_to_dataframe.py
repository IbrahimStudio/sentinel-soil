from pathlib import Path
import pandas as pd

from sentinel_soil.io.raster_to_dataframe import raster_to_dataframe

# pd.set_option('display.float_format', '{:.20f}'.format)


def main():
    # Path to the GeoTIFF you just generated
    tif_path = "data/raw/point_patch_100m_2024-06-15.tif"

    band_names = ["B02", "B03", "B04", "B08"]

    df = raster_to_dataframe(tif_path, band_names)

    print(df.head())
    print(f"DataFrame shape: {df.shape}")

    # Save as pickle
    out_path = Path("data/derived/point_patch_100m_2024-06-15.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_pickle(out_path)

    print(f"Saved DataFrame pickle → {out_path}")


if __name__ == "__main__":

    import rasterio

    with rasterio.open("data/raw/point_patch_100m_2024-06-15.tif") as src:
        arr = src.read()
        print(arr.min(), arr.max())

    main()
