import pandas as pd
import rasterio
from rasterio.transform import xy
from typing import List


def raster_to_dataframe(
    tif_path: str,
    band_names: List[str],
) -> pd.DataFrame:
    """
    Convert a multiband GeoTIFF into a pandas DataFrame.

    Each row = one pixel
    """

    with rasterio.open(tif_path) as src:
        data = src.read()  # shape: (bands, height, width)
        height = src.height
        width = src.width
        transform = src.transform

        rows = []

        for r in range(height):
            for c in range(width):
                x, y = xy(transform, r, c)

                pixel = {
                    "row": r,
                    "col": c,
                    "x": float(x),
                    "y": float(y),
                }

                for i, band in enumerate(band_names):
                    pixel[band] = float(data[i, r, c])

                rows.append(pixel)

    df = pd.DataFrame(rows)
    return df
