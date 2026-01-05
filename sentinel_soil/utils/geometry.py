from pyproj import Transformer
from typing import Tuple


def bbox_around_point(
    lat: float,
    lon: float,
    size_m: float,
    src_epsg: int = 4326,
    dst_epsg: int = 32632,
) -> Tuple[Tuple[float, float, float, float], int]:
    """
    Build a square bbox of size_m x size_m meters around a lat/lon point.

    Returns:
      (minx, miny, maxx, maxy), dst_epsg
    """

    transformer = Transformer.from_crs(
        f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True
    )

    x, y = transformer.transform(lon, lat)

    half = size_m / 2.0

    bbox = (
        x - half,
        y - half,
        x + half,
        y + half,
    )

    return bbox, dst_epsg
