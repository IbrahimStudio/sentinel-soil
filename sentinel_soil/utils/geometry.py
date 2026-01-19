from pyproj import Transformer
from typing import Tuple

def bbox_for_grid_around_point(
    *,
    lat: float,
    lon: float,
    width_px: int,
    height_px: int,
    res_m: float = 10.0,
    src_epsg: int = 4326,
    dst_epsg: int = 32632,
) -> Tuple[Tuple[float, float, float, float], int]:
    """
    Build a bbox sized exactly to (width_px * res_m) by (height_px * res_m),
    centered on (lat, lon), in dst_epsg meters.
    """
    transformer = Transformer.from_crs(
        f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True
    )
    x, y = transformer.transform(lon, lat)

    half_w = (width_px * res_m) / 2.0
    half_h = (height_px * res_m) / 2.0

    bbox = (x - half_w, y - half_h, x + half_w, y + half_h)
    return bbox, dst_epsg

def grid_for_n_pixels(n: int) -> Tuple[int, int]:
    """
    Minimal grid that contains at least n pixels.
    For n=1 -> (1,1)
    For n=2 -> (2,2) recommended (gives 4, then you select 2)
    """
    if n <= 1:
        return (1, 1)
    if n == 2:
        return (3, 3)   # robust nearest selection
    # generic fallback: square-ish grid
    import math
    s = math.ceil(math.sqrt(n))
    return (s, s)