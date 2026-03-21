# sentinel_soil/utils/geometry.py
from __future__ import annotations

from typing import Optional, Tuple
from pyproj import Transformer


def utm_epsg_from_latlon(lat: float, lon: float) -> int:
    """
    Compute the correct UTM EPSG for a WGS84 lat/lon point.

    Northern hemisphere: EPSG 32601..32660
    Southern hemisphere: EPSG 32701..32760
    """
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Invalid latitude: {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Invalid longitude: {lon}")

    zone = int((lon + 180.0) // 6.0) + 1  # 1..60
    if zone < 1 or zone > 60:
        raise ValueError(f"Invalid UTM zone computed from lon={lon}: {zone}")

    return (32600 + zone) if lat >= 0 else (32700 + zone)


def bbox_for_grid_around_point(
    *,
    lat: float,
    lon: float,
    width_px: int,
    height_px: int,
    res_m: float = 10.0,
    src_epsg: int = 4326,
    dst_epsg: Optional[int] = None,
) -> Tuple[Tuple[float, float, float, float], int]:
    """
    Build a bbox sized exactly (width_px*res_m) by (height_px*res_m), centered on (lat, lon).

    Returns:
      bbox = (minx, miny, maxx, maxy) in dst_epsg coordinates (meters for UTM)
      epsg = dst_epsg

    This output is directly compatible with:
      BBox(bbox=bbox, crs=CRS(crs_epsg))
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError("width_px and height_px must be > 0")
    if res_m <= 0:
        raise ValueError("res_m must be > 0")

    # Optional but recommended if you want the point to sit on the center pixel
    # (odd dimensions => clear center pixel)
    # If you want to allow even windows, just remove these two checks.
    if width_px % 2 == 0 or height_px % 2 == 0:
        raise ValueError("width_px and height_px should be odd to center the window on the point")

    if dst_epsg is None:
        dst_epsg = utm_epsg_from_latlon(lat, lon)

    transformer = Transformer.from_crs(
        f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True
    )
    x, y = transformer.transform(lon, lat)

    half_w = (width_px * res_m) / 2.0
    half_h = (height_px * res_m) / 2.0

    bbox = (x - half_w, y - half_h, x + half_w, y + half_h)
    return bbox, dst_epsg
