#!/usr/bin/env python3
"""
Data Models and DTOs for Statistics API

Defines structured data models for requests, responses, and processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

# Feature columns mapping (B0..B17)
FEATURE_COLS = [
    "B02", "B03", "B04", "B08", "B11", "B12",             # 0..5 raw bands
    "NDVI", "NDWI", "MNDWI", "NDMI", "BSI",               # 6..10 indices
    "BRIGHT", "ALBEDO_PROXY",                             # 11..12 brightness
    "RED", "SWIR1", "SWIR2",                              # 13..15 raw bands
    "RED_SWIR1_RATIO", "SWIR1_SWIR2_RATIO"                # 16..17 ratios
]

@dataclass
class BoundingBox:
    """Geographic bounding box"""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def to_list(self) -> List[float]:
        """Convert to list format [min_lon, min_lat, max_lon, max_lat]"""
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]

    @classmethod
    def from_list(cls, bbox_list: List[float]) -> BoundingBox:
        """Create from list format"""
        if len(bbox_list) != 4:
            raise ValueError("Bounding box must have exactly 4 elements")
        return cls(
            min_lon=bbox_list[0],
            min_lat=bbox_list[1],
            max_lon=bbox_list[2],
            max_lat=bbox_list[3]
        )

@dataclass
class TimeWindow:
    """Time window for statistics request"""
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD

@dataclass
class StatisticsRequest:
    """Complete statistics API request parameters"""
    bbox: BoundingBox
    time_window: TimeWindow
    interval: str = "P1D"  # ISO 8601 duration format
    evalscript: str = ""
    resolution: int = 20
    mosaicking_order: str = "leastCC"

@dataclass
class DataMaskStats:
    """Data mask statistics"""
    sample_count: int
    no_data_count: int

    @property
    def total_count(self) -> int:
        """Total pixel count"""
        return self.sample_count + self.no_data_count

    @property
    def coverage(self) -> float:
        """Coverage ratio (0.0 to 1.0)"""
        if self.total_count == 0:
            return 0.0
        return self.sample_count / self.total_count

@dataclass
class FeatureStats:
    """Statistics for a single feature/band"""
    p50: Optional[float] = None  # 50th percentile (median)

@dataclass
class IntervalStats:
    """Statistics for a single time interval"""
    from_time: Optional[str] = None
    to_time: Optional[str] = None
    data_mask: Optional[DataMaskStats] = None
    features: Dict[str, FeatureStats] = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}

@dataclass
class StatisticsResponse:
    """Complete statistics API response"""
    request_id: Optional[str] = None
    status: Optional[str] = None
    intervals: List[IntervalStats] = None

    def __post_init__(self):
        if self.intervals is None:
            self.intervals = []

@dataclass
class DailyStatsRecord:
    """Parsed daily statistics record with metadata"""
    lat: float
    lon: float
    bbox_epsg4326: List[float]
    query_start_date: str
    query_end_date: str
    aggregation_interval: str
    from_time: Optional[str] = None
    to_time: Optional[str] = None
    sample_count: int = 0
    no_data_count: int = 0
    coverage: float = 0.0
    p50: Dict[str, Optional[float]] = None  # feature_name -> p50 value

    def __post_init__(self):
        if self.p50 is None:
            self.p50 = {}

@dataclass
class AggregatedStatsRecord:
    """Aggregated statistics across multiple days"""
    lat: float
    lon: float
    bbox_epsg4326: List[float]
    query_start_date: str
    query_end_date: str
    aggregation_interval: str
    coverage_threshold: float
    n_days_total: int
    n_days_kept: int
    kept_ratio: float
    coverage_median_kept: Optional[float] = None
    coverage_min_kept: Optional[float] = None
    p50_aggregated: Dict[str, Optional[float]] = None  # feature_name -> median p50

    def __post_init__(self):
        if self.p50_aggregated is None:
            self.p50_aggregated = {}

@dataclass
class JobSpec:
    """Job specification for batch processing"""
    job_id: str
    point_id: str
    lat: float
    lon: float
    start_date: str
    end_date: str
    bbox: BoundingBox
    # Filtering thresholds
    ndvi_threshold: float = 0.2
    mndwi_threshold: float = 0.0
    sun_zenith_threshold: float = 70.0
    coverage_threshold: float = 0.8
    scl_exclude_classes: List[int] = None

    def __post_init__(self):
        if self.scl_exclude_classes is None:
            # Default: exclude clouds, shadows, water, snow/ice
            self.scl_exclude_classes = [3, 6, 8, 9, 10, 11]

@dataclass
class JobResult:
    """Result of a single job execution"""
    status: str  # "SUCCESS" or "FAILED"
    job_id: str
    point_id: str
    error: Optional[str] = None
    raw_path: Optional[str] = None
    daily_rows: List[DailyStatsRecord] = None
    kept_rows: List[DailyStatsRecord] = None
    aggregated: Optional[AggregatedStatsRecord] = None

    def __post_init__(self):
        if self.daily_rows is None:
            self.daily_rows = []
        if self.kept_rows is None:
            self.kept_rows = []

def bbox_around_point_m(lat: float, lon: float, size_m: float) -> BoundingBox:
    """
    Create a bounding box around a point with given size in meters

    ⚠️  IMPORTANT: This function creates a bounding box in EPSG:4326 (degrees)
    that approximates the requested meter size. However, when used with
    Sentinel Hub Statistics API, the resolution (resx/resy) is interpreted
    in the units of the CRS, not meters.

    For true meter-based analysis, consider:
    1. Using EPSG:3857 (Web Mercator) with proper coordinate transformation
    2. Using create_meter_based_request() for proper meter-based requests

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        size_m: Size of square bounding box in meters

    Returns:
        BoundingBox centered on the point in EPSG:4326 (degrees)

    Example:
        # For a 30m x 30m area (will create ~0.00027° x 0.00027° bbox)
        bbox = bbox_around_point_m(45.0, 10.0, 30.0)

        # But with resx=resy=10, Sentinel Hub interprets this as 10 DEGREES per pixel!
        # This mismatch causes zero-pixel responses.
    """
    half = size_m / 2.0
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    dlat = half / meters_per_deg_lat
    dlon = half / meters_per_deg_lon

    return BoundingBox(
        min_lon=lon - dlon,
        min_lat=lat - dlat,
        max_lon=lon + dlon,
        max_lat=lat + dlat
    )

def create_meter_based_request_config(
    lat: float,
    lon: float,
    size_m: float,
    resolution_m: int = 10
) -> Dict[str, Any]:
    """
    Create a proper meter-based request configuration for Sentinel Hub Statistics API

    This function addresses the fundamental CRS/resolution mismatch by:
    1. Using EPSG:3857 (Web Mercator) for true meter-based coordinates
    2. Properly setting resolution in meters
    3. Calculating the correct pixel grid

    Args:
        lat: Latitude in decimal degrees (EPSG:4326)
        lon: Longitude in decimal degrees (EPSG:4326)
        size_m: Size of square area of interest in meters
        resolution_m: Desired resolution in meters (e.g., 10 for 10m pixels)

    Returns:
        Dictionary with complete request configuration including:
        - bbox: Bounding box in EPSG:3857 (meters)
        - resx/resy: Resolution in meters
        - pixel_grid: Expected pixel dimensions
        - crs: Coordinate reference system

    Example:
        # For 30m x 30m area with 10m pixels (3x3 grid)
        config = create_meter_based_request_config(45.0, 10.0, 30.0, 10)
        # Returns: bbox in EPSG:3857, resx=resy=10, expected 3x3 pixels
    """
    try:
        import pyproj
    except ImportError:
        raise ImportError(
            "pyproj library required for meter-based requests. "
            "Install with: pip install pyproj"
        )

    # Convert WGS84 (EPSG:4326) to Web Mercator (EPSG:3857)
    wgs84 = pyproj.CRS("EPSG:4326")
    web_mercator = pyproj.CRS("EPSG:3857")
    transformer = pyproj.Transformer.from_crs(wgs84, web_mercator)

    # Convert center point to EPSG:3857
    x_center, y_center = transformer.transform(lat, lon)

    # Calculate bounding box in meters (EPSG:3857)
    half_size = size_m / 2.0
    min_x = x_center - half_size
    min_y = y_center - half_size
    max_x = x_center + half_size
    max_y = y_center + half_size

    # Calculate expected pixel grid
    pixels_per_side = size_m / resolution_m
    total_pixels = pixels_per_side ** 2

    return {
        "bbox": [min_x, min_y, max_x, max_y],
        "crs": "http://www.opengis.net/def/crs/EPSG/0/3857",
        "resx": resolution_m,
        "resy": resolution_m,
        "pixel_grid": {
            "width": pixels_per_side,
            "height": pixels_per_side,
            "total": total_pixels
        },
        "size_m": size_m,
        "resolution_m": resolution_m,
        "notes": (
            f"Configured for {size_m}m × {size_m}m area with {resolution_m}m pixels. "
            f"Expected: {pixels_per_side:.1f} × {pixels_per_side:.1f} = {total_pixels:.0f} pixels."
        )
    }

def calculate_pixel_config_for_epsg4326(
    lat: float,
    size_m: float,
    resolution_m: int = 10
) -> Dict[str, Any]:
    """
    Calculate the equivalent pixel-based configuration for EPSG:4326 requests

    Since Sentinel Hub interprets resx/resy in degrees for EPSG:4326,
    this function helps you understand the relationship between meters and degrees.

    Args:
        lat: Latitude for calculation (affects longitude conversion)
        size_m: Desired area size in meters
        resolution_m: Desired resolution in meters

    Returns:
        Dictionary with degree-based equivalents and warnings

    Example:
        # Understand what 10m resolution means in degrees at 45° latitude
        config = calculate_pixel_config_for_epsg4326(45.0, 30.0, 10)
    """
    # Calculate degrees per meter at this latitude
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = 111_320.0 * math.cos(math.radians(lat))

    # Convert meters to degrees
    size_deg_lat = size_m / meters_per_degree_lat
    size_deg_lon = size_m / meters_per_degree_lon
    resolution_deg_lat = resolution_m / meters_per_degree_lat
    resolution_deg_lon = resolution_m / meters_per_degree_lon

    # Calculate pixel dimensions
    pixels_lat = size_deg_lat / resolution_deg_lat
    pixels_lon = size_deg_lon / resolution_deg_lon

    # Warnings
    warnings = []
    if pixels_lat < 3 or pixels_lon < 3:
        warnings.append(
            f"Very small pixel grid: {pixels_lat:.1f} × {pixels_lon:.1f} pixels. "
            "This will likely result in zero valid pixels after filtering."
        )
    elif pixels_lat < 10 or pixels_lon < 10:
        warnings.append(
            f"Small pixel grid: {pixels_lat:.1f} × {pixels_lon:.1f} pixels. "
            "Consider larger area or coarser resolution for better statistics."
        )

    return {
        "size_m": size_m,
        "resolution_m": resolution_m,
        "equivalent_degrees": {
            "size_deg_lat": size_deg_lat,
            "size_deg_lon": size_deg_lon,
            "resolution_deg_lat": resolution_deg_lat,
            "resolution_deg_lon": resolution_deg_lon
        },
        "pixel_grid": {
            "lat_pixels": pixels_lat,
            "lon_pixels": pixels_lon,
            "total_pixels": pixels_lat * pixels_lon
        },
        "warnings": warnings,
        "recommendation": (
            "For reliable results, consider using EPSG:3857 with "
            "create_meter_based_request_config() for true meter-based analysis."
        )
    }

def parse_date(d: str) -> str:
    """
    Parse and validate date string

    Args:
        d: Date string in YYYY-MM-DD format

    Returns:
        Validated date string

    Raises:
        ValueError: If date format is invalid
    """
    pd.to_datetime(d, format="%Y-%m-%d")
    return d