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

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        size_m: Size of square bounding box in meters

    Returns:
        BoundingBox centered on the point
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