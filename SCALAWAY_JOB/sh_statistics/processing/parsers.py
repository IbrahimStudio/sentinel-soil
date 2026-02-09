#!/usr/bin/env python3
"""
Response Parsers for Statistics API

Handles parsing of Sentinel Hub Statistical API responses into structured data models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

from ..models import (
    FEATURE_COLS,
    DailyStatsRecord,
    AggregatedStatsRecord,
    DataMaskStats,
    StatisticsResponse
)

def _as_float_or_none(x: Any) -> Optional[float]:
    """
    Convert various types to float, handling special cases

    Args:
        x: Value to convert

    Returns:
        Float value or None if conversion fails or value is invalid
    """
    if x is None:
        return None

    # Sentinel Hub often returns "NaN" as a string in JSON
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("nan", "null", "", "nan"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    # numeric
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None
        return xf
    except Exception:
        return None

def _get_percentile50(outputs: Dict[str, Any], band_key: str) -> Optional[float]:
    """
    Extract 50th percentile value from feature statistics

    Args:
        outputs: Outputs section from Sentinel Hub response
        band_key: Band key (e.g., "B0", "B1", etc.)

    Returns:
        50th percentile value or None if not available
    """
    try:
        value = outputs["features"]["bands"][band_key]["stats"]["percentiles"]["50.0"]
        return _as_float_or_none(value)
    except Exception:
        return None

def _get_datamask_counts(outputs: Dict[str, Any], evalscript_type: str = "features") -> DataMaskStats:
    """
    Extract data mask statistics from response

    Args:
        outputs: Outputs section from Sentinel Hub response
        evalscript_type: Type of evalscript ("features" or "only_scl")

    Returns:
        DataMaskStats object with sample and no-data counts
    """
    try:
        if evalscript_type == "only_scl":
            # For only_scl.js, calculate coverage based on actual feature bands
            # since there's no dataMask band in the response
            sample_count = 0
            no_data_count = 0

            # Check if we have valid data in any of the feature bands
            has_valid_data = False
            for i in range(18):  # Check all 18 feature bands
                try:
                    band_stats = outputs["features"]["bands"][f"B{i}"]["stats"]
                    sample_count = int(band_stats.get("sampleCount", 0) or 0)
                    no_data_count = int(band_stats.get("noDataCount", 0) or 0)
                    if sample_count > 0:
                        has_valid_data = True
                        break
                except Exception:
                    continue

            # If no valid data found in any band, return 0 counts
            if not has_valid_data:
                return DataMaskStats(sample_count=0, no_data_count=0)

            return DataMaskStats(sample_count=sample_count, no_data_count=no_data_count)
        else:
            # For features.js, use dataMask (original behavior)
            stats = outputs.get("dataMask", {}).get("bands", {}).get("B0", {}).get("stats", {}) or {}
            sample = int(stats.get("sampleCount", 0) or 0)
            nodata = int(stats.get("noDataCount", 0) or 0)
            return DataMaskStats(sample_count=sample, no_data_count=nodata)
    except Exception:
        return DataMaskStats(sample_count=0, no_data_count=0)

def parse_statistics_response(sh_json: Dict[str, Any]) -> StatisticsResponse:
    """
    Parse raw Sentinel Hub JSON response into structured model

    Args:
        sh_json: Raw JSON response from Sentinel Hub API

    Returns:
        StatisticsResponse object with parsed data
    """
    response = StatisticsResponse(
        request_id=sh_json.get("requestId"),
        status=sh_json.get("status")
    )

    for item in sh_json.get("data", []):
        interval_obj = item.get("interval", {})
        outputs = item.get("outputs", {})

        interval_stats = DataMaskStats(
            sample_count=0,
            no_data_count=0
        )

        # Parse data mask
        datamask_stats = _get_datamask_counts(outputs)
        interval_stats = datamask_stats

        # Parse features
        features = {}
        for i, name in enumerate(FEATURE_COLS):
            p50_value = _get_percentile50(outputs, f"B{i}")
            features[name] = p50_value

        response.intervals.append({
            "from_time": interval_obj.get("from"),
            "to_time": interval_obj.get("to"),
            "data_mask": datamask_stats,
            "features": features
        })

    return response

def parse_daily_records(
    sh_json: Dict[str, Any],
    *,
    lat: float,
    lon: float,
    bbox: List[float],
    start_date: str,
    end_date: str,
    interval: str,
    evalscript_type: str = "features"  # "features" or "only_scl"
) -> List[DailyStatsRecord]:
    """
    Parse Sentinel Hub response into daily statistics records

    Args:
        sh_json: Raw JSON response from Sentinel Hub API
        lat: Latitude of the point
        lon: Longitude of the point
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
        start_date: Start date of the query
        end_date: End date of the query
        interval: Aggregation interval
        evalscript_type: Type of evalscript ("features" or "only_scl")

    Returns:
        List of DailyStatsRecord objects
    """
    out_rows: List[DailyStatsRecord] = []

    for item in sh_json.get("data", []):
        interval_obj = item.get("interval", {})
        outputs = item.get("outputs", {})

        # Get data mask statistics with evalscript type
        datamask_stats = _get_datamask_counts(outputs, evalscript_type)
        coverage = datamask_stats.coverage

        # Create daily record
        row = DailyStatsRecord(
            lat=lat,
            lon=lon,
            bbox_epsg4326=bbox,
            query_start_date=start_date,
            query_end_date=end_date,
            aggregation_interval=interval,
            from_time=interval_obj.get("from"),
            to_time=interval_obj.get("to"),
            sample_count=datamask_stats.sample_count,
            no_data_count=datamask_stats.no_data_count,
            coverage=coverage,
            p50={}
        )

        # Parse all feature values
        for i, name in enumerate(FEATURE_COLS):
            row.p50[name] = _get_percentile50(outputs, f"B{i}")

        out_rows.append(row)

    return out_rows

def _median(vals: List[Optional[float]]) -> Optional[float]:
    """
    Calculate median of a list of values, ignoring None values

    Args:
        vals: List of float values (may contain None)

    Returns:
        Median value or None if no valid values
    """
    import pandas as pd
    vv = [v for v in vals if v is not None]
    if not vv:
        return None
    return float(pd.Series(vv).median())

def aggregate_records(
    daily_rows: List[DailyStatsRecord],
    *,
    coverage_threshold: float,
) -> Tuple[List[DailyStatsRecord], AggregatedStatsRecord]:
    """
    Aggregate daily records with scene-level filtering

    Args:
        daily_rows: List of daily statistics records
        coverage_threshold: Minimum coverage threshold for keeping records

    Returns:
        Tuple of (kept_daily_rows, aggregated_record)
    """
    total = len(daily_rows)
    kept = [r for r in daily_rows if (r.coverage is not None and r.coverage >= coverage_threshold)]

    # Create base aggregated record from first daily record
    base = daily_rows[0] if total else None
    if not base:
        # Return empty results if no input
        empty_agg = AggregatedStatsRecord(
            lat=0.0,
            lon=0.0,
            bbox_epsg4326=[],
            query_start_date="",
            query_end_date="",
            aggregation_interval="",
            coverage_threshold=coverage_threshold,
            n_days_total=0,
            n_days_kept=0,
            kept_ratio=0.0,
            coverage_median_kept=None,
            coverage_min_kept=None,
            p50_aggregated={}
        )
        return kept, empty_agg

    agg = AggregatedStatsRecord(
        lat=base.lat,
        lon=base.lon,
        bbox_epsg4326=base.bbox_epsg4326,
        query_start_date=base.query_start_date,
        query_end_date=base.query_end_date,
        aggregation_interval=base.aggregation_interval,
        coverage_threshold=coverage_threshold,
        n_days_total=total,
        n_days_kept=len(kept),
        kept_ratio=(len(kept) / total) if total else 0.0,
        coverage_median_kept=_median([r.coverage for r in kept]) if kept else None,
        coverage_min_kept=min([r.coverage for r in kept]) if kept else None,
        p50_aggregated={}
    )

    # Calculate median values for each feature across kept days
    for name in FEATURE_COLS:
        agg.p50_aggregated[name] = _median([r.p50.get(name) for r in kept]) if kept else None

    return kept, agg

# Simple feature columns for raw bands only (B0..B5)
SIMPLE_FEATURE_COLS = [
    "B02", "B03", "B04", "B08", "B11", "B12"              # 0..5 raw bands only
]

def parse_daily_records_simple(
    sh_json: Dict[str, Any],
    *,
    lat: float,
    lon: float,
    bbox: List[float],
    start_date: str,
    end_date: str,
    interval: str,
) -> List[DailyStatsRecord]:
    """
    Parse Sentinel Hub response into daily statistics records (simple 6-band version)

    Args:
        sh_json: Raw JSON response from Sentinel Hub API
        lat: Latitude of the point
        lon: Longitude of the point
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
        start_date: Start date of the query
        end_date: End date of the query
        interval: Aggregation interval

    Returns:
        List of DailyStatsRecord objects
    """
    out_rows: List[DailyStatsRecord] = []

    for item in sh_json.get("data", []):
        interval_obj = item.get("interval", {})
        outputs = item.get("outputs", {})

        # For simple evalscript, calculate coverage based on actual feature bands
        # since there's no dataMask band
        sample_count = 0
        no_data_count = 0

        # Check if we have valid data in any of the feature bands
        has_valid_data = False
        for i, name in enumerate(SIMPLE_FEATURE_COLS):
            try:
                band_stats = outputs["features"]["bands"][f"B{i}"]["stats"]
                sample_count = int(band_stats.get("sampleCount", 0) or 0)
                no_data_count = int(band_stats.get("noDataCount", 0) or 0)
                if sample_count > 0:
                    has_valid_data = True
                    break
            except Exception:
                continue

        # Calculate coverage based on sample vs total pixels
        total_count = sample_count + no_data_count
        coverage = sample_count / total_count if total_count > 0 else 0.0

        # If no valid data found in any band, set coverage to 0
        if not has_valid_data:
            coverage = 0.0

        # Create daily record
        row = DailyStatsRecord(
            lat=lat,
            lon=lon,
            bbox_epsg4326=bbox,
            query_start_date=start_date,
            query_end_date=end_date,
            aggregation_interval=interval,
            from_time=interval_obj.get("from"),
            to_time=interval_obj.get("to"),
            sample_count=sample_count,
            no_data_count=no_data_count,
            coverage=coverage,
            p50={}
        )

        # Parse only the 6 raw bands
        for i, name in enumerate(SIMPLE_FEATURE_COLS):
            row.p50[name] = _get_percentile50(outputs, f"B{i}")

        out_rows.append(row)

    return out_rows

def aggregate_records_simple(
    daily_rows: List[DailyStatsRecord],
    *,
    coverage_threshold: float,
) -> Tuple[List[DailyStatsRecord], AggregatedStatsRecord]:
    """
    Aggregate daily records with scene-level filtering (simple 6-band version)

    Args:
        daily_rows: List of daily statistics records
        coverage_threshold: Minimum coverage threshold for keeping records

    Returns:
        Tuple of (kept_daily_rows, aggregated_record)
    """
    total = len(daily_rows)
    kept = [r for r in daily_rows if (r.coverage is not None and r.coverage >= coverage_threshold)]

    # Create base aggregated record from first daily record
    base = daily_rows[0] if total else None
    if not base:
        # Return empty results if no input
        empty_agg = AggregatedStatsRecord(
            lat=0.0,
            lon=0.0,
            bbox_epsg4326=[],
            query_start_date="",
            query_end_date="",
            aggregation_interval="",
            coverage_threshold=coverage_threshold,
            n_days_total=0,
            n_days_kept=0,
            kept_ratio=0.0,
            coverage_median_kept=None,
            coverage_min_kept=None,
            p50_aggregated={}
        )
        return kept, empty_agg

    agg = AggregatedStatsRecord(
        lat=base.lat,
        lon=base.lon,
        bbox_epsg4326=base.bbox_epsg4326,
        query_start_date=base.query_start_date,
        query_end_date=base.query_end_date,
        aggregation_interval=base.aggregation_interval,
        coverage_threshold=coverage_threshold,
        n_days_total=total,
        n_days_kept=len(kept),
        kept_ratio=(len(kept) / total) if total else 0.0,
        coverage_median_kept=_median([r.coverage for r in kept]) if kept else None,
        coverage_min_kept=min([r.coverage for r in kept]) if kept else None,
        p50_aggregated={}
    )

    # Calculate median values for each of the 6 raw bands across kept days
    for name in SIMPLE_FEATURE_COLS:
        agg.p50_aggregated[name] = _median([r.p50.get(name) for r in kept]) if kept else None

    return kept, agg