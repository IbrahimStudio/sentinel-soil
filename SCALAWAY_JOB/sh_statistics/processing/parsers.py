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


def get_coverage_from_outputs(outputs: dict) -> float:
    stats = outputs.get("valid", {}).get("bands", {}).get("B0", {}).get("stats", {}) or {}
    return float(stats.get("mean", 0.0) or 0.0)


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
            # ✅ FIXED: Accumulate counts across ALL bands, not just the first one with data
            total_sample_count = 0
            total_no_data_count = 0

            # Accumulate counts across all feature bands
            for i in range(18):  # Check all 18 feature bands
                try:
                    band_stats = outputs["features"]["bands"][f"B{i}"]["stats"]
                    sample_count = int(band_stats.get("sampleCount", 0) or 0)
                    no_data_count = int(band_stats.get("noDataCount", 0) or 0)
                    total_sample_count += sample_count
                    total_no_data_count += no_data_count
                except Exception:
                    continue

            # Return accumulated counts
            return DataMaskStats(sample_count=total_sample_count, no_data_count=total_no_data_count)
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


def _safe_get(d: Dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

def _get_band_stats(outputs: Dict[str, Any], output_id: str, band_id: str) -> Dict[str, Any]:
    stats = _safe_get(outputs, output_id, "bands", band_id, "stats")
    return stats if isinstance(stats, dict) else {}

def _get_stat(outputs: Dict[str, Any], output_id: str, band_id: str, stat_name: str) -> Optional[float]:
    v = _safe_get(outputs, output_id, "bands", band_id, "stats", stat_name)
    return float(v) if isinstance(v, (int, float)) else None

def _get_percentile(outputs: Dict[str, Any], output_id: str, band_id: str, k: int) -> Optional[float]:
    pct = _safe_get(outputs, output_id, "bands", band_id, "stats", "percentiles")
    if not isinstance(pct, dict):
        return None

    # Sentinel Hub often uses "50.0" keys (strings). Be robust.
    candidates = [str(k), f"{k}.0", f"{float(k):.1f}"]
    for key in candidates:
        if key in pct and isinstance(pct[key], (int, float)):
            return float(pct[key])
    return None

def _set_if_attr(obj: Any, attr: str, value: Any) -> bool:
    if hasattr(obj, attr):
        setattr(obj, attr, value)
        return True
    return False

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
) -> List["DailyStatsRecord"]:
    """
    Parse Sentinel Hub response into daily statistics records.

    Backward compatible:
    - Always populates row.p50[name]
    - coverage computed from valid.mean
    - Extra stats added if DailyStatsRecord supports them, else stored in extra_stats when available.
    """
    out_rows: List["DailyStatsRecord"] = []

    for item in sh_json.get("data", []):
        interval_obj = item.get("interval", {})
        outputs = item.get("outputs", {})

        # Coverage: mean(valid) in [0..1]
        # (valid is UINT8 0/1; mean is coverage)
        coverage = _get_stat(outputs, "valid", "B0", "mean")
        if coverage is None:
            # fallback to your existing helper if present in your codebase
            try:
                coverage = get_coverage_from_outputs(outputs)  # type: ignore[name-defined]
            except Exception:
                coverage = 0.0

        row = DailyStatsRecord(
            lat=lat,
            lon=lon,
            bbox_epsg4326=bbox,
            query_start_date=start_date,
            query_end_date=end_date,
            aggregation_interval=interval,
            from_time=interval_obj.get("from"),
            to_time=interval_obj.get("to"),
            coverage=float(coverage),
            p50={}
        )

        # Prepare containers for richer stats
        p10: Dict[str, Optional[float]] = {}
        p25: Dict[str, Optional[float]] = {}
        p75: Dict[str, Optional[float]] = {}
        p90: Dict[str, Optional[float]] = {}
        mean: Dict[str, Optional[float]] = {}
        stdev: Dict[str, Optional[float]] = {}
        vmin: Dict[str, Optional[float]] = {}
        vmax: Dict[str, Optional[float]] = {}

        # Optional daily qc counters (these are per-output; same for all bands)
        features_sample_count = _get_stat(outputs, "features", "B0", "sampleCount")
        features_no_data_count = _get_stat(outputs, "features", "B0", "noDataCount")
        valid_sample_count = _get_stat(outputs, "valid", "B0", "sampleCount")

        # Parse all feature values (bands are named B0..B{n-1})
        for i, name in enumerate(FEATURE_COLS):
            band_id = f"B{i}"

            # Keep existing behavior: p50 always present
            row.p50[name] = _get_percentile(outputs, "features", band_id, 50)

            # Extra reducers
            mean[name] = _get_stat(outputs, "features", band_id, "mean")
            stdev[name] = _get_stat(outputs, "features", band_id, "stDev")
            vmin[name] = _get_stat(outputs, "features", band_id, "min")
            vmax[name] = _get_stat(outputs, "features", band_id, "max")

            p10[name] = _get_percentile(outputs, "features", band_id, 10)
            p25[name] = _get_percentile(outputs, "features", band_id, 25)
            p75[name] = _get_percentile(outputs, "features", band_id, 75)
            p90[name] = _get_percentile(outputs, "features", band_id, 90)

        # Attach extra stats in a backward-compatible way
        attached = False
        attached |= _set_if_attr(row, "mean", mean)
        attached |= _set_if_attr(row, "stdev", stdev)
        attached |= _set_if_attr(row, "min", vmin)
        attached |= _set_if_attr(row, "max", vmax)
        attached |= _set_if_attr(row, "p10", p10)
        attached |= _set_if_attr(row, "p25", p25)
        attached |= _set_if_attr(row, "p75", p75)
        attached |= _set_if_attr(row, "p90", p90)

        # QC counters (optional)
        attached |= _set_if_attr(row, "features_sample_count", features_sample_count)
        attached |= _set_if_attr(row, "features_no_data_count", features_no_data_count)
        attached |= _set_if_attr(row, "valid_sample_count", valid_sample_count)

        if not attached:
            # If your dataclass doesn't have the above fields, store everything in a single dict if possible
            extra = {
                "mean": mean,
                "stdev": stdev,
                "min": vmin,
                "max": vmax,
                "p10": p10,
                "p25": p25,
                "p75": p75,
                "p90": p90,
                "features_sample_count": features_sample_count,
                "features_no_data_count": features_no_data_count,
                "valid_sample_count": valid_sample_count,
            }
            _set_if_attr(row, "extra_stats", extra)

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

from typing import Dict, List, Optional, Tuple

def _as_float_or_none(x: object) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None

def _median(vals: List[Optional[float]]) -> Optional[float]:
    clean = sorted([v for v in vals if isinstance(v, (int, float)) and v == v])
    if not clean:
        return None
    n = len(clean)
    mid = n // 2
    if n % 2 == 1:
        return float(clean[mid])
    return float((clean[mid - 1] + clean[mid]) / 2.0)

def _min(vals: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if isinstance(v, (int, float)) and v == v]
    return float(min(clean)) if clean else None

def _max(vals: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if isinstance(v, (int, float)) and v == v]
    return float(max(clean)) if clean else None

def _mean(vals: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if isinstance(v, (int, float)) and v == v]
    return float(sum(clean) / len(clean)) if clean else None

def _iqr(vals: List[Optional[float]]) -> Optional[float]:
    # IQR = P75 - P25, using percentiles already computed per-day if available
    # (this helper is for aggregating daily scalars; when using it, pass [p75-p25] per day)
    clean = [v for v in vals if isinstance(v, (int, float)) and v == v]
    return float(_median(clean)) if clean else None  # not used directly; see below

def _get_dict_attr(obj: object, attr: str) -> Optional[Dict[str, Optional[float]]]:
    d = getattr(obj, attr, None)
    return d if isinstance(d, dict) else None

def aggregate_records(
    daily_rows: List["DailyStatsRecord"],
    *,
    coverage_threshold: float,
) -> Tuple[List["DailyStatsRecord"], "AggregatedStatsRecord"]:
    """
    Aggregate daily records with scene-level filtering.

    Keeps backward compatibility:
    - Still produces agg.p50_aggregated (median of daily p50 across kept days)
    Adds richer aggregated summaries if AggregatedStatsRecord supports them:
    - mean/min/max/median for (daily mean, daily stdev, daily p10/p25/p75/p90)
    - robust variability proxies: median daily stdev; median daily IQR (p75 - p25)
    """
    total = len(daily_rows)
    kept = [r for r in daily_rows if (r.coverage is not None and r.coverage >= coverage_threshold)]

    base = daily_rows[0] if total else None
    if not base:
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
        coverage_min_kept=_min([r.coverage for r in kept]) if kept else None,
        p50_aggregated={}
    )

    # --- Backward-compatible aggregation: median of daily p50 across kept days
    for name in FEATURE_COLS:
        agg.p50_aggregated[name] = _median([_as_float_or_none(r.p50.get(name)) for r in kept]) if kept else None

    # --- New: aggregate additional per-day reducers if present on DailyStatsRecord
    daily_mean = _get_dict_attr(base, "mean")
    daily_stdev = _get_dict_attr(base, "stdev")
    daily_min = _get_dict_attr(base, "min")
    daily_max = _get_dict_attr(base, "max")
    daily_p10 = _get_dict_attr(base, "p10")
    daily_p25 = _get_dict_attr(base, "p25")
    daily_p75 = _get_dict_attr(base, "p75")
    daily_p90 = _get_dict_attr(base, "p90")

    has_extra = any([daily_mean, daily_stdev, daily_min, daily_max, daily_p10, daily_p25, daily_p75, daily_p90])

    if has_extra and kept:
        # Compute daily IQR (p75 - p25) per feature if available
        daily_iqr_vals: Dict[str, Optional[float]] = {}
        if daily_p25 and daily_p75:
            for name in FEATURE_COLS:
                iqr_list: List[Optional[float]] = []
                for r in kept:
                    p25d = _get_dict_attr(r, "p25")
                    p75d = _get_dict_attr(r, "p75")
                    if not p25d or not p75d:
                        continue
                    a = _as_float_or_none(p25d.get(name))
                    b = _as_float_or_none(p75d.get(name))
                    iqr_list.append((b - a) if (a is not None and b is not None) else None)
                daily_iqr_vals[name] = _median(iqr_list)

        # Helper to build aggregated dicts
        def agg_from_daily_dict(attr: str, reducer) -> Dict[str, Optional[float]]:
            out: Dict[str, Optional[float]] = {}
            for name in FEATURE_COLS:
                vals: List[Optional[float]] = []
                for r in kept:
                    d = _get_dict_attr(r, attr)
                    if not d:
                        continue
                    vals.append(_as_float_or_none(d.get(name)))
                out[name] = reducer(vals)
            return out

        extra_payload = {
            # central tendency of daily reducers
            "mean_median": agg_from_daily_dict("mean", _median) if daily_mean else None,
            "mean_mean":   agg_from_daily_dict("mean", _mean)   if daily_mean else None,
            "stdev_median": agg_from_daily_dict("stdev", _median) if daily_stdev else None,

            # range/robust range across kept days
            "mean_min": agg_from_daily_dict("mean", _min) if daily_mean else None,
            "mean_max": agg_from_daily_dict("mean", _max) if daily_mean else None,

            # aggregated percentiles (median across days of daily percentiles)
            "p10_aggregated": agg_from_daily_dict("p10", _median) if daily_p10 else None,
            "p25_aggregated": agg_from_daily_dict("p25", _median) if daily_p25 else None,
            "p75_aggregated": agg_from_daily_dict("p75", _median) if daily_p75 else None,
            "p90_aggregated": agg_from_daily_dict("p90", _median) if daily_p90 else None,

            # daily IQR as a robust variability proxy (median across days)
            "iqr_median": daily_iqr_vals if daily_iqr_vals else None,
        }

        # Attach to AggregatedStatsRecord in a compatibility-safe way:
        # - if fields exist, set them
        # - else stash into `extra_stats` if present
        attached_any = False
        for k, v in extra_payload.items():
            if v is None:
                continue
            if hasattr(agg, k):
                setattr(agg, k, v)
                attached_any = True

        if not attached_any and hasattr(agg, "extra_stats"):
            # keep it structured
            agg.extra_stats = extra_payload  # type: ignore[attr-defined]

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