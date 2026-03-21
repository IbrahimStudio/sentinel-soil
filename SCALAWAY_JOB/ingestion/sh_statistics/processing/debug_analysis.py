#!/usr/bin/env python3
"""
Data Loss Analysis Tools for Statistics API

Provides utilities to analyze where data is being lost in the filtering pipeline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..models import FEATURE_COLS, DailyStatsRecord

# Extended feature list for debug version
DEBUG_FEATURE_COLS = FEATURE_COLS + [
    "DEBUG_SCL_PASS",    # 1.0 if passed SCL filter, 0.0 if failed
    "DEBUG_SZA_PASS",    # 1.0 if passed SZA filter, 0.0 if failed
    "DEBUG_NDVI_PASS",   # 1.0 if passed NDVI filter, 0.0 if failed
    "DEBUG_MNDWI_PASS"   # 1.0 if passed MNDWI filter, 0.0 if failed
]

@dataclass
class FilterStageStats:
    """Statistics for a single filter stage"""
    stage_name: str
    pixels_passed: int
    pixels_failed: int
    pass_rate: float

    @property
    def total_pixels(self) -> int:
        return self.pixels_passed + self.pixels_failed

@dataclass
class DailyFilterAnalysis:
    """Filter analysis for a single day"""
    date: str
    total_pixels: int
    valid_pixels: int
    coverage: float
    filter_stages: List[FilterStageStats]
    kept_for_aggregation: bool
    rejection_reason: Optional[str] = None

@dataclass
class DataLossReport:
    """Complete data loss analysis report"""
    point_id: str
    lat: float
    lon: float
    total_days: int
    days_with_data: int
    days_rejected: int
    daily_analyses: List[DailyFilterAnalysis]
    overall_filter_efficiency: Dict[str, FilterStageStats]

def analyze_data_loss_from_raw_response(
    raw_response: Dict[str, Any],
    *,
    lat: float,
    lon: float,
    point_id: str,
    coverage_threshold: float = 0.8,
    use_debug_evalscript: bool = False
) -> DataLossReport:
    """
    Analyze data loss from raw Sentinel Hub API response

    Args:
        raw_response: Raw JSON response from Sentinel Hub API
        lat: Latitude of the point
        lon: Longitude of the point
        point_id: Point identifier
        coverage_threshold: Coverage threshold used for day rejection
        use_debug_evalscript: Whether debug evalscript was used (22 bands vs 18)

    Returns:
        DataLossReport with detailed analysis
    """
    feature_cols = DEBUG_FEATURE_COLS if use_debug_evalscript else FEATURE_COLS
    daily_analyses = []

    # Initialize overall filter stats
    overall_filters = {
        "SCL": FilterStageStats("SCL", 0, 0, 0.0),
        "SZA": FilterStageStats("SZA", 0, 0, 0.0),
        "NDVI": FilterStageStats("NDVI", 0, 0, 0.0),
        "MNDWI": FilterStageStats("MNDWI", 0, 0, 0.0)
    }

    for item in raw_response.get("data", []):
        interval_obj = item.get("interval", {})
        outputs = item.get("outputs", {})
        date = interval_obj.get("from", "unknown")

        # Get data mask statistics
        try:
            stats = outputs.get("dataMask", {}).get("bands", {}).get("B0", {}).get("stats", {}) or {}
            sample_count = int(stats.get("sampleCount", 0) or 0)
            no_data_count = int(stats.get("noDataCount", 0) or 0)
            total_pixels = sample_count + no_data_count
            coverage = (sample_count / total_pixels) if total_pixels > 0 else 0.0
        except Exception:
            total_pixels = 0
            sample_count = 0
            coverage = 0.0

        # Parse debug information if available
        filter_stages = []
        debug_data_available = False

        if use_debug_evalscript and total_pixels > 0:
            debug_data_available = True
            try:
                # Parse debug bands (last 4 bands)
                debug_bands = outputs.get("features", {}).get("bands", {})
                scl_pass = debug_bands.get("B18", {}).get("stats", {}).get("percentiles", {}).get("50.0", 0.0)
                sza_pass = debug_bands.get("B19", {}).get("stats", {}).get("percentiles", {}).get("50.0", 0.0)
                ndvi_pass = debug_bands.get("B20", {}).get("stats", {}).get("percentiles", {}).get("50.0", 0.0)
                mndwi_pass = debug_bands.get("B21", {}).get("stats", {}).get("percentiles", {}).get("50.0", 0.0)

                # Convert to pixel counts (assuming median represents typical pixel)
                scl_pass_pixels = int(scl_pass * total_pixels)
                sza_pass_pixels = int(sza_pass * total_pixels)
                ndvi_pass_pixels = int(ndvi_pass * total_pixels)
                mndwi_pass_pixels = int(mndwi_pass * total_pixels)

                # Calculate filter stage stats
                filter_stages = [
                    FilterStageStats("SCL", scl_pass_pixels, total_pixels - scl_pass_pixels,
                                   scl_pass_pixels / total_pixels if total_pixels > 0 else 0.0),
                    FilterStageStats("SZA", sza_pass_pixels, scl_pass_pixels - sza_pass_pixels,
                                   sza_pass_pixels / scl_pass_pixels if scl_pass_pixels > 0 else 0.0),
                    FilterStageStats("NDVI", ndvi_pass_pixels, sza_pass_pixels - ndvi_pass_pixels,
                                   ndvi_pass_pixels / sza_pass_pixels if sza_pass_pixels > 0 else 0.0),
                    FilterStageStats("MNDWI", mndwi_pass_pixels, ndvi_pass_pixels - mndwi_pass_pixels,
                                   mndwi_pass_pixels / ndvi_pass_pixels if ndvi_pass_pixels > 0 else 0.0)
                ]

                # Update overall stats
                overall_filters["SCL"].pixels_passed += scl_pass_pixels
                overall_filters["SCL"].pixels_failed += (total_pixels - scl_pass_pixels)
                overall_filters["SZA"].pixels_passed += sza_pass_pixels
                overall_filters["SZA"].pixels_failed += (scl_pass_pixels - sza_pass_pixels)
                overall_filters["NDVI"].pixels_passed += ndvi_pass_pixels
                overall_filters["NDVI"].pixels_failed += (sza_pass_pixels - ndvi_pass_pixels)
                overall_filters["MNDWI"].pixels_passed += mndwi_pass_pixels
                overall_filters["MNDWI"].pixels_failed += (ndvi_pass_pixels - mndwi_pass_pixels)

            except Exception:
                debug_data_available = False

        # If no debug data, create basic filter stages
        if not debug_data_available:
            filter_stages = [
                FilterStageStats("SCL", sample_count, no_data_count, coverage),
                FilterStageStats("SZA", sample_count, no_data_count, coverage),
                FilterStageStats("NDVI", sample_count, no_data_count, coverage),
                FilterStageStats("MNDWI", sample_count, no_data_count, coverage)
            ]

        # Determine if day was kept for aggregation
        kept_for_aggregation = coverage >= coverage_threshold
        rejection_reason = None
        if not kept_for_aggregation:
            rejection_reason = f"Coverage {coverage:.3f} < threshold {coverage_threshold}"

        daily_analysis = DailyFilterAnalysis(
            date=date,
            total_pixels=total_pixels,
            valid_pixels=sample_count,
            coverage=coverage,
            filter_stages=filter_stages,
            kept_for_aggregation=kept_for_aggregation,
            rejection_reason=rejection_reason
        )

        daily_analyses.append(daily_analysis)

    # Calculate overall pass rates
    for stage_name, stats in overall_filters.items():
        if stats.total_pixels > 0:
            stats.pass_rate = stats.pixels_passed / (stats.pixels_passed + stats.pixels_failed)

    # Count days with data vs rejected
    days_with_data = sum(1 for day in daily_analyses if day.valid_pixels > 0)
    days_rejected = sum(1 for day in daily_analyses if not day.kept_for_aggregation and day.valid_pixels > 0)

    return DataLossReport(
        point_id=point_id,
        lat=lat,
        lon=lon,
        total_days=len(daily_analyses),
        days_with_data=days_with_data,
        days_rejected=days_rejected,
        daily_analyses=daily_analyses,
        overall_filter_efficiency=overall_filters
    )

def generate_data_loss_report(
    report: DataLossReport,
    *,
    output_dir: Optional[Path] = None,
    include_detailed_daily: bool = True
) -> Dict[str, Any]:
    """
    Generate a comprehensive data loss report

    Args:
        report: DataLossReport object
        output_dir: Optional directory to save report files
        include_detailed_daily: Whether to include detailed daily analysis

    Returns:
        Dictionary with report data
    """
    # Create summary
    summary = {
        "point_id": report.point_id,
        "location": {"lat": report.lat, "lon": report.lon},
        "total_days_analyzed": report.total_days,
        "days_with_valid_data": report.days_with_data,
        "days_rejected_by_coverage": report.days_rejected,
        "data_recovery_rate": report.days_with_data / report.total_days if report.total_days > 0 else 0.0,
        "overall_filter_efficiency": {
            stage_name: {
                "pixels_passed": stats.pixels_passed,
                "pixels_failed": stats.pixels_failed,
                "pass_rate": stats.pass_rate,
                "total_pixels": stats.total_pixels
            }
            for stage_name, stats in report.overall_filter_efficiency.items()
        }
    }

    # Add detailed analysis if requested
    if include_detailed_daily:
        summary["daily_analysis"] = []
        for day in report.daily_analyses:
            day_summary = {
                "date": day.date,
                "total_pixels": day.total_pixels,
                "valid_pixels": day.valid_pixels,
                "coverage": day.coverage,
                "kept_for_aggregation": day.kept_for_aggregation,
                "filter_stages": {
                    stage.stage_name: {
                        "pixels_passed": stage.pixels_passed,
                        "pixels_failed": stage.pixels_failed,
                        "pass_rate": stage.pass_rate
                    }
                    for stage in day.filter_stages
                }
            }
            if day.rejection_reason:
                day_summary["rejection_reason"] = day.rejection_reason
            summary["daily_analysis"].append(day_summary)

    # Save to file if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"data_loss_report_{report.point_id}.json"
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary

def analyze_data_loss_from_files(
    raw_response_files: List[Path],
    *,
    coverage_threshold: float = 0.8,
    use_debug_evalscript: bool = False,
    output_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Analyze data loss from multiple raw response files

    Args:
        raw_response_files: List of paths to raw response JSON files
        coverage_threshold: Coverage threshold used
        use_debug_evalscript: Whether debug evalscript was used
        output_dir: Optional directory to save reports

    Returns:
        List of data loss reports
    """
    reports = []

    for file_path in raw_response_files:
        try:
            raw_response = json.loads(file_path.read_text(encoding="utf-8"))

            # Extract point info from filename (format: point_id__job_id.json)
            point_id = file_path.stem.split("__")[0]

            # Get lat/lon from response metadata or use defaults
            lat = raw_response.get("metadata", {}).get("lat", 0.0)
            lon = raw_response.get("metadata", {}).get("lon", 0.0)

            # Generate analysis
            report = analyze_data_loss_from_raw_response(
                raw_response,
                lat=lat,
                lon=lon,
                point_id=point_id,
                coverage_threshold=coverage_threshold,
                use_debug_evalscript=use_debug_evalscript
            )

            # Generate and save report
            report_data = generate_data_loss_report(
                report,
                output_dir=output_dir,
                include_detailed_daily=True
            )

            reports.append(report_data)

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            continue

    return reports

def create_data_loss_summary(
    reports: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create a summary across multiple data loss reports

    Args:
        reports: List of data loss report dictionaries

    Returns:
        Summary dictionary
    """
    if not reports:
        return {"error": "No reports provided"}

    # Calculate overall statistics
    total_points = len(reports)
    total_days = sum(r["total_days_analyzed"] for r in reports)
    total_days_with_data = sum(r["days_with_valid_data"] for r in reports)
    total_days_rejected = sum(r["days_rejected_by_coverage"] for r in reports)

    # Calculate filter efficiency averages
    filter_stats = {}
    for report in reports:
        for stage_name, stats in report["overall_filter_efficiency"].items():
            if stage_name not in filter_stats:
                filter_stats[stage_name] = {
                    "total_passed": 0,
                    "total_failed": 0,
                    "total_pixels": 0
                }
            filter_stats[stage_name]["total_passed"] += stats["pixels_passed"]
            filter_stats[stage_name]["total_failed"] += stats["pixels_failed"]
            filter_stats[stage_name]["total_pixels"] += stats["total_pixels"]

    # Calculate averages
    for stage_name, stats in filter_stats.items():
        if stats["total_pixels"] > 0:
            stats["average_pass_rate"] = stats["total_passed"] / stats["total_pixels"]

    return {
        "summary": {
            "total_points_analyzed": total_points,
            "total_days_analyzed": total_days,
            "total_days_with_valid_data": total_days_with_data,
            "total_days_rejected_by_coverage": total_days_rejected,
            "overall_data_recovery_rate": total_days_with_data / total_days if total_days > 0 else 0.0,
            "average_data_recovery_rate_per_point": total_days_with_data / total_points if total_points > 0 else 0.0
        },
        "filter_efficiency": filter_stats,
        "point_level_summary": [
            {
                "point_id": r["point_id"],
                "location": r["location"],
                "data_recovery_rate": r["data_recovery_rate"],
                "days_analyzed": r["total_days_analyzed"],
                "days_with_data": r["days_with_valid_data"],
                "days_rejected": r["days_rejected_by_coverage"]
            }
            for r in reports
        ]
    }