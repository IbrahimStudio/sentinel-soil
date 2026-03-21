#!/usr/bin/env python3
"""
Coverage Analysis Core Logic

Computes statistics that quantify how Sentinel Hub filtering reduces data availability.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sh_statistics.client import StatisticsApiClient, create_client_from_env
from sh_statistics.models import (
    CoverageConfig,
    CoverageStats,
    CoverageResult,
    create_meter_based_request_config
)
# Import is not needed - we read the evalscript file directly

def _get_evalscript_path() -> str:
    """Get the path to the coverage analysis evalscript"""
    return os.path.join(os.path.dirname(__file__), "..", "evalscripts", "coverage_analysis.js")

def _read_evalscript() -> str:
    """Read the coverage analysis evalscript"""
    evalscript_path = _get_evalscript_path()
    with open(evalscript_path, 'r') as f:
        return f.read()

def _parse_coverage_response(
    sh_response: Dict[str, Any],
    aoi_id: str,
    lat: float,
    lon: float,
    date: str
) -> CoverageStats:
    """
    Parse Sentinel Hub response for coverage analysis

    Args:
        sh_response: Raw Sentinel Hub JSON response
        aoi_id: AOI identifier
        lat: Latitude
        lon: Longitude
        date: Date for this record

    Returns:
        CoverageStats object with computed metrics
    """
    # Extract statistics from the response
    data = sh_response.get("data", [])

    if not data:
        # Return zero stats if no data
        return CoverageStats(
            aoi_id=aoi_id,
            lat=lat,
            lon=lon,
            date=date,
            observable_fraction=0.0,
            kept_scl_abs=0.0,
            kept_scl_ndvi_abs=0.0,
            saved_scl=0.0,
            saved_scl_ndvi=0.0,
            sample_count=0,
            no_data_count=0
        )

    item = data[0]  # We expect one interval per request
    outputs = item.get("outputs", {})

    # Extract mean values for each mask
    def get_mask_mean(output_id: str) -> float:
        try:
            # Navigate to the mask mean value
            mask_stats = outputs.get(output_id, {}).get("bands", {}).get("B0", {}).get("stats", {})
            return float(mask_stats.get("mean", 0.0))
        except (AttributeError, KeyError, ValueError):
            return 0.0

    def get_mask_counts(output_id: str) -> Tuple[int, int]:
        try:
            mask_stats = outputs.get(output_id, {}).get("bands", {}).get("B0", {}).get("stats", {})
            sample_count = int(mask_stats.get("sampleCount", 0))
            no_data_count = int(mask_stats.get("noDataCount", 0))
            return sample_count, no_data_count
        except (AttributeError, KeyError, ValueError):
            return 0, 0

    # Get mask statistics
    observable_fraction = get_mask_mean("dataMask")
    kept_scl_abs = get_mask_mean("scl_ok")
    kept_scl_ndvi_abs = get_mask_mean("scl_ok_ndvi")

    # Get sample counts from dataMask (most comprehensive)
    sample_count, no_data_count = get_mask_counts("dataMask")

    # Compute derived metrics
    if observable_fraction > 0:
        saved_scl = kept_scl_abs / observable_fraction
        saved_scl_ndvi = kept_scl_ndvi_abs / observable_fraction
    else:
        saved_scl = 0.0
        saved_scl_ndvi = 0.0

    return CoverageStats(
        aoi_id=aoi_id,
        lat=lat,
        lon=lon,
        date=date,
        observable_fraction=observable_fraction,
        kept_scl_abs=kept_scl_abs,
        kept_scl_ndvi_abs=kept_scl_ndvi_abs,
        saved_scl=saved_scl,
        saved_scl_ndvi=saved_scl_ndvi,
        sample_count=sample_count,
        no_data_count=no_data_count
    )

def compute_coverage_metrics(
    client: StatisticsApiClient,
    aoi_id: str,
    lat: float,
    lon: float,
    date: str,
    config: CoverageConfig
) -> CoverageStats:
    """
    Compute coverage metrics for a single AOI and date

    Args:
        client: Sentinel Hub Statistics API client
        aoi_id: AOI identifier
        lat: Latitude
        lon: Longitude
        date: Date in YYYY-MM-DD format
        config: Coverage analysis configuration

    Returns:
        CoverageStats object with computed metrics
    """
    # Read evalscript
    evalscript = _read_evalscript()

    # Create meter-based request configuration
    meter_config = create_meter_based_request_config(
        lat=lat,
        lon=lon,
        size_m=config.size_m,
        resolution_m=config.resolution_m
    )

    # Make the request
    response = client.request_statistics_meter_based(
        lat=lat,
        lon=lon,
        size_m=config.size_m,
        resolution_m=config.resolution_m,
        start_date=date,
        end_date=date,  # Single day
        interval=config.interval,
        evalscript=evalscript,
        mosaicking_order=config.mosaicking_order
    )

    # Parse the response
    return _parse_coverage_response(response, aoi_id, lat, lon, date)

def aggregate_coverage_results(
    coverage_stats_list: List[CoverageStats],
    config: CoverageConfig,
    start_date: str,
    end_date: str
) -> CoverageResult:
    """
    Aggregate coverage results across multiple AOIs and dates

    Args:
        coverage_stats_list: List of CoverageStats objects
        config: Coverage analysis configuration
        start_date: Start date of analysis
        end_date: End date of analysis

    Returns:
        CoverageResult object with aggregated statistics
    """
    result = CoverageResult(
        config=config,
        start_date=start_date,
        end_date=end_date,
        coverage_stats=coverage_stats_list
    )

    # Compute summary statistics
    result.compute_summary()

    return result

def run_coverage_analysis(
    aois: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    config: Optional[CoverageConfig] = None,
    client: Optional[StatisticsApiClient] = None,
    output_dir: Optional[str] = None
) -> CoverageResult:
    """
    Run complete coverage analysis for multiple AOIs and date range

    Args:
        aois: List of AOI dictionaries with 'aoi_id', 'lat', 'lon'
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        config: Coverage analysis configuration (optional)
        client: Sentinel Hub client (optional, will create if None)
        output_dir: Directory to save raw responses (optional)

    Returns:
        CoverageResult object with complete analysis results
    """
    # Use default config if none provided
    if config is None:
        config = CoverageConfig()

    # Create client if none provided
    if client is None:
        client = create_client_from_env()

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = [d.strftime('%Y-%m-%d') for d in date_range]

    all_coverage_stats = []

    # Process each AOI and date
    for aoi in aois:
        aoi_id = aoi['aoi_id']
        lat = aoi['lat']
        lon = aoi['lon']

        print(f"Processing AOI {aoi_id} at ({lat}, {lon})")

        for date in dates:
            print(f"  Processing date {date}")

            try:
                # Compute coverage metrics
                coverage_stats = compute_coverage_metrics(
                    client=client,
                    aoi_id=aoi_id,
                    lat=lat,
                    lon=lon,
                    date=date,
                    config=config
                )

                all_coverage_stats.append(coverage_stats)

                # Save raw response if output_dir provided
                if output_dir:
                    response = client.request_statistics_meter_based(
                        lat=lat,
                        lon=lon,
                        size_m=config.size_m,
                        resolution_m=config.resolution_m,
                        start_date=date,
                        end_date=date,
                        interval=config.interval,
                        evalscript=_read_evalscript(),
                        mosaicking_order=config.mosaicking_order
                    )

                    # Save raw response
                    os.makedirs(output_dir, exist_ok=True)
                    response_file = os.path.join(output_dir, f"coverage_raw_{aoi_id}_{date}.json")
                    with open(response_file, 'w') as f:
                        json.dump(response, f, indent=2)

            except Exception as e:
                print(f"    Error processing {aoi_id} on {date}: {e}")
                continue

    # Aggregate results
    result = aggregate_coverage_results(
        coverage_stats_list=all_coverage_stats,
        config=config,
        start_date=start_date,
        end_date=end_date
    )

    return result