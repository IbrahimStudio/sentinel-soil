#!/usr/bin/env python3
"""
Refactored Statistics API Client

Single point statistics API client using the new statistics module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from statistics.client import create_client_from_env
from statistics.models import bbox_around_point_m, parse_date
from statistics.processing.parsers import parse_daily_records, aggregate_records

def main() -> None:
    """Main entry point for single point statistics API"""
    ap = argparse.ArgumentParser(description="Sentinel Hub Statistics API Client")
    ap.add_argument("--lat", type=float, required=True, help="Latitude in decimal degrees")
    ap.add_argument("--lon", type=float, required=True, help="Longitude in decimal degrees")
    ap.add_argument("--start_date", type=parse_date, required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end_date", type=parse_date, required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--interval", type=str, default="P1D", help="Aggregation interval (e.g., P1D for daily)")
    ap.add_argument("--size_m", type=float, default=30.0, help="Bounding box size in meters")
    ap.add_argument("--coverage_threshold", type=float, default=0.8, help="Minimum coverage threshold")
    ap.add_argument("--evalscript_path", type=str, required=True, help="Path to evalscript file")
    ap.add_argument("--out_dir", type=str, default="out_stats_json", help="Output directory")
    args = ap.parse_args()

    # Read evalscript
    evalscript = Path(args.evalscript_path).read_text(encoding="utf-8")

    # Create bounding box
    bbox = bbox_around_point_m(args.lat, args.lon, args.size_m)

    # Create API client
    client = create_client_from_env()

    try:
        # Execute API request
        response = client.request_statistics(
            bbox=bbox.to_list(),
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            evalscript=evalscript,
            res=20,
            mosaicking_order="leastCC"
        )

        # Prepare output directory
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Write raw response
        raw_response_path = out_dir / "raw_response.json"
        raw_response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")

        # 2) Parse daily records
        daily_rows = parse_daily_records(
            response,
            lat=args.lat,
            lon=args.lon,
            bbox=bbox.to_list(),
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval
        )

        daily_parsed_path = out_dir / "daily_parsed.json"
        daily_parsed_path.write_text(
            json.dumps([row.__dict__ for row in daily_rows], indent=2),
            encoding="utf-8"
        )

        # 3) Aggregate records
        kept_rows, agg_row = aggregate_records(
            daily_rows,
            coverage_threshold=args.coverage_threshold
        )

        daily_kept_path = out_dir / "daily_kept.json"
        daily_kept_path.write_text(
            json.dumps([row.__dict__ for row in kept_rows], indent=2),
            encoding="utf-8"
        )

        aggregated_path = out_dir / "aggregated_one_row.json"
        aggregated_path.write_text(
            json.dumps(agg_row.__dict__, indent=2),
            encoding="utf-8"
        )

        print(f"Wrote: {raw_response_path}")
        print(f"Wrote: {daily_parsed_path}")
        print(f"Wrote: {daily_kept_path}")
        print(f"Wrote: {aggregated_path}")
        print(f"Daily rows: {len(daily_rows)} | Kept (coverage>={args.coverage_threshold}): {len(kept_rows)}")

    except Exception as e:
        print(f"Error executing statistics API request: {e}")
        raise

if __name__ == "__main__":
    main()