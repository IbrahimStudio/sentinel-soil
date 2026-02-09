#!/usr/bin/env python3
"""
Scaleway Batch Statistics from XLSX

Batch processing of statistics API requests from Excel files using the new statistics module
with Scaleway object storage for results.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sh_statistics.batch.xlsx_processor import create_xlsx_processor
from sh_statistics.batch.scaleway_workers import create_scaleway_worker

def _parse_sheet_arg(sheet_arg: str) -> Any:
    """Parse sheet argument (can be name or index)"""
    if sheet_arg.isdigit():
        return int(sheet_arg)
    return sheet_arg

def _calculate_time_window_dates(survey_date: str, time_window: int) -> Tuple[str, str]:
    """
    Calculate start and end dates based on survey date and time window

    Args:
        survey_date: Survey date in YYYY-MM-DD format or datetime format
        time_window: Total time window in days

    Returns:
        Tuple of (start_date, end_date) strings in YYYY-MM-DD format
    """
    try:
        # Try parsing as date-only format first
        survey_dt = datetime.strptime(survey_date, "%Y-%m-%d").date()
    except ValueError:
        try:
            # Try parsing as datetime format (with time component)
            survey_dt = datetime.strptime(survey_date, "%Y-%m-%d %H:%M:%S").date()
        except ValueError:
            try:
                # Try parsing as datetime format with potential milliseconds
                survey_dt = datetime.strptime(survey_date, "%Y-%m-%d %H:%M:%S.%f").date()
            except ValueError as e:
                # If all parsing fails, try to extract just the date portion
                if " " in survey_date:
                    date_part = survey_date.split(" ")[0]
                    survey_dt = datetime.strptime(date_part, "%Y-%m-%d").date()
                else:
                    raise ValueError(f"Could not parse date from '{survey_date}': {e}")

    half_window = time_window // 2
    start_date = (survey_dt - timedelta(days=half_window)).strftime("%Y-%m-%d")
    end_date = (survey_dt + timedelta(days=half_window)).strftime("%Y-%m-%d")
    return start_date, end_date

def _progress_callback(current: int, total: int, point_id: str, status: str) -> None:
    """Progress callback for worker execution"""
    print(f"[{current}/{total}] {status} point_id={point_id}")

def main() -> None:
    """Main entry point for Scaleway batch processing from XLSX"""
    ap = argparse.ArgumentParser(description="Scaleway Batch Statistics API from XLSX")
    ap.add_argument("--xlsx", type=str, required=True, help="Path to input Excel file")
    ap.add_argument("--sheet", type=str, default="0", help='Sheet name or index (e.g., "0" or "Sheet1")')
    ap.add_argument("--limit", type=int, default=-1, help="Process only first N rows (-1 for all)")
    ap.add_argument("--workers", type=int, default=3, help="Number of parallel workers")
    ap.add_argument("--evalscript_path", type=str, required=True, help="Path to evalscript file")
    ap.add_argument("--bbox_size_m", type=float, default=30.0, help="Bounding box size in meters")
    ap.add_argument("--interval", type=str, default="P1D", help="Aggregation interval")
    ap.add_argument("--res", type=int, default=10, help="Spatial resolution in meters (10, 20, or 60)")
    ap.add_argument("--start_date", type=str, default=None, help="Start date for all points (YYYY-MM-DD). If not provided, uses time_window around SURVEY_DATE")
    ap.add_argument("--end_date", type=str, default=None, help="End date for all points (YYYY-MM-DD). If not provided, uses time_window around SURVEY_DATE")
    ap.add_argument("--time_window", type=int, default=None, help="Time window in days around SURVEY_DATE (e.g., 30 for 15 days before/after, 730 for 1 year before/after)")
    ap.add_argument("--storage_prefix", type=str, default="batch_results", help="Storage prefix for results")
    # Filtering thresholds
    ap.add_argument("--ndvi_threshold", type=float, default=0.2, help="NDVI threshold for filtering")
    ap.add_argument("--mndwi_threshold", type=float, default=0.0, help="MNDWI threshold for filtering")
    ap.add_argument("--sun_zenith_threshold", type=float, default=70.0, help="Sun zenith angle threshold for filtering")
    ap.add_argument("--coverage_threshold", type=float, default=0.8, help="Minimum coverage threshold")
    ap.add_argument("--scl_exclude_classes", type=str, default="3,6,8,9,10,11", help="SCL classes to exclude (comma-separated)")
    args = ap.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Read evalscript
        evalscript = Path(args.evalscript_path).read_text(encoding="utf-8")

        # Parse SCL exclude classes
        scl_exclude_classes = [int(x.strip()) for x in args.scl_exclude_classes.split(",") if x.strip()]

        # Parse sheet argument
        sheet = _parse_sheet_arg(args.sheet)

        # Handle time window logic
        if args.time_window is not None:
            if args.start_date is not None or args.end_date is not None:
                logger.warning("Both time_window and explicit dates provided. time_window will take precedence.")
            # We'll handle the time window logic in a custom processor
            from sh_statistics.batch.xlsx_processor import XlsxBatchProcessor, XlsxBatchConfig
            import pandas as pd

            # Read and validate input file
            df = pd.read_excel(args.xlsx, sheet_name=sheet)

            # Check required columns
            required_cols = ["POINT_ID", "TH_LAT", "TH_LONG", "SURVEY_DATE"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns in XLSX: {missing}")

            # Apply limit if specified
            if args.limit > 0:
                df = df.head(args.limit)

            # Create jobs with time window around SURVEY_DATE
            jobs = []
            for _, row in df.iterrows():
                try:
                    point_id = str(row["POINT_ID"]).strip()
                    if not point_id:
                        continue  # Skip empty point IDs

                    lat = float(row["TH_LAT"])
                    lon = float(row["TH_LONG"])
                    survey_date_str = str(row["SURVEY_DATE"])

                    # Calculate time window dates
                    if not args.start_date or not args.end_date:
                        start_date, end_date = _calculate_time_window_dates(survey_date_str, args.time_window)
                        print(f"Point {point_id}: SURVEY_DATE={survey_date_str}, Time window: {start_date} to {end_date}")

                    # Create bounding box
                    from sh_statistics.models import bbox_around_point_m
                    bbox = bbox_around_point_m(lat, lon, args.bbox_size_m)

                    # Create job specification
                    from sh_statistics.models import JobSpec
                    import uuid
                    job_id = uuid.uuid4().hex[:10]
                    job = JobSpec(
                        job_id=job_id,
                        point_id=point_id,
                        lat=lat,
                        lon=lon,
                        start_date=start_date,
                        end_date=end_date,
                        bbox=bbox,
                        ndvi_threshold=args.ndvi_threshold,
                        mndwi_threshold=args.mndwi_threshold,
                        sun_zenith_threshold=args.sun_zenith_threshold,
                        coverage_threshold=args.coverage_threshold,
                        scl_exclude_classes=scl_exclude_classes
                    )

                    jobs.append(job)

                except Exception as e:
                    logger.error(f"Error processing row: {e}")
                    continue

            print(f"Prepared {len(jobs)} jobs from {args.xlsx} using time window of {args.time_window} days")

            # Create Scaleway worker and execute jobs
            worker = create_scaleway_worker(
                evalscript=evalscript,
                interval=args.interval,
                resolution=args.res,
                coverage_threshold=args.coverage_threshold,
                max_workers=args.workers,
                storage_prefix=args.storage_prefix
            )

            # Execute jobs
            results = worker.execute_jobs(
                jobs=jobs,
                progress_callback=_progress_callback
            )

            # Print summary
            success_count = sum(1 for r in results if r.status == "SUCCESS")
            failure_count = len(results) - success_count

            print(f"\nProcessing complete:")
            print(f"Success: {success_count}/{len(results)}")
            print(f"Failed: {failure_count}/{len(results)}")
            print(f"Results stored in Scaleway object storage with prefix: {args.storage_prefix}/")
        else:
            # Use original logic with fixed dates
            if args.start_date is None or args.end_date is None:
                raise ValueError("Either time_window or both start_date and end_date must be provided")

            xlsx_processor = create_xlsx_processor(
                xlsx_path=Path(args.xlsx),
                sheet=sheet,
                start_date=args.start_date,
                end_date=args.end_date,
                bbox_size_m=args.bbox_size_m,
                limit=args.limit,
                ndvi_threshold=args.ndvi_threshold,
                mndwi_threshold=args.mndwi_threshold,
                sun_zenith_threshold=args.sun_zenith_threshold,
                coverage_threshold=args.coverage_threshold,
                scl_exclude_classes=scl_exclude_classes
            )
            jobs = xlsx_processor.process()
            print(f"Prepared {len(jobs)} jobs from {args.xlsx}. workers={args.workers}")
            print(f"Time frame: {args.start_date} to {args.end_date}")
            # Print filtering thresholds info
            print(f"Filtering thresholds:")
            print(f"  NDVI: {args.ndvi_threshold}")
            print(f"  MNDWI: {args.mndwi_threshold}")
            print(f"  Sun Zenith: {args.sun_zenith_threshold}°")
            print(f"  Coverage: {args.coverage_threshold}")
            print(f"  SCL Exclude: {scl_exclude_classes}")

            # Create Scaleway worker
            worker = create_scaleway_worker(
                evalscript=evalscript,
                interval=args.interval,
                resolution=args.res,
                coverage_threshold=args.coverage_threshold,
                max_workers=args.workers,
                storage_prefix=args.storage_prefix
            )

            # Execute jobs
            results = worker.execute_jobs(
                jobs=jobs,
                progress_callback=_progress_callback
            )

            # Print summary
            success_count = sum(1 for r in results if r.status == "SUCCESS")
            failure_count = len(results) - success_count

            print(f"\nProcessing complete:")
            print(f"Success: {success_count}/{len(results)}")
            print(f"Failed: {failure_count}/{len(results)}")
            print(f"Results stored in Scaleway object storage with prefix: {args.storage_prefix}/")

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise

if __name__ == "__main__":
    main()