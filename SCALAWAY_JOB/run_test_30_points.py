#!/usr/bin/env python3
"""
Test run with 30 data points using meter-based approach
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

from statistics.models import JobSpec, create_meter_based_request_config
from statistics.client import create_client_from_env
from statistics.batch.workers import StatisticsWorker
from statistics.processing.parsers import parse_daily_records, aggregate_records

def create_test_jobs_from_xlsx(xlsx_path: str, limit: int = 30) -> List[JobSpec]:
    """
    Create test jobs from XLSX file using meter-based approach
    """
    import pandas as pd
    from statistics.models import bbox_around_point_m

    # Read Excel file
    df = pd.read_excel(xlsx_path)

    # Limit to specified number of rows
    if limit > 0:
        df = df.head(limit)

    jobs = []
    for i, row in df.iterrows():
        try:
            point_id = str(row["POINT_ID"]).strip()
            lat = float(row["TH_LAT"])
            lon = float(row["TH_LONG"])

            # Parse survey date and create time window
            survey_date = pd.to_datetime(row["SURVEY_DATE"])
            center = survey_date.normalize()
            start_date = (center - timedelta(days=15)).strftime("%Y-%m-%d")
            end_date = (center + timedelta(days=15)).strftime("%Y-%m-%d")

            # Create job using meter-based approach
            job_id = f"test_{i:03d}"
            job = JobSpec(
                job_id=job_id,
                point_id=point_id,
                lat=lat,
                lon=lon,
                start_date=start_date,
                end_date=end_date,
                bbox=bbox_around_point_m(lat, lon, 30.0)  # 30m bbox
            )

            jobs.append(job)
            print(f"Created job {i+1}/{limit}: {point_id} at ({lat}, {lon})")

        except Exception as e:
            print(f"⚠️  Skipping row {i}: {e}")
            continue

    return jobs

def execute_meter_based_job(job: JobSpec, evalscript: str) -> dict:
    """
    Execute a single job using meter-based approach
    """
    try:
        # Create client
        client = create_client_from_env()

        # Create meter-based request configuration
        config = create_meter_based_request_config(
            job.lat, job.lon, size_m=30.0, resolution_m=10
        )

        print(f"📊 Job {job.point_id}: {config['notes']}")

        # Execute meter-based request
        response = client.request_statistics_meter_based(
            lat=job.lat,
            lon=job.lon,
            size_m=30.0,
            resolution_m=10,
            start_date=job.start_date,
            end_date=job.end_date,
            interval="P1D",
            evalscript=evalscript,
            mosaicking_order="leastCC"
        )

        # Parse and aggregate results
        daily_rows = parse_daily_records(
            sh_json=response,
            lat=job.lat,
            lon=job.lon,
            bbox=job.bbox.to_list(),
            start_date=job.start_date,
            end_date=job.end_date,
            interval="P1D"
        )

        kept_rows, aggregated = aggregate_records(daily_rows, coverage_threshold=0.8)

        return {
            "status": "SUCCESS",
            "job_id": job.job_id,
            "point_id": job.point_id,
            "daily_count": len(daily_rows),
            "kept_count": len(kept_rows),
            "aggregated": aggregated.__dict__ if aggregated else None,
            "response": response
        }

    except Exception as e:
        return {
            "status": "FAILED",
            "job_id": job.job_id,
            "point_id": job.point_id,
            "error": str(e)
        }

def main():
    """Run test with 30 data points"""
    print("🚀 Starting 30-point test with meter-based approach...")
    print("=" * 60)

    # Configuration
    XLSX_PATH = "gabri_filters.xlsx"
    EVALSCRIPT_PATH = "statistics/evalscripts/features.js"
    OUTPUT_DIR = "test_30_points_output"
    LIMIT = 30

    try:
        # Read evalscript
        evalscript = Path(EVALSCRIPT_PATH).read_text(encoding="utf-8")
        print(f"✓ Loaded evalscript from {EVALSCRIPT_PATH}")

        # Create jobs
        jobs = create_test_jobs_from_xlsx(XLSX_PATH, LIMIT)
        print(f"✓ Created {len(jobs)} jobs from {XLSX_PATH}")

        if not jobs:
            print("❌ No jobs created. Check input file format.")
            return 1

        # Create output directory
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Execute jobs sequentially (for better debugging)
        results = []
        success_count = 0

        for i, job in enumerate(jobs, start=1):
            print(f"\n🔄 Processing job {i}/{len(jobs)}: {job.point_id}")

            result = execute_meter_based_job(job, evalscript)
            results.append(result)

            if result["status"] == "SUCCESS":
                success_count += 1
                print(f"✅ SUCCESS: {result['daily_count']} daily records, {result['kept_count']} kept")
            else:
                print(f"❌ FAILED: {result['error']}")

            # Save raw response for debugging
            if result.get("response"):
                response_file = out_dir / f"{job.point_id}_response.json"
                with open(response_file, "w", encoding="utf-8") as f:
                    json.dump(result["response"], f, indent=2)

            # Save aggregated result
            if result.get("aggregated"):
                agg_file = out_dir / f"{job.point_id}_aggregated.json"
                with open(agg_file, "w", encoding="utf-8") as f:
                    json.dump(result["aggregated"], f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY:")
        print(f"Total jobs: {len(jobs)}")
        print(f"Success: {success_count}/{len(jobs)}")
        print(f"Failure: {len(jobs) - success_count}/{len(jobs)}")
        print(f"Output directory: {out_dir}")

        # Calculate success rate
        success_rate = (success_count / len(jobs)) * 100 if jobs else 0
        print(f"Success rate: {success_rate:.1f}%")

        if success_count > 0:
            print(f"\n✅ Test completed successfully!")
            print(f"Check output files in: {out_dir}")
        else:
            print(f"\n❌ All jobs failed. Check error messages above.")

        return 0

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

# Add missing import
from typing import List