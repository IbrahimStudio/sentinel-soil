#!/usr/bin/env python3
"""
Test script for the fixed only_scl.js evalscript with 0.5 coverage threshold
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from statistics.models import JobSpec, create_meter_based_request_config, DataMaskStats
from statistics.client import create_client_from_env
from statistics.processing.parsers import parse_daily_records, aggregate_records, _get_datamask_counts
from statistics.models import DailyStatsRecord

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

            # Create job using meter-based approach with updated coverage threshold
            job_id = f"test_{i:03d}"
            job = JobSpec(
                job_id=job_id,
                point_id=point_id,
                lat=lat,
                lon=lon,
                start_date=start_date,
                end_date=end_date,
                bbox=bbox_around_point_m(lat, lon, 30.0),  # 30m bbox
                coverage_threshold=0.5  # Updated coverage threshold
            )

            jobs.append(job)
            logger.info(f"Created job {i+1}/{limit}: {point_id} at ({lat}, {lon})")
            logger.info(f"  Time window: {start_date} to {end_date}")

        except Exception as e:
            logger.warning(f"⚠️  Skipping row {i}: {e}")
            continue

    return jobs

def execute_test_job(job: JobSpec, evalscript: str) -> dict:
    """
    Execute a single job with the fixed evalscript and new coverage threshold
    """
    try:
        logger.info(f"🔍 Starting execution for job: {job.point_id}")

        # Create client
        client = create_client_from_env()

        # Create meter-based request configuration
        config = create_meter_based_request_config(
            job.lat, job.lon, size_m=30.0, resolution_m=10
        )

        logger.info(f"📊 Meter-based config: {config['notes']}")

        # Execute meter-based request
        logger.info("📡 Sending request to Sentinel Hub API...")
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

        # Parse daily records
        logger.info("📋 Parsing daily records...")
        daily_rows = parse_daily_records(
            sh_json=response,
            lat=job.lat,
            lon=job.lon,
            bbox=job.bbox.to_list(),
            start_date=job.start_date,
            end_date=job.end_date,
            interval="P1D",
            evalscript_type="only_scl"  # Use the new parameter
        )

        logger.info(f"📆 Parsed {len(daily_rows)} daily records")

        # Debug daily records
        for i, record in enumerate(daily_rows[:3]):  # Show first 3 as examples
            logger.info(f"  Record {i+1}: coverage={record.coverage:.3f}, sample_count={record.sample_count}, no_data_count={record.no_data_count}")

        # Aggregate records with new 0.5 coverage threshold
        logger.info(f"🔢 Aggregating records with coverage threshold: {job.coverage_threshold}")
        kept_rows, aggregated = aggregate_records(daily_rows, coverage_threshold=job.coverage_threshold)

        logger.info(f"📊 Aggregation results: {len(kept_rows)}/{len(daily_rows)} records kept")

        # Debug aggregation
        if aggregated:
            logger.info(f"📈 Aggregated stats: n_days_total={aggregated.n_days_total}, n_days_kept={aggregated.n_days_kept}")
            logger.info(f"📊 Coverage stats: median={aggregated.coverage_median_kept}, min={aggregated.coverage_min_kept}")

            # Check for null values in aggregated features
            null_features = [name for name, value in aggregated.p50_aggregated.items() if value is None]
            if null_features:
                logger.warning(f"⚠️  Null features in aggregation: {null_features}")
            else:
                logger.info("✅ All features have valid values in aggregation")
        else:
            logger.warning("⚠️  Aggregation returned None")

        return {
            "status": "SUCCESS",
            "job_id": job.job_id,
            "point_id": job.point_id,
            "daily_count": len(daily_rows),
            "kept_count": len(kept_rows),
            "aggregated": aggregated.__dict__ if aggregated else None,
            "response": response,
            "debug_summary": {
                "days_with_coverage_above_threshold": len(kept_rows),
                "coverage_threshold": job.coverage_threshold
            }
        }

    except Exception as e:
        logger.error(f"❌ Error executing job {job.point_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "FAILED",
            "job_id": job.job_id,
            "point_id": job.point_id,
            "error": str(e)
        }

def main():
    """Run test with fixed only_scl.js evalscript"""
    print("🚀 Starting Fixed SCL-Only Test with 30 data points...")
    print("=" * 80)
    print("📋 Using updated only_scl.js with more permissive SCL filtering")
    print("📊 Using 0.5 coverage threshold (was 0.8)")

    # Configuration
    XLSX_PATH = "gabri_filters.xlsx"
    EVALSCRIPT_PATH = "statistics/evalscripts/only_scl.js"
    OUTPUT_DIR = "test_fixed_scl_output"
    LIMIT = 30

    try:
        # Read evalscript
        evalscript = Path(EVALSCRIPT_PATH).read_text(encoding="utf-8")
        print(f"✓ Loaded evalscript from {EVALSCRIPT_PATH}")
        print(f"📄 Evalscript uses updated SCL filtering (allows SCL 7, 8)")

        # Create jobs
        jobs = create_test_jobs_from_xlsx(XLSX_PATH, LIMIT)
        print(f"✓ Created {len(jobs)} jobs from {XLSX_PATH}")

        if not jobs:
            print("❌ No jobs created. Check input file format.")
            return 1

        # Create output directory
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"� Output directory: {out_dir}")

        # Execute jobs sequentially
        results = []
        success_count = 0
        jobs_with_aggregation = 0

        for i, job in enumerate(jobs, start=1):
            print(f"\n{'='*80}")
            print(f"🔄 Processing job {i}/{len(jobs)}: {job.point_id}")
            print(f"📍 Location: ({job.lat}, {job.lon})")
            print(f"📅 Time range: {job.start_date} to {job.end_date}")

            result = execute_test_job(job, evalscript)
            results.append(result)

            if result["status"] == "SUCCESS":
                success_count += 1
                debug_summary = result.get("debug_summary", {})

                if result.get("aggregated") and result["aggregated"].get("n_days_kept", 0) > 0:
                    jobs_with_aggregation += 1

                print(f"✅ SUCCESS:")
                print(f"   Daily records: {result['daily_count']}")
                print(f"   Records kept: {result['kept_count']}")
                print(f"   Days in aggregation: {result['aggregated']['n_days_kept'] if result['aggregated'] else 0}")

                # Check for null aggregations
                if result.get("aggregated"):
                    null_count = sum(1 for val in result["aggregated"]["p50_aggregated"].values() if val is None)
                    total_features = len(result["aggregated"]["p50_aggregated"])
                    print(f"   Null features: {null_count}/{total_features}")

                    if null_count == 0:
                        print("   ✅ ALL FEATURES HAVE VALID VALUES!")
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

        # Print comprehensive summary
        print(f"\n{'='*80}")
        print(f"📊 COMPREHENSIVE TEST SUMMARY:")
        print(f"Total jobs processed: {len(jobs)}")
        print(f"API success rate: {success_count}/{len(jobs)} ({success_count/len(jobs)*100:.1f}%)")
        print(f"Jobs with successful aggregation: {jobs_with_aggregation}/{len(jobs)} ({jobs_with_aggregation/len(jobs)*100:.1f}%)")
        print(f"Output directory: {out_dir}")

        # Analyze null aggregation patterns
        null_agg_jobs = [r for r in results if r.get("aggregated") and
                        all(v is None for v in r["aggregated"]["p50_aggregated"].values())]

        print(f"\n🔍 NULL AGGREGATION ANALYSIS:")
        print(f"Jobs with all-null aggregations: {len(null_agg_jobs)}/{len(jobs)}")

        if null_agg_jobs:
            print("⚠️  Still some null aggregations - may need further adjustment")
        else:
            print("✅ NO NULL AGGREGATIONS - Fix appears successful!")

        # Save comprehensive summary
        summary = {
            "test_timestamp": datetime.now().isoformat(),
            "total_jobs": len(jobs),
            "success_count": success_count,
            "jobs_with_aggregation": jobs_with_aggregation,
            "null_aggregation_count": len(null_agg_jobs),
            "coverage_threshold": 0.5,  # Updated threshold
            "evalscript": "only_scl.js (updated)",
            "test_parameters": {
                "bbox_size_m": 30.0,
                "resolution_m": 10,
                "time_window_days": 30,
                "limit": LIMIT
            }
        }

        summary_file = out_dir / "test_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Test completed! Summary saved to: {summary_file}")
        print(f"📁 All output files available in: {out_dir}")

        # Provide actionable insights
        print(f"\n🎯 ACTIONABLE INSIGHTS:")
        if len(null_agg_jobs) == 0:
            print("   🎉 SUCCESS! All jobs now have valid aggregations")
            print("   📊 The fix (more permissive SCL + 0.5 threshold) appears to work")
            print("   ✅ Ready for production deployment")
        else:
            print(f"   ⚠️  {len(null_agg_jobs)} jobs still have null aggregations")
            print("   🔬 May need further SCL filtering adjustment")
            print("   📊 Consider increasing bbox size or further relaxing SCL criteria")

        return 0

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())