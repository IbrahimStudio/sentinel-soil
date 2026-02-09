#!/usr/bin/env python3
"""
Debug Test for only_scl.js Evalscript

This script tests the SCL-only evalscript with 30 data points to understand
why aggregations are returning null. It includes enhanced logging and debugging
information to trace the data flow through the processing pipeline.
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

def create_debug_jobs_from_xlsx(xlsx_path: str, limit: int = 30) -> List[JobSpec]:
    """
    Create debug jobs from XLSX file using meter-based approach
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
            job_id = f"debug_{i:03d}"
            job = JobSpec(
                job_id=job_id,
                point_id=point_id,
                lat=lat,
                lon=lon,
                start_date=start_date,
                end_date=end_date,
                bbox=bbox_around_point_m(lat, lon, 30.0),  # 30m bbox
                coverage_threshold=0.8  # Default coverage threshold
            )

            jobs.append(job)
            logger.info(f"Created job {i+1}/{limit}: {point_id} at ({lat}, {lon})")
            logger.info(f"  Time window: {start_date} to {end_date}")

        except Exception as e:
            logger.warning(f"⚠️  Skipping row {i}: {e}")
            continue

    return jobs

def debug_coverage_analysis(sh_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform detailed coverage analysis on the raw Sentinel Hub response
    """
    debug_info = {
        'total_intervals': len(sh_json.get('data', [])),
        'intervals_with_data': 0,
        'intervals_analyzed': [],
        'overall_stats': {
            'total_pixels': 0,
            'valid_pixels': 0,
            'coverage_distribution': []
        }
    }

    for interval_idx, item in enumerate(sh_json.get('data', [])):
        interval_obj = item.get('interval', {})
        outputs = item.get('outputs', {})

        # Get data mask statistics
        datamask_stats = _get_datamask_counts(outputs)
        coverage = datamask_stats.coverage

        # Analyze each feature band
        features = outputs.get('features', {}).get('bands', {})
        band_stats = {}

        for band_name, band_data in features.items():
            stats = band_data.get('stats', {})
            sample_count = stats.get('sampleCount', 0)
            no_data_count = stats.get('noDataCount', 0)
            p50_value = stats.get('percentiles', {}).get('50.0')

            band_stats[band_name] = {
                'sample_count': sample_count,
                'no_data_count': no_data_count,
                'total_pixels': sample_count + no_data_count,
                'coverage': sample_count / (sample_count + no_data_count) if (sample_count + no_data_count) > 0 else 0.0,
                'p50_value': p50_value
            }

        interval_info = {
            'interval': f"{interval_obj.get('from')} to {interval_obj.get('to')}",
            'data_mask_coverage': coverage,
            'sample_count': datamask_stats.sample_count,
            'no_data_count': datamask_stats.no_data_count,
            'band_stats': band_stats,
            'has_valid_data': any(
                band_info['sample_count'] > 0
                for band_info in band_stats.values()
            )
        }

        debug_info['intervals_analyzed'].append(interval_info)

        if interval_info['has_valid_data']:
            debug_info['intervals_with_data'] += 1

        # Update overall stats
        debug_info['overall_stats']['total_pixels'] += datamask_stats.total_count
        debug_info['overall_stats']['valid_pixels'] += datamask_stats.sample_count
        debug_info['overall_stats']['coverage_distribution'].append(coverage)

    # Calculate overall coverage
    total_pixels = debug_info['overall_stats']['total_pixels']
    valid_pixels = debug_info['overall_stats']['valid_pixels']
    debug_info['overall_stats']['overall_coverage'] = valid_pixels / total_pixels if total_pixels > 0 else 0.0

    return debug_info

def execute_debug_job(job: JobSpec, evalscript: str) -> dict:
    """
    Execute a single job with enhanced debugging information
    """
    try:
        logger.info(f"🔍 Starting debug execution for job: {job.point_id}")

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

        # Perform detailed coverage analysis
        logger.info("🔬 Analyzing coverage and data availability...")
        coverage_debug = debug_coverage_analysis(response)

        logger.info(f"📈 Coverage analysis: {coverage_debug['intervals_with_data']}/{coverage_debug['total_intervals']} intervals have valid data")
        logger.info(f"📊 Overall coverage: {coverage_debug['overall_stats']['overall_coverage']:.3f}")

        # Parse daily records
        logger.info("📋 Parsing daily records...")
        daily_rows = parse_daily_records(
            sh_json=response,
            lat=job.lat,
            lon=job.lon,
            bbox=job.bbox.to_list(),
            start_date=job.start_date,
            end_date=job.end_date,
            interval="P1D"
        )

        logger.info(f"📆 Parsed {len(daily_rows)} daily records")

        # Debug daily records
        for i, record in enumerate(daily_rows[:3]):  # Show first 3 as examples
            logger.info(f"  Record {i+1}: coverage={record.coverage:.3f}, sample_count={record.sample_count}, no_data_count={record.no_data_count}")

        # Aggregate records
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
            "coverage_debug": coverage_debug,
            "debug_summary": {
                "has_valid_data": coverage_debug['intervals_with_data'] > 0,
                "overall_coverage": coverage_debug['overall_stats']['overall_coverage'],
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
    """Run debug test with 30 data points using only_scl.js"""
    print("🚀 Starting SCL-Only Debug Test with 30 data points...")
    print("=" * 80)

    # Configuration
    XLSX_PATH = "gabri_filters.xlsx"
    EVALSCRIPT_PATH = "statistics/evalscripts/only_scl.js"
    OUTPUT_DIR = "debug_scl_test_output"
    LIMIT = 30

    try:
        # Read evalscript
        evalscript = Path(EVALSCRIPT_PATH).read_text(encoding="utf-8")
        print(f"✓ Loaded evalscript from {EVALSCRIPT_PATH}")
        print(f"📄 Evalscript uses SCL-only filtering (no sun zenith, NDVI, or MNDWI filters)")

        # Create jobs
        jobs = create_debug_jobs_from_xlsx(XLSX_PATH, LIMIT)
        print(f"✓ Created {len(jobs)} jobs from {XLSX_PATH}")

        if not jobs:
            print("❌ No jobs created. Check input file format.")
            return 1

        # Create output directory
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {out_dir}")

        # Execute jobs sequentially (for better debugging)
        results = []
        success_count = 0
        jobs_with_data = 0
        jobs_with_aggregation = 0

        for i, job in enumerate(jobs, start=1):
            print(f"\n{'='*80}")
            print(f"🔄 Processing job {i}/{len(jobs)}: {job.point_id}")
            print(f"📍 Location: ({job.lat}, {job.lon})")
            print(f"📅 Time range: {job.start_date} to {job.end_date}")

            result = execute_debug_job(job, evalscript)
            results.append(result)

            if result["status"] == "SUCCESS":
                success_count += 1
                debug_summary = result.get("debug_summary", {})

                if debug_summary.get("has_valid_data", False):
                    jobs_with_data += 1

                if result.get("aggregated") and result["aggregated"].get("n_days_kept", 0) > 0:
                    jobs_with_aggregation += 1

                print(f"✅ SUCCESS:")
                print(f"   Daily records: {result['daily_count']}")
                print(f"   Records kept: {result['kept_count']}")
                print(f"   Has valid data: {debug_summary.get('has_valid_data', False)}")
                print(f"   Overall coverage: {debug_summary.get('overall_coverage', 0.0):.3f}")
                print(f"   Days in aggregation: {result['aggregated']['n_days_kept'] if result['aggregated'] else 0}")

                # Check for null aggregations
                if result.get("aggregated"):
                    null_count = sum(1 for val in result["aggregated"]["p50_aggregated"].values() if val is None)
                    total_features = len(result["aggregated"]["p50_aggregated"])
                    print(f"   Null features: {null_count}/{total_features}")

                    if null_count == total_features:
                        print("   ⚠️  ALL FEATURES ARE NULL - This is the issue we're investigating!")
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

            # Save debug coverage analysis
            if result.get("coverage_debug"):
                debug_file = out_dir / f"{job.point_id}_coverage_debug.json"
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump(result["coverage_debug"], f, indent=2)

        # Print comprehensive summary
        print(f"\n{'='*80}")
        print(f"📊 COMPREHENSIVE TEST SUMMARY:")
        print(f"Total jobs processed: {len(jobs)}")
        print(f"API success rate: {success_count}/{len(jobs)} ({success_count/len(jobs)*100:.1f}%)")
        print(f"Jobs with valid data: {jobs_with_data}/{len(jobs)} ({jobs_with_data/len(jobs)*100:.1f}%)")
        print(f"Jobs with successful aggregation: {jobs_with_aggregation}/{len(jobs)} ({jobs_with_aggregation/len(jobs)*100:.1f}%)")
        print(f"Output directory: {out_dir}")

        # Analyze null aggregation patterns
        null_agg_jobs = [r for r in results if r.get("aggregated") and
                        all(v is None for v in r["aggregated"]["p50_aggregated"].values())]

        print(f"\n🔍 NULL AGGREGATION ANALYSIS:")
        print(f"Jobs with all-null aggregations: {len(null_agg_jobs)}/{len(jobs)}")

        if null_agg_jobs:
            print("🎯 This appears to be the core issue!")
            print("🔬 Investigating patterns...")

            # Look for common patterns
            coverage_patterns = []
            for job_result in null_agg_jobs:
                debug_summary = job_result.get("debug_summary", {})
                coverage_patterns.append(debug_summary.get("overall_coverage", 0.0))

            avg_coverage = sum(coverage_patterns) / len(coverage_patterns) if coverage_patterns else 0
            print(f"   Average coverage in null jobs: {avg_coverage:.3f}")
            print(f"   Coverage threshold: {jobs[0].coverage_threshold}")

            if avg_coverage < jobs[0].coverage_threshold:
                print("   💡 INSIGHT: Coverage is below threshold, causing all days to be filtered out!")
            else:
                print("   🤔 Coverage seems sufficient, but aggregation is still null")

        # Save comprehensive summary
        summary = {
            "test_timestamp": datetime.now().isoformat(),
            "total_jobs": len(jobs),
            "success_count": success_count,
            "jobs_with_data": jobs_with_data,
            "jobs_with_aggregation": jobs_with_aggregation,
            "null_aggregation_count": len(null_agg_jobs),
            "coverage_threshold": jobs[0].coverage_threshold if jobs else None,
            "evalscript": "only_scl.js",
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
        if len(null_agg_jobs) > 0:
            print("   1. NULL AGGREGATIONS DETECTED - This is the issue to investigate")
            print("   2. Check coverage_debug files for detailed filtering analysis")
            print("   3. Consider adjusting coverage threshold or bbox size")
            print("   4. Examine SCL filtering patterns in the evalscript")

        return 0

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())