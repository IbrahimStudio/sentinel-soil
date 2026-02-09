#!/usr/bin/env python3
"""
Test Script for Scaleway Batch Processing

Tests the batch processing system with a small subset of points
to validate everything works before full deployment.
"""

from __future__ import annotations

import logging
from pathlib import Path

from statistics.batch.xlsx_processor import create_xlsx_processor
from statistics.batch.scaleway_workers import create_scaleway_worker

def test_few_points():
    """Test the batch processing with just 3 points"""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        print("🧪 Starting test with 3 points from gabri_filters.xlsx")

        # Read evalscript
        evalscript_path = Path("statistics/evalscripts/features.js")
        if not evalscript_path.exists():
            raise FileNotFoundError(f"Evalscript not found: {evalscript_path}")

        evalscript = evalscript_path.read_text(encoding="utf-8")
        print(f"✅ Loaded evalscript: {evalscript_path.name}")

        # Test with just 3 points
        xlsx_path = Path("gabri_filters.xlsx")
        if not xlsx_path.exists():
            raise FileNotFoundError(f"XLSX file not found: {xlsx_path}")

        # Create XLSX processor with test configuration
        xlsx_processor = create_xlsx_processor(
            xlsx_path=xlsx_path,
            sheet=0,  # First sheet
            start_date="2015-01-01",  
            end_date="201-12-31",    
            bbox_size_m=30.0,
            limit=3,  # Only process 3 points
            ndvi_threshold=0.2,
            mndwi_threshold=0.0,
            sun_zenith_threshold=70.0,
            coverage_threshold=0.5,  # Lower threshold for testing
            scl_exclude_classes=[3, 6, 8, 9, 10, 11]
        )

        # Process Excel file to create jobs
        jobs = xlsx_processor.process()
        print(f"✅ Prepared {len(jobs)} test jobs")

        if len(jobs) == 0:
            print("❌ No jobs created - check input data")
            return False

        # Show job details
        for i, job in enumerate(jobs, 1):
            print(f"   Job {i}: {job.point_id} at ({job.lat:.5f}, {job.lon:.5f})")
            print(f"            {job.start_date} to {job.end_date}")

        # Create Scaleway worker with test configuration
        worker = create_scaleway_worker(
            evalscript=evalscript,
            interval="P1D",
            resolution=10,  # 10m resolution for better data
            coverage_threshold=0.5,
            max_workers=2,  # Use 2 workers for test (less resource intensive)
            storage_prefix="test_batch_results"
        )

        print("🚀 Starting test execution with 2 workers...")

        # Execute jobs with progress tracking
        def progress_callback(current, total, point_id, status):
            print(f"   [{current}/{total}] {status} - {point_id}")

        results = worker.execute_jobs(
            jobs=jobs,
            progress_callback=progress_callback
        )

        # Analyze results
        success_count = sum(1 for r in results if r.status == "SUCCESS")
        failure_count = len(results) - success_count

        print(f"\n📊 Test Results:")
        print(f"   Success: {success_count}/{len(results)}")
        print(f"   Failed: {failure_count}/{len(results)}")

        # Show details for each result
        for i, result in enumerate(results, 1):
            print(f"\n   Result {i}: {result.point_id} - {result.status}")
            if result.status == "FAILED":
                print(f"      Error: {result.error}")
            else:
                if result.aggregated:
                    print(f"      Days total: {result.aggregated.n_days_total}")
                    print(f"      Days kept: {result.aggregated.n_days_kept}")
                    print(f"      Kept ratio: {result.aggregated.kept_ratio:.2f}")
                    if result.aggregated.coverage_median_kept is not None:
                        print(f"      Coverage median: {result.aggregated.coverage_median_kept:.3f}")
                    else:
                        print(f"      Coverage median: None (no days kept)")

        # Success criteria
        success_rate = success_count / len(results) if len(results) > 0 else 0
        if success_rate >= 0.67:  # At least 2/3 success rate
            print(f"\n✅ Test PASSED - Success rate: {success_rate:.1%}")
            print("   System is ready for full deployment!")
            return True
        else:
            print(f"\n❌ Test FAILED - Success rate: {success_rate:.1%}")
            print("   Please check the errors above before deploying")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_few_points()
    exit(0 if success else 1)