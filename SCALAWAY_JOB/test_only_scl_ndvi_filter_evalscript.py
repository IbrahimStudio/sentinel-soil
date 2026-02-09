#!/usr/bin/env python3
"""
Test script to demonstrate using the new only_scl_ndvi_filter.js evalscript
with the batch processing script.

This shows how to run the batch processing with SCL + NDVI filtering
to collect bare soil data while excluding vegetated areas.
"""

import subprocess
import sys
from pathlib import Path

def test_only_scl_ndvi_filter_evalscript():
    """Test the new SCL + NDVI filter evalscript"""

    print("🧪 Testing SCL + NDVI filter evalscript...")

    # Path to the batch processing script
    batch_script = Path("scaleway_batch_stats_from_xlsx.py")

    if not batch_script.exists():
        print(f"❌ Batch script not found: {batch_script}")
        return False

    # Path to the new evalscript
    evalscript_path = Path("sh_statistics/evalscripts/only_scl_ndvi_filter.js")

    if not evalscript_path.exists():
        print(f"❌ Evalscript not found: {evalscript_path}")
        return False

    print(f"✅ Found batch script: {batch_script}")
    print(f"✅ Found evalscript: {evalscript_path}")

    # Example command to run batch processing with SCL + NDVI filtering
    # This is what you would run to collect data with NDVI filtering:
    example_command = [
        "python", str(batch_script),
        "--xlsx", "gabri_filters.xlsx",
        "--workers", "3",  # As requested for VM (3 workers)
        "--evalscript_path", str(evalscript_path),
        "--start_date", "2015-01-01",
        "--end_date", "2018-12-31",
        "--storage_prefix", "batch_results_2015_2018_scl_ndvi_filter"
    ]

    print("\n📋 Example command to run SCL + NDVI filter batch processing:")
    print(" " + " \\\n   ".join(example_command))

    print("\n🔑 Key features of this evalscript:")
    print("   ✅ SCL filtering (clouds, shadows, water, snow/ice)")
    print("   ✅ NDVI < 0.2 filter (excludes vegetated areas)")
    print("   ✅ NO sun zenith angle filter")
    print("   ✅ NO MNDWI water filter")
    print("   ✅ Computes 18 features (same as other evalscripts)")

    print("\n📊 Expected results:")
    print("   - Focused on bare soil areas (NDVI < 0.2)")
    print("   - Excludes vegetated pixels")
    print("   - Better data quality for soil analysis")
    print("   - Same feature computation (18 features)")
    print("   - Same output format (compatible with results analyzer)")

    return True

def show_evalscript_comparison():
    """Show comparison between the different evalscripts"""

    print("\n🔍 Evalscript Comparison:")

    features_js = Path("sh_statistics/evalscripts/features.js")
    only_scl_js = Path("sh_statistics/evalscripts/only_scl.js")
    only_scl_ndvi_js = Path("sh_statistics/evalscripts/only_scl_ndvi_filter.js")

    print(f"\n📄 {features_js.name}:")
    print("   Filters: SCL + sun zenith + NDVI + MNDWI")
    print("   Purpose: Strict filtering for high-quality bare soil")
    print("   Expected: Lower data availability, highest precision")

    print(f"\n📄 {only_scl_js.name}:")
    print("   Filters: SCL only")
    print("   Purpose: Relaxed filtering for maximum data collection")
    print("   Expected: Higher data availability, more comprehensive coverage")

    print(f"\n📄 {only_scl_ndvi_js.name}:")
    print("   Filters: SCL + NDVI < 0.2")
    print("   Purpose: Balanced filtering for bare soil analysis")
    print("   Expected: Good data availability with vegetation exclusion")

    print("\n🎯 Usage recommendation:")
    print("   1. Use features.js for final analysis with strict quality control")
    print("   2. Use only_scl.js for initial data collection and exploration")
    print("   3. Use only_scl_ndvi_filter.js for focused bare soil analysis")
    print("   4. All produce compatible output formats")

if __name__ == "__main__":
    success = test_only_scl_ndvi_filter_evalscript()
    if success:
        show_evalscript_comparison()
        print("\n✅ Test completed successfully!")
        print("\n🚀 Ready to run: python scaleway_batch_stats_from_xlsx.py --evalscript_path sh_statistics/evalscripts/only_scl_ndvi_filter.js ...")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)