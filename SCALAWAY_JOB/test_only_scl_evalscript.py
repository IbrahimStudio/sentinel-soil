#!/usr/bin/env python3
"""
Test script to demonstrate using the new only_scl.js evalscript
with the batch processing script.

This shows how to run the batch processing with relaxed filtering
(SCL-only) to collect more data.
"""

import subprocess
import sys
from pathlib import Path

def test_only_scl_evalscript():
    """Test the new SCL-only evalscript"""

    print("🧪 Testing SCL-only evalscript...")

    # Path to the batch processing script
    batch_script = Path("scaleway_batch_stats_from_xlsx.py")

    if not batch_script.exists():
        print(f"❌ Batch script not found: {batch_script}")
        return False

    # Path to the new evalscript
    evalscript_path = Path("statistics/evalscripts/only_scl.js")

    if not evalscript_path.exists():
        print(f"❌ Evalscript not found: {evalscript_path}")
        return False

    print(f"✅ Found batch script: {batch_script}")
    print(f"✅ Found evalscript: {evalscript_path}")

    # Example command to run batch processing with SCL-only filtering
    # This is what you would run to collect data with relaxed filters:
    example_command = [
        "python", str(batch_script),
        "--xlsx", "gabri_filters.xlsx",
        "--workers", "3",  # As requested for VM (3 workers)
        "--evalscript_path", str(evalscript_path),
        "--start_date", "2015-01-01",
        "--end_date", "2018-12-31",
        "--storage_prefix", "batch_results_2015_2018_scl_only"
    ]

    print("\n📋 Example command to run SCL-only batch processing:")
    print(" " + " \\\n   ".join(example_command))

    print("\n🔑 Key differences from strict filtering:")
    print("   ✅ Removed sun zenith angle filter (< 70°)")
    print("   ✅ Removed NDVI < 0.2 vegetation filter")
    print("   ✅ Removed MNDWI < 0.0 water filter")
    print("   ✅ Kept SCL filtering (clouds, shadows, water, snow/ice)")
    print("   ✅ Should collect significantly more data points")

    print("\n📊 Expected results:")
    print("   - Higher data availability rate")
    print("   - More points with valid data")
    print("   - Better temporal coverage")
    print("   - Same feature computation (18 features)")
    print("   - Same output format (compatible with results analyzer)")

    return True

def show_evalscript_comparison():
    """Show comparison between the two evalscripts"""

    print("\n🔍 Evalscript Comparison:")

    features_js = Path("statistics/evalscripts/features.js")
    only_scl_js = Path("statistics/evalscripts/only_scl.js")

    print(f"\n📄 {features_js.name}:")
    print("   Filters: SCL + sun zenith + NDVI + MNDWI")
    print("   Purpose: Strict filtering for high-quality bare soil")
    print("   Expected: Lower data availability, higher precision")

    print(f"\n📄 {only_scl_js.name}:")
    print("   Filters: SCL only")
    print("   Purpose: Relaxed filtering for maximum data collection")
    print("   Expected: Higher data availability, more comprehensive coverage")

    print("\n🎯 Usage recommendation:")
    print("   1. Use features.js for final analysis with strict quality control")
    print("   2. Use only_scl.js for initial data collection and exploration")
    print("   3. Both produce compatible output formats")

if __name__ == "__main__":
    success = test_only_scl_evalscript()
    if success:
        show_evalscript_comparison()
        print("\n✅ Test completed successfully!")
        print("\n🚀 Ready to run: python scaleway_batch_stats_from_xlsx.py --evalscript_path statistics/evalscripts/only_scl.js ...")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)