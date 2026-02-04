#!/usr/bin/env python3
"""
Test script for refactored statistics module
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all modules can be imported correctly"""
    print("Testing imports...")

    try:
        # Test core modules
        from statistics.client import StatisticsApiClient, create_client_from_env
        from statistics.models import (
            FEATURE_COLS, BoundingBox, bbox_around_point_m,
            DailyStatsRecord, AggregatedStatsRecord
        )
        from statistics.processing.parsers import (
            parse_daily_records, aggregate_records
        )
        from statistics.batch.xlsx_processor import create_xlsx_processor
        from statistics.batch.workers import create_worker

        print("✓ All imports successful")

        # Test that FEATURE_COLS is correct
        expected_features = [
            "B02", "B03", "B04", "B08", "B11", "B12",
            "NDVI", "NDWI", "MNDWI", "NDMI", "BSI",
            "BRIGHT", "ALBEDO_PROXY",
            "RED", "SWIR1", "SWIR2",
            "RED_SWIR1_RATIO", "SWIR1_SWIR2_RATIO"
        ]
        assert FEATURE_COLS == expected_features, f"FEATURE_COLS mismatch: {FEATURE_COLS}"
        print(f"✓ FEATURE_COLS correct: {len(FEATURE_COLS)} features")

        # Test bbox creation
        bbox = bbox_around_point_m(45.0, 10.0, 30.0)
        assert isinstance(bbox, BoundingBox)
        assert len(bbox.to_list()) == 4
        print(f"✓ Bounding box creation works: {bbox.to_list()}")

        return True

    except Exception as e:
        print(f"✗ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_evalscript_exists():
    """Test that evalscript file exists and has correct content"""
    print("\nTesting evalscript...")

    try:
        evalscript_path = Path("statistics/evalscripts/features.js")
        assert evalscript_path.exists(), f"Evalscript not found: {evalscript_path}"

        content = evalscript_path.read_text(encoding="utf-8")
        assert "//VERSION=3" in content, "Missing VERSION=3 comment"
        assert "function setup()" in content, "Missing setup function"
        assert "function evaluatePixel" in content, "Missing evaluatePixel function"
        assert "features: new Array(18).fill(0)" in content, "Missing features array"

        print(f"✓ Evalscript exists and has correct structure: {len(content)} characters")

        return True

    except Exception as e:
        print(f"✗ Evalscript test failed: {e}")
        traceback.print_exc()
        return False

def test_module_structure():
    """Test that module structure is correct"""
    print("\nTesting module structure...")

    try:
        # Check that all expected files exist
        expected_files = [
            "statistics/__init__.py",
            "statistics/client.py",
            "statistics/models.py",
            "statistics/evalscripts/__init__.py",
            "statistics/evalscripts/features.js",
            "statistics/processing/__init__.py",
            "statistics/processing/parsers.py",
            "statistics/batch/__init__.py",
            "statistics/batch/xlsx_processor.py",
            "statistics/batch/workers.py"
        ]

        for file_path in expected_files:
            path = Path(file_path)
            assert path.exists(), f"Missing file: {file_path}"
            print(f"✓ {file_path} exists")

        return True

    except Exception as e:
        print(f"✗ Module structure test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Running refactored code tests...\n")

    tests = [
        test_imports,
        test_evalscript_exists,
        test_module_structure
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\n{'='*50}")
    print(f"Test Results: {sum(results)}/{len(results)} passed")

    if all(results):
        print("✓ All tests passed! Refactoring successful.")
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())