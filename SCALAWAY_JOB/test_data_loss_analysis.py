#!/usr/bin/env python3
"""
Test script for data loss analysis tools
"""

import sys
import traceback
from pathlib import Path

def test_data_loss_analysis():
    """Test the data loss analysis functionality"""
    print("Testing data loss analysis tools...")

    try:
        # Test imports
        from statistics.processing.debug_analysis import (
            DEBUG_FEATURE_COLS,
            FilterStageStats,
            DailyFilterAnalysis,
            DataLossReport,
            analyze_data_loss_from_raw_response,
            generate_data_loss_report,
            create_data_loss_summary
        )

        print("✓ Data loss analysis imports successful")

        # Test that debug feature list is correct
        expected_debug_features = [
            "B02", "B03", "B04", "B08", "B11", "B12",
            "NDVI", "NDWI", "MNDWI", "NDMI", "BSI",
            "BRIGHT", "ALBEDO_PROXY",
            "RED", "SWIR1", "SWIR2",
            "RED_SWIR1_RATIO", "SWIR1_SWIR2_RATIO",
            "DEBUG_SCL_PASS", "DEBUG_SZA_PASS", "DEBUG_NDVI_PASS", "DEBUG_MNDWI_PASS"
        ]

        assert DEBUG_FEATURE_COLS == expected_debug_features, f"DEBUG_FEATURE_COLS mismatch"
        print(f"✓ Debug feature list correct: {len(DEBUG_FEATURE_COLS)} features")

        # Test data classes
        filter_stage = FilterStageStats("TEST", 100, 50, 0.666)
        assert filter_stage.total_pixels == 150
        print("✓ FilterStageStats class works correctly")

        # Test with sample data
        sample_response = {
            "data": [
                {
                    "interval": {"from": "2023-01-01T00:00:00Z", "to": "2023-01-02T00:00:00Z"},
                    "outputs": {
                        "dataMask": {
                            "bands": {
                                "B0": {
                                    "stats": {
                                        "sampleCount": 80,
                                        "noDataCount": 20
                                    }
                                }
                            }
                        },
                        "features": {
                            "bands": {
                                # Simulate debug bands showing filter stages
                                "B18": {"stats": {"percentiles": {"50.0": 0.8}}},  # SCL pass rate
                                "B19": {"stats": {"percentiles": {"50.0": 0.7}}},  # SZA pass rate
                                "B20": {"stats": {"percentiles": {"50.0": 0.6}}},  # NDVI pass rate
                                "B21": {"stats": {"percentiles": {"50.0": 0.5}}}   # MNDWI pass rate
                            }
                        }
                    }
                }
            ]
        }

        # Test analysis with debug evalscript
        report = analyze_data_loss_from_raw_response(
            sample_response,
            lat=45.0,
            lon=10.0,
            point_id="TEST_POINT",
            coverage_threshold=0.8,
            use_debug_evalscript=True
        )

        assert report.point_id == "TEST_POINT"
        assert report.total_days == 1
        assert len(report.daily_analyses) == 1
        print("✓ Data loss analysis works with debug evalscript")

        # Test report generation
        report_data = generate_data_loss_report(report)
        assert "point_id" in report_data
        assert "overall_filter_efficiency" in report_data
        print("✓ Report generation works correctly")

        # Test summary creation
        summary = create_data_loss_summary([report_data])
        assert "summary" in summary
        assert "filter_efficiency" in summary
        print("✓ Summary creation works correctly")

        return True

    except Exception as e:
        print(f"✗ Data loss analysis test failed: {e}")
        traceback.print_exc()
        return False

def test_real_data_analysis():
    """Test with real data files if available"""
    print("\nTesting with real data files...")

    try:
        from statistics.processing.debug_analysis import analyze_data_loss_from_files

        # Check if we have any raw response files
        raw_dir = Path("out_batch_json/raw_response")
        if raw_dir.exists():
            raw_files = list(raw_dir.glob("*.json"))
            if raw_files:
                print(f"✓ Found {len(raw_files)} raw response files")

                # Analyze first few files
                test_files = raw_files[:3]
                reports = analyze_data_loss_from_files(
                    test_files,
                    coverage_threshold=0.8,
                    use_debug_evalscript=False,
                    output_dir=Path("debug_reports")
                )

                print(f"✓ Analyzed {len(reports)} files successfully")
                return True
            else:
                print("⚠ No raw response files found (this is normal for fresh setup)")
                return True
        else:
            print("⚠ No raw response directory found (this is normal for fresh setup)")
            return True

    except Exception as e:
        print(f"✗ Real data analysis test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all data loss analysis tests"""
    print("Running data loss analysis tests...\n")

    tests = [
        test_data_loss_analysis,
        test_real_data_analysis
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\n{'='*50}")
    print(f"Data Loss Analysis Test Results: {sum(results)}/{len(results)} passed")

    if all(results):
        print("✓ All data loss analysis tests passed!")
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())