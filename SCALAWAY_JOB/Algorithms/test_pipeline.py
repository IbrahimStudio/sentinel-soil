#!/usr/bin/env python3
"""
Test script to validate the pipeline components.
"""

import sys
import os
from pathlib import Path

# Add the Algorithms directory to Python path so we can import from src
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.io.scaleway_s3 import get_s3_client
        print("✓ Scaleway S3 client")

        from src.io.parse_aggregated_json import get_json_parser
        print("✓ JSON parser")

        from src.data.build_dataset import get_dataset_builder
        print("✓ Dataset builder")

        from src.analysis.pca import get_pca_analyzer
        print("✓ PCA analyzer")

        from src.modeling.train import get_model_trainer
        print("✓ Model trainer")

        from src.utils.validate import get_data_validator
        print("✓ Data validator")

        print("All imports successful!")

    except Exception as e:
        print(f"Import error: {e}")
        return False

    return True

def test_s3_client():
    """Test S3 client initialization."""
    print("\nTesting S3 client...")

    try:
        from src.io.scaleway_s3 import get_s3_client

        # Set up environment variables from vm.env
        vm_env_path = Path('../vm.env')
        if vm_env_path.exists():
            with open(vm_env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if key not in os.environ:
                            os.environ[key] = value

        s3_client = get_s3_client()
        print(f"✓ S3 client initialized for bucket: {s3_client.client.bucket}")

        # Test listing objects (just a few to avoid long wait)
        prefix = 'soil-sentinel/batch_results_2015_2018_scl_ndvi/aggregated/'
        objects = s3_client.list_objects(prefix)
        print(f"✓ Found {len(objects)} objects in S3 prefix")

    except Exception as e:
        print(f"S3 client error: {e}")
        return False

    return True

def test_json_parser():
    """Test JSON parser with sample data."""
    print("\nTesting JSON parser...")

    try:
        from src.io.parse_aggregated_json import get_json_parser

        # Create sample JSON data
        sample_json = {
            "point_id": "test_point",
            "lat": 45.0,
            "lon": 12.0,
            "n_days_total": 10,
            "n_days_kept": 8,
            "kept_ratio": 0.8,
            "p50_aggregated": {
                "B02": 0.1,
                "B03": 0.2,
                "B04": 0.3,
                "NDVI": 0.5,
                "NDWI": 0.4
            }
        }

        parser = get_json_parser()
        features = parser.extract_features_from_json(sample_json, "test_point")

        print(f"✓ Extracted {len(features)} features from sample JSON")
        print(f"  Features: {list(features.keys())}")

    except Exception as e:
        print(f"JSON parser error: {e}")
        return False

    return True

def test_dataset_builder():
    """Test dataset builder with sample data."""
    print("\nTesting dataset builder...")

    try:
        from src.data.build_dataset import get_dataset_builder
        import pandas as pd
        import tempfile

        # Create sample features DataFrame
        features_data = {
            'point_id': ['point1', 'point2', 'point3'],
            'lat': [45.0, 46.0, 47.0],
            'lon': [12.0, 13.0, 14.0],
            'p50_B02': [0.1, 0.2, 0.3],
            'p50_NDVI': [0.5, 0.6, 0.7]
        }
        features_df = pd.DataFrame(features_data).set_index('point_id')

        # Create sample Excel data
        excel_data = {
            'POINT_ID': ['point1', 'point2', 'point4'],  # point3 missing
            'clay': [20.0, 25.0, 30.0],
            'silt': [40.0, 45.0, 50.0],
            'sand': [40.0, 30.0, 20.0]
        }
        excel_df = pd.DataFrame(excel_data).set_index('POINT_ID')

        builder = get_dataset_builder()
        joined_df = builder.join_datasets(features_df, excel_df)

        print(f"✓ Joined dataset: {len(joined_df)} rows (expected 2, since point3 missing from Excel)")
        print(f"  Columns: {list(joined_df.columns)}")

    except Exception as e:
        print(f"Dataset builder error: {e}")
        return False

    return True

def main():
    """Run all tests."""
    print("🧪 Testing Soil Texture Prediction Pipeline")
    print("=" * 50)

    tests = [
        test_imports,
        test_s3_client,
        test_json_parser,
        test_dataset_builder
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\nTo run the complete pipeline:")
        print("cd Algorithms")
        print("python -m scripts.run_all run-all")
    else:
        print("❌ Some tests failed. Check the error messages above.")

    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)