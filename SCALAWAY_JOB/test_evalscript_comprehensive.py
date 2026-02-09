#!/usr/bin/env python3
"""
Comprehensive test script to check evalscript behavior and data availability
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from sh_statistics.client import create_client_from_env

def test_simple_evalscript():
    """Test with a simple evalscript that returns all bands without filtering"""

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv("vm.env")

    # Simple evalscript that returns all bands
    simple_evalscript = """
    function setup() {
        return {
            input: [{
                bands: ["B02","B03","B04","B08","B11","B12","SCL","dataMask"],
                units: "DN"
            }],
            output: [
                { id: "features", bands: 8, sampleType: "FLOAT32" },
                { id: "dataMask", bands: 1, sampleType: "UINT8" }
            ]
        };
    }

    function evaluatePixel(s) {
        // Return all bands without filtering
        var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
        var nbr2 = (s.B11 - s.B12) / (s.B11 + s.B12 + 1e-6);

        return {
            features: [s.B02, s.B03, s.B04, s.B08, s.B11, s.B12, ndvi, nbr2],
            dataMask: [1]  // Keep all pixels
        };
    }
    """

    # Test coordinates and dates
    test_lat = 45.0
    test_lon = 10.0
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    print("=== Testing Simple Evalscript (No Filtering) ===")
    print(f"Coordinates: {test_lat}, {test_lon}")
    print(f"Date range: {start_date} to {end_date}")

    try:
        client = create_client_from_env()

        response = client.request_statistics_meter_based(
            lat=test_lat,
            lon=test_lon,
            size_m=30.0,
            resolution_m=10,
            start_date=start_date,
            end_date=end_date,
            interval="P1D",
            evalscript=simple_evalscript,
            mosaicking_order="leastCC"
        )

        print(f"Response status: {response.get('status', 'N/A')}")

        if 'data' in response and isinstance(response['data'], list) and len(response['data']) > 0:
            first_item = response['data'][0]
            print(f"First data item keys: {list(first_item.keys())}")

            if 'features' in first_item and 'stats' in first_item['features']:
                stats = first_item['features']['stats']
                print(f"Features stats: {json.dumps(stats, indent=2)}")

                # Check if we have actual data
                if stats.get('sampleCount', 0) > 0:
                    print("✓ Data is available!")
                    return True
                else:
                    print("✗ No data samples found")
                    return False
            else:
                print("✗ No features stats found")
                return False
        else:
            print("✗ No data in response")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_original_evalscript():
    """Test with the original soc_eval_1.js"""

    # Read original evalscript
    evalscript_path = Path("sh_statistics/evalscripts/soc_eval_1.js")
    with open(evalscript_path, 'r') as f:
        original_evalscript = f.read()

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv("vm.env")

    # Test coordinates and dates
    test_lat = 45.0
    test_lon = 10.0
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    print("\n=== Testing Original Evalscript (With Filtering) ===")
    print(f"Coordinates: {test_lat}, {test_lon}")
    print(f"Date range: {start_date} to {end_date}")

    try:
        client = create_client_from_env()

        response = client.request_statistics_meter_based(
            lat=test_lat,
            lon=test_lon,
            size_m=30.0,
            resolution_m=10,
            start_date=start_date,
            end_date=end_date,
            interval="P1D",
            evalscript=original_evalscript,
            mosaicking_order="leastCC"
        )

        print(f"Response status: {response.get('status', 'N/A')}")

        if 'data' in response and isinstance(response['data'], list) and len(response['data']) > 0:
            first_item = response['data'][0]
            print(f"First data item keys: {list(first_item.keys())}")

            if 'features' in first_item and 'stats' in first_item['features']:
                stats = first_item['features']['stats']
                print(f"Features stats: {json.dumps(stats, indent=2)}")

                # Check if we have actual data
                if stats.get('sampleCount', 0) > 0:
                    print("✓ Filtered data is available!")
                    return True
                else:
                    print("✗ No filtered data samples found (filtering may be too strict)")
                    return False
            else:
                print("✗ No features stats found")
                return False
        else:
            print("✗ No data in response")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    simple_success = test_simple_evalscript()
    original_success = test_original_evalscript()

    print("\n=== Summary ===")
    print(f"Simple evalscript (no filtering): {'✓ Success' if simple_success else '✗ Failed'}")
    print(f"Original evalscript (with filtering): {'✓ Success' if original_success else '✗ Failed'}")

    if simple_success and not original_success:
        print("\n🔍 Analysis: The evalscript is working, but the filtering criteria may be too strict")
        print("   for the test location/date range. This could result in empty objects being stored.")
    elif not simple_success:
        print("\n❌ Issue: Even the simple evalscript is not returning data. There may be a")
        print("   connectivity or configuration issue with SentinelHub.")
    else:
        print("\n✅ Both evalscripts are working correctly!")