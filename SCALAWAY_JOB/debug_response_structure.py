#!/usr/bin/env python3
"""
Debug script to examine the exact response structure from SentinelHub
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from sh_statistics.client import create_client_from_env

def debug_response_structure():
    """Debug the exact response structure from SentinelHub"""

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv("vm.env")

    # Simple evalscript
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
        var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
        var nbr2 = (s.B11 - s.B12) / (s.B11 + s.B12 + 1e-6);

        return {
            features: [s.B02, s.B03, s.B04, s.B08, s.B11, s.B12, ndvi, nbr2],
            dataMask: [1]
        };
    }
    """

    # Test coordinates and dates
    test_lat = 45.0
    test_lon = 10.0
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")  # Shorter period

    print(f"Testing with coordinates: {test_lat}, {test_lon}")
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

        print(f"\n=== Response Structure ===")
        print(f"Response keys: {list(response.keys())}")
        print(f"Response status: {response.get('status', 'N/A')}")

        if 'data' in response:
            data = response['data']
            print(f"Data type: {type(data)}")
            print(f"Data length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")

            if isinstance(data, list) and len(data) > 0:
                print(f"\n=== First Data Item ===")
                first_item = data[0]
                print(f"First item keys: {list(first_item.keys())}")

                for key, value in first_item.items():
                    print(f"\n--- {key} ---")
                    print(f"Type: {type(value)}")

                    if key == 'outputs':
                        print(f"Outputs keys: {list(value.keys())}")
                        for output_key, output_value in value.items():
                            print(f"\n--- Output: {output_key} ---")
                            print(f"Type: {type(output_value)}")
                            if isinstance(output_value, dict):
                                print(f"Keys: {list(output_value.keys())}")
                                if 'stats' in output_value:
                                    print(f"Stats: {output_value['stats']}")

        # Save full response to file for inspection
        with open('sentinelhub_response_debug.json', 'w') as f:
            json.dump(response, f, indent=2)

        print(f"\n✓ Full response saved to sentinelhub_response_debug.json")

        return response

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_response_structure()