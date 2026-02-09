#!/usr/bin/env python3
"""
Test script to check if soc_eval_1.js returns data from SentinelHub
"""

import os
import json
from datetime import datetime, timedelta
from sh_statistics.client import create_client_from_env
from pathlib import Path

def test_evalscript_response():
    """Test if the evalscript returns data from SentinelHub"""

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv("vm.env")

    # Read evalscript from file
    evalscript_path = Path("sh_statistics/evalscripts/soc_eval_1.js")
    with open(evalscript_path, 'r') as f:
        soc_eval_1 = f.read()

    # Test coordinates (somewhere in Italy with likely bare soil)
    test_lat = 45.0
    test_lon = 10.0

    # Test date range (recent data)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"Testing evalscript with coordinates: {test_lat}, {test_lon}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Evalscript length: {len(soc_eval_1)} characters")

    try:
        # Create client
        client = create_client_from_env()

        # Test with meter-based request (30m x 30m area, 10m resolution)
        response = client.request_statistics_meter_based(
            lat=test_lat,
            lon=test_lon,
            size_m=30.0,
            resolution_m=10,
            start_date=start_date,
            end_date=end_date,
            interval="P1D",
            evalscript=soc_eval_1,
            mosaicking_order="leastCC"
        )

        print("\n=== SentinelHub Response ===")
        print(f"Response type: {type(response)}")
        print(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")

        if isinstance(response, dict):
            print(f"Response length: {len(response)}")

            # Check if response contains data
            if 'data' in response:
                print(f"Data found in response!")
                data = response['data']
                print(f"Data type: {type(data)}")
                print(f"Data length: {len(data) if isinstance(data, (list, dict)) else 'Not a list/dict'}")

                if isinstance(data, list) and len(data) > 0:
                    print(f"First data item: {data[0]}")
                elif isinstance(data, dict):
                    print(f"Data keys: {list(data.keys())}")
            else:
                print("No 'data' key in response")

            # Check for error messages
            if 'error' in response:
                print(f"ERROR in response: {response['error']}")

            # Print full response for debugging
            print("\n=== Full Response ===")
            print(json.dumps(response, indent=2, default=str))

        else:
            print(f"Unexpected response format: {response}")

        return response

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_evalscript_response()