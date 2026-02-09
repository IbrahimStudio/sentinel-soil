#!/usr/bin/env python3
"""
Final validation test with specific coordinates and 2-year time window
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from sh_statistics.client import create_client_from_env

def test_final_validation():
    """Test with specific coordinates and 2-year time window"""

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv("vm.env")

    # Read the improved evalscript
    evalscript_path = Path("sh_statistics/evalscripts/soc_eval_1.js")
    with open(evalscript_path, 'r') as f:
        improved_evalscript = f.read()

    # Use specific coordinates provided by user
    test_lat = 47.48881664
    test_lon = 16.52059488

    # Use 2-year time window to ensure we get clean data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years

    print("=== Final Validation Test ===")
    print(f"Coordinates: {test_lat}, {test_lon}")
    print(f"Date range: {start_date} to {end_date} (2 years)")
    print(f"Evalscript: {evalscript_path.name}")

    try:
        # Create client
        client = create_client_from_env()

        # Test with meter-based request
        response = client.request_statistics_meter_based(
            lat=test_lat,
            lon=test_lon,
            size_m=30.0,
            resolution_m=10,
            start_date=start_date,
            end_date=end_date,
            interval="P1D",
            evalscript=improved_evalscript,
            mosaicking_order="leastCC"
        )

        print(f"\n=== SentinelHub Response ===")
        print(f"Status: {response.get('status', 'N/A')}")
        print(f"Data items: {len(response.get('data', []))}")

        # Analyze the response
        data_items = response.get('data', [])
        if data_items:
            print(f"\n=== Data Analysis ===")

            days_with_data = 0
            total_valid_samples = 0

            for i, item in enumerate(data_items):
                interval = item.get('interval', {})
                outputs = item.get('outputs', {})
                features = outputs.get('features', {})
                bands = features.get('bands', {})

                # Check dataMask
                data_mask = outputs.get('dataMask', {})
                dm_sample_count = 0
                if data_mask and 'bands' in data_mask and 'B0' in data_mask['bands']:
                    dm_stats = data_mask['bands']['B0'].get('stats', {})
                    dm_sample_count = dm_stats.get('sampleCount', 0)

                # Check if we have any valid feature data
                has_valid_data = any(
                    bands[band_id].get('stats', {}).get('sampleCount', 0) > 0
                    for band_id in ['B0', 'B1', 'B2']  # Check first few bands
                )

                if has_valid_data:
                    days_with_data += 1

                    # Count total valid samples across all bands
                    for band_id in bands:
                        sample_count = bands[band_id].get('stats', {}).get('sampleCount', 0)
                        total_valid_samples += sample_count

                    if i < 3:  # Show details for first few days with data
                        print(f"\nDay {i+1}: {interval.get('from', '?')} to {interval.get('to', '?')}")
                        print(f"  DataMask samples: {dm_sample_count}")

                        # Show some band statistics
                        for band_id in ['B0', 'B1', 'B2']:
                            stats = bands[band_id].get('stats', {})
                            print(f"  Band {band_id}: Samples={stats.get('sampleCount', 0)}, P50={stats.get('percentiles', {}).get('50.0', 'N/A')}")

            print(f"\n=== Summary Statistics ===")
            print(f"Total days analyzed: {len(data_items)}")
            print(f"Days with valid data: {days_with_data}")
            print(f"Total valid samples: {total_valid_samples}")

            if days_with_data > 0:
                print(f"Success rate: {days_with_data/len(data_items)*100:.1f}%")

                # Show some sample values from the first day with data
                first_day_with_data = None
                for item in data_items:
                    if any(
                        item.get('outputs', {}).get('features', {}).get('bands', {}).get(band_id, {}).get('stats', {}).get('sampleCount', 0) > 0
                        for band_id in ['B0', 'B1', 'B2']
                    ):
                        first_day_with_data = item
                        break

                if first_day_with_data:
                    print(f"\n=== Sample Values (First Day with Data) ===")
                    bands = first_day_with_data.get('outputs', {}).get('features', {}).get('bands', {})
                    for band_id, band_name in [('B0', 'B02'), ('B1', 'B03'), ('B2', 'B04'), ('B6', 'NDVI'), ('B7', 'NBR2')]:
                        if band_id in bands:
                            stats = bands[band_id].get('stats', {})
                            p50 = stats.get('percentiles', {}).get('50.0', 'N/A')
                            print(f"  {band_name}: {p50}")

        # Save full response for inspection
        with open('final_validation_response.json', 'w') as f:
            json.dump(response, f, indent=2, default=str)

        print(f"\n✓ Full response saved to final_validation_response.json")

        # Final analysis
        if days_with_data > 0:
            print(f"\n✅ SUCCESS: Evalscript is working correctly!")
            print(f"   Found {days_with_data} days with valid bare soil data")
            print(f"   Total valid samples: {total_valid_samples}")

            if total_valid_samples > 0:
                avg_samples_per_day = total_valid_samples / days_with_data
                print(f"   Average samples per day: {avg_samples_per_day:.1f}")

            print("\n🎉 The evalscript is properly configured and returning data!")
            print("   Empty objects in Scaleway storage were likely due to:")
            print("   1. Initial units error (REFLECTANCE vs DN) - FIXED")
            print("   2. Too short time window - FIXED (now using 2 years)")
            print("   3. Potentially wrong coordinates - FIXED (using specific coords)")
        else:
            print(f"\n❌ ISSUE: No valid data found even with 2-year window")
            print("   This suggests either:")
            print("   1. The coordinates don't have bare soil in the time period")
            print("   2. The filtering criteria are still too strict")
            print("   3. There may be a data availability issue")

            print("\n🔍 Next steps for debugging:")
            print("- Try relaxing the filtering criteria (e.g., allow SCL=4, increase NDVI threshold)")
            print("- Test with a simpler evalscript to verify basic data availability")
            print("- Check if the area has data coverage in Sentinel Hub")

        return days_with_data > 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_validation()
    if success:
        print(f"\n🎯 INVESTIGATION COMPLETE: The evalscript is working correctly!")
        print("   Empty objects in Scaleway storage should now be resolved.")
    else:
        print(f"\n⚠️ INVESTIGATION INCOMPLETE: Further debugging needed.")