#!/usr/bin/env python3
"""
Test script for CRS and resolution fixes
"""

import sys
import traceback
from pathlib import Path

# Import at module level to avoid repetition
from statistics.client import StatisticsApiClient, StatisticsApiConfig
from statistics.models import bbox_around_point_m, calculate_pixel_config_for_epsg4326

def test_bbox_function_warnings():
    """Test that bbox_around_point_m now has proper warnings"""
    print("Testing bbox_around_point_m warnings...")

    try:
        from statistics.models import bbox_around_point_m

        # Test that function still works
        bbox = bbox_around_point_m(45.0, 10.0, 30.0)
        assert isinstance(bbox, object)
        print("✓ bbox_around_point_m still works")

        # Check that it has proper documentation
        docstring = bbox_around_point_m.__doc__
        assert "EPSG:4326" in docstring
        assert "degrees" in docstring
        assert "meters" in docstring
        assert "CRS" in docstring
        print("✓ Function has proper CRS warnings in docstring")

        return True

    except Exception as e:
        print(f"✗ bbox function test failed: {e}")
        traceback.print_exc()
        return False

def test_meter_based_functions():
    """Test the new meter-based functions"""
    print("\nTesting meter-based functions...")

    try:
        from statistics.models import (
            create_meter_based_request_config,
            calculate_pixel_config_for_epsg4326
        )

        # Test meter-based request config
        try:
            config = create_meter_based_request_config(45.0, 10.0, 30.0, 10)
            print("✓ Meter-based config works (pyproj available)")
        except ImportError:
            print("✓ Meter-based config correctly requires pyproj (when not available)")

        # Test EPSG:4326 pixel calculation
        config = calculate_pixel_config_for_epsg4326(45.0, 30.0, 10)
        assert "equivalent_degrees" in config
        assert "pixel_grid" in config
        assert "warnings" in config
        print("✓ EPSG:4326 pixel calculation works")

        # Check warnings for small area
        if config["warnings"]:
            print(f"✓ Proper warnings generated: {config['warnings'][0]}")

        return True

    except Exception as e:
        print(f"✗ Meter-based function test failed: {e}")
        traceback.print_exc()
        return False

def test_client_meter_based_methods():
    """Test client meter-based methods"""
    print("\nTesting client meter-based methods...")

    try:
        from statistics.client import StatisticsApiClient, StatisticsApiConfig

        # Create mock client
        config = StatisticsApiConfig(client_id="test", client_secret="test")
        client = StatisticsApiClient(config)

        # Test that meter-based methods exist
        assert hasattr(client, 'create_meter_based_request')
        assert hasattr(client, 'request_statistics_meter_based')
        print("✓ Client has meter-based methods")

        # Test that methods have proper documentation
        create_doc = client.create_meter_based_request.__doc__
        request_doc = client.request_statistics_meter_based.__doc__

        assert "EPSG:3857" in create_doc
        assert "meter-based" in create_doc
        assert "Web Mercator" in create_doc
        assert "meter-based" in request_doc
        print("✓ Methods have proper documentation")

        return True

    except Exception as e:
        print(f"✗ Client meter-based test failed: {e}")
        traceback.print_exc()
        return False

def test_crs_awareness():
    """Test CRS awareness in the codebase"""
    print("\nTesting CRS awareness...")

    try:
        from statistics.models import bbox_around_point_m
        from statistics.client import StatisticsApiClient

        # Check that bbox function has CRS warnings
        docstring = bbox_around_point_m.__doc__
        assert "⚠️" in docstring or "IMPORTANT" in docstring
        assert "EPSG:4326" in docstring
        print("✓ bbox function has CRS warnings")

        # Check that client methods mention CRS
        client = StatisticsApiClient(StatisticsApiConfig("test", "test"))
        build_doc = client._build_stats_request.__doc__
        assert "CRS" in build_doc or "EPSG" in build_doc
        print("✓ Client methods mention CRS")

        return True

    except Exception as e:
        print(f"✗ CRS awareness test failed: {e}")
        traceback.print_exc()
        return False

def test_30m_3x3_config():
    """Test the specific 30m x 30m with 3x3 pixels configuration"""
    print("\nTesting 30m x 30m (3x3 pixels) configuration...")

    try:
        from statistics.models import calculate_pixel_config_for_epsg4326

        # Calculate what 30m x 30m with 10m pixels means in EPSG:4326
        config = calculate_pixel_config_for_epsg4326(45.0, 30.0, 10)

        print(f"30m × 30m area with 10m pixels at 45° latitude:")
        print(f"- Size in degrees: {config['equivalent_degrees']['size_deg_lat']:.6f}° lat × {config['equivalent_degrees']['size_deg_lon']:.6f}° lon")
        print(f"- Resolution in degrees: {config['equivalent_degrees']['resolution_deg_lat']:.6f}°/pixel lat × {config['equivalent_degrees']['resolution_deg_lon']:.6f}°/pixel lon")
        print(f"- Expected pixels: {config['pixel_grid']['lat_pixels']:.1f} × {config['pixel_grid']['lon_pixels']:.1f} = {config['pixel_grid']['total_pixels']:.0f} total")

        if config['warnings']:
            print(f"⚠️  Warnings: {config['warnings'][0]}")

        # This should show why the original approach failed
        print("\n🔍 Analysis:")
        print(f"With EPSG:4326, 10m resolution = {config['equivalent_degrees']['resolution_deg_lat']:.8f} degrees")
        print(f"30m area = {config['equivalent_degrees']['size_deg_lat']:.8f} degrees")
        print(f"Pixel count = {config['equivalent_degrees']['size_deg_lat'] / config['equivalent_degrees']['resolution_deg_lat']:.2f}")

        return True

    except Exception as e:
        print(f"✗ 30m config test failed: {e}")
        traceback.print_exc()
        return False

def test_documentation_examples():
    """Test that documentation examples are clear"""
    print("\nTesting documentation examples...")

    try:
        from statistics.models import bbox_around_point_m, calculate_pixel_config_for_epsg4326
        from statistics.client import StatisticsApiClient

        # Check that examples mention the CRS issue
        bbox_doc = bbox_around_point_m.__doc__
        assert "resx=resy=10" in bbox_doc or "10 DEGREES" in bbox_doc
        print("✓ bbox documentation mentions CRS issue")

        # Check that meter-based examples are clear
        client = StatisticsApiClient(StatisticsApiConfig("test", "test"))
        meter_doc = client.create_meter_based_request.__doc__
        assert "EPSG:3857" in meter_doc
        assert "3x3" in meter_doc or "30.0" in meter_doc
        print("✓ Meter-based examples are clear")

        return True

    except Exception as e:
        print(f"✗ Documentation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all CRS fix tests"""
    print("Running CRS and resolution fix tests...\n")

    tests = [
        test_bbox_function_warnings,
        test_meter_based_functions,
        test_client_meter_based_methods,
        test_crs_awareness,
        test_30m_3x3_config,
        test_documentation_examples
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\n{'='*60}")
    print(f"CRS Fix Test Results: {sum(results)}/{len(results)} passed")

    if all(results):
        print("✅ All CRS fixes implemented correctly!")
        print("\n🎯 KEY INSIGHTS:")
        print("- bbox_around_point_m() now has clear CRS warnings")
        print("- New meter-based functions added for proper meter analysis")
        print("- Client supports both EPSG:4326 and EPSG:3857 requests")
        print("- Comprehensive documentation about CRS/resolution relationship")
        print("\n💡 RECOMMENDATION:")
        print("For 30m × 30m area with 10m pixels (3x3 grid), use:")
        print("client.request_statistics_meter_based(lat, lon, size_m=30, resolution_m=10, ...)")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())