#!/usr/bin/env python3
"""
Configuration Validation and Recommendations for Sentinel Hub Statistics API

Provides validation functions and recommended configurations for different use cases.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Recommended configurations for different use cases
RECOMMENDED_CONFIGS = {
    "bare_soil_analysis": {
        "description": "Optimized for bare soil detection and analysis",
        "resolution": 10,
        "bbox_size_m": 30,
        "coverage_threshold": 0.7,
        "mosaicking_order": "leastCC",
        "notes": "Larger bbox (200m) provides robust statistics for soil features"
    },
    "vegetation_monitoring": {
        "description": "Optimized for vegetation health monitoring",
        "resolution": 10,
        "bbox_size_m": 150,
        "coverage_threshold": 0.8,
        "mosaicking_order": "leastCC",
        "notes": "Balanced configuration for vegetation indices"
    },
    "water_body_analysis": {
        "description": "Optimized for water body detection and monitoring",
        "resolution": 10,
        "bbox_size_m": 100,
        "coverage_threshold": 0.6,
        "mosaicking_order": "leastCC",
        "notes": "Lower coverage threshold for water bodies which may be partially obscured"
    },
    "urban_analysis": {
        "description": "Optimized for urban feature analysis",
        "resolution": 10,
        "bbox_size_m": 120,
        "coverage_threshold": 0.75,
        "mosaicking_order": "leastCC",
        "notes": "Medium bbox size for heterogeneous urban environments"
    },
    "high_resolution": {
        "description": "High resolution analysis (10m pixels)",
        "resolution": 10,
        "bbox_size_m": 200,
        "coverage_threshold": 0.7,
        "mosaicking_order": "leastCC",
        "notes": "Best for detailed analysis using visible/NIR bands"
    },
    "medium_resolution": {
        "description": "Medium resolution analysis (20m pixels)",
        "resolution": 20,
        "bbox_size_m": 400,
        "coverage_threshold": 0.7,
        "mosaicking_order": "leastCC",
        "notes": "Good for SWIR analysis, larger bbox compensates for lower resolution"
    }
}

def validate_resolution(resolution: int) -> bool:
    """
    Validate that resolution is supported by Sentinel Hub

    Args:
        resolution: Resolution in meters

    Returns:
        True if valid

    Raises:
        ValueError: If resolution is invalid
    """
    valid_resolutions = [10, 20, 60]
    if resolution not in valid_resolutions:
        raise ValueError(
            f"Invalid resolution {resolution}m. "
            f"Must be one of: {valid_resolutions} (native Sentinel-2 resolutions)"
        )
    return True

def validate_bbox_size(bbox_size_m: float) -> bool:
    """
    Validate bounding box size

    Args:
        bbox_size_m: Bounding box size in meters

    Returns:
        True if valid

    Raises:
        ValueError: If bbox size is invalid
    """
    if bbox_size_m <= 0:
        raise ValueError(f"Bounding box size must be positive, got {bbox_size_m}m")
    if bbox_size_m < 10:
        raise ValueError(f"Bounding box size {bbox_size_m}m is too small. Minimum 10m recommended")
    return True

def validate_config_compatibility(resolution: int, bbox_size_m: float) -> bool:
    """
    Validate that resolution and bounding box size are compatible

    Args:
        resolution: Resolution in meters
        bbox_size_m: Bounding box size in meters

    Returns:
        True if compatible

    Raises:
        ValueError: If configuration is incompatible
    """
    validate_resolution(resolution)
    validate_bbox_size(bbox_size_m)

    # Calculate expected pixel dimensions
    pixels = bbox_size_m / resolution
    total_pixels = pixels ** 2

    # Minimum viable configuration
    if pixels < 3:  # Less than 3x3 = 9 pixels
        raise ValueError(
            f"Too few pixels: {bbox_size_m}m bbox with {resolution}m resolution = {pixels:.1f} pixels. "
            f"Minimum recommended: {3*resolution}m bbox for {resolution}m resolution."
        )

    # Warn about suboptimal configurations
    if pixels < 10:  # Less than 10x10 = 100 pixels
        print(
            f"⚠️  Warning: {bbox_size_m}m bbox with {resolution}m resolution = {pixels:.1f} pixels. "
            f"Consider using larger bbox (e.g., {10*resolution}m) for better statistical reliability. "
            f"Current configuration provides only {total_pixels:.0f} total pixels."
        )

    return True

def calculate_pixel_count(resolution: int, bbox_size_m: float) -> Tuple[int, int]:
    """
    Calculate pixel dimensions and total count

    Args:
        resolution: Resolution in meters
        bbox_size_m: Bounding box size in meters

    Returns:
        Tuple of (pixels_per_side, total_pixels)
    """
    pixels_per_side = bbox_size_m / resolution
    total_pixels = pixels_per_side ** 2
    return int(pixels_per_side), int(total_pixels)

def get_recommended_config(use_case: str) -> Dict[str, any]:
    """
    Get recommended configuration for a specific use case

    Args:
        use_case: Use case name (e.g., "bare_soil_analysis")

    Returns:
        Recommended configuration dictionary

    Raises:
        ValueError: If use case is not recognized
    """
    if use_case not in RECOMMENDED_CONFIGS:
        available = list(RECOMMENDED_CONFIGS.keys())
        raise ValueError(
            f"Unknown use case '{use_case}'. Available use cases: {available}"
        )

    config = RECOMMENDED_CONFIGS[use_case].copy()
    config["use_case"] = use_case

    # Add calculated metrics
    pixels, total_pixels = calculate_pixel_count(
        config["resolution"], config["bbox_size_m"]
    )
    config["expected_pixels"] = pixels
    config["expected_total_pixels"] = total_pixels

    return config

def list_available_configs() -> List[str]:
    """
    List available recommended configurations

    Returns:
        List of available use case names
    """
    return list(RECOMMENDED_CONFIGS.keys())

def create_config_guide() -> str:
    """
    Create a formatted configuration guide

    Returns:
        Formatted string with configuration guidance
    """
    guide_lines = [
        "📋 SENTINEL HUB STATISTICS API CONFIGURATION GUIDE",
        "=" * 60,
        "",
        "🎯 RESOLUTION OPTIONS:",
        "- 10m: Best for visible/NIR bands (B02, B03, B04, B08)",
        "- 20m: Required for SWIR bands (B11, B12)",
        "- 60m: Low resolution, not recommended for most use cases",
        "",
        "📏 BOUNDING BOX GUIDELINES:",
        "- Minimum: 30m (but often results in 0 pixels with filtering)",
        "- Recommended: 100-200m for robust statistics",
        "- Formula: pixels = bbox_size_m / resolution",
        "",
        "✅ RECOMMENDED CONFIGURATIONS:"
    ]

    for use_case, config in RECOMMENDED_CONFIGS.items():
        pixels, total_pixels = calculate_pixel_count(
            config["resolution"], config["bbox_size_m"]
        )
        guide_lines.extend([
            f"",
            f"🔹 {use_case.upper()}:",
            f"  Resolution: {config['resolution']}m",
            f"  BBox Size: {config['bbox_size_m']}m",
            f"  Pixels: {pixels}x{pixels} = {total_pixels} total",
            f"  Coverage Threshold: {config['coverage_threshold']}",
            f"  Notes: {config['notes']}"
        ])

    guide_lines.extend([
        "",
        "⚠️  COMMON PITFALLS:",
        "- Using 20m resolution with 30m bbox = 1.5 pixels → often 0 valid pixels",
        "- Using 10m resolution with 30m bbox = 3 pixels → very sparse data",
        "- Forgetting that filtering (SCL, SZA, NDVI, MNDWI) reduces valid pixels",
        "",
        "💡 BEST PRACTICES:",
        "- Use 10m resolution for most soil/vegetation analysis",
        "- Use bbox_size_m ≥ 100m for reliable statistics",
        "- Consider mosaicking_order='leastCC' for better cloud handling",
        "- Adjust coverage_threshold based on expected data availability"
    ])

    return "\n".join(guide_lines)

def print_config_guide() -> None:
    """Print the configuration guide to console"""
    print(create_config_guide())