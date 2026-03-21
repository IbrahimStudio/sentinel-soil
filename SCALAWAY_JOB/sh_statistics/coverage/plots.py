#!/usr/bin/env python3
"""
Coverage Analysis Visualization

Generates plots for coverage analysis results.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from sh_statistics.models import CoverageResult

def plot_histogram(
    coverage_result: CoverageResult,
    output_file: Optional[str] = None,
    title: str = "Distribution of Saved Fractions"
) -> Figure:
    """
    Plot histogram of saved fractions across AOIs

    Args:
        coverage_result: CoverageResult object
        output_file: Output file path (optional)
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    if not coverage_result.coverage_stats:
        raise ValueError("No coverage stats available for plotting")

    # Extract data
    saved_scl = [stats.saved_scl for stats in coverage_result.coverage_stats]
    saved_scl_ndvi = [stats.saved_scl_ndvi for stats in coverage_result.coverage_stats]

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot histograms
    plt.hist(saved_scl, bins=20, alpha=0.7, label='SCL Filtering', color='blue')
    plt.hist(saved_scl_ndvi, bins=20, alpha=0.7, label='SCL+NDVI Filtering', color='orange')

    # Add labels and title
    plt.xlabel('Saved Fraction')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add summary statistics
    summary_text = (
        f"SCL Filtering: mean={coverage_result.summary['mean_saved_scl']:.3f}, "
        f"median={coverage_result.summary['median_saved_scl']:.3f}\n"
        f"SCL+NDVI Filtering: mean={coverage_result.summary['mean_saved_scl_ndvi']:.3f}, "
        f"median={coverage_result.summary['median_saved_scl_ndvi']:.3f}"
    )

    plt.text(0.02, 0.95, summary_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save if output file provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved histogram to {output_file}")
    else:
        plt.tight_layout()

    return plt.gcf()

def plot_ecdf(
    coverage_result: CoverageResult,
    output_file: Optional[str] = None,
    title: str = "Empirical CDF of Saved Fractions"
) -> Figure:
    """
    Plot ECDF of saved fractions across AOIs

    Args:
        coverage_result: CoverageResult object
        output_file: Output file path (optional)
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    if not coverage_result.coverage_stats:
        raise ValueError("No coverage stats available for plotting")

    # Extract data
    saved_scl = np.array([stats.saved_scl for stats in coverage_result.coverage_stats])
    saved_scl_ndvi = np.array([stats.saved_scl_ndvi for stats in coverage_result.coverage_stats])

    # Sort data
    saved_scl_sorted = np.sort(saved_scl)
    saved_scl_ndvi_sorted = np.sort(saved_scl_ndvi)

    # Compute ECDF
    n = len(saved_scl_sorted)
    y = np.arange(1, n+1) / n

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot ECDF
    plt.plot(saved_scl_sorted, y, label='SCL Filtering', color='blue')
    plt.plot(saved_scl_ndvi_sorted, y, label='SCL+NDVI Filtering', color='orange')

    # Add labels and title
    plt.xlabel('Saved Fraction')
    plt.ylabel('Cumulative Probability')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add reference lines
    plt.axvline(x=coverage_result.summary['median_saved_scl'], color='blue', linestyle='--', alpha=0.7)
    plt.axvline(x=coverage_result.summary['median_saved_scl_ndvi'], color='orange', linestyle='--', alpha=0.7)

    # Save if output file provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved ECDF to {output_file}")
    else:
        plt.tight_layout()

    return plt.gcf()

def plot_time_series(
    coverage_result: CoverageResult,
    output_file: Optional[str] = None,
    title: str = "Time Series of Average Saved Fractions"
) -> Figure:
    """
    Plot time series of average saved fractions over time

    Args:
        coverage_result: CoverageResult object
        output_file: Output file path (optional)
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    if not coverage_result.coverage_stats:
        raise ValueError("No coverage stats available for plotting")

    # Group by date and compute averages
    df = pd.DataFrame([asdict(stats) for stats in coverage_result.coverage_stats])
    df['date'] = pd.to_datetime(df['date'])

    # Group by date and compute mean
    daily_stats = df.groupby('date').agg({
        'saved_scl': 'mean',
        'saved_scl_ndvi': 'mean'
    }).reset_index()

    # Create figure
    plt.figure(figsize=(14, 7))

    # Plot time series
    plt.plot(daily_stats['date'], daily_stats['saved_scl'],
             label='SCL Filtering', color='blue', marker='o', markersize=4)
    plt.plot(daily_stats['date'], daily_stats['saved_scl_ndvi'],
             label='SCL+NDVI Filtering', color='orange', marker='s', markersize=4)

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Average Saved Fraction')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save if output file provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved time series to {output_file}")
    else:
        plt.tight_layout()

    return plt.gcf()

def plot_scatter_comparison(
    coverage_result: CoverageResult,
    output_file: Optional[str] = None,
    title: str = "SCL vs SCL+NDVI Saved Fractions"
) -> Figure:
    """
    Plot scatter comparison of SCL vs SCL+NDVI filtering

    Args:
        coverage_result: CoverageResult object
        output_file: Output file path (optional)
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    if not coverage_result.coverage_stats:
        raise ValueError("No coverage stats available for plotting")

    # Extract data
    saved_scl = [stats.saved_scl for stats in coverage_result.coverage_stats]
    saved_scl_ndvi = [stats.saved_scl_ndvi for stats in coverage_result.coverage_stats]

    # Create figure
    plt.figure(figsize=(10, 10))

    # Plot scatter
    scatter = plt.scatter(saved_scl, saved_scl_ndvi, alpha=0.6,
                         c=saved_scl_ndvi, cmap='viridis')

    # Add reference lines
    max_val = max(max(saved_scl), max(saved_scl_ndvi))
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

    # Add labels and title
    plt.xlabel('Saved Fraction (SCL Filtering)')
    plt.ylabel('Saved Fraction (SCL+NDVI Filtering)')
    plt.title(title)
    plt.colorbar(scatter, label='SCL+NDVI Saved Fraction')
    plt.grid(True, alpha=0.3)

    # Save if output file provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved scatter plot to {output_file}")
    else:
        plt.tight_layout()

    return plt.gcf()

def generate_coverage_plots(
    coverage_result: CoverageResult,
    output_dir: str,
    prefix: str = "coverage"
) -> Dict[str, str]:
    """
    Generate all coverage plots and save to directory

    Args:
        coverage_result: CoverageResult object
        output_dir: Output directory
        prefix: File prefix

    Returns:
        Dictionary mapping plot type to file path
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_files = {}

    # Generate histogram
    hist_file = os.path.join(output_dir, f"{prefix}_histogram.png")
    plot_histogram(coverage_result, output_file=hist_file)
    plot_files['histogram'] = hist_file

    # Generate ECDF
    ecdf_file = os.path.join(output_dir, f"{prefix}_ecdf.png")
    plot_ecdf(coverage_result, output_file=ecdf_file)
    plot_files['ecdf'] = ecdf_file

    # Generate time series
    ts_file = os.path.join(output_dir, f"{prefix}_time_series.png")
    plot_time_series(coverage_result, output_file=ts_file)
    plot_files['time_series'] = ts_file

    # Generate scatter plot
    scatter_file = os.path.join(output_dir, f"{prefix}_scatter.png")
    plot_scatter_comparison(coverage_result, output_file=scatter_file)
    plot_files['scatter'] = scatter_file

    return plot_files

def asdict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to dict (helper function)"""
    if hasattr(obj, '__dataclass_fields__'):
        return {f.name: asdict(getattr(obj, f.name)) for f in obj.__dataclass_fields__.values()}
    elif isinstance(obj, (list, tuple)):
        return [asdict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: asdict(v) for k, v in obj.items()}
    else:
        return obj