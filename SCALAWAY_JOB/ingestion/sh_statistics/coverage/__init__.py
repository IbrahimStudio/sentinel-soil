#!/usr/bin/env python3
"""
Coverage Analysis Module

Quantifies how Sentinel Hub filtering reduces data availability for ML pipelines.
"""

from .analysis import (
    compute_coverage_metrics,
    aggregate_coverage_results,
    run_coverage_analysis
)
from .plots import (
    generate_coverage_plots,
    plot_histogram,
    plot_time_series
)
from .storage import (
    persist_coverage_results,
    upload_coverage_results
)
# CoverageConfig is imported from sh_statistics.models
