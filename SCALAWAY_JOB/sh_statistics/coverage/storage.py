#!/usr/bin/env python3
"""
Coverage Analysis Storage Integration

Handles persistence of coverage analysis results to Scaleway object store.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sh_pipeline.storage import storage_from_env
from sh_statistics.models import CoverageResult

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

def persist_coverage_results(
    coverage_result: CoverageResult,
    local_output_dir: str,
    s3_prefix: str,
    include_raw_responses: bool = True
) -> Dict[str, str]:
    """
    Persist coverage analysis results to local directory and Scaleway object store

    Args:
        coverage_result: CoverageResult object
        local_output_dir: Local directory to save files
        s3_prefix: S3 prefix for object store
        include_raw_responses: Whether to include raw response files

    Returns:
        Dictionary mapping file types to S3 keys
    """
    # Create local output directory
    os.makedirs(local_output_dir, exist_ok=True)

    uploaded_files = {}

    # 1. Save summarized JSON
    summary_json_file = os.path.join(local_output_dir, "coverage_summary.json")
    with open(summary_json_file, 'w') as f:
        json.dump({
            'config': asdict(coverage_result.config),
            'start_date': coverage_result.start_date,
            'end_date': coverage_result.end_date,
            'summary': coverage_result.summary,
            'n_stats': len(coverage_result.coverage_stats)
        }, f, indent=2)

    # Upload to Scaleway
    storage_client = storage_from_env()
    s3_summary_key = f"{s3_prefix}/coverage_summary.json"
    storage_client.upload_file(summary_json_file, s3_summary_key)
    uploaded_files['summary_json'] = s3_summary_key

    # 2. Save detailed CSV
    csv_file = os.path.join(local_output_dir, "coverage_stats.csv")
    df = pd.DataFrame([asdict(stats) for stats in coverage_result.coverage_stats])
    df.to_csv(csv_file, index=False)

    s3_csv_key = f"{s3_prefix}/coverage_stats.csv"
    storage_client.upload_file(csv_file, s3_csv_key)
    uploaded_files['csv'] = s3_csv_key

    # 3. Save detailed Parquet
    parquet_file = os.path.join(local_output_dir, "coverage_stats.parquet")
    df.to_parquet(parquet_file)

    s3_parquet_key = f"{s3_prefix}/coverage_stats.parquet"
    storage_client.upload_file(parquet_file, s3_parquet_key)
    uploaded_files['parquet'] = s3_parquet_key

    # 4. Upload raw responses if available and requested
    if include_raw_responses:
        raw_dir = os.path.join(local_output_dir, "raw_responses")
        if os.path.exists(raw_dir):
            s3_raw_prefix = f"{s3_prefix}/raw_responses"
            uploaded_count = storage_client.upload_tree(raw_dir, s3_raw_prefix)
            uploaded_files['raw_responses'] = f"{s3_raw_prefix}/ (uploaded {uploaded_count} files)"

    # 5. Upload plots if available
    plots_dir = os.path.join(local_output_dir, "plots")
    if os.path.exists(plots_dir):
        s3_plots_prefix = f"{s3_prefix}/plots"
        uploaded_count = storage_client.upload_tree(plots_dir, s3_plots_prefix)
        uploaded_files['plots'] = f"{s3_plots_prefix}/ (uploaded {uploaded_count} files)"

    return uploaded_files

def upload_coverage_results(
    local_dir: str,
    s3_prefix: str,
    include_raw: bool = True,
    include_plots: bool = True
) -> Dict[str, str]:
    """
    Upload existing coverage results from local directory to Scaleway

    Args:
        local_dir: Local directory containing results
        s3_prefix: S3 prefix for object store
        include_raw: Whether to include raw response files
        include_plots: Whether to include plot files

    Returns:
        Dictionary mapping file types to S3 keys
    """
    storage_client = storage_from_env()
    uploaded_files = {}

    # Upload summary JSON
    summary_file = os.path.join(local_dir, "coverage_summary.json")
    if os.path.exists(summary_file):
        s3_key = f"{s3_prefix}/coverage_summary.json"
        storage_client.upload_file(summary_file, s3_key)
        uploaded_files['summary_json'] = s3_key

    # Upload CSV
    csv_file = os.path.join(local_dir, "coverage_stats.csv")
    if os.path.exists(csv_file):
        s3_key = f"{s3_prefix}/coverage_stats.csv"
        storage_client.upload_file(csv_file, s3_key)
        uploaded_files['csv'] = s3_key

    # Upload Parquet
    parquet_file = os.path.join(local_dir, "coverage_stats.parquet")
    if os.path.exists(parquet_file):
        s3_key = f"{s3_prefix}/coverage_stats.parquet"
        storage_client.upload_file(parquet_file, s3_key)
        uploaded_files['parquet'] = s3_key

    # Upload raw responses
    if include_raw:
        raw_dir = os.path.join(local_dir, "raw_responses")
        if os.path.exists(raw_dir):
            s3_raw_prefix = f"{s3_prefix}/raw_responses"
            uploaded_count = storage_client.upload_tree(raw_dir, s3_raw_prefix)
            uploaded_files['raw_responses'] = f"{s3_raw_prefix}/ (uploaded {uploaded_count} files)"

    # Upload plots
    if include_plots:
        plots_dir = os.path.join(local_dir, "plots")
        if os.path.exists(plots_dir):
            s3_plots_prefix = f"{s3_prefix}/plots"
            uploaded_count = storage_client.upload_tree(plots_dir, s3_plots_prefix)
            uploaded_files['plots'] = f"{s3_plots_prefix}/ (uploaded {uploaded_count} files)"

    return uploaded_files

def save_local_results(
    coverage_result: CoverageResult,
    output_dir: str,
    raw_responses_dir: Optional[str] = None,
    plots_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Save coverage results to local directory

    Args:
        coverage_result: CoverageResult object
        output_dir: Output directory
        raw_responses_dir: Directory containing raw responses (optional)
        plots_dir: Directory containing plots (optional)

    Returns:
        Dictionary mapping file types to local paths
    """
    os.makedirs(output_dir, exist_ok=True)

    saved_files = {}

    # Save summary JSON
    summary_file = os.path.join(output_dir, "coverage_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'config': asdict(coverage_result.config),
            'start_date': coverage_result.start_date,
            'end_date': coverage_result.end_date,
            'summary': coverage_result.summary,
            'n_stats': len(coverage_result.coverage_stats)
        }, f, indent=2)
    saved_files['summary_json'] = summary_file

    # Save CSV
    csv_file = os.path.join(output_dir, "coverage_stats.csv")
    df = pd.DataFrame([asdict(stats) for stats in coverage_result.coverage_stats])
    df.to_csv(csv_file, index=False)
    saved_files['csv'] = csv_file

    # Save Parquet
    parquet_file = os.path.join(output_dir, "coverage_stats.parquet")
    df.to_parquet(parquet_file)
    saved_files['parquet'] = parquet_file

    # Copy raw responses if provided
    if raw_responses_dir and os.path.exists(raw_responses_dir):
        import shutil
        dest_raw_dir = os.path.join(output_dir, "raw_responses")
        # Remove existing directory if it exists
        if os.path.exists(dest_raw_dir):
            shutil.rmtree(dest_raw_dir)
        shutil.copytree(raw_responses_dir, dest_raw_dir)
        saved_files['raw_responses'] = dest_raw_dir

    # Copy plots if provided
    if plots_dir and os.path.exists(plots_dir):
        import shutil
        dest_plots_dir = os.path.join(output_dir, "plots")
        # Remove existing directory if it exists
        if os.path.exists(dest_plots_dir):
            shutil.rmtree(dest_plots_dir)
        shutil.copytree(plots_dir, dest_plots_dir)
        saved_files['plots'] = dest_plots_dir

    return saved_files