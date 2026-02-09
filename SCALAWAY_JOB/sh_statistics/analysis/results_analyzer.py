#!/usr/bin/env python3
"""
Results Analyzer Module

Analyzes aggregated statistics results and joins them with source data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pipeline.storage import storage_from_env
from dotenv import load_dotenv

load_dotenv('vm.env')

# Feature names for analysis
FEATURE_NAMES = [
    "B02", "B03", "B04", "B08", "B11", "B12",  # Raw bands
    "NDVI", "NDWI", "MNDWI", "NDMI", "BSI",    # Spectral indices
    "BRIGHT", "ALBEDO_PROXY",                   # Brightness
    "RED", "SWIR1", "SWIR2",                    # Raw bands
    "RED_SWIR1_RATIO", "SWIR1_SWIR2_RATIO"     # Ratios
]

@dataclass
class AggregatedStatistics:
    """Comprehensive statistics computed from aggregated results"""
    total_points: int = 0
    points_with_data: int = 0
    total_aggregated_medians: int = 0
    medians_by_feature: Dict[str, int] = None
    coverage_stats: Dict[str, Any] = None
    feature_availability: Dict[str, float] = None
    time_period: Tuple[str, str] = ("", "")

    def __post_init__(self):
        if self.medians_by_feature is None:
            self.medians_by_feature = {feature: 0 for feature in FEATURE_NAMES}
        if self.coverage_stats is None:
            self.coverage_stats = {
                "min_coverage": None,
                "max_coverage": None,
                "avg_coverage": None,
                "median_coverage": None
            }
        if self.feature_availability is None:
            self.feature_availability = {feature: 0.0 for feature in FEATURE_NAMES}

@dataclass
class AnalysisConfig:
    """Configuration for results analysis"""
    storage_prefix: str = "batch_results"
    xlsx_path: Optional[Path] = None
    output_dir: Path = Path("analysis_output")
    min_coverage_threshold: float = 0.1

class ResultsAnalyzer:
    """
    Analyzes aggregated statistics results and provides comprehensive reporting
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.storage_client = None
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Storage client is initialized lazily only when needed

        # Caching mechanism for aggregated results
        self._results_cache: List[Dict[str, Any]] = []
        self._cache_loaded = False
        self._cache_hits = 0
        self._cache_misses = 0

    def _load_aggregated_results(self) -> List[Dict[str, Any]]:
        """
        Load aggregated results from Scalaway storage with caching

        Returns:
            List of aggregated result dictionaries
        """
        # Check cache first
        if self._cache_loaded and self._results_cache:
            self._cache_hits += 1
            print(f"🔄 Using cached results (cache hits: {self._cache_hits}, misses: {self._cache_misses})")
            return self._results_cache.copy()  # Return a copy to prevent external modifications

        # Cache miss - load from Scalaway
        self._cache_misses += 1
        results = []

        try:
            # Initialize storage client if not already initialized
            if self.storage_client is None:
                self.storage_client = storage_from_env()

            # List all aggregated files from Scalaway S3
            prefix = f"soil-sentinel/{self.config.storage_prefix}/aggregated/"
            print(f"🔍 Loading aggregated results from: s3://{prefix}")

            # List objects from S3
            objects = self.storage_client.list_objects(prefix)

            if not objects:
                print(f"⚠️  No objects found in s3://{prefix}")
                self._cache_loaded = True  # Mark cache as loaded (even if empty)
                return results

            # Download and parse each JSON file
            for obj_key in objects:
                try:
                    # Extract point_id from the object key (filename without extension)
                    point_id = Path(obj_key).stem
                    print(f"📥 Downloading: {obj_key}")

                    # Get the JSON content
                    json_content = self.storage_client.get_text(obj_key)
                    data = json.loads(json_content)

                    # Add point_id to the data
                    data['point_id'] = point_id
                    results.append(data)

                except Exception as e:
                    print(f"⚠️  Warning: Could not process {obj_key}: {e}")
                    continue

            # Cache the results
            self._results_cache = results
            self._cache_loaded = True

            print(f"✅ Loaded {len(results)} aggregated results from Scalaway (cached for future use)")
            return results.copy()  # Return a copy

        except Exception as e:
            print(f"❌ Error loading aggregated results from Scalaway: {e}")
            self._cache_loaded = True  # Mark cache as loaded to prevent repeated failures
            return []

    def _load_from_scaleway(self, prefix: str) -> List[Dict[str, Any]]:
        """
        Load results from Scalaway S3 storage

        Args:
            prefix: S3 prefix to list objects from

        Returns:
            List of result dictionaries
        """
        results = []

        try:
            # List objects from S3
            # Note: This would use self.storage_client.list_objects() in real implementation
            # For now, we'll return empty list as we don't have actual S3 access in this environment

            return results

        except Exception as e:
            print(f"Error loading from Scalaway: {e}")
            return []

    def compute_statistics(self) -> AggregatedStatistics:
        """
        Compute comprehensive statistics from aggregated results

        Returns:
            AggregatedStatistics object with computed metrics
        """
        stats = AggregatedStatistics()

        # Load aggregated results
        results = self._load_aggregated_results()

        if not results:
            print("⚠️  No aggregated results found")
            return stats

        stats.total_points = len(results)

        # Initialize feature counters
        for feature in FEATURE_NAMES:
            stats.medians_by_feature[feature] = 0

        coverage_values = []
        points_with_data = 0

        for result in results:
            # Update time period
            if not stats.time_period[0] or result['query_start_date'] < stats.time_period[0]:
                stats.time_period = (result['query_start_date'], stats.time_period[1])
            if not stats.time_period[1] or result['query_end_date'] > stats.time_period[1]:
                stats.time_period = (stats.time_period[0], result['query_end_date'])

            # Count points with data
            if result['n_days_kept'] > 0:
                points_with_data += 1

            # Count aggregated medians
            p50_data = result.get('p50_aggregated', {})
            if p50_data:
                for feature, value in p50_data.items():
                    if value is not None:
                        stats.medians_by_feature[feature] += 1
                        stats.total_aggregated_medians += 1

            # Collect coverage statistics
            if result['coverage_median_kept'] is not None:
                coverage_values.append(result['coverage_median_kept'])

        stats.points_with_data = points_with_data

        # Compute coverage statistics
        if coverage_values:
            stats.coverage_stats['min_coverage'] = min(coverage_values)
            stats.coverage_stats['max_coverage'] = max(coverage_values)
            stats.coverage_stats['avg_coverage'] = sum(coverage_values) / len(coverage_values)
            stats.coverage_stats['median_coverage'] = sorted(coverage_values)[len(coverage_values) // 2]

        # Compute feature availability percentages
        if stats.total_points > 0:
            for feature in FEATURE_NAMES:
                stats.feature_availability[feature] = stats.medians_by_feature[feature] / stats.total_points

        return stats

    def _load_source_xlsx(self) -> pd.DataFrame:
        """
        Load source Excel data for joining

        Returns:
            DataFrame with source data
        """
        if not self.config.xlsx_path or not self.config.xlsx_path.exists():
            raise FileNotFoundError(f"Source XLSX not found: {self.config.xlsx_path}")

        return pd.read_excel(self.config.xlsx_path)

    def join_with_source_xlsx(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Join aggregated results with source Excel data

        Args:
            output_path: Optional path to save joined data

        Returns:
            Joined DataFrame
        """
        # Load source data
        source_df = self._load_source_xlsx()

        # Load aggregated results
        results = self._load_aggregated_results()
        if not results:
            raise ValueError("No aggregated results available for joining")

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Extract p50 values to separate columns
        p50_df = pd.json_normalize(results_df['p50_aggregated'])
        results_df = pd.concat([results_df, p50_df], axis=1)
        results_df = results_df.drop('p50_aggregated', axis=1)

        # Ensure POINT_ID is available in source data
        if 'POINT_ID' not in source_df.columns:
            # Try common variations
            for col in ['point_id', 'Point_ID', 'ID']:
                if col in source_df.columns:
                    source_df = source_df.rename(columns={col: 'POINT_ID'})
                    break
            else:
                raise ValueError("Could not find POINT_ID column in source data")

        # Convert point_id to match source format
        results_df['POINT_ID'] = results_df['point_id']

        # Perform the join
        joined_df = pd.merge(
            source_df,
            results_df,
            on='POINT_ID',
            how='left'
        )

        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joined_df.to_excel(output_path, index=False)
            print(f"✅ Joined data saved to: {output_path}")

        return joined_df

    def generate_summary_report(self, report_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive summary report

        Args:
            report_path: Optional path to save JSON report

        Returns:
            Dictionary with summary report
        """
        # Compute statistics
        stats = self.compute_statistics()

        # Generate report
        report = {
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "data_source": f"s3://{self.config.storage_prefix}/aggregated/",
            "time_period": {
                "start_date": stats.time_period[0],
                "end_date": stats.time_period[1]
            },
            "summary_statistics": {
                "total_points_processed": stats.total_points,
                "points_with_valid_data": stats.points_with_data,
                "data_availability_rate": stats.points_with_data / stats.total_points if stats.total_points > 0 else 0,
                "total_aggregated_medians": stats.total_aggregated_medians,
                "average_medians_per_point": stats.total_aggregated_medians / stats.points_with_data if stats.points_with_data > 0 else 0
            },
            "coverage_statistics": stats.coverage_stats,
            "feature_statistics": {
                "total_features": len(FEATURE_NAMES),
                "features_with_data": sum(1 for count in stats.medians_by_feature.values() if count > 0),
                "median_availability_by_feature": stats.feature_availability,
                "median_counts_by_feature": stats.medians_by_feature
            },
            "quality_metrics": {
                "points_with_sufficient_coverage": stats.points_with_data,
                "coverage_success_rate": stats.points_with_data / stats.total_points if stats.total_points > 0 else 0
            }
        }

        # Save report if path provided
        if report_path:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"✅ Summary report saved to: {report_path}")

        return report

    def analyze_and_report(self, output_prefix: str = "analysis"):
        """
        Complete analysis pipeline: compute stats, join data, generate reports

        Args:
            output_prefix: Prefix for output files
        """
        print("🔍 Starting comprehensive analysis...")

        # Create output directory
        output_dir = self.output_dir / output_prefix
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Compute statistics
        print("📊 Computing statistics...")
        stats = self.compute_statistics()

        print(f"   Total points processed: {stats.total_points}")
        print(f"   Points with valid data: {stats.points_with_data}")
        print(f"   Total aggregated medians: {stats.total_aggregated_medians}")
        if stats.total_points > 0:
            print(f"   Data availability rate: {stats.points_with_data / stats.total_points:.1%}")
        else:
            print(f"   Data availability rate: 0.0%")

        # 2. Generate summary report
        print("📋 Generating summary report...")
        report = self.generate_summary_report(output_dir / f"{output_prefix}_report.json")

        # 3. Join with source data (if source XLSX provided)
        if self.config.xlsx_path:
            print("🔗 Joining with source data...")
            joined_df = self.join_with_source_xlsx(output_dir / f"{output_prefix}_joined.xlsx")
            print(f"   Joined dataset shape: {joined_df.shape}")

        print(f"✅ Analysis complete! Results saved to: {output_dir}/")

        return {
            "statistics": stats,
            "report": report,
            "output_directory": output_dir
        }

def create_results_analyzer(
    storage_prefix: str = "batch_results",
    xlsx_path: Optional[Path] = None,
    output_dir: Path = Path("analysis_output")
) -> ResultsAnalyzer:
    """
    Factory function to create ResultsAnalyzer

    Args:
        storage_prefix: Prefix for storage location
        xlsx_path: Path to source Excel file for joining
        output_dir: Directory for analysis outputs

    Returns:
        Configured ResultsAnalyzer instance
    """
    config = AnalysisConfig(
        storage_prefix=storage_prefix,
        xlsx_path=xlsx_path,
        output_dir=output_dir
    )

    return ResultsAnalyzer(config)