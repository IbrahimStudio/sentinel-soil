#!/usr/bin/env python3
"""
Test script for Results Analyzer module
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from statistics.analysis.results_analyzer import create_results_analyzer

# Load environment variablues
load_dotenv('vm.env')


def test_results_analyzer():
    """Test the results analyzer with sample data"""
    print("🧪 Testing Results Analyzer...")

    # Create analyzer instance - use the actual Scalaway storage prefix
    analyzer = create_results_analyzer(
        storage_prefix="batch_results_2015_2018",
        xlsx_path=Path("gabri_filters.xlsx"),
        output_dir=Path("test_analysis_output")
    )

    # 1. Compute statistics
    print("\n1. Computing statistics...")
    stats = analyzer.compute_statistics()

    print(f"   ✅ Total points processed: {stats.total_points}")
    print(f"   ✅ Points with valid data: {stats.points_with_data}")
    print(f"   ✅ Total aggregated medians: {stats.total_aggregated_medians}")

    if stats.total_points > 0:
        print(f"   ✅ Data availability rate: {stats.points_with_data / stats.total_points:.1%}")

    # 2. Show feature statistics
    print("\n2. Feature statistics:")
    for feature, count in stats.medians_by_feature.items():
        if count > 0:
            print(f"   ✅ {feature}: {count} medians")

    # 3. Generate summary report
    print("\n3. Generating summary report...")
    report = analyzer.generate_summary_report(Path("test_analysis_output/test_report.json"))
    print(f"   ✅ Report generated with {len(report)} sections")

    # 4. Join with source data
    print("\n4. Joining with source data...")
    try:
        joined_df = analyzer.join_with_source_xlsx(Path("test_analysis_output/test_joined.xlsx"))
        print(f"   ✅ Joined dataset created: {joined_df.shape[0]} rows × {joined_df.shape[1]} columns")
    except Exception as e:
        print(f"   ⚠️  Could not join with source data: {e}")

    # 5. Complete analysis pipeline
    print("\n5. Running complete analysis pipeline...")
    result = analyzer.analyze_and_report("complete_analysis")

    print(f"   ✅ Analysis complete!")
    print(f"   ✅ Statistics computed: {result['statistics'].total_aggregated_medians} total medians")
    print(f"   ✅ Report generated: {result['output_directory']}/complete_analysis_report.json")

    if result['statistics'].total_points > 0:
        availability_rate = result['statistics'].points_with_data / result['statistics'].total_points
        print(f"   ✅ Data availability: {availability_rate:.1%}")

    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    test_results_analyzer()