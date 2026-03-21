#!/usr/bin/env python3
"""
Coverage Analysis CLI Entrypoint

Command-line interface for running coverage analysis.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from sh_statistics.coverage.analysis import run_coverage_analysis
from sh_statistics.coverage.plots import generate_coverage_plots
from sh_statistics.coverage.storage import persist_coverage_results
from sh_statistics.models import CoverageConfig

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run coverage analysis to quantify how Sentinel Hub filtering reduces data availability"
    )

    # Required arguments
    parser.add_argument(
        "--aois", required=True,
        help="JSON file containing AOI definitions or inline JSON string"
    )
    parser.add_argument(
        "--start-date", required=True,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date", required=True,
        help="End date in YYYY-MM-DD format"
    )

    # Output options
    parser.add_argument(
        "--output-dir", default="coverage_results",
        help="Local output directory (default: coverage_results)"
    )
    parser.add_argument(
        "--s3-prefix", default="coverage_analysis",
        help="S3 prefix for Scaleway object store (default: coverage_analysis)"
    )
    parser.add_argument(
        "--skip-scaleway", action="store_true",
        help="Skip uploading to Scaleway"
    )

    # Configuration options
    parser.add_argument(
        "--ndvi-threshold", type=float, default=0.2,
        help="NDVI threshold (default: 0.2)"
    )
    parser.add_argument(
        "--resolution", type=int, default=10,
        help="Resolution in meters (default: 10)"
    )
    parser.add_argument(
        "--size", type=float, default=30.0,
        help="AOI size in meters (default: 30.0)"
    )
    parser.add_argument(
        "--interval", default="P1D",
        help="Aggregation interval (default: P1D)"
    )
    parser.add_argument(
        "--mosaicking-order", default="leastCC",
        help="Mosaicking order (default: leastCC)"
    )
    parser.add_argument(
        "--scl-exclude", nargs='+', type=int,
        help="SCL classes to exclude (default: [3, 6, 8, 9, 10, 11])"
    )

    # Optional flags
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--skip-raw", action="store_true",
        help="Skip saving raw responses"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()

def load_aois(aois_input: str) -> List[Dict[str, Any]]:
    """
    Load AOI definitions from file or JSON string

    Args:
        aois_input: Path to JSON file or inline JSON string

    Returns:
        List of AOI dictionaries
    """
    try:
        # Try to parse as JSON string first
        aois = json.loads(aois_input)
        if isinstance(aois, list):
            return aois
        else:
            return [aois]
    except json.JSONDecodeError:
        # Try to read from file
        try:
            with open(aois_input, 'r') as f:
                aois = json.load(f)
                if isinstance(aois, list):
                    return aois
                else:
                    return [aois]
        except Exception as e:
            raise ValueError(f"Could not parse AOIs from '{aois_input}': {e}")

def create_config_from_args(args: argparse.Namespace) -> CoverageConfig:
    """Create CoverageConfig from command line arguments"""
    config = CoverageConfig(
        ndvi_threshold=args.ndvi_threshold,
        resolution_m=args.resolution,
        size_m=args.size,
        interval=args.interval,
        mosaicking_order=args.mosaicking_order
    )

    if args.scl_exclude:
        config.scl_exclude_classes = args.scl_exclude

    return config

def main():
    """Main CLI entrypoint"""
    try:
        args = parse_args()

        if args.verbose:
            print("=== Coverage Analysis ===")
            print(f"Start date: {args.start_date}")
            print(f"End date: {args.end_date}")
            print(f"Output directory: {args.output_dir}")
            print(f"S3 prefix: {args.s3_prefix}")
            print(f"Skip Scaleway: {args.skip_scaleway}")

        # Load AOIs
        if args.verbose:
            print(f"Loading AOIs from: {args.aois}")

        aois = load_aois(args.aois)

        if args.verbose:
            print(f"Loaded {len(aois)} AOIs:")
            for aoi in aois[:5]:  # Show first 5
                print(f"  - {aoi.get('aoi_id', 'unknown')}: ({aoi.get('lat', '?')}, {aoi.get('lon', '?')})")
            if len(aois) > 5:
                print(f"  ... and {len(aois) - 5} more")

        # Create configuration
        config = create_config_from_args(args)

        if args.verbose:
            print(f"Configuration:")
            print(f"  NDVI threshold: {config.ndvi_threshold}")
            print(f"  Resolution: {config.resolution_m}m")
            print(f"  AOI size: {config.size_m}m")
            print(f"  Interval: {config.interval}")
            print(f"  SCL exclude classes: {config.scl_exclude_classes}")

        # Run coverage analysis
        if args.verbose:
            print("Running coverage analysis...")

        raw_output_dir = os.path.join(args.output_dir, "raw_responses") if not args.skip_raw else None

        coverage_result = run_coverage_analysis(
            aois=aois,
            start_date=args.start_date,
            end_date=args.end_date,
            config=config,
            output_dir=raw_output_dir
        )

        if args.verbose:
            print("Coverage analysis completed!")
            print(f"Processed {len(coverage_result.coverage_stats)} AOI-date combinations")
            print(f"Summary statistics:")
            for key, value in coverage_result.summary.items():
                print(f"  {key}: {value}")

        # Generate plots
        if not args.skip_plots:
            if args.verbose:
                print("Generating plots...")

            plots_dir = os.path.join(args.output_dir, "plots")
            plot_files = generate_coverage_plots(
                coverage_result=coverage_result,
                output_dir=plots_dir,
                prefix="coverage"
            )

            if args.verbose:
                print(f"Generated {len(plot_files)} plots in {plots_dir}")
        else:
            plots_dir = None

        # Persist results
        if not args.skip_scaleway:
            if args.verbose:
                print("Uploading results to Scaleway...")

            uploaded_files = persist_coverage_results(
                coverage_result=coverage_result,
                local_output_dir=args.output_dir,
                s3_prefix=args.s3_prefix,
                include_raw_responses=not args.skip_raw
            )

            if args.verbose:
                print("Uploaded files:")
                for file_type, s3_key in uploaded_files.items():
                    print(f"  {file_type}: {s3_key}")
        else:
            if args.verbose:
                print("Skipping Scaleway upload as requested")

        # Save summary of what was done
        run_summary = {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'n_aois': len(aois),
            'n_results': len(coverage_result.coverage_stats),
            'config': {
                'ndvi_threshold': config.ndvi_threshold,
                'resolution_m': config.resolution_m,
                'size_m': config.size_m,
                'interval': config.interval,
                'scl_exclude_classes': config.scl_exclude_classes
            },
            'output_dir': args.output_dir,
            's3_prefix': args.s3_prefix,
            'skip_scaleway': args.skip_scaleway,
            'skip_plots': args.skip_plots,
            'skip_raw': args.skip_raw,
            'summary_stats': coverage_result.summary
        }

        summary_file = os.path.join(args.output_dir, "run_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(run_summary, f, indent=2)

        if args.verbose:
            print(f"Run summary saved to {summary_file}")

        print(f"\n=== Coverage Analysis Complete ===")
        print(f"Results saved to: {args.output_dir}")
        print(f"Processed {len(coverage_result.coverage_stats)} AOI-date combinations")
        print(f"Mean SCL saved fraction: {coverage_result.summary['mean_saved_scl']:.3f}")
        print(f"Mean SCL+NDVI saved fraction: {coverage_result.summary['mean_saved_scl_ndvi']:.3f}")

        if not args.skip_scaleway:
            print(f"Results uploaded to Scaleway with prefix: {args.s3_prefix}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()