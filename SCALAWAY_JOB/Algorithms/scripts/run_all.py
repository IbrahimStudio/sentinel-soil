#!/usr/bin/env python3
"""
Main CLI entrypoint for the soil texture prediction pipeline.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# Add the Algorithms directory to Python path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add the parent directory to Python path so we can import from pipeline
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Now import from src
from src.io.scaleway_s3 import get_s3_client
from src.io.parse_aggregated_json import get_json_parser
from src.data.build_dataset import get_dataset_builder
from src.analysis.pca import get_pca_analyzer
from src.modeling.train import get_model_trainer
from src.utils.validate import get_data_validator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up environment variables from vm.env if not already set."""
    vm_env_path = Path('../vm.env')
    if vm_env_path.exists():
        logger.info(f"Loading environment variables from {vm_env_path}")
        with open(vm_env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key not in os.environ:
                        os.environ[key] = value
                        logger.debug(f"Set {key} from vm.env")

def download_features(args):
    """Download features from Scaleway S3."""
    logger.info("Starting feature download from Scaleway S3")

    # Set up paths
    cache_path = Path('outputs/cache/features_raw.parquet')

    # Check if we should use cache
    if not args.refresh and cache_path.exists():
        logger.info(f"Using cached features from: {cache_path}")
        return

    try:
        # Initialize S3 client
        s3_client = get_s3_client()

        # Get S3 prefix from config or use default
        s3_prefix = args.prefix or 'soil-sentinel/batch_results_2015_2018_scl_only/aggregated/'

        # Download all JSON objects
        logger.info(f"Downloading JSON objects from prefix: {s3_prefix}")
        json_objects = s3_client.download_all_json_objects(s3_prefix)

        if not json_objects:
            logger.warning("No JSON objects found in S3 prefix")
            return

        # Parse JSON objects
        json_parser = get_json_parser()
        features_df = json_parser.parse_json_objects(json_objects)

        if features_df.empty:
            logger.warning("No features extracted from JSON objects")
            return

        # Save to cache
        json_parser.save_features_to_cache(features_df, cache_path)
        logger.info(f"Feature download completed. Saved {len(features_df)} records to cache.")

    except Exception as e:
        logger.error(f"Error in feature download: {e}")
        raise

def build_dataset(args):
    """Build dataset by joining Sentinel-2 features with Excel targets."""
    logger.info("Starting dataset building")

    try:
        # Load features from cache
        cache_path = Path('outputs/cache/features_raw.parquet')
        json_parser = get_json_parser()
        features_df = json_parser.load_features_from_cache(cache_path)

        if features_df is None or features_df.empty:
            logger.error("No features found in cache. Run download-features first.")
            return

        logger.info(f"Loaded {len(features_df)} features from cache")

        # Read Excel data
        excel_path = Path('data/gabri_filters.xlsx')
        if not excel_path.exists():
            logger.error(f"Excel file not found: {excel_path}")
            return

        dataset_builder = get_dataset_builder()
        excel_df = dataset_builder.read_excel_data(excel_path)

        # Join datasets
        dataset_df = dataset_builder.join_datasets(features_df, excel_df)

        if dataset_df.empty:
            logger.warning("Joined dataset is empty")
            return

        # Validate dataset
        validator = get_data_validator()
        required_columns = {
            'point_id': 'Point identifier',
            'lat': 'Latitude',
            'lon': 'Longitude',
            'p50_': 'Sentinel-2 band/index features',
            'clay': 'Clay content target',
            'silt': 'Silt content target',
            'sand': 'Sand content target'
        }
        validation = validator.validate_dataset(dataset_df, required_columns)

        if validation.get('errors'):
            logger.error(f"Dataset validation errors: {validation['errors']}")
            return

        # Save dataset
        output_dir = Path('outputs/datasets')
        dataset_builder.save_dataset(dataset_df, output_dir)

        logger.info(f"Dataset building completed. Final dataset: {len(dataset_df)} rows, {len(dataset_df.columns)} columns")

    except Exception as e:
        logger.error(f"Error in dataset building: {e}")
        raise

def run_pca(args):
    """Run PCA analysis on the feature matrix."""
    logger.info("Starting PCA analysis")

    try:
        # Load dataset
        dataset_path = Path('outputs/datasets/dataset.parquet')
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}. Run build-dataset first.")
            return

        dataset_df = pd.read_parquet(dataset_path)
        logger.info(f"Loaded dataset: {len(dataset_df)} rows, {len(dataset_df.columns)} columns")

        # Run PCA
        pca_analyzer = get_pca_analyzer(n_components=args.n_components)
        features_df = pca_analyzer.prepare_features(dataset_df)
        pca_analyzer.run_pca(features_df)
        pca_df = pca_analyzer.get_pca_results(features_df)

        # Create output directory
        output_dir = Path('outputs/analysis/pca')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results and plots
        pca_analyzer.plot_explained_variance(output_dir)
        pca_analyzer.plot_pca_scatter(pca_df, dataset_df, output_dir)
        pca_analyzer.save_pca_results(pca_df, output_dir)
        pca_report = pca_analyzer.create_pca_report(output_dir)

        logger.info(f"PCA analysis completed. Explained variance: {pca_report['explained_variance']}")

    except Exception as e:
        logger.error(f"Error in PCA analysis: {e}")
        raise

def train_models(args):
    """Train ML models for soil texture prediction."""
    logger.info("Starting model training")

    try:
        # Load dataset
        dataset_path = Path('outputs/datasets/dataset.parquet')
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}. Run build-dataset first.")
            return

        dataset_df = pd.read_parquet(dataset_path)
        logger.info(f"Loaded dataset: {len(dataset_df)} rows, {len(dataset_df.columns)} columns")

        # Train models
        model_trainer = get_model_trainer()
        results = model_trainer.train_all_models(dataset_df, use_pca=args.use_pca)

        # Save overall training report
        report_path = Path('outputs/models/training_report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Model training completed. Results saved to: {report_path}")

    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

def run_all(args):
    """Run the complete pipeline end-to-end."""
    logger.info("Starting complete pipeline execution")

    try:
        # Run all steps in order
        download_features(args)
        build_dataset(args)
        run_pca(args)
        train_models(args)

        logger.info("Complete pipeline execution finished successfully")

    except Exception as e:
        logger.error(f"Error in complete pipeline: {e}")
        raise

def main():
    """Main entry point."""
    setup_environment()

    parser = argparse.ArgumentParser(
        description="Soil Texture Prediction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Download features command
    download_parser = subparsers.add_parser('download-features', help='Download features from Scaleway S3')
    download_parser.add_argument(
        '--refresh',
        action='store_true',
        help="Force re-download from S3 (ignore cache)"
    )
    download_parser.add_argument(
        '--prefix',
        default=None,
        help="S3 prefix for JSON files"
    )

    # Build dataset command
    subparsers.add_parser('build-dataset', help='Build dataset by joining features with Excel targets')

    # PCA command
    pca_parser = subparsers.add_parser('pca', help='Run PCA analysis')
    pca_parser.add_argument(
        '--n-components',
        type=int,
        default=10,
        help="Number of PCA components to compute"
    )

    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument(
        '--use-pca',
        action='store_true',
        help="Use PCA features instead of raw features for modeling"
    )

    # Run all command
    run_all_parser = subparsers.add_parser('run-all', help='Run complete pipeline end-to-end')
    run_all_parser.add_argument(
        '--refresh',
        action='store_true',
        help="Force re-download from S3 (ignore cache)"
    )
    run_all_parser.add_argument(
        '--prefix',
        default=None,
        help="S3 prefix for JSON files"
    )
    run_all_parser.add_argument(
        '--n-components',
        type=int,
        default=2,
        help="Number of PCA components to compute"
    )
    run_all_parser.add_argument(
        '--use-pca',
        action='store_true',
        help="Use PCA features instead of raw features for modeling"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logger.info(f"Starting command: {args.command}")

    # Execute the appropriate function based on the command
    if args.command == 'download-features':
        download_features(args)
    elif args.command == 'build-dataset':
        build_dataset(args)
    elif args.command == 'pca':
        run_pca(args)
    elif args.command == 'train':
        train_models(args)
    elif args.command == 'run-all':
        run_all(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == '__main__':
    # main()

    run_pca()