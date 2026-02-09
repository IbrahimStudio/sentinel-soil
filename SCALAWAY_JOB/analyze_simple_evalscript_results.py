#!/usr/bin/env python3
"""
Analysis script to read and process JSONL results from Scaleway object storage
and convert them to CSV format.

Enhanced version with command-line arguments for bucket prefix and unique output naming.
"""

import json
import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pipeline.storage import storage_from_env

# Import seaborn with alias to avoid conflict with local statistics module
import seaborn as sns

def load_environment():
    """Load environment variables from vm.env file"""
    try:
        from dotenv import load_dotenv
        load_dotenv('vm.env')
    except ImportError:
        # Fallback: manually load environment variables if dotenv not available
        env_path = Path('vm.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

def get_scaleway_storage_client():
    """Create and return a Scaleway storage client"""
    return storage_from_env()

def list_jsonl_objects(storage_client, prefix: str) -> List[str]:
    """
    List all JSONL objects in the specified prefix

    Args:
        storage_client: S3StorageClient instance
        prefix: S3 prefix to search for JSONL files

    Returns:
        List of object keys
    """
    print(f"📋 Listing JSONL objects with prefix: {prefix}")
    try:
        objects = storage_client.list_objects(prefix)
        jsonl_objects = [obj for obj in objects if obj.endswith('.jsonl')]
        print(f"✅ Found {len(jsonl_objects)} JSONL objects")
        return jsonl_objects
    except Exception as e:
        print(f"❌ Error listing objects: {e}")
        raise

def read_jsonl_from_storage(storage_client, object_key: str) -> List[Dict]:
    """
    Read a JSONL file from Scaleway storage and parse it into a list of dictionaries

    Args:
        storage_client: S3StorageClient instance
        object_key: S3 object key

    Returns:
        List of parsed JSON objects
    """
    print(f"📥 Reading JSONL file: {object_key}")
    try:
        content = storage_client.get_text(object_key)
        lines = content.strip().split('\n')
        return [json.loads(line) for line in lines if line.strip()]
    except Exception as e:
        print(f"❌ Error reading {object_key}: {e}")
        raise

def flatten_json_record(record: Dict) -> Dict:
    """
    Flatten a JSON record to extract all relevant fields for CSV output

    Args:
        record: Input JSON record

    Returns:
        Flattened dictionary with all fields
    """
    # Extract basic fields
    flat_record = {
        'lat': record.get('lat'),
        'lon': record.get('lon'),
        'query_start_date': record.get('query_start_date'),
        'query_end_date': record.get('query_end_date'),
        'aggregation_interval': record.get('aggregation_interval'),
        'from_time': record.get('from_time'),
        'to_time': record.get('to_time'),
        'sample_count': record.get('sample_count'),
        'no_data_count': record.get('no_data_count'),
        'coverage': record.get('coverage')
    }

    # Extract p50 values for all bands
    p50_data = record.get('p50', {})
    for band, value in p50_data.items():
        flat_record[f'p50_{band}'] = value

    # Extract bbox coordinates
    bbox = record.get('bbox_epsg4326', [])
    if bbox and len(bbox) == 4:
        flat_record['bbox_min_x'] = bbox[0]
        flat_record['bbox_min_y'] = bbox[1]
        flat_record['bbox_max_x'] = bbox[2]
        flat_record['bbox_max_y'] = bbox[3]

    return flat_record

def process_all_jsonl_files(storage_client, prefix: str, output_csv: str) -> None:
    """
    Process all JSONL files from the specified prefix and save to CSV

    Args:
        storage_client: S3StorageClient instance
        prefix: S3 prefix to search for JSONL files
        output_csv: Path to output CSV file
    """
    print(f"🚀 Starting processing of JSONL files from prefix: {prefix}")

    # List all JSONL objects
    jsonl_objects = list_jsonl_objects(storage_client, prefix)

    if not jsonl_objects:
        print("❌ No JSONL files found in the specified prefix")
        return

    all_records = []
    total_records = 0

    # Process each JSONL file
    for i, obj_key in enumerate(jsonl_objects, 1):
        print(f"\n📖 Processing file {i}/{len(jsonl_objects)}: {obj_key}")

        try:
            # Read JSONL file
            records = read_jsonl_from_storage(storage_client, obj_key)

            # Flatten and collect records
            for record in records:
                flat_record = flatten_json_record(record)
                all_records.append(flat_record)

            total_records += len(records)
            print(f"✅ Processed {len(records)} records from {obj_key}")

        except Exception as e:
            print(f"⚠️  Skipping {obj_key} due to error: {e}")
            continue

    print(f"\n📊 Total records processed: {total_records}")

    if not all_records:
        print("❌ No records were processed successfully")
        return

    # Convert to DataFrame and save as CSV
    print(f"💾 Saving results to CSV: {output_csv}")

    df = pd.DataFrame(all_records)

    # Reorder columns for better readability
    column_order = [
        'lat', 'lon',
        'bbox_min_x', 'bbox_min_y', 'bbox_max_x', 'bbox_max_y',
        'query_start_date', 'query_end_date',
        'aggregation_interval', 'from_time', 'to_time',
        'sample_count', 'no_data_count', 'coverage'
    ]

    # Add all p50_* columns
    p50_columns = [col for col in df.columns if col.startswith('p50_')]
    column_order.extend(sorted(p50_columns))

    # Filter to only include columns that exist
    column_order = [col for col in column_order if col in df.columns]

    df = df[column_order]

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ Successfully saved {len(df)} records to {output_csv}")

    # Show some statistics
    print(f"\n📈 Statistics:")
    print(f"  - Total records: {len(df)}")
    print(f"  - Date range: {df['from_time'].min()} to {df['from_time'].max()}")
    print(f"  - Unique locations: {df[['lat', 'lon']].drop_duplicates().shape[0]}")
    print(f"  - Average coverage: {df['coverage'].mean():.3f}")

def compute_basic_statistics(csv_path: str) -> Dict[str, Any]:
    """
    Compute basic statistics from the CSV file

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dictionary containing computed statistics
    """
    print(f"📊 Computing statistics from: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter out records with NaN values in key indices (NDVI and MNDWI)
    df_clean = df[df['p50_NDVI'].notna() & df['p50_MNDWI'].notna()]

    if len(df_clean) == 0:
        print("⚠️  No records with valid NDVI/MNDWI data found")
        return {}

    stats = {}

    # Basic statistics for NDVI and MNDWI
    ndvi_col = 'p50_NDVI'
    mndwi_col = 'p50_MNDWI'

    if ndvi_col in df_clean.columns:
        stats['ndvi'] = {
            'mean': df_clean[ndvi_col].mean(),
            'median': df_clean[ndvi_col].median(),
            'std': df_clean[ndvi_col].std(),
            'min': df_clean[ndvi_col].min(),
            'max': df_clean[ndvi_col].max(),
            'count': len(df_clean[df_clean[ndvi_col].notna()])
        }

    if mndwi_col in df_clean.columns:
        stats['mndwi'] = {
            'mean': df_clean[mndwi_col].mean(),
            'median': df_clean[mndwi_col].median(),
            'std': df_clean[mndwi_col].std(),
            'min': df_clean[mndwi_col].min(),
            'max': df_clean[mndwi_col].max(),
            'count': len(df_clean[df_clean[mndwi_col].notna()])
        }

    # Additional statistics
    stats['general'] = {
        'total_records': len(df),
        'records_with_data': len(df_clean),
        'data_coverage_percentage': len(df_clean) / len(df) * 100 if len(df) > 0 else 0,
        'date_range': f"{df['from_time'].min()} to {df['from_time'].max()}",
        'unique_locations': df[['lat', 'lon']].drop_duplicates().shape[0]
    }

    # Statistics for other important bands
    important_bands = ['p50_B02', 'p50_B03', 'p50_B04', 'p50_B08', 'p50_B11', 'p50_B12']
    band_stats = {}

    for band in important_bands:
        if band in df_clean.columns:
            band_stats[band.replace('p50_', '')] = {
                'mean': df_clean[band].mean(),
                'median': df_clean[band].median(),
                'std': df_clean[band].std()
            }

    stats['bands'] = band_stats

    print(f"✅ Statistics computed successfully")
    print(f"   - Records with data: {stats['general']['records_with_data']}")
    print(f"   - Data coverage: {stats['general']['data_coverage_percentage']:.1f}%")
    print(f"   - NDVI mean: {stats['ndvi']['mean']:.4f}")
    print(f"   - MNDWI mean: {stats['mndwi']['mean']:.4f}")

    return stats

def plot_statistics(csv_path: str, output_dir: str = "stats_plots") -> None:
    """
    Generate visualizations from the CSV data

    Args:
        csv_path: Path to the CSV file
        output_dir: Directory to save plots
    """
    print(f"📈 Generating visualizations from: {csv_path}")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)

    # Filter out records with NaN values in key indices (NDVI and MNDWI)
    df_clean = df[df['p50_NDVI'].notna() & df['p50_MNDWI'].notna()]

    if len(df_clean) == 0:
        print("⚠️  No records with valid NDVI/MNDWI data found for plotting")
        return

    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("viridis")

    # 1. NDVI and MNDWI distribution plots
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df_clean['p50_NDVI'], bins=50, kde=True)
    plt.title('NDVI Distribution')
    plt.xlabel('NDVI Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(df_clean['p50_MNDWI'], bins=50, kde=True)
    plt.title('MNDWI Distribution')
    plt.xlabel('MNDWI Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ndvi_mndwi_distribution.png")
    plt.close()
    print(f"✅ Saved NDVI/MNDWI distribution plot")

    # 2. Temporal analysis - NDVI over time
    df_clean['date'] = pd.to_datetime(df_clean['from_time']).dt.date

    # Group by date and compute mean NDVI
    daily_ndvi = df_clean.groupby('date')['p50_NDVI'].mean().reset_index()

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=daily_ndvi, x='date', y='p50_NDVI')
    plt.title('Daily Average NDVI Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average NDVI')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/daily_ndvi_trend.png")
    plt.close()
    print(f"✅ Saved daily NDVI trend plot")

    # 3. Correlation matrix for spectral bands
    band_columns = [col for col in df_clean.columns if col.startswith('p50_') and col != 'p50_ALBEDO_PROXY']
    if len(band_columns) > 1:
        corr_matrix = df_clean[band_columns].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Spectral Band Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/band_correlation_matrix.png")
        plt.close()
        print(f"✅ Saved band correlation matrix plot")

    # 4. Coverage analysis
    plt.figure(figsize=(12, 6))
    sns.histplot(df['coverage'], bins=50)
    plt.title('Data Coverage Distribution')
    plt.xlabel('Coverage')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/coverage_distribution.png")
    plt.close()
    print(f"✅ Saved coverage distribution plot")

    # 5. Box plot of important bands
    important_bands = ['p50_B02', 'p50_B03', 'p50_B04', 'p50_B08', 'p50_B11', 'p50_B12', 'p50_NDVI', 'p50_MNDWI']
    important_bands = [band for band in important_bands if band in df_clean.columns]

    if len(important_bands) > 0:
        plt.figure(figsize=(14, 8))
        df_melted = df_clean[important_bands].melt(var_name='Band', value_name='Value')
        sns.boxplot(data=df_melted, x='Band', y='Value')
        plt.title('Spectral Band Value Distribution')
        plt.xlabel('Band')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/band_boxplot.png")
        plt.close()
        print(f"✅ Saved spectral band box plot")

    print(f"🎨 All visualizations saved to: {output_dir}")

def analyze_csv_data(csv_path: str, output_dir: str = "stats_plots", generate_plots: bool = True) -> Dict[str, Any]:
    """
    Main analysis function that computes statistics and generates plots

    Args:
        csv_path: Path to the CSV file
        output_dir: Directory to save plots
        generate_plots: Whether to generate visualizations

    Returns:
        Dictionary containing computed statistics
    """
    print("🔬 Starting CSV data analysis")
    print("=" * 40)

    # Compute statistics
    stats = compute_basic_statistics(csv_path)

    # Generate plots if requested
    if generate_plots:
        try:
            plot_statistics(csv_path, output_dir)
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")
            print("   Continuing with statistics only...")

    print("\n📊 Analysis Summary:")
    print(f"   - Total records: {stats.get('general', {}).get('total_records', 0)}")
    print(f"   - Records with data: {stats.get('general', {}).get('records_with_data', 0)}")
    print(f"   - Data coverage: {stats.get('general', {}).get('data_coverage_percentage', 0):.1f}%")
    print(f"   - Unique locations: {stats.get('general', {}).get('unique_locations', 0)}")

    if 'ndvi' in stats:
        print(f"\n🌱 NDVI Statistics:")
        print(f"   - Mean: {stats['ndvi']['mean']:.4f}")
        print(f"   - Median: {stats['ndvi']['median']:.4f}")
        print(f"   - Std Dev: {stats['ndvi']['std']:.4f}")
        print(f"   - Range: [{stats['ndvi']['min']:.4f}, {stats['ndvi']['max']:.4f}]")

    if 'mndwi' in stats:
        print(f"\n💧 MNDWI Statistics:")
        print(f"   - Mean: {stats['mndwi']['mean']:.4f}")
        print(f"   - Median: {stats['mndwi']['median']:.4f}")
        print(f"   - Std Dev: {stats['mndwi']['std']:.4f}")
        print(f"   - Range: [{stats['mndwi']['min']:.4f}, {stats['mndwi']['max']:.4f}]")

    print(f"\n🎉 Analysis complete!")

    return stats

def main():
    print("🔍 Scaleway JSONL Results Analyzer")
    print("=" * 50)

    # Load environment and create storage client
    load_environment()

    try:
        storage_client = get_scaleway_storage_client()
        print(f"✅ Connected to Scaleway storage bucket: {storage_client.bucket}")

        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Analyze Scaleway JSONL results")
        parser.add_argument("--prefix", type=str, required=True,
                           help="Scaleway bucket prefix (e.g., 'soil-sentinel/batch_results_2015_2018_scl_ndvi/daily_parsed')")
        parser.add_argument("--output_name", type=str, default=None,
                           help="Base name for output files (default: derived from prefix)")
        args = parser.parse_args()

        # Generate unique output names to prevent overwriting
        if args.output_name is None:
            # Extract a unique identifier from the prefix
            prefix_parts = args.prefix.split('/')
            unique_id = prefix_parts[-2] if len(prefix_parts) > 1 else "analysis"
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            args.output_name = f"{unique_id}_{timestamp}"

        # Configuration with unique filenames
        prefix = args.prefix
        output_csv = f"{args.output_name}.csv"
        output_dir = f"{args.output_name}_plots"

        print(f"📋 Configuration:")
        print(f"   - Input prefix: {prefix}")
        print(f"   - Output CSV: {output_csv}")
        print(f"   - Output plots directory: {output_dir}")

        # Process all JSONL files
        process_all_jsonl_files(storage_client, prefix, output_csv)

        # Analyze the generated CSV with unique plot directory
        stats = analyze_csv_data(output_csv, output_dir, generate_plots=True)

        print(f"\n🎉 Complete analysis complete!")
        print(f"   - Results saved to: {output_csv}")
        print(f"   - Plots saved to: {output_dir}")
        print(f"   - Total records processed: {stats.get('general', {}).get('total_records', 0)}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())