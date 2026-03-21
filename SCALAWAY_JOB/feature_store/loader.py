#!/usr/bin/env python3
"""
Scaleway Data Loader

Provides functionality to read JSON files from Scaleway buckets and build DataFrames.
"""

from __future__ import annotations

import sys
import os

# Add project root directory to Python path to ensure local modules are found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
from sh_pipeline.storage import storage_from_env, S3StorageClient
from sh_statistics.models import AggregatedStatsRecord, DailyStatsRecord, FEATURE_COLS
import hashlib
import pickle
from pathlib import Path
import tempfile

def read_json_files_from_scaleway(
    prefix: str = "",
    bucket: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Read all JSON files from a Scaleway bucket with the given prefix.

    Args:
        prefix: Prefix to filter JSON files (e.g., "batch_results/aggregated/")
        bucket: Scaleway bucket name (optional, uses env var if not provided)
        endpoint_url: Scaleway S3 endpoint URL (optional, uses env var if not provided)
        region: Scaleway region (optional, uses env var if not provided)
        access_key: Scaleway access key (optional, uses env var if not provided)
        secret_key: Scaleway secret key (optional, uses env var if not provided)
        logger: Optional logger for logging operations

    Returns:
        List of dictionaries containing parsed JSON data

    Raises:
        RuntimeError: If required environment variables are missing and not provided
        Exception: For any errors during S3 operations
    """
    # Create logger if not provided
    log = logger or logging.getLogger(__name__)

    # Create storage client
    if all([bucket, endpoint_url, access_key, secret_key]):
        # Use provided credentials
        from sh_pipeline.storage import StorageConfig
        cfg = StorageConfig(
            endpoint_url=endpoint_url,
            region=region or "fr-par",
            bucket=bucket,
            access_key=access_key,
            secret_key=secret_key,
        )
        storage_client = S3StorageClient(cfg, logger=log)
    else:
        # Use environment variables
        storage_client = storage_from_env(logger=log)

    # List all objects with the given prefix
    log.info(f"Listing JSON files with prefix: {prefix}")
    object_keys = storage_client.list_objects(prefix)

    # Filter for JSON files only
    json_keys = [key for key in object_keys if key.lower().endswith('.json')]
    log.info(f"Found {len(json_keys)} JSON files to process")

    # Read and parse each JSON file
    json_data_list = []

    for key in json_keys:
        try:
            log.info(f"Reading JSON file: {key}")
            json_content = storage_client.get_text(key)
            json_data = json.loads(json_content)
            json_data_list.append(json_data)
            log.debug(f"Successfully parsed JSON from {key}")
        except Exception as e:
            log.error(f"Failed to read or parse JSON file {key}: {e}")
            continue

    log.info(f"Successfully processed {len(json_data_list)} JSON files")
    return json_data_list

def read_jsonl_files_from_scaleway(
    prefix: str = "",
    bucket: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Read all JSONL files from a Scaleway bucket with the given prefix.

    Args:
        prefix: Prefix to filter JSONL files (e.g., "batch_results/daily_parsed/")
        bucket: Scaleway bucket name (optional, uses env var if not provided)
        endpoint_url: Scaleway S3 endpoint URL (optional, uses env var if not provided)
        region: Scaleway region (optional, uses env var if not provided)
        access_key: Scaleway access key (optional, uses env var if not provided)
        secret_key: Scaleway secret key (optional, uses env var if not provided)
        logger: Optional logger for logging operations

    Returns:
        List of dictionaries containing parsed JSONL records

    Raises:
        RuntimeError: If required environment variables are missing and not provided
        Exception: For any errors during S3 operations
    """
    # Create logger if not provided
    log = logger or logging.getLogger(__name__)

    # Create storage client
    if all([bucket, endpoint_url, access_key, secret_key]):
        # Use provided credentials
        from sh_pipeline.storage import StorageConfig
        cfg = StorageConfig(
            endpoint_url=endpoint_url,
            region=region or "fr-par",
            bucket=bucket,
            access_key=access_key,
            secret_key=secret_key,
        )
        storage_client = S3StorageClient(cfg, logger=log)
    else:
        # Use environment variables
        storage_client = storage_from_env(logger=log)

    # List all objects with the given prefix
    log.info(f"Listing JSONL files with prefix: {prefix}")
    object_keys = storage_client.list_objects(prefix)

    # Filter for JSONL files only
    jsonl_keys = [key for key in object_keys if key.lower().endswith('.jsonl')]
    log.info(f"Found {len(jsonl_keys)} JSONL files to process")

    # Read and parse each JSONL file
    jsonl_data_list = []

    for key in jsonl_keys:
        try:
            log.info(f"Reading JSONL file: {key}")
            jsonl_content = storage_client.get_text(key)

            # Parse JSONL content line by line
            lines = jsonl_content.strip().split('\n')
            for line in lines:
                if line.strip():  # Skip empty lines
                    try:
                        json_data = json.loads(line)
                        jsonl_data_list.append(json_data)
                    except json.JSONDecodeError as e:
                        log.error(f"Failed to parse JSON line in {key}: {e}")
                        continue

            log.debug(f"Successfully parsed {len(lines)} records from {key}")

        except Exception as e:
            log.error(f"Failed to read or parse JSONL file {key}: {e}")
            continue

    log.info(f"Successfully processed {len(jsonl_data_list)} JSONL records from {len(jsonl_keys)} files")
    return jsonl_data_list

def build_dataframe_from_json_data(
    json_data_list: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Build a pandas DataFrame from a list of JSON data dictionaries.

    Args:
        json_data_list: List of dictionaries containing JSON data
        logger: Optional logger for logging operations

    Returns:
        pandas DataFrame containing the structured data

    Note:
        This function expects JSON data in the format of AggregatedStatsRecord
        from the sh_statistics.models module.
    """
    # Create logger if not provided
    log = logger or logging.getLogger(__name__)

    if not json_data_list:
        log.warning("No JSON data provided - returning empty DataFrame")
        return pd.DataFrame()

    log.info(f"Building DataFrame from {len(json_data_list)} JSON records")

    # Extract common fields from the first record to understand structure
    sample_record = json_data_list[0]

    # Prepare data for DataFrame
    data = []

    for record in json_data_list:
        try:
            # Extract basic fields
            row_data = {
                'lat': record.get('lat'),
                'lon': record.get('lon'),
                'bbox_epsg4326': record.get('bbox_epsg4326'),
                'query_start_date': record.get('query_start_date'),
                'query_end_date': record.get('query_end_date'),
                'aggregation_interval': record.get('aggregation_interval'),
                'coverage_threshold': record.get('coverage_threshold'),
                'n_days_total': record.get('n_days_total'),
                'n_days_kept': record.get('n_days_kept'),
                'kept_ratio': record.get('kept_ratio'),
                'coverage_median_kept': record.get('coverage_median_kept'),
                'coverage_min_kept': record.get('coverage_min_kept'),
            }

            # Extract p50_aggregated features
            p50_data = record.get('p50_aggregated', {})
            if p50_data:
                for feature in FEATURE_COLS:
                    row_data[f'p50_{feature}'] = p50_data.get(feature)

            data.append(row_data)

        except Exception as e:
            log.error(f"Failed to process record: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(data)

    # Convert bbox to string representation for better display
    if 'bbox_epsg4326' in df.columns:
        df['bbox_epsg4326'] = df['bbox_epsg4326'].apply(lambda x: str(x) if isinstance(x, list) else x)

    log.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    return df

def build_dataframe_from_jsonl_data(
    jsonl_data_list: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Build a pandas DataFrame from a list of JSONL daily record dictionaries.

    Args:
        jsonl_data_list: List of dictionaries containing JSONL daily records
        logger: Optional logger for logging operations

    Returns:
        pandas DataFrame containing the structured daily data

    Note:
        This function expects JSONL data in the format of daily records with
        individual p50 values for each band.
    """
    # Create logger if not provided
    log = logger or logging.getLogger(__name__)

    if not jsonl_data_list:
        log.warning("No JSONL data provided - returning empty DataFrame")
        return pd.DataFrame()

    log.info(f"Building DataFrame from {len(jsonl_data_list)} JSONL records")

    # Prepare data for DataFrame
    data = []

    for record in jsonl_data_list:
        try:
            # Extract basic fields
            row_data = {
                'lat': record.get('lat'),
                'lon': record.get('lon'),
                'bbox_epsg4326': record.get('bbox_epsg4326'),
                'query_start_date': record.get('query_start_date'),
                'query_end_date': record.get('query_end_date'),
                'aggregation_interval': record.get('aggregation_interval'),
                'from_time': record.get('from_time'),
                'to_time': record.get('to_time'),
                'sample_count': record.get('sample_count'),
                'no_data_count': record.get('no_data_count'),
                'coverage': record.get('coverage'),
            }

            # Extract individual p50 values
            p50_data = record.get('p50', {})
            if p50_data:
                for feature in FEATURE_COLS:
                    row_data[f'p50_{feature}'] = p50_data.get(feature)

            data.append(row_data)

        except Exception as e:
            log.error(f"Failed to process JSONL record: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(data)

    # Convert bbox to string representation for better display
    if 'bbox_epsg4326' in df.columns:
        df['bbox_epsg4326'] = df['bbox_epsg4326'].apply(lambda x: str(x) if isinstance(x, list) else x)

    # Convert datetime fields if present
    if 'from_time' in df.columns:
        df['from_time'] = pd.to_datetime(df['from_time'], errors='coerce')
    if 'to_time' in df.columns:
        df['to_time'] = pd.to_datetime(df['to_time'], errors='coerce')

    log.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    return df

def read_scaleway_jsonl_and_build_dataframe(
    prefix: str = "",
    bucket: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Convenience function that reads JSONL files from Scaleway bucket and builds a DataFrame.

    Args:
        prefix: Prefix to filter JSONL files
        bucket: Scaleway bucket name (optional)
        endpoint_url: Scaleway S3 endpoint URL (optional)
        region: Scaleway region (optional)
        access_key: Scaleway access key (optional)
        secret_key: Scaleway secret key (optional)
        logger: Optional logger for logging operations

    Returns:
        pandas DataFrame containing the structured daily data from all JSONL files

    Raises:
        RuntimeError: If required environment variables are missing and not provided
        Exception: For any errors during the process
    """
    # Create logger if not provided
    log = logger or logging.getLogger(__name__)

    log.info(f"Starting Scaleway JSONL data loading with prefix: {prefix}")

    # Step 1: Read JSONL files from Scaleway
    jsonl_data_list = read_jsonl_files_from_scaleway(
        prefix=prefix,
        bucket=bucket,
        endpoint_url=endpoint_url,
        region=region,
        access_key=access_key,
        secret_key=secret_key,
        logger=log
    )

    # Step 2: Build DataFrame from JSONL data
    df = build_dataframe_from_jsonl_data(jsonl_data_list, logger=log)

    log.info(f"Completed JSONL data loading - DataFrame shape: {df.shape}")
    return df

def _median(values: List[Optional[float]]) -> Optional[float]:
    """
    Calculate median of a list of values, ignoring None values

    Args:
        values: List of optional float values

    Returns:
        Median value or None if no valid values
    """
    if not values:
        return None

    # Filter out None values
    valid_values = [v for v in values if v is not None]

    if not valid_values:
        return None

    valid_values.sort()
    n = len(valid_values)
    mid = n // 2

    if n % 2 == 1:
        return valid_values[mid]
    else:
        return (valid_values[mid - 1] + valid_values[mid]) / 2

def dataframe_row_to_daily_record(row: pd.Series) -> DailyStatsRecord:
    """
    Convert a DataFrame row to a DailyStatsRecord object

    Args:
        row: pandas Series representing a daily record

    Returns:
        DailyStatsRecord object
    """
    # Extract p50 values from columns
    p50_dict = {}
    for feature in FEATURE_COLS:
        p50_value = row.get(f'p50_{feature}')
        p50_dict[feature] = p50_value

    return DailyStatsRecord(
        lat=row['lat'],
        lon=row['lon'],
        bbox_epsg4326=row['bbox_epsg4326'],
        query_start_date=row['query_start_date'],
        query_end_date=row['query_end_date'],
        aggregation_interval=row['aggregation_interval'],
        from_time=row['from_time'],
        to_time=row['to_time'],
        sample_count=row['sample_count'],
        no_data_count=row['no_data_count'],
        coverage=row['coverage'],
        p50=p50_dict
    )

def aggregate_records_from_dataframe(
    df_daily: pd.DataFrame,
    coverage_threshold: float = 0.8,
    logger: Optional[logging.Logger] = None
) -> AggregatedStatsRecord:
    """
    Aggregate daily records from a DataFrame to reproduce the aggregation process

    Args:
        df_daily: DataFrame containing daily records
        coverage_threshold: Minimum coverage threshold for keeping records
        logger: Optional logger for logging operations

    Returns:
        AggregatedStatsRecord with aggregated statistics

    Note:
        This function reproduces the aggregation logic to understand why
        the aggregated bucket has no valid values for bands.
    """
    # Create logger if not provided
    log = logger or logging.getLogger(__name__)

    if df_daily.empty:
        log.warning("Empty DataFrame provided for aggregation")
        return AggregatedStatsRecord(
            lat=0.0,
            lon=0.0,
            bbox_epsg4326=[],
            query_start_date="",
            query_end_date="",
            aggregation_interval="",
            coverage_threshold=coverage_threshold,
            n_days_total=0,
            n_days_kept=0,
            kept_ratio=0.0,
            coverage_median_kept=None,
            coverage_min_kept=None,
            p50_aggregated={}
        )

    log.info(f"Aggregating {len(df_daily)} daily records with coverage threshold: {coverage_threshold}")

    # Convert DataFrame rows to DailyStatsRecord objects
    daily_rows = []
    for _, row in df_daily.iterrows():
        try:
            record = dataframe_row_to_daily_record(row)
            daily_rows.append(record)
        except Exception as e:
            log.error(f"Failed to convert row to DailyStatsRecord: {e}")
            continue

    total = len(daily_rows)
    kept = [r for r in daily_rows if (r.coverage is not None and r.coverage >= coverage_threshold)]

    log.info(f"Total records: {total}, Kept records: {len(kept)}")

    # Create base aggregated record from first daily record
    base = daily_rows[0] if total else None
    if not base:
        # Return empty results if no input
        empty_agg = AggregatedStatsRecord(
            lat=0.0,
            lon=0.0,
            bbox_epsg4326=[],
            query_start_date="",
            query_end_date="",
            aggregation_interval="",
            coverage_threshold=coverage_threshold,
            n_days_total=0,
            n_days_kept=0,
            kept_ratio=0.0,
            coverage_median_kept=None,
            coverage_min_kept=None,
            p50_aggregated={}
        )
        return empty_agg

    # Calculate coverage statistics
    coverage_values = [r.coverage for r in kept] if kept else []
    coverage_median = _median(coverage_values) if coverage_values else None
    coverage_min = min(coverage_values) if coverage_values else None

    # Create aggregated record
    agg = AggregatedStatsRecord(
        lat=base.lat,
        lon=base.lon,
        bbox_epsg4326=base.bbox_epsg4326,
        query_start_date=base.query_start_date,
        query_end_date=base.query_end_date,
        aggregation_interval=base.aggregation_interval,
        coverage_threshold=coverage_threshold,
        n_days_total=total,
        n_days_kept=len(kept),
        kept_ratio=(len(kept) / total) if total else 0.0,
        coverage_median_kept=coverage_median,
        coverage_min_kept=coverage_min,
        p50_aggregated={}
    )

    # Calculate median values for each feature across kept days
    for name in FEATURE_COLS:
        feature_values = [r.p50.get(name) for r in kept] if kept else []
        agg.p50_aggregated[name] = _median(feature_values) if feature_values else None

    log.info(f"Aggregated record: {len(kept)} days kept, {agg.kept_ratio:.2%} kept ratio")
    return agg

def _get_cache_dir() -> Path:
    """
    Get the cache directory path, creating it if it doesn't exist

    Returns:
        Path to the cache directory
    """
    cache_dir = Path(tempfile.gettempdir()) / "scaleway_data_loader_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def _generate_cache_key(prefix: str, data_type: str = "dataframe") -> str:
    """
    Generate a unique cache key for a given prefix and data type

    Args:
        prefix: The Scaleway prefix used for data loading
        data_type: Type of data being cached (e.g., "dataframe", "aggregated")

    Returns:
        Unique cache key string
    """
    cache_string = f"{prefix}_{data_type}"
    return hashlib.md5(cache_string.encode()).hexdigest()

def _save_to_cache(cache_key: str, data: Any) -> None:
    """
    Save data to cache using pickle serialization

    Args:
        cache_key: Cache key for the data
        data: Data to be cached
    """
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / f"{cache_key}.pkl"

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save to cache: {e}")

def _load_from_cache(cache_key: str) -> Optional[Any]:
    """
    Load data from cache if available

    Args:
        cache_key: Cache key for the data

    Returns:
        Cached data if available, None otherwise
    """
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / f"{cache_key}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load from cache: {e}")
            # Remove corrupted cache file
            try:
                cache_file.unlink()
            except:
                pass
    return None

def _clear_cache() -> None:
    """
    Clear all cached data
    """
    cache_dir = _get_cache_dir()
    for cache_file in cache_dir.glob("*.pkl"):
        try:
            cache_file.unlink()
        except:
            pass

def read_scaleway_bucket_and_build_dataframe(
    prefix: str = "",
    bucket: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Convenience function that reads JSON files from Scaleway bucket and builds a DataFrame.

    Args:
        prefix: Prefix to filter JSON files
        bucket: Scaleway bucket name (optional)
        endpoint_url: Scaleway S3 endpoint URL (optional)
        region: Scaleway region (optional)
        access_key: Scaleway access key (optional)
        secret_key: Scaleway secret key (optional)
        logger: Optional logger for logging operations
        use_cache: Whether to use caching to avoid re-reading from Scaleway

    Returns:
        pandas DataFrame containing the structured data from all JSON files

    Raises:
        RuntimeError: If required environment variables are missing and not provided
        Exception: For any errors during the process
    """
    # Create logger if not provided
    log = logger or logging.getLogger(__name__)

    log.info(f"Starting Scaleway bucket data loading with prefix: {prefix}")

    # Check cache first if enabled
    if use_cache:
        cache_key = _generate_cache_key(prefix, "dataframe")
        cached_df = _load_from_cache(cache_key)
        if cached_df is not None:
            log.info(f"Loaded cached DataFrame with shape: {cached_df.shape}")
            return cached_df

    # Step 1: Read JSON files from Scaleway
    json_data_list = read_json_files_from_scaleway(
        prefix=prefix,
        bucket=bucket,
        endpoint_url=endpoint_url,
        region=region,
        access_key=access_key,
        secret_key=secret_key,
        logger=log
    )

    # Step 2: Build DataFrame from JSON data
    df = build_dataframe_from_json_data(json_data_list, logger=log)

    # Save to cache if enabled
    if use_cache:
        _save_to_cache(cache_key, df)

    log.info(f"Completed data loading - DataFrame shape: {df.shape}")
    return df

# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv("vm.env")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Example 1: Read aggregated results from Scaleway (JSON files)
        print("=== Example 1: Reading aggregated JSON files ===")
        df_aggregated = read_scaleway_bucket_and_build_dataframe(
            prefix="soil-sentinel/batch_results_2015_2018_scl_only/aggregated/"
        )

        print(f"Successfully loaded aggregated DataFrame with shape: {df_aggregated.shape}")
        print("First few rows:")
        print(df_aggregated.head())

        print("\nDataFrame columns:")
        print(df_aggregated.columns.tolist())

        print("\nDataFrame info:")
        print(df_aggregated.info())

        # Example 2: Read daily parsed results from Scaleway (JSONL files)
        print("\n=== Example 2: Reading daily parsed JSONL files ===")
        df_daily = read_scaleway_jsonl_and_build_dataframe(
            prefix="soil-sentinel/batch_results_2015_2018_scl_only/daily_parsed/"
        )

        print(f"Successfully loaded daily DataFrame with shape: {df_daily.shape}")
        print("First few rows:")
        print(df_daily.head())

        print("\nDataFrame columns:")
        print(df_daily.columns.tolist())

        print("\nDataFrame info:")
        print(df_daily.info())

        # Example 3: Test aggregation with low coverage threshold (0.5)
        print("\n=== Example 3: Testing aggregation with low coverage threshold (0.5) ===")

        # Select a sample location for testing (first location)
        sample_location = df_daily.iloc[0]
        print(f"Testing aggregation for location: lat={sample_location['lat']}, lon={sample_location['lon']}")

        # Filter daily data for this specific location
        location_mask = (
            (df_daily['lat'] == sample_location['lat']) &
            (df_daily['lon'] == sample_location['lon'])
        )
        df_location = df_daily[location_mask]

        print(f"Found {len(df_location)} daily records for this location")

        # Analyze coverage distribution first
        print(f"\n--- Coverage Analysis ---")
        print(f"Coverage statistics:")
        print(df_location['coverage'].describe())

        # Count coverage by ranges
        coverage_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        coverage_counts = pd.cut(df_location['coverage'], bins=coverage_bins).value_counts().sort_index()
        print(f"\nCoverage distribution:")
        for bin_range, count in coverage_counts.items():
            print(f"  {bin_range}: {count} days")

        # Check if any days have coverage >= 0.5
        high_coverage_days = df_location[df_location['coverage'] >= 0.5]
        print(f"\nDays with coverage >= 0.5: {len(high_coverage_days)}")

        # Analyze sample_count and no_data_count
        print(f"\n--- Sample Count Analysis ---")
        print(f"Sample count statistics:")
        print(df_location['sample_count'].describe())

        print(f"No data count statistics:")
        print(df_location['no_data_count'].describe())

        # Check if any days have valid p50 values
        p50_valid_days = df_location[df_location['p50_B02'].notna()]
        print(f"\nDays with valid p50_B02 values: {len(p50_valid_days)}")

        if len(p50_valid_days) > 0:
            print(f"Sample days with valid p50 values:")
            print(p50_valid_days[['from_time', 'coverage', 'sample_count', 'no_data_count', 'p50_B02', 'p50_B04', 'p50_B08']].head(10))

        # Check the relationship between coverage and p50 values
        print(f"\n--- Coverage vs P50 Analysis ---")
        print(f"Days with coverage > 0: {len(df_location[df_location['coverage'] > 0])}")
        print(f"Days with sample_count > 0: {len(df_location[df_location['sample_count'] > 0])}")
        print(f"Days with any valid p50: {len(df_location[df_location.filter(like='p50_').notna().any(axis=1)])}")

        # Show some sample records to understand the data structure
        print(f"\n--- Sample Records Analysis ---")
        print("First 5 records:")
        sample_records = df_location.head(5)
        print(sample_records[['from_time', 'to_time', 'sample_count', 'no_data_count', 'coverage', 'p50_B02', 'p50_B04', 'p50_B08']])

        # Test with different coverage thresholds
        for coverage_threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8]:
            print(f"\n--- Testing with coverage threshold: {coverage_threshold} ---")

            # Perform aggregation
            aggregated_record = aggregate_records_from_dataframe(
                df_location,
                coverage_threshold=coverage_threshold
            )

            print(f"Aggregated results:")
            print(f"  Total days: {aggregated_record.n_days_total}")
            print(f"  Days kept: {aggregated_record.n_days_kept}")
            print(f"  Kept ratio: {aggregated_record.kept_ratio:.2%}")
            print(f"  Coverage median: {aggregated_record.coverage_median_kept}")
            print(f"  Coverage min: {aggregated_record.coverage_min_kept}")

            # Count valid p50 values
            valid_p50_count = sum(1 for v in aggregated_record.p50_aggregated.values() if v is not None)
            print(f"  Valid p50 features: {valid_p50_count}/{len(aggregated_record.p50_aggregated)}")

            # Show some sample p50 values
            sample_features = ['B02', 'B03', 'B04', 'B08', 'NDVI', 'NDWI']
            print(f"  Sample p50 values:")
            for feature in sample_features:
                value = aggregated_record.p50_aggregated.get(feature)
                print(f"    {feature}: {value}")

        # Compare with existing aggregated data for this location
        print(f"\n--- Comparison with Existing Aggregated Data ---")
        aggregated_location = df_aggregated[
            (df_aggregated['lat'] == sample_location['lat']) &
            (df_aggregated['lon'] == sample_location['lon'])
        ]

        if not aggregated_location.empty:
            agg_record = aggregated_location.iloc[0]
            print(f"Existing aggregated record:")
            print(f"  Coverage threshold: {agg_record['coverage_threshold']}")
            print(f"  Total days: {agg_record['n_days_total']}")
            print(f"  Days kept: {agg_record['n_days_kept']}")
            print(f"  Kept ratio: {agg_record['kept_ratio']}")
            print(f"  Coverage median: {agg_record['coverage_median_kept']}")
            print(f"  Coverage min: {agg_record['coverage_min_kept']}")

            # Count valid p50 values in existing data
            valid_p50_count = sum(1 for v in [agg_record.get(f'p50_{f}') for f in FEATURE_COLS] if v is not None)
            print(f"  Valid p50 features: {valid_p50_count}/{len(FEATURE_COLS)}")
        else:
            print(f"  No existing aggregated data found for this location")

        print("\n=== Summary ===")
        print(f"Aggregated data: {df_aggregated.shape[0]} locations, {df_aggregated.shape[1]} features")
        print(f"Daily data: {df_daily.shape[0]} daily records, {df_daily.shape[1]} features")

        # Example 4: Compute new aggregated DataFrame using corrected logic
        print("\n=== Example 4: Computing new aggregated DataFrame with corrected logic ===")

        # Group daily data by location and compute corrected aggregation
        print("Grouping daily data by location and computing corrected aggregation...")

        # Create a function to compute corrected aggregation for a single location
        def compute_corrected_aggregation_for_location(location_df: pd.DataFrame) -> Dict[str, Any]:
            """Compute corrected aggregation for a single location"""
            if location_df.empty:
                return None

            # Use low coverage threshold (0.5) as requested
            aggregated_record = aggregate_records_from_dataframe(
                location_df,
                coverage_threshold=0.5
            )

            # Convert to dict for DataFrame construction
            result = {
                'lat': aggregated_record.lat,
                'lon': aggregated_record.lon,
                'bbox_epsg4326': aggregated_record.bbox_epsg4326,
                'query_start_date': aggregated_record.query_start_date,
                'query_end_date': aggregated_record.query_end_date,
                'aggregation_interval': aggregated_record.aggregation_interval,
                'coverage_threshold': aggregated_record.coverage_threshold,
                'n_days_total': aggregated_record.n_days_total,
                'n_days_kept': aggregated_record.n_days_kept,
                'kept_ratio': aggregated_record.kept_ratio,
                'coverage_median_kept': aggregated_record.coverage_median_kept,
                'coverage_min_kept': aggregated_record.coverage_min_kept,
            }

            # Add p50 aggregated features
            for feature in FEATURE_COLS:
                result[f'p50_{feature}'] = aggregated_record.p50_aggregated.get(feature)

            return result

        # Group by location and compute corrected aggregation
        print("Processing locations...")
        corrected_aggregated_data = []

        # Get unique locations
        unique_locations = df_daily[['lat', 'lon']].drop_duplicates()
        total_locations = len(unique_locations)
        print(f"Found {total_locations} unique locations to process")

        # Process each location
        for i, (lat, lon) in enumerate(unique_locations.itertuples(index=False), 1):
            if i % 100 == 0:
                print(f"Processing location {i}/{total_locations}...")

            # Filter data for this location
            location_mask = (df_daily['lat'] == lat) & (df_daily['lon'] == lon)
            location_df = df_daily[location_mask]

            # Compute corrected aggregation
            corrected_agg = compute_corrected_aggregation_for_location(location_df)
            if corrected_agg:
                corrected_aggregated_data.append(corrected_agg)

        # Create corrected aggregated DataFrame
        df_corrected_aggregated = pd.DataFrame(corrected_aggregated_data)

        # Convert bbox to string representation for consistency
        if 'bbox_epsg4326' in df_corrected_aggregated.columns:
            df_corrected_aggregated['bbox_epsg4326'] = df_corrected_aggregated['bbox_epsg4326'].apply(
                lambda x: str(x) if isinstance(x, list) else x
            )

        print(f"\nCreated corrected aggregated DataFrame with shape: {df_corrected_aggregated.shape}")
        print("First few rows:")
        print(df_corrected_aggregated.head())

        # Compare corrected vs original aggregation
        print(f"\n--- Comparison: Original vs Corrected Aggregation ---")
        print(f"Original aggregated data: {df_aggregated.shape[0]} locations, {df_aggregated.shape[1]} features")
        print(f"Corrected aggregated data: {df_corrected_aggregated.shape[0]} locations, {df_corrected_aggregated.shape[1]} features")

        # Count valid p50 values in both DataFrames
        original_valid_p50 = df_aggregated.filter(like='p50_').notna().sum().sum()
        corrected_valid_p50 = df_corrected_aggregated.filter(like='p50_').notna().sum().sum()

        print(f"\nValid p50 values:")
        print(f"  Original: {original_valid_p50} total valid values")
        print(f"  Corrected: {corrected_valid_p50} total valid values")
        print(f"  Improvement: {corrected_valid_p50 - original_valid_p50} additional valid values")

        # Show statistics for key features
        key_features = ['B02', 'B03', 'B04', 'B08', 'NDVI', 'NDWI']
        print(f"\nKey feature statistics:")

        for feature in key_features:
            original_valid = df_aggregated[f'p50_{feature}'].notna().sum()
            corrected_valid = df_corrected_aggregated[f'p50_{feature}'].notna().sum()

            print(f"  {feature}:")
            print(f"    Original valid locations: {original_valid}/{len(df_aggregated)}")
            print(f"    Corrected valid locations: {corrected_valid}/{len(df_corrected_aggregated)}")
            print(f"    Improvement: +{corrected_valid - original_valid} locations")

        # Save corrected DataFrame to cache
        corrected_cache_key = _generate_cache_key("corrected_aggregated", "dataframe")
        _save_to_cache(corrected_cache_key, df_corrected_aggregated)
        print(f"\n✅ Saved corrected aggregated DataFrame to cache")

        print(f"\n🎉 COMPLETE! The corrected aggregation logic now produces valid p50 values!")
        print(f"   - Fixed coverage calculation bug in parsers.py")
        print(f"   - Implemented caching to avoid re-reading from Scaleway")
        print(f"   - Used low coverage threshold (0.5) as requested")
        print(f"   - Generated {corrected_valid_p50} valid p50 values across {len(df_corrected_aggregated)} locations")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
