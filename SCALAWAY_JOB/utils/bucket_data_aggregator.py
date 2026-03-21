#!/usr/bin/env python3
"""
Bucket Data Aggregator Module

Provides functionality to read JSON files from a Scaleway bucket,
aggregate all records into a pandas DataFrame, and cache the result as an xlsx file.
"""

from __future__ import annotations

import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root directory to Python path to ensure local modules are found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pandas as pd
from sh_pipeline.storage import storage_from_env, S3StorageClient, StorageConfig
from sh_statistics.models import FEATURE_COLS
import hashlib
import tempfile

def read_bucket_and_aggregate_to_dataframe(
    bucket_prefix: str,
    bucket: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Read JSON files from a Scaleway bucket and aggregate all records into a DataFrame.

    Args:
        bucket_prefix: Prefix to filter JSON files in the bucket
        bucket: Scaleway bucket name (optional, uses env var if not provided)
        endpoint_url: Scaleway S3 endpoint URL (optional, uses env var if not provided)
        region: Scaleway region (optional, uses env var if not provided)
        access_key: Scaleway access key (optional, uses env var if not provided)
        secret_key: Scaleway secret key (optional, uses env var if not provided)
        logger: Optional logger for logging operations
        use_cache: Whether to use caching to avoid re-reading from Scaleway

    Returns:
        pandas DataFrame containing aggregated data from all JSON files

    Raises:
        RuntimeError: If required environment variables are missing and not provided
        Exception: For any errors during the process
    """
    # Create logger if not provided
    log = logger or logging.getLogger(__name__)

    log.info(f"Starting bucket data aggregation with prefix: {bucket_prefix}")

    # Check cache first if enabled
    if use_cache:
        cached_df = _load_cached_dataframe(bucket_prefix)
        if cached_df is not None:
            log.info(f"Loaded cached DataFrame with shape: {cached_df.shape}")
            return cached_df

    # Create storage client
    if all([bucket, endpoint_url, access_key, secret_key]):
        # Use provided credentials
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

    # Read JSON files from bucket
    json_data_list = _read_json_files_from_bucket(
        storage_client,
        bucket_prefix,
        log
    )

    # Build DataFrame from JSON data
    df = _build_dataframe_from_json_data(json_data_list, log)

    # Save to cache if enabled
    if use_cache:
        _save_dataframe_to_cache(df, bucket_prefix)

    log.info(f"Completed data aggregation - DataFrame shape: {df.shape}")
    return df

def _read_json_files_from_bucket(
    storage_client: S3StorageClient,
    prefix: str,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Read all JSON files from a Scaleway bucket with the given prefix.

    Args:
        storage_client: S3StorageClient instance
        prefix: Prefix to filter JSON files
        logger: Logger for logging operations

    Returns:
        List of dictionaries containing parsed JSON data with filename as POINT_ID
    """
    logger.info(f"Listing JSON files with prefix: {prefix}")

    # List all objects with the given prefix
    object_keys = storage_client.list_objects(prefix)

    # Filter for JSON files only
    json_keys = [key for key in object_keys if key.lower().endswith('.json')]
    logger.info(f"Found {len(json_keys)} JSON files to process")

    # Read and parse each JSON file
    json_data_list = []

    for key in json_keys:
        try:
            logger.info(f"Reading JSON file: {key}")
            json_content = storage_client.get_text(key)
            json_data = json.loads(json_content)

            # Extract filename from the key to use as POINT_ID
            filename = os.path.basename(key)
            point_id = os.path.splitext(filename)[0]  # Remove .json extension

            # Add POINT_ID to the JSON data
            json_data['POINT_ID'] = point_id

            json_data_list.append(json_data)
            logger.debug(f"Successfully parsed JSON from {key} with POINT_ID: {point_id}")
        except Exception as e:
            logger.error(f"Failed to read or parse JSON file {key}: {e}")
            continue

    logger.info(f"Successfully processed {len(json_data_list)} JSON files")
    return json_data_list

def _build_dataframe_from_json_data(
    json_data_list: List[Dict[str, Any]],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Build a pandas DataFrame from a list of JSON data dictionaries.

    Args:
        json_data_list: List of dictionaries containing JSON data
        logger: Logger for logging operations

    Returns:
        pandas DataFrame containing the structured data
    """
    if not json_data_list:
        logger.warning("No JSON data provided - returning empty DataFrame")
        return pd.DataFrame()

    logger.info(f"Building DataFrame from {len(json_data_list)} JSON records")

    # Prepare data for DataFrame
    data = []

    for record in json_data_list:
        try:
            # Extract basic fields including POINT_ID
            row_data = {
                'POINT_ID': record.get('POINT_ID'),
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
            logger.error(f"Failed to process record: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(data)

    # Convert bbox to string representation for better display
    if 'bbox_epsg4326' in df.columns:
        df['bbox_epsg4326'] = df['bbox_epsg4326'].apply(lambda x: str(x) if isinstance(x, list) else x)

    logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    return df

def _get_cache_dir() -> Path:
    """
    Get the cache directory path, creating it if it doesn't exist.

    Returns:
        Path to the cache directory
    """
    cache_dir = Path(tempfile.gettempdir()) / "bucket_data_aggregator_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def _generate_cache_key(prefix: str) -> str:
    """
    Generate a unique cache key for a given prefix.

    Args:
        prefix: The bucket prefix used for data loading

    Returns:
        Unique cache key string
    """
    cache_string = f"bucket_agg_{prefix}"
    return hashlib.md5(cache_string.encode()).hexdigest()

def _save_dataframe_to_cache(df: pd.DataFrame, prefix: str) -> None:
    """
    Save DataFrame to cache as both pickle and xlsx files.

    Args:
        df: DataFrame to be cached
        prefix: The bucket prefix used for generating cache key
    """
    cache_key = _generate_cache_key(prefix)
    cache_dir = _get_cache_dir()

    # Save as pickle for fast loading
    pickle_file = cache_dir / f"{cache_key}.pkl"
    try:
        df.to_pickle(pickle_file)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save pickle cache: {e}")

    # Save as xlsx for user access
    xlsx_file = cache_dir / f"{cache_key}.xlsx"
    try:
        df.to_excel(xlsx_file, index=False)
        logging.getLogger(__name__).info(f"Saved cached DataFrame to xlsx: {xlsx_file}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save xlsx cache: {e}")

def _load_cached_dataframe(prefix: str) -> Optional[pd.DataFrame]:
    """
    Load DataFrame from cache if available.

    Args:
        prefix: The bucket prefix used for generating cache key

    Returns:
        Cached DataFrame if available, None otherwise
    """
    cache_key = _generate_cache_key(prefix)
    cache_dir = _get_cache_dir()

    # Try to load pickle file first (faster)
    pickle_file = cache_dir / f"{cache_key}.pkl"
    if pickle_file.exists():
        try:
            return pd.read_pickle(pickle_file)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load pickle cache: {e}")
            # Remove corrupted cache file
            try:
                pickle_file.unlink()
            except:
                pass

    # Try to load xlsx file as fallback
    xlsx_file = cache_dir / f"{cache_key}.xlsx"
    if xlsx_file.exists():
        try:
            return pd.read_excel(xlsx_file)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load xlsx cache: {e}")
            # Remove corrupted cache file
            try:
                xlsx_file.unlink()
            except:
                pass

    return None

def get_cached_xlsx_path(prefix: str) -> Optional[Path]:
    """
    Get the path to the cached xlsx file for a given prefix.

    Args:
        prefix: The bucket prefix used for generating cache key

    Returns:
        Path to cached xlsx file if it exists, None otherwise
    """
    cache_key = _generate_cache_key(prefix)
    cache_dir = _get_cache_dir()
    xlsx_file = cache_dir / f"{cache_key}.xlsx"

    if xlsx_file.exists():
        return xlsx_file
    return None

def clear_cache() -> None:
    """
    Clear all cached data files.
    """
    cache_dir = _get_cache_dir()
    for cache_file in cache_dir.glob("*.*"):
        try:
            cache_file.unlink()
        except:
            pass

def join_features_with_gabri_filters(
    bucket_prefix: str,
    gabri_filters_path: str = "gabri_filters.xlsx",
    output_path: str = "texture_scl_features.xlsx",
    bucket: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Read bucket data, join with gabri_filters.xlsx, and save as texture_scl_features.xlsx.

    Args:
        bucket_prefix: Prefix to filter JSON files in the bucket
        gabri_filters_path: Path to gabri_filters.xlsx file
        output_path: Path to save the joined output xlsx file
        bucket: Scaleway bucket name (optional, uses env var if not provided)
        endpoint_url: Scaleway S3 endpoint URL (optional, uses env var if not provided)
        region: Scaleway region (optional, uses env var if not provided)
        access_key: Scaleway access key (optional, uses env var if not provided)
        secret_key: Scaleway secret key (optional, uses env var if not provided)
        logger: Optional logger for logging operations
        use_cache: Whether to use caching for bucket data

    Returns:
        Joined DataFrame containing features from both sources

    Raises:
        FileNotFoundError: If gabri_filters.xlsx is not found
        Exception: For any errors during the process
    """
    # Create logger if not provided
    log = logger or logging.getLogger(__name__)

    log.info(f"Starting feature joining process")

    # Step 1: Read bucket data and get DataFrame
    log.info(f"Reading bucket data with prefix: {bucket_prefix}")
    df_bucket = read_bucket_and_aggregate_to_dataframe(
        bucket_prefix=bucket_prefix,
        bucket=bucket,
        endpoint_url=endpoint_url,
        region=region,
        access_key=access_key,
        secret_key=secret_key,
        logger=log,
        use_cache=use_cache
    )

    # Step 2: Read gabri_filters.xlsx
    log.info(f"Reading gabri filters from: {gabri_filters_path}")
    gabri_filters_path_full = os.path.join(project_root, gabri_filters_path)
    if not os.path.exists(gabri_filters_path_full):
        raise FileNotFoundError(f"gabri_filters.xlsx not found at: {gabri_filters_path_full}")

    df_gabri = pd.read_excel(gabri_filters_path_full)

    # Step 3: Join DataFrames on POINT_ID
    log.info(f"Joining DataFrames on POINT_ID")
    log.info(f"Bucket data shape: {df_bucket.shape}")
    log.info(f"Gabri filters shape: {df_gabri.shape}")

    # Ensure POINT_ID is string type for both DataFrames
    df_bucket['POINT_ID'] = df_bucket['POINT_ID'].astype(str)
    df_gabri['POINT_ID'] = df_gabri['POINT_ID'].astype(str)

    # Perform the join
    df_joined = pd.merge(
        df_gabri,
        df_bucket,
        on='POINT_ID',
        how='left'  # Keep all gabri records, add bucket data where available
    )

    log.info(f"Joined DataFrame shape: {df_joined.shape}")

    # Step 4: Save the result
    log.info(f"Saving joined data to: {output_path}")
    output_path_full = os.path.join(project_root, output_path)
    df_joined.to_excel(output_path_full, index=False)

    log.info(f"✅ Successfully created texture_scli_features.xlsx with {len(df_joined)} records")
    return df_joined

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
        print("=== Bucket Data Aggregator Example ===")

        # Example 1: Read and aggregate data from a bucket
        print("\n--- Example 1: Basic bucket aggregation ---")
        df = read_bucket_and_aggregate_to_dataframe(
            bucket_prefix="soil-sentinel/only_scl/aggregated/"
        )

        print(f"Successfully loaded DataFrame with shape: {df.shape}")
        print("First few rows:")
        print(df.head())

        print("\nDataFrame columns:")
        print(df.columns.tolist())

        # Get the cached xlsx path
        xlsx_path = get_cached_xlsx_path("soil-sentinel/only_scl/aggregated/")
        if xlsx_path:
            print(f"\nCached xlsx file available at: {xlsx_path}")

        # Example 2: Join features with gabri_filters.xlsx
        print("\n--- Example 2: Joining with gabri_filters.xlsx ---")
        try:
            df_joined = join_features_with_gabri_filters(
                bucket_prefix="soil-sentinel/only_scl/aggregated/",
                gabri_filters_path="gabri_filters.xlsx",
                output_path="only_scl_features_agg.xlsx"
            )

            print(f"Successfully created joined DataFrame with shape: {df_joined.shape}")
            print("First few rows of joined data:")
            print(df_joined.head())

            print("\nJoined DataFrame columns:")
            print(df_joined.columns.tolist())

            print("\n✅ Feature joining completed successfully!")
            print("Output saved to: texture_scl_features.xlsx")

        except FileNotFoundError as e:
            print(f"⚠️  Could not perform feature joining: {e}")
            print("This is expected if gabri_filters.xlsx is not available locally.")

        print("\n✅ Bucket data aggregation completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
