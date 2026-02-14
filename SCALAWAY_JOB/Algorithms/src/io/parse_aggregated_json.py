import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path
import json
import re

# Set up logging
logger = logging.getLogger(__name__)

class AggregatedJSONParser:
    """Parser for Sentinel-2 aggregated JSON files."""

    def __init__(self):
        """Initialize the parser."""
        # Expected feature keys from p50_aggregated
        self.expected_bands = [
            'B02', 'B03', 'B04', 'B08', 'B11', 'B12',
            'NDVI', 'NDWI', 'MNDWI', 'NDMI', 'BSI',
            'BRIGHT', 'ALBEDO_PROXY', 'RED', 'SWIR1', 'SWIR2',
            'RED_SWIR1_RATIO', 'SWIR1_SWIR2_RATIO'
        ]

    def extract_features_from_json(self, json_data: Dict[str, Any], point_id: str) -> Dict[str, Any]:
        """Extract features from a single JSON object.

        Args:
            json_data: Parsed JSON data
            point_id: Point identifier (from filename)

        Returns:
            Dictionary of extracted features
        """
        features = {
            'point_id': point_id,
            'lat': json_data.get('lat'),
            'lon': json_data.get('lon'),
            'n_days_total': json_data.get('n_days_total'),
            'n_days_kept': json_data.get('n_days_kept'),
            'kept_ratio': json_data.get('kept_ratio'),
            'coverage_median_kept': json_data.get('coverage_median_kept'),
            'coverage_min_kept': json_data.get('coverage_min_kept')
        }

        # Extract band/index features from p50_aggregated
        p50_data = json_data.get('p50_aggregated', {})
        if p50_data:
            for band in self.expected_bands:
                if band in p50_data:
                    features[f'p50_{band}'] = p50_data[band]

        return features

    def parse_json_objects(self, json_objects: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Parse multiple JSON objects into a DataFrame.

        Args:
            json_objects: Dictionary mapping object keys to JSON data

        Returns:
            DataFrame with extracted features
        """
        records = []

        for key, json_data in json_objects.items():
            try:
                # Extract point_id from filename (remove extension and path)
                point_id = self._extract_point_id_from_key(key)
                if not point_id:
                    logger.warning(f"Could not extract point_id from key: {key}")
                    continue

                features = self.extract_features_from_json(json_data, point_id)
                records.append(features)

            except Exception as e:
                logger.error(f"Error parsing JSON object {key}: {e}")
                continue

        if not records:
            logger.warning("No valid records found in JSON objects")
            return pd.DataFrame()

        # Create DataFrame and set point_id as index
        df = pd.DataFrame(records)
        df.set_index('point_id', inplace=True)

        logger.info(f"Parsed {len(df)} records from {len(json_objects)} JSON objects")
        return df

    def _extract_point_id_from_key(self, key: str) -> Optional[str]:
        """Extract point_id from S3 object key.

        Args:
            key: S3 object key (e.g., 'path/to/45542620.json')

        Returns:
            Extracted point_id or None if not found
        """
        # Remove file extension
        filename = Path(key).stem

        # Extract the numeric part (point_id should be numeric)
        match = re.search(r'(\d+)', filename)
        if match:
            return match.group(1)

        # If no numeric part found, return the whole filename
        return filename if filename else None

    def save_features_to_cache(self, features_df: pd.DataFrame, cache_path: Path) -> None:
        """Save features DataFrame to parquet cache.

        Args:
            features_df: DataFrame with extracted features
            cache_path: Path to save cache file
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(cache_path)
        logger.info(f"Saved features cache to: {cache_path}")

    def load_features_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load features DataFrame from parquet cache.

        Args:
            cache_path: Path to cache file

        Returns:
            Loaded DataFrame or None if file doesn't exist
        """
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded features cache from: {cache_path}")
            return df
        return None

def get_json_parser() -> AggregatedJSONParser:
    """Get a configured JSON parser instance."""
    return AggregatedJSONParser()