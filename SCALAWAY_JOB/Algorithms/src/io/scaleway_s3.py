import os
import logging
from typing import List, Dict, Any, Optional
import json
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# Import from existing pipeline
from pipeline.storage import storage_from_env

# Set up logging
logger = logging.getLogger(__name__)

class ScalewayS3Client:
    """Wrapper around existing pipeline.storage.S3StorageClient for JSON data ingestion."""

    def __init__(self):
        """Initialize the S3 client using existing pipeline infrastructure."""
        # Use existing environment variables from vm.env
        if not all([
            os.environ.get('SCALEWAY_S3_ENDPOINT'),
            os.environ.get('SCALEWAY_S3_BUCKET'),
            os.environ.get('SCALEWAY_ACCESS_KEY'),
            os.environ.get('SCALEWAY_SECRET_KEY')
        ]):
            raise ValueError("Missing required Scaleway S3 environment variables")

        self.client = storage_from_env()
        logger.info(f"Initialized Scaleway S3 client for bucket: {self.client.bucket}")

    def list_objects(self, prefix: str) -> List[str]:
        """List all objects under the given prefix.

        Args:
            prefix: S3 prefix to list objects under

        Returns:
            List of object keys
        """
        return self.client.list_objects(prefix)

    def download_json_object(self, key: str) -> Dict[str, Any]:
        """Download and parse a JSON object from S3.

        Args:
            key: S3 object key

        Returns:
            Parsed JSON data as dictionary
        """
        try:
            content = self.client.get_text(key)
            data = json.loads(content)
            logger.debug(f"Downloaded JSON object: {key}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from object {key}: {e}")
            raise

    def download_all_json_objects(self, prefix: str, file_extension: str = '.json') -> Dict[str, Dict[str, Any]]:
        """Download all JSON objects under the given prefix.

        Args:
            prefix: S3 prefix to search under
            file_extension: File extension to filter by

        Returns:
            Dictionary mapping object keys to parsed JSON data
        """
        object_keys = self.list_objects(prefix)
        json_objects = {}

        for key in tqdm(object_keys, desc="Downloading JSON objects"):
            if key.endswith(file_extension):
                try:
                    data = self.download_json_object(key)
                    json_objects[key] = data
                except Exception as e:
                    logger.warning(f"Skipping object {key} due to error: {e}")

        logger.info(f"Downloaded {len(json_objects)} JSON objects")
        return json_objects

def get_s3_client() -> ScalewayS3Client:
    """Get a configured Scaleway S3 client instance."""
    return ScalewayS3Client()