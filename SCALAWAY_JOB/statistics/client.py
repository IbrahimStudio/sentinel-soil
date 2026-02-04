#!/usr/bin/env python3
"""
Sentinel Hub Statistics API Client

Provides a clean interface for interacting with Sentinel Hub's Statistical API.
Handles authentication, request building, response parsing, and error handling.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

import pandas as pd
import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# Load environment variables
load_dotenv("vm.env")

# Sentinel Hub API endpoints
TOKEN_URL = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
STATS_URL = "https://services.sentinel-hub.com/api/v1/statistics"

@dataclass
class StatisticsApiConfig:
    """Configuration for Statistics API client"""
    client_id: str
    client_secret: str
    token_url: str = TOKEN_URL
    stats_url: str = STATS_URL
    max_retries: int = 3
    timeout: int = 30

class StatisticsApiClient:
    """
    Client for Sentinel Hub Statistical API

    Features:
    - OAuth2 authentication with automatic token refresh
    - Request building with validation
    - Response parsing and error handling
    - Retry logic for transient failures
    - Type hints for better IDE support
    """

    def __init__(self, config: StatisticsApiConfig):
        self.config = config
        self.session: Optional[OAuth2Session] = None

    def _get_oauth_session(self) -> OAuth2Session:
        """Create or refresh OAuth2 session"""
        if self.session is None:
            client = BackendApplicationClient(client_id=self.config.client_id)
            self.session = OAuth2Session(client=client)
            self.session.register_compliance_hook(
                "access_token_response",
                self._sentinelhub_compliance_hook
            )

        # Check if token needs refresh or fetch new one
        if not self.session.token or self.session.token.is_expired():
            self.session.fetch_token(
                token_url=self.config.token_url,
                client_secret=self.config.client_secret,
                include_client_id=True,
            )

        return self.session

    def _sentinelhub_compliance_hook(self, resp: requests.Response) -> requests.Response:
        """Ensure Sentinel Hub API responses are valid"""
        resp.raise_for_status()
        return resp

    def _build_stats_request(
        self,
        *,
        bbox: List[float],
        start_date: str,
        end_date: str,
        interval: str,
        evalscript: str,
        res: int = 20,
        mosaicking_order: str = "leastCC",
    ) -> Dict[str, Any]:
        """
        Build a Statistics API request payload

        Args:
            bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Aggregation interval (e.g., "P1D" for daily)
            evalscript: JavaScript evalscript for feature computation
            res: Resolution in meters
            mosaicking_order: Mosaicking order strategy

        Returns:
            Dictionary with complete API request payload
        """
        return {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {"mosaickingOrder": mosaicking_order},
                    }
                ],
            },
            "aggregation": {
                "timeRange": {
                    "from": f"{start_date}T00:00:00Z",
                    # "to" behaves as exclusive -> add 1 day
                    "to": (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
                },
                "aggregationInterval": {"of": interval},
                "evalscript": evalscript,
                "resx": res,
                "resy": res,
            },
            "calculations": {
                "features": {
                    "statistics": {"default": {"percentiles": {"k": [50]}}}
                },
                "dataMask": {
                    "statistics": {"default": {}}
                },
            },
        }

    def request_statistics(
        self,
        *,
        bbox: List[float],
        start_date: str,
        end_date: str,
        interval: str,
        evalscript: str,
        res: int = 20,
        mosaicking_order: str = "leastCC",
    ) -> Dict[str, Any]:
        """
        Execute a Statistics API request

        Args:
            bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Aggregation interval (e.g., "P1D" for daily)
            evalscript: JavaScript evalscript for feature computation
            res: Resolution in meters
            mosaicking_order: Mosaicking order strategy

        Returns:
            Parsed JSON response from Sentinel Hub API

        Raises:
            requests.exceptions.HTTPError: For HTTP errors
            requests.exceptions.RequestException: For connection/network errors
            ValueError: For invalid parameters
        """
        # Validate parameters
        if len(bbox) != 4:
            raise ValueError("bbox must have exactly 4 elements: [min_lon, min_lat, max_lon, max_lat]")

        if not evalscript.strip():
            raise ValueError("evalscript cannot be empty")

        # Build request
        request_payload = self._build_stats_request(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            evalscript=evalscript,
            res=res,
            mosaicking_order=mosaicking_order,
        )

        # Execute request with retries
        for attempt in range(self.config.max_retries):
            try:
                session = self._get_oauth_session()
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }

                response = session.post(
                    self.config.stats_url,
                    headers=headers,
                    json=request_payload,
                    timeout=self.config.timeout
                )

                # Parse and return response
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise
                # Wait before retry (exponential backoff)
                import time
                time.sleep(2 ** attempt)

    def close(self):
        """Close the OAuth2 session"""
        if self.session:
            self.session.close()
            self.session = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure session is closed"""
        self.close()

def create_client_from_env() -> StatisticsApiClient:
    """
    Create a StatisticsApiClient using environment variables

    Expects:
        SH_CLIENT_ID: Sentinel Hub client ID
        SH_CLIENT_SECRET: Sentinel Hub client secret

    Returns:
        Configured StatisticsApiClient instance

    Raises:
        ValueError: If required environment variables are missing
    """
    client_id = os.getenv("SH_CLIENT_ID")
    client_secret = os.getenv("SH_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("Missing environment variables SH_CLIENT_ID and/or SH_CLIENT_SECRET")

    config = StatisticsApiConfig(
        client_id=client_id,
        client_secret=client_secret
    )

    return StatisticsApiClient(config)