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
        # Handle different OAuth2 token implementations
        token = self.session.token
        token_expired = False

        if not token:
            token_expired = True
        else:
            # Check for is_expired method (some implementations)
            if hasattr(token, 'is_expired') and callable(getattr(token, 'is_expired', None)):
                token_expired = token.is_expired()
            # Check for expires_at attribute (common implementation)
            elif hasattr(token, 'expires_at') and token.get('expires_at'):
                import time
                current_time = int(time.time())
                token_expired = current_time >= token['expires_at']
            # If we can't determine expiration, assume it's still valid
            else:
                token_expired = False

        if token_expired:
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

    def _validate_config(self, res: int, bbox_size_m: Optional[float] = None) -> None:
        """
        Validate resolution and bounding box configuration

        Args:
            res: Resolution in meters
            bbox_size_m: Optional bounding box size in meters for additional validation

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate resolution
        valid_resolutions = [10, 20, 60]
        if res not in valid_resolutions:
            raise ValueError(
                f"Invalid resolution {res}m. Must be one of: {valid_resolutions}"
            )

        # Validate bbox size if provided
        if bbox_size_m is not None:
            if bbox_size_m <= 0:
                raise ValueError(f"Bounding box size must be positive, got {bbox_size_m}m")

            # Calculate expected pixel count
            pixels = bbox_size_m / res
            if pixels < 3:  # Minimum 3x3 = 9 pixels recommended
                raise ValueError(
                    f"Too few pixels: {bbox_size_m}m bbox with {res}m resolution = {pixels:.1f} pixels. "
                    f"Recommend at least {3*res}m bbox for {res}m resolution."
                )

            # Warn about potentially sparse data
            if pixels < 10:  # Less than 100 pixels total
                print(
                    f"⚠️  Warning: {bbox_size_m}m bbox with {res}m resolution = {pixels:.1f} pixels. "
                    f"Consider using larger bbox (e.g., {10*res}m) for better statistical reliability."
                )


# DEPRECATED VERSION: only median here
    # def _build_stats_request(
    #     self,
    #     *,
    #     bbox: List[float],
    #     start_date: str,
    #     end_date: str,
    #     interval: str,
    #     evalscript: str,
    #     res: int = 10,
    #     mosaicking_order: str = "leastCC",
    #     crs: str = "http://www.opengis.net/def/crs/EPSG/0/4326",
    # ) -> Dict[str, Any]:
    #     """
    #     Build a Statistics API request payload

    #     Args:
    #         bbox: Bounding box coordinates
    #         start_date: Start date in YYYY-MM-DD format
    #         end_date: End date in YYYY-MM-DD format
    #         interval: Aggregation interval (e.g., "P1D" for daily)
    #         evalscript: JavaScript evalscript for feature computation
    #         res: Resolution in units of the CRS (meters for EPSG:3857, degrees for EPSG:4326)
    #         mosaicking_order: Mosaicking order strategy
    #         crs: Coordinate reference system

    #     Returns:
    #         Dictionary with complete API request payload

    #     Important:
    #         - For EPSG:4326 (default), res is in DEGREES
    #         - For EPSG:3857, res is in METERS
    #         - Use create_meter_based_request() for true meter-based analysis
    #     """
    #     return {
    #         "input": {
    #             "bounds": {
    #                 "bbox": bbox,
    #                 "properties": {"crs": crs},
    #             },
    #             "data": [
    #                 {
    #                     "type": "sentinel-2-l2a",
    #                     "dataFilter": {"mosaickingOrder": mosaicking_order},
    #                 }
    #             ],
    #         },
    #         "aggregation": {
    #             "timeRange": {
    #                 "from": f"{start_date}T00:00:00Z",
    #                 # "to" behaves as exclusive -> add 1 day
    #                 "to": (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
    #             },
    #             "aggregationInterval": {"of": interval},
    #             "evalscript": evalscript,
    #             "resx": res,
    #             "resy": res,
    #         },
    #         "calculations": {
    #             "features": {
    #                 "statistics": {"default": {"percentiles": {"k": [50]}}}
    #             },
    #             "valid": {
    #                 "statistics": {"default": {}}
    #             },
    #             "dataMask": {
    #                 "statistics": {"default": {}}
    #             },
    #         },
    #     }

    def _build_stats_request(
        self,
        *,
        bbox: List[float],
        start_date: str,
        end_date: str,
        interval: str,
        evalscript: str,
        res: int = 10,
        mosaicking_order: str = "leastCC",
        crs: str = "http://www.opengis.net/def/crs/EPSG/0/4326",
    ) -> Dict[str, Any]:
        return {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": crs},
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
                    "statistics": {
                        "default": {
                            # core reducers
                            "mean": True,
                            "stDev": True,
                            "min": True,
                            "max": True,
                            # good extra signal for ML: distribution shape
                            "percentiles": {"k": [10, 25, 50, 75, 90]},
                            # sometimes useful for QC/debugging
                            "sampleCount": True,
                            "noDataCount": True,
                        }
                    }
                },
                "valid": {
                    "statistics": {
                        "default": {
                            # coverage = mean(valid) in [0..1]
                            "mean": True,
                            "sampleCount": True,
                        }
                    }
                },
                "dataMask": {
                    "statistics": {
                        "default": {
                            # optional; useful if you want to compare with valid
                            "mean": True,
                            "sampleCount": True,
                        }
                    }
                },
            },
        }


    def create_meter_based_request(
        self,
        *,
        lat: float,
        lon: float,
        size_m: float,
        resolution_m: int = 10,
        start_date: str,
        end_date: str,
        interval: str,
        evalscript: str,
        mosaicking_order: str = "leastCC",
    ) -> Dict[str, Any]:
        """
        Create a proper meter-based Statistics API request

        This method addresses the fundamental CRS/resolution mismatch by:
        1. Using EPSG:3857 (Web Mercator) for true meter-based coordinates
        2. Setting resolution in meters (not degrees)
        3. Calculating the correct bounding box in meter-based CRS

        Args:
            lat: Latitude in decimal degrees (EPSG:4326)
            lon: Longitude in decimal degrees (EPSG:4326)
            size_m: Size of square area of interest in meters
            resolution_m: Desired resolution in meters
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Aggregation interval (e.g., "P1D" for daily)
            evalscript: JavaScript evalscript for feature computation
            mosaicking_order: Mosaicking order strategy

        Returns:
            Dictionary with complete API request payload

        Example:
            # For 30m x 30m area with 10m pixels (3x3 grid)
            request = client.create_meter_based_request(
                lat=45.0, lon=10.0, size_m=30.0, resolution_m=10,
                start_date="2023-01-01", end_date="2023-01-31",
                interval="P1D", evalscript=evalscript
            )
        """
        try:
            from sh_statistics.models import create_meter_based_request_config
        except ImportError:
            raise ImportError(
                "Meter-based requests require sh_statistics.models module. "
                "Ensure the sh_statistics package is properly installed."
            )

        # Get meter-based configuration
        config = create_meter_based_request_config(
            lat, lon, size_m, resolution_m
        )

        # Build the request with meter-based CRS and resolution
        return self._build_stats_request(
            bbox=config["bbox"],
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            evalscript=evalscript,
            res=resolution_m,  # Resolution in METERS for EPSG:3857
            mosaicking_order=mosaicking_order,
            crs=config["crs"]   # Use EPSG:3857 for meter-based
        )

    def request_statistics_meter_based(
        self,
        *,
        lat: float,
        lon: float,
        size_m: float,
        resolution_m: int = 10,
        start_date: str,
        end_date: str,
        interval: str,
        evalscript: str,
        mosaicking_order: str = "leastCC",
    ) -> Dict[str, Any]:
        """
        Execute a meter-based Statistics API request

        This is a convenience method that combines create_meter_based_request()
        with actual API execution.

        Args:
            lat: Latitude in decimal degrees (EPSG:4326)
            lon: Longitude in decimal degrees (EPSG:4326)
            size_m: Size of square area of interest in meters
            resolution_m: Desired resolution in meters
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Aggregation interval (e.g., "P1D" for daily)
            evalscript: JavaScript evalscript for feature computation
            mosaicking_order: Mosaicking order strategy

        Returns:
            Parsed JSON response from Sentinel Hub API

        Example:
            # For 30m x 30m area with 10m pixels
            response = client.request_statistics_meter_based(
                lat=45.0, lon=10.0, size_m=30.0, resolution_m=10,
                start_date="2023-01-01", end_date="2023-01-31",
                interval="P1D", evalscript=evalscript
            )
        """
        # Build meter-based request
        request_payload = self.create_meter_based_request(
            lat=lat,
            lon=lon,
            size_m=size_m,
            resolution_m=resolution_m,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            evalscript=evalscript,
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

    def request_statistics(
        self,
        *,
        bbox: List[float],
        start_date: str,
        end_date: str,
        interval: str,
        evalscript: str,
        res: int = 10,
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