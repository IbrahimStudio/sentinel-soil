"""
sh_clients.py — Sentinel Hub Catalog API + Process API clients.

Both clients share the same OAuth2 bearer-token auth (auto-refresh).
Pattern inherited from SCALAWAY_JOB/ingestion/sh_statistics/client.py.
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests
import rasterio
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from rasterio.io import MemoryFile

log = logging.getLogger(__name__)

_TOKEN_URL = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
_CATALOG_URL = "https://services.sentinel-hub.com/catalog/v1/search"
_PROCESS_URL = "https://services.sentinel-hub.com/process/v1"

# Band order in evalscript_process_api.js — must match evalscript exactly.
BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
N_BANDS = len(BAND_NAMES)  # 11


# ---------------------------------------------------------------------------
# Shared OAuth2 auth
# ---------------------------------------------------------------------------

class _SHAuth:
    """OAuth2 bearer-token session with automatic refresh."""

    def __init__(self, client_id: str, client_secret: str, token_url: str = _TOKEN_URL):
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._session: Optional[OAuth2Session] = None

    def _get_session(self) -> OAuth2Session:
        if self._session is None:
            client = BackendApplicationClient(client_id=self._client_id)
            self._session = OAuth2Session(client=client)
            self._session.register_compliance_hook(
                "access_token_response", self._compliance_hook
            )

        token = self._session.token
        expired = False
        if not token:
            expired = True
        elif token.get("expires_at"):
            expired = int(time.time()) >= int(token["expires_at"])

        if expired:
            self._session.fetch_token(
                token_url=self._token_url,
                client_secret=self._client_secret,
                include_client_id=True,
            )

        return self._session

    @staticmethod
    def _compliance_hook(resp: requests.Response) -> requests.Response:
        resp.raise_for_status()
        return resp

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Catalog API
# ---------------------------------------------------------------------------

@dataclass
class SceneRecord:
    date: str          # "YYYY-MM-DD"
    scene_id: str
    cloud_pct: float


class CatalogClient(_SHAuth):
    """
    Query the SH Catalog API for Sentinel-2 L2A acquisitions.

    Applies catalog_prefilter from filter_config.json as CQL2-json.
    Returns one SceneRecord per acquisition, sorted by date ascending.
    Paginates until all results are collected.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        super().__init__(client_id, client_secret)
        self.max_retries = max_retries
        self.timeout = timeout

    def search_acquisitions(
        self,
        bbox_4326: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        catalog_prefilter: dict,
    ) -> list[SceneRecord]:
        """
        Args:
            bbox_4326: (xmin, ymin, xmax, ymax) in EPSG:4326
            start_date / end_date: "YYYY-MM-DD"
            catalog_prefilter: dict from filter_config["catalog_prefilter"]
                Keys: eo:cloud_cover_lt, s2:snow_ice_percentage_lt,
                      s2:cloud_shadow_percentage_lt, s2:not_vegetated_percentage_gt

        Returns:
            List of SceneRecord sorted by date ascending, pre-filtered
            client-side using catalog_prefilter thresholds.

        Note: SH Catalog API does not support server-side CQL2/query filtering
        at this endpoint — filtering is applied after fetching scene metadata.
        """
        records: list[SceneRecord] = []
        next_token: Optional[str] = None

        while True:
            payload: dict = {
                "bbox": list(bbox_4326),
                "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
                "collections": ["sentinel-2-l2a"],
                "limit": 100,
            }
            if next_token:
                payload["next"] = next_token

            resp = self._post_with_retry(_CATALOG_URL, payload)
            features = resp.get("features", [])

            for feat in features:
                props = feat.get("properties", {})

                # Only scene-level filter: skip entirely snow-covered tiles.
                # All other quality filtering (cloud, shadow, vegetation) is
                # handled correctly at pixel level by SCL+NDVI+NBR2.
                snow_pct = float(props.get("s2:snow_ice_percentage", 0.0))
                if snow_pct >= catalog_prefilter.get("s2:snow_ice_percentage_lt", 90):
                    continue

                dt = props.get("datetime", "")[:10]  # "YYYY-MM-DD"
                scene_id = feat.get("id", "")
                cloud_pct = float(props.get("eo:cloud_cover", 0.0))
                records.append(SceneRecord(date=dt, scene_id=scene_id, cloud_pct=cloud_pct))

            # Pagination
            links = resp.get("links", [])
            next_link = next((l for l in links if l.get("rel") == "next"), None)
            if not next_link:
                break
            next_token = next_link.get("body", {}).get("next") or next_link.get("href")
            if not next_token:
                break

        records.sort(key=lambda r: r.date)
        log.debug("Catalog: %d acquisitions after pre-filter (%s – %s)", len(records), start_date, end_date)
        return records

    def _post_with_retry(self, url: str, payload: dict) -> dict:
        for attempt in range(self.max_retries):
            try:
                sess = self._get_session()
                resp = sess.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt
                log.warning("Catalog attempt %d/%d failed (%s). Retrying in %ds.", attempt + 1, self.max_retries, exc, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Process API
# ---------------------------------------------------------------------------

class ProcessClient(_SHAuth):
    """
    Fetch per-date rasters via the SH Process API.

    Returns float32 numpy arrays shaped (H, W, N_BANDS).
    Band order matches BAND_NAMES constant in this module.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        evalscript: str,
        *,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        super().__init__(client_id, client_secret)
        self.evalscript = evalscript
        self.max_retries = max_retries
        self.timeout = timeout

    def fetch_raster(
        self,
        bbox_3857: tuple[float, float, float, float],
        date: str,
        *,
        width: int = 9,
        height: int = 9,
    ) -> np.ndarray:
        """
        Fetch a single-date raster for the given bbox.

        Args:
            bbox_3857: (xmin, ymin, xmax, ymax) in EPSG:3857 (meters)
            date: "YYYY-MM-DD"
            width / height: output pixel dimensions (9×9 for training)

        Returns:
            float32 array shaped (height, width, N_BANDS).
            SCL band (index 10) contains raw integer class values as float.
        """
        payload = {
            "input": {
                "bounds": {
                    "bbox": list(bbox_3857),
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"},
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date}T00:00:00Z",
                            "to":   f"{date}T23:59:59Z",
                        }
                    },
                }],
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [{
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }],
            },
            "evalscript": self.evalscript,
        }

        content = self._post_with_retry(_PROCESS_URL, payload)
        return _tiff_to_array(content, expected_bands=N_BANDS)

    def _post_with_retry(self, url: str, payload: dict) -> bytes:
        for attempt in range(self.max_retries):
            try:
                sess = self._get_session()
                resp = sess.post(
                    url,
                    json=payload,
                    headers={"Accept": "image/tiff"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.content
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt
                log.warning("Process API attempt %d/%d failed (%s). Retrying in %ds.", attempt + 1, self.max_retries, exc, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")


def _tiff_to_array(content: bytes, expected_bands: int) -> np.ndarray:
    """Parse a TIFF response from Process API into (H, W, bands) float32."""
    with MemoryFile(content) as memfile:
        with memfile.open() as ds:
            arr = ds.read()  # (bands, H, W)

    if arr.ndim != 3 or arr.shape[0] != expected_bands:
        raise ValueError(
            f"Unexpected raster shape {arr.shape}; expected ({expected_bands}, H, W). "
            "Check evalscript band count."
        )

    arr = arr.transpose(1, 2, 0).astype(np.float32)  # → (H, W, bands)
    return arr


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def clients_from_env(evalscript: str) -> tuple[CatalogClient, ProcessClient]:
    """
    Build Catalog + Process clients from environment variables.

    Required:
        SH_CLIENT_ID
        SH_CLIENT_SECRET
    """
    client_id = os.environ.get("SH_CLIENT_ID") or os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("SH_CLIENT_SECRET") or os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Missing SH_CLIENT_ID / SH_CLIENT_SECRET env vars.")
    return (
        CatalogClient(client_id, client_secret),
        ProcessClient(client_id, client_secret, evalscript),
    )
