"""
sh_clients.py — Sentinel Hub Catalog API + Process API clients.

Both clients share the same OAuth2 bearer-token auth (auto-refresh).
"""

from __future__ import annotations

import io
import json
import logging
import os
import tarfile
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

# Sentinel Hub commercial endpoints
_SH_TOKEN_URL   = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
_SH_CATALOG_URL = "https://services.sentinel-hub.com/catalog/v1/search"
_SH_PROCESS_URL = "https://services.sentinel-hub.com/process/v1"

# CDSE (Copernicus Data Space Ecosystem) endpoints
_CDSE_TOKEN_URL   = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
_CDSE_CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
_CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Defaults (kept for backwards compat)
_TOKEN_URL   = _SH_TOKEN_URL
_CATALOG_URL = _SH_CATALOG_URL
_PROCESS_URL = _SH_PROCESS_URL

# Band order in evalscripts — must match exactly.
BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
N_BANDS = len(BAND_NAMES)  # 11

# Must match MAX_SCENES in evalscript_multitemporal.js
_MT_MAX_SCENES = 400


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

    Used primarily to get a scene count before making the multi-temporal
    Process API request (to detect if MAX_SCENES would be exceeded).
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        token_url: str = _TOKEN_URL,
        catalog_url: str = _CATALOG_URL,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        super().__init__(client_id, client_secret, token_url)
        self._catalog_url = catalog_url
        self.max_retries = max_retries
        self.timeout = timeout

    def search_acquisitions(
        self,
        bbox_4326: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        catalog_prefilter: dict,
    ) -> list[SceneRecord]:
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

            resp = self._post_with_retry(self._catalog_url, payload)
            features = resp.get("features", [])

            for feat in features:
                props = feat.get("properties", {})
                snow_pct = float(props.get("s2:snow_ice_percentage", 0.0))
                if snow_pct >= catalog_prefilter.get("s2:snow_ice_percentage_lt", 90):
                    continue
                dt = props.get("datetime", "")[:10]
                scene_id = feat.get("id", "")
                cloud_pct = float(props.get("eo:cloud_cover", 0.0))
                records.append(SceneRecord(date=dt, scene_id=scene_id, cloud_pct=cloud_pct))

            links = resp.get("links", [])
            next_link = next((l for l in links if l.get("rel") == "next"), None)
            if not next_link:
                break
            next_token = next_link.get("body", {}).get("next") or next_link.get("href")
            if not next_token:
                break

        records.sort(key=lambda r: r.date)
        log.debug("Catalog: %d acquisitions (%s – %s)", len(records), start_date, end_date)
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
                log.warning("Catalog attempt %d/%d failed (%s). Retrying in %ds.",
                            attempt + 1, self.max_retries, exc, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Process API — multi-temporal and per-date
# ---------------------------------------------------------------------------

class ProcessClient(_SHAuth):
    """
    Fetch rasters via the SH Process API.

    Two modes:
      fetch_all_dates()  — one request for the entire time window using
                           ORBIT mosaicking (evalscript_multitemporal.js).
                           Returns (N_dates, H, W, N_BANDS) + date list.
      fetch_raster()     — one request per specific date (legacy fallback).
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        evalscript: str,
        multitemporal_evalscript: str,
        *,
        token_url: str = _TOKEN_URL,
        process_url: str = _PROCESS_URL,
        max_retries: int = 3,
        timeout: int = 120,
        mt_timeout: int = 600,
    ):
        super().__init__(client_id, client_secret, token_url)
        self._process_url = process_url
        self.evalscript = evalscript
        self.multitemporal_evalscript = multitemporal_evalscript
        self.max_retries = max_retries
        self.timeout = timeout
        self.mt_timeout = mt_timeout  # multi-temporal requests return large TIFFs

    # ------------------------------------------------------------------
    # Multi-temporal: all dates via yearly-chunked requests
    # ------------------------------------------------------------------

    def fetch_all_dates(
        self,
        bbox_3857: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        *,
        width: int = 9,
        height: int = 9,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Fetch all Sentinel-2 orbit passes for the bbox/time range.

        Uses ORBIT mosaicking: evaluatePixel() receives one raw sample per orbit
        pass — NO server-side aggregation or compositing.  If two tiles overlap
        the point on the same orbit, SH picks the one covering the bbox centre
        deterministically.  All quality filtering (SCL, NDVI, NBR2) still runs
        client-side in the feature store.

        The time range is automatically split into 12-month chunks so that
        tile-overlap points (which can have 150+ orbits per year) never hit the
        MAX_SCENES cap inside the evalscript.

        Returns:
            rasters: float32  (N_total, height, width, N_BANDS)
            dates:   list[str] of "YYYY-MM-DD", length = N_total
        """
        if not self.multitemporal_evalscript:
            raise RuntimeError(
                "multitemporal_evalscript not set — pass --evalscript-mt to main.py"
            )

        from datetime import date as _date, timedelta

        s = _date.fromisoformat(start_date)
        e = _date.fromisoformat(end_date)

        # Build 12-month chunks: [s, s+1yr), [s+1yr, s+2yr), …, [last, e]
        chunks: list[tuple[str, str]] = []
        cs = s
        while cs < e:
            try:
                ce = _date(cs.year + 1, cs.month, cs.day)
            except ValueError:
                # Feb 29 → Feb 28 in non-leap year
                ce = _date(cs.year + 1, cs.month, cs.day - 1)
            if ce > e:
                ce = e
            chunks.append((cs.isoformat(), ce.isoformat()))
            cs = ce + timedelta(days=1)

        all_rasters: list[np.ndarray] = []
        all_dates:   list[str]        = []

        for cs_str, ce_str in chunks:
            r, d = self._fetch_window(bbox_3857, cs_str, ce_str, width=width, height=height)
            if r.shape[0] > 0:
                all_rasters.append(r)
                all_dates.extend(d)

        if not all_rasters:
            return np.zeros((0, height, width, N_BANDS), dtype=np.float32), []

        return np.concatenate(all_rasters, axis=0), all_dates

    def _fetch_window(
        self,
        bbox_3857: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        *,
        width: int = 9,
        height: int = 9,
    ) -> tuple[np.ndarray, list[str]]:
        """Single multi-temporal request for one time window."""
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
                            "from": f"{start_date}T00:00:00Z",
                            "to":   f"{end_date}T23:59:59Z",
                        }
                    },
                }],
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [
                    {"identifier": "default",  "format": {"type": "image/tiff"}},
                    {"identifier": "userdata", "format": {"type": "application/json"}},
                ],
            },
            "evalscript": self.multitemporal_evalscript,
        }

        content = self._post_with_retry(self._process_url, payload, accept="application/tar",
                                        timeout=self.mt_timeout)
        return _parse_multitemporal_tar(content)

    # ------------------------------------------------------------------
    # Per-date fallback (kept for debugging)
    # ------------------------------------------------------------------

    def fetch_raster(
        self,
        bbox_3857: tuple[float, float, float, float],
        date: str,
        *,
        width: int = 9,
        height: int = 9,
    ) -> np.ndarray:
        """Fetch a single-date raster. Returns float32 (height, width, N_BANDS)."""
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
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
            "evalscript": self.evalscript,
        }

        content = self._post_with_retry(self._process_url, payload, accept="image/tiff")
        return _tiff_to_array(content, expected_bands=N_BANDS)

    # ------------------------------------------------------------------
    # Shared retry logic
    # ------------------------------------------------------------------

    def _post_with_retry(self, url: str, payload: dict, *, accept: str,
                         timeout: Optional[int] = None) -> bytes:
        max_attempts = max(self.max_retries, 8)
        effective_timeout = timeout if timeout is not None else self.timeout
        for attempt in range(max_attempts):
            try:
                sess = self._get_session()
                resp = sess.post(url, json=payload, headers={"Accept": accept},
                                 timeout=effective_timeout)
                resp.raise_for_status()
                return resp.content
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    if attempt == max_attempts - 1:
                        raise
                    retry_after = int(exc.response.headers.get("Retry-After", 60))
                    wait = max(retry_after, 60)
                    log.warning(
                        "Process API rate-limited (429). Waiting %ds before retry %d/%d.",
                        wait, attempt + 1, max_attempts,
                    )
                    time.sleep(wait)
                else:
                    if attempt == max_attempts - 1:
                        raise
                    wait = 2 ** min(attempt, 5)
                    log.warning("Process API attempt %d/%d failed (%s). Retrying in %ds.",
                                attempt + 1, max_attempts, exc, wait)
                    time.sleep(wait)
            except requests.RequestException as exc:
                if attempt == max_attempts - 1:
                    raise
                wait = 2 ** min(attempt, 5)
                log.warning("Process API attempt %d/%d failed (%s). Retrying in %ds.",
                            attempt + 1, max_attempts, exc, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Tar response parser for multi-temporal requests
# ---------------------------------------------------------------------------

def _parse_multitemporal_tar(content: bytes) -> tuple[np.ndarray, list[str]]:
    """
    Parse the application/tar response from a multi-temporal Process API call.

    The tar contains:
      default.tif   — (MAX_SCENES * N_BANDS, H, W) float32 TIFF, padded with -9999
      userdata.json — {"n_scenes": <int>, "dates": ["YYYY-MM-DD", ...]}

    Returns (rasters, dates) where rasters has shape (n_actual, H, W, N_BANDS).
    """
    tiff_data: Optional[bytes] = None
    userdata:  Optional[dict]  = None

    with tarfile.open(fileobj=io.BytesIO(content)) as tf:
        for member in tf.getmembers():
            name = member.name.lower()
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            if name.endswith(".tif") or name.endswith(".tiff"):
                tiff_data = fobj.read()
            elif name == "userdata.json":
                userdata = json.loads(fobj.read().decode())

    if tiff_data is None or userdata is None:
        raise ValueError(
            "Multi-temporal tar response missing expected files. "
            f"userdata present: {userdata is not None}, tiff present: {tiff_data is not None}. "
            "Check that updateOutputMetadata is defined in the evalscript "
            "and that Accept: application/tar was used."
        )

    dates = userdata.get("dates", [])
    n_actual = len(dates)

    if n_actual == 0:
        return np.zeros((0, 9, 9, N_BANDS), dtype=np.float32), []

    if n_actual >= _MT_MAX_SCENES - 10:
        log.warning(
            "Scene count %d is near or at MAX_SCENES=%d — some orbits may have been "
            "silently truncated. Consider adding a seasonal filter (--season-months).",
            n_actual, _MT_MAX_SCENES,
        )

    with MemoryFile(tiff_data) as memfile:
        with memfile.open() as ds:
            arr = ds.read()  # (MAX_SCENES * N_BANDS, H, W)

    total_bands, H, W = arr.shape
    if total_bands % N_BANDS != 0:
        raise ValueError(f"TIFF has {total_bands} bands — not a multiple of N_BANDS={N_BANDS}.")

    max_scenes = total_bands // N_BANDS
    # Reshape: (max_scenes * N_BANDS, H, W) → (max_scenes, N_BANDS, H, W)
    arr = arr.reshape(max_scenes, N_BANDS, H, W)
    # Trim padding: keep only actual scenes
    arr = arr[:n_actual]
    # Transpose to match per-date npz format: (n_actual, H, W, N_BANDS)
    arr = arr.transpose(0, 2, 3, 1).astype(np.float32)

    return arr, dates


# ---------------------------------------------------------------------------
# Per-date TIFF parser (used by fetch_raster)
# ---------------------------------------------------------------------------

def _tiff_to_array(content: bytes, expected_bands: int) -> np.ndarray:
    """Parse a single-date TIFF response into (H, W, bands) float32."""
    with MemoryFile(content) as memfile:
        with memfile.open() as ds:
            arr = ds.read()  # (bands, H, W)

    if arr.ndim != 3 or arr.shape[0] != expected_bands:
        raise ValueError(
            f"Unexpected raster shape {arr.shape}; expected ({expected_bands}, H, W)."
        )

    return arr.transpose(1, 2, 0).astype(np.float32)  # → (H, W, bands)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def clients_from_env(
    evalscript: str,
    multitemporal_evalscript: str = "",
) -> tuple[CatalogClient, ProcessClient]:
    """
    Build Catalog + Process clients from environment variables.

    Prefers SH_CLIENT_ID/SH_CLIENT_SECRET (Sentinel Hub commercial).
    Falls back to CDSE_CLIENT_ID/CDSE_CLIENT_SECRET and switches all
    endpoints to the CDSE deployment automatically.
    """
    sh_id     = os.environ.get("SH_CLIENT_ID")
    sh_secret = os.environ.get("SH_CLIENT_SECRET")
    cdse_id     = os.environ.get("CDSE_CLIENT_ID")
    cdse_secret = os.environ.get("CDSE_CLIENT_SECRET")

    if sh_id and sh_secret:
        client_id, client_secret = sh_id, sh_secret
        token_url   = _SH_TOKEN_URL
        catalog_url = _SH_CATALOG_URL
        process_url = _SH_PROCESS_URL
        log.info("Using Sentinel Hub commercial endpoints.")
    elif cdse_id and cdse_secret:
        client_id, client_secret = cdse_id, cdse_secret
        token_url   = _CDSE_TOKEN_URL
        catalog_url = _CDSE_CATALOG_URL
        process_url = _CDSE_PROCESS_URL
        log.info("Using CDSE endpoints.")
    else:
        raise RuntimeError("Missing credentials: set SH_CLIENT_ID/SH_CLIENT_SECRET or CDSE_CLIENT_ID/CDSE_CLIENT_SECRET.")

    return (
        CatalogClient(client_id, client_secret, token_url=token_url, catalog_url=catalog_url),
        ProcessClient(client_id, client_secret, evalscript, multitemporal_evalscript,
                      token_url=token_url, process_url=process_url),
    )
