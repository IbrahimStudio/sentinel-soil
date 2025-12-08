from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import json
import requests

from sentinel_soil.config import AppConfig
from sentinel_soil.clients.auth import CDSEAuthClient


@dataclass
class ProcessRequestParams:
    bbox: List[float]      # [minLon, minLat, maxLon, maxLat] in EPSG:4326
    crs_epsg: int          # e.g. 4326
    from_date: str         # "YYYY-MM-DD"
    to_date: str           # "YYYY-MM-DD"
    max_cloud: int         # 0..100
    bands: List[str]       # ["B02","B03",...]
    width: int = 512       # output width in pixels
    height: int = 512      # output height in pixels


class SentinelHubProcessClient:
    def __init__(self, cfg: AppConfig, auth_client: CDSEAuthClient):
        self.cfg = cfg
        self.auth_client = auth_client
        self.process_url = cfg.sentinelhub.process_url

    def _build_evalscript(self, bands: List[str]) -> str:
        """
        Build a simple evalscript that outputs the requested bands as FLOAT32.
        """
        bands_list = ", ".join(f'"{b}"' for b in bands)
        # JS array to return all bands in order
        return f"""//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: [{bands_list}],
      units: "REFLECTANCE"
    }}],
    output: {{
      bands: {len(bands)},
      sampleType: "FLOAT32"
    }}
  }};
}}

function evaluatePixel(sample) {{
  return [{", ".join(f"sample.{b}" for b in bands)}];
}}
"""

    def build_request_body(self, p: ProcessRequestParams) -> Dict[str, Any]:
        crs_url = f"http://www.opengis.net/def/crs/EPSG/0/{p.crs_epsg}"

        body = {
            "input": {
                "bounds": {
                    "bbox": p.bbox,
                    "properties": {
                        "crs": crs_url
                    }
                },
                "data": [
                    {
                        "type": self.cfg.sentinel2.collection,  # "sentinel-2-l2a"
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{p.from_date}T00:00:00Z",
                                "to": f"{p.to_date}T23:59:59Z",
                            },
                            "maxCloudCoverage": p.max_cloud,
                        },
                        # "processing": {}  # can add mosaickingOrder etc. later
                    }
                ]
            },
            "output": {
                "width": p.width,
                "height": p.height,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {
                            "type": "image/tiff"
                        }
                    }
                ]
            },
            "evalscript": self._build_evalscript(p.bands),
        }
        return body

    def download_geotiff_bytes(self, params: ProcessRequestParams) -> bytes:
        body = self.build_request_body(params)
        headers = {
            "Authorization": f"Bearer {self.auth_client.get_token()}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            self.process_url,
            headers=headers,
            data=json.dumps(body),
            timeout=120,
        )
        resp.raise_for_status()
        return resp.content
