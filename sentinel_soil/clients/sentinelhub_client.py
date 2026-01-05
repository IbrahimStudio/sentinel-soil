from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
from sentinelhub import (
    SentinelHubRequest,
    DataCollection,
    MimeType,
    CRS,
    BBox,
    SHConfig,
)

@dataclass
class SentinelHubCredentials:
    client_id: str
    client_secret: str
    token_url: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    base_url: str = "https://sh.dataspace.copernicus.eu"


class SentinelHubClient:
    """
    Thin wrapper around sentinelhub SDK.
    """

    def __init__(self, credentials: SentinelHubCredentials):
        self.config = SHConfig()
        self.config.sh_client_id = credentials.client_id
        self.config.sh_client_secret = credentials.client_secret
        self.config.sh_token_url = credentials.token_url
        self.config.sh_base_url = credentials.base_url

    def request_tiff(
        self,
        evalscript: str,
        bbox: Tuple[float, float, float, float],
        crs_epsg: int,
        time_interval: Tuple[str, str],
        size: Tuple[int, int],
        max_cloud_coverage: int,
        output_folder: str,
        mosaicking_order: str = "leastCC",
    ) -> None:
        """
        Executes a Sentinel Hub Process API request and saves data to disk.
        """

        bbox_obj = BBox(bbox=bbox, crs=CRS(crs_epsg))

        data_collection = DataCollection.SENTINEL2_L2A.define_from(
            name="sentinel-2-l2a",
            service_url=self.config.sh_base_url,
        )

        request = SentinelHubRequest(
            data_folder=output_folder,
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=time_interval,
                    other_args={
                        "dataFilter": {
                            "mosaickingOrder": mosaicking_order,
                            "maxCloudCoverage": max_cloud_coverage,
                        }
                    },
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF),
                SentinelHubRequest.output_response("userdata", MimeType.JSON),
            ],
            bbox=bbox_obj,
            size=size,
            config=self.config,
        )

        request.get_data(save_data=True)
