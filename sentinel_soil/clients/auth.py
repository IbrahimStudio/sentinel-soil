from dataclasses import dataclass
from typing import Optional
import time
import requests

from sentinel_soil.config import CDSEConfig


@dataclass
class Token:
    access_token: str
    expires_at: float  # epoch seconds


class CDSEAuthClient:
    def __init__(self, cfg: CDSEConfig):
        self.cfg = cfg
        self._token: Optional[Token] = None

    def _fetch_token(self) -> Token:
        data = {
            "grant_type": "client_credentials",
            "client_id": self.cfg.client_id,
        }
        if self.cfg.client_secret:
            data["client_secret"] = self.cfg.client_secret

        resp = requests.post(self.cfg.auth_url, data=data, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        access_token = payload["access_token"]
        expires_in = payload.get("expires_in", 300)  # seconds

        return Token(
            access_token=access_token,
            expires_at=time.time() + expires_in * 0.9,  # refresh a bit early
        )

    def get_token(self) -> str:
        if self._token is None or time.time() >= self._token.expires_at:
            self._token = self._fetch_token()
        return self._token.access_token

    def auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self.get_token()}"}
