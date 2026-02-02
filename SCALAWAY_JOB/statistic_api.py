#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv("vm.env")

import pandas as pd
import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

TOKEN_URL = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
STATS_URL = "https://services.sentinel-hub.com/api/v1/statistics"

# Your evalscript returns 18 values in "features" (B0..B17)
FEATURE_COLS = [
    "B02", "B03", "B04", "B08", "B11", "B12",             # 0..5 raw
    "NDVI", "NDWI", "MNDWI", "NDMI", "BSI",               # 6..10
    "BRIGHT", "ALBEDO_PROXY",                             # 11..12
    "RED", "SWIR1", "SWIR2",                              # 13..15
    "RED_SWIR1_RATIO", "SWIR1_SWIR2_RATIO"                # 16..17
]


def sentinelhub_compliance_hook(resp: requests.Response) -> requests.Response:
    resp.raise_for_status()
    return resp


def parse_date(d: str) -> str:
    pd.to_datetime(d, format="%Y-%m-%d")
    return d


def bbox_around_point_m(lat: float, lon: float, size_m: float) -> List[float]:
    half = size_m / 2.0
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    dlat = half / meters_per_deg_lat
    dlon = half / meters_per_deg_lon
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def get_oauth_session(client_id: str, client_secret: str) -> OAuth2Session:
    oauth = OAuth2Session(client=BackendApplicationClient(client_id=client_id))
    oauth.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)
    oauth.fetch_token(
        token_url=TOKEN_URL,
        client_secret=client_secret,
        include_client_id=True,
    )
    return oauth


def build_stats_request(
    *,
    bbox: List[float],
    start_date: str,
    end_date: str,
    interval: str,
    evalscript: str,
    res: int = 20,
) -> Dict[str, Any]:
    return {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {"mosaickingOrder": "mostRecent"},
                }
            ],
        },
        "aggregation": {
            "timeRange": {
                "from": f"{start_date}T00:00:00Z",
                "to": (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
            },
            "aggregationInterval": {"of": interval},
            "evalscript": evalscript,
            "resx": res,
            "resy": res,
        },
        # keys must match output IDs in evalscript: "features" and "dataMask"
        "calculations": {
            "features": {
                "statistics": {"default": {"percentiles": {"k": [50]}}}
            },
            "dataMask": {
                "statistics": {"default": {}}
            },
        },
    }


def _get_percentile50(outputs: Dict[str, Any], band_key: str) -> Optional[float]:
    try:
        return outputs["features"]["bands"][band_key]["stats"]["percentiles"]["50.0"]
    except Exception:
        return None


def _get_datamask_counts(outputs: Dict[str, Any]) -> Tuple[int, int]:
    stats = outputs.get("dataMask", {}).get("bands", {}).get("B0", {}).get("stats", {}) or {}
    sample = int(stats.get("sampleCount", 0) or 0)
    nodata = int(stats.get("noDataCount", 0) or 0)
    return sample, nodata


def parse_daily_dicts(
    sh_json: Dict[str, Any],
    *,
    lat: float,
    lon: float,
    bbox: List[float],
    start_date: str,
    end_date: str,
    interval: str,
) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []

    for item in sh_json.get("data", []):
        interval_obj = item.get("interval", {})
        outputs = item.get("outputs", {})

        sample, nodata = _get_datamask_counts(outputs)
        denom = sample + nodata
        coverage = (sample / denom) if denom > 0 else 0.0

        row: Dict[str, Any] = {
            "lat": lat,
            "lon": lon,
            "bbox_epsg4326": bbox,
            "query_start_date": start_date,
            "query_end_date": end_date,
            "aggregation_interval": interval,
            "from": interval_obj.get("from"),
            "to": interval_obj.get("to"),
            "sampleCount": sample,
            "noDataCount": nodata,
            "coverage": coverage,
            "p50": {},  # feature_name -> p50
        }

        for i, name in enumerate(FEATURE_COLS):
            row["p50"][name] = _get_percentile50(outputs, f"B{i}")

        out_rows.append(row)

    return out_rows


def aggregate_one_row(
    daily_rows: List[Dict[str, Any]],
    *,
    coverage_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Scene-level filter: keep rows with coverage >= threshold.
    Aggregation: median across kept days for each p50 feature.
    """
    total = len(daily_rows)
    kept = [r for r in daily_rows if (r.get("coverage") is not None and r["coverage"] >= coverage_threshold)]

    def _median(vals: List[Optional[float]]) -> Optional[float]:
        vv = [v for v in vals if v is not None]
        if not vv:
            return None
        return float(pd.Series(vv).median())

    # base metadata
    base = daily_rows[0] if total else {}
    agg: Dict[str, Any] = {
        "lat": base.get("lat"),
        "lon": base.get("lon"),
        "bbox_epsg4326": base.get("bbox_epsg4326"),
        "query_start_date": base.get("query_start_date"),
        "query_end_date": base.get("query_end_date"),
        "aggregation_interval": base.get("aggregation_interval"),
        "coverage_threshold": coverage_threshold,
        "n_days_total": total,
        "n_days_kept": len(kept),
        "kept_ratio": (len(kept) / total) if total else 0.0,
        "coverage_median_kept": _median([r.get("coverage") for r in kept]) if kept else None,
        "coverage_min_kept": min([r.get("coverage", 0.0) for r in kept]) if kept else None,
        "p50_aggregated": {},  # feature_name -> median over kept days
    }

    for name in FEATURE_COLS:
        agg["p50_aggregated"][name] = _median([r.get("p50", {}).get(name) for r in kept]) if kept else None

    return kept, agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--start_date", type=parse_date, required=True)  # YYYY-MM-DD
    ap.add_argument("--end_date", type=parse_date, required=True)    # YYYY-MM-DD
    ap.add_argument("--interval", type=str, default="P1D")
    ap.add_argument("--size_m", type=float, default=30.0)
    ap.add_argument("--coverage_threshold", type=float, default=0.8)
    ap.add_argument("--evalscript_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="out_stats_json")
    args = ap.parse_args()

    client_id = os.getenv("SH_CLIENT_ID")
    client_secret = os.getenv("SH_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SystemExit("Missing env vars SH_CLIENT_ID / SH_CLIENT_SECRET")

    evalscript = Path(args.evalscript_path).read_text(encoding="utf-8")
    bbox = bbox_around_point_m(args.lat, args.lon, args.size_m)

    oauth = get_oauth_session(client_id, client_secret)
    stats_request = build_stats_request(
        bbox=bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        evalscript=evalscript,
        res=20,
    )

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    resp = oauth.request("POST", STATS_URL, headers=headers, json=stats_request)
    sh_json = resp.json()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) raw response
    (out_dir / "raw_response.json").write_text(json.dumps(sh_json, indent=2), encoding="utf-8")

    # 2) parsed daily
    daily_rows = parse_daily_dicts(
        sh_json,
        lat=args.lat,
        lon=args.lon,
        bbox=bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
    )
    (out_dir / "daily_parsed.json").write_text(json.dumps(daily_rows, indent=2), encoding="utf-8")

    # 3) scene-level filter + aggregated one-row
    kept_rows, agg_row = aggregate_one_row(daily_rows, coverage_threshold=args.coverage_threshold)
    (out_dir / "daily_kept.json").write_text(json.dumps(kept_rows, indent=2), encoding="utf-8")
    (out_dir / "aggregated_one_row.json").write_text(json.dumps(agg_row, indent=2), encoding="utf-8")

    print(f"Wrote: {out_dir / 'raw_response.json'}")
    print(f"Wrote: {out_dir / 'daily_parsed.json'}")
    print(f"Wrote: {out_dir / 'daily_kept.json'}")
    print(f"Wrote: {out_dir / 'aggregated_one_row.json'}")
    print(f"Daily rows: {len(daily_rows)} | Kept (coverage>={args.coverage_threshold}): {len(kept_rows)}")


if __name__ == "__main__":
    main()
