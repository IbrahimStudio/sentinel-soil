#!/usr/bin/env python3
"""
batch_stats_from_xlsx.py

Reads an XLSX, and for each row runs a Sentinel Hub Statistical API request
(using an evalscript that outputs:
  - "features" with 18 bands (B0..B17)
  - "dataMask" with 1 band
)

Outputs (JSON only):
- out_dir/raw_response/<point_id>__<job_id>.json           (raw API response)
- out_dir/daily_parsed.jsonl                              (one JSON per interval per point)
- out_dir/daily_kept.jsonl                                (intervals kept after scene-level coverage filter)
- out_dir/aggregated_one_row.jsonl                        (one JSON per point: median over kept intervals)
- out_dir/errors.jsonl                                    (failures)

Workers: default 3.

Required XLSX columns: POINT_ID, TH_LAT, TH_LONG, SURVEY_DATE
Time window: +/- WINDOW_DAYS around SURVEY_DATE (inclusive), aggregated daily (P1D).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# ---------- Sentinel Hub endpoints ----------
TOKEN_URL = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
STATS_URL = "https://services.sentinel-hub.com/api/v1/statistics"

# ---------- Feature mapping (B0..B17) ----------
FEATURE_COLS = [
    "B02", "B03", "B04", "B08", "B11", "B12",             # 0..5 raw
    "NDVI", "NDWI", "MNDWI", "NDMI", "BSI",               # 6..10
    "BRIGHT", "ALBEDO_PROXY",                             # 11..12
    "RED", "SWIR1", "SWIR2",                              # 13..15
    "RED_SWIR1_RATIO", "SWIR1_SWIR2_RATIO"                # 16..17
]

REQUIRED_COLS = ["POINT_ID", "TH_LAT", "TH_LONG", "SURVEY_DATE"]


# ---------------------- Helpers ----------------------
def _normalize_point_id(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _bbox_around_point_m(lat: float, lon: float, size_m: float) -> List[float]:
    half = size_m / 2.0
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    dlat = half / meters_per_deg_lat
    dlon = half / meters_per_deg_lon
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def _sentinelhub_compliance_hook(resp: requests.Response) -> requests.Response:
    resp.raise_for_status()
    return resp


def _oauth_session(client_id: str, client_secret: str) -> OAuth2Session:
    oauth = OAuth2Session(client=BackendApplicationClient(client_id=client_id))
    oauth.register_compliance_hook("access_token_response", _sentinelhub_compliance_hook)
    oauth.fetch_token(
        token_url=TOKEN_URL,
        client_secret=client_secret,
        include_client_id=True,
    )
    return oauth


def _build_stats_request(
    *,
    bbox: List[float],
    start_date: str,  # YYYY-MM-DD
    end_date: str,    # YYYY-MM-DD
    interval: str,
    evalscript: str,
    res: int,
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
                    "dataFilter": {"mosaickingOrder": "leastCC"},
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


def _as_float_or_none(x):
    if x is None:
        return None
    # Sentinel Hub often returns "NaN" as a string in JSON
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("nan", "null", "", "NaN"):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    # numeric
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None
        return xf
    except Exception:
        return None


def _get_percentile50(outputs, band_key):
    try:
        x = outputs["features"]["bands"][band_key]["stats"]["percentiles"]["50.0"]
        return _as_float_or_none(x)
    except Exception:
        return None


def _get_datamask_counts(outputs: Dict[str, Any]) -> Tuple[int, int]:
    try:
        stats = outputs.get("dataMask", {}).get("bands", {}).get("B0", {}).get("stats", {}) or {}
        sample = int(stats.get("sampleCount", 0) or 0)
        nodata = int(stats.get("noDataCount", 0) or 0)
        return sample, nodata
    except Exception:
        return 0, 0


def _parse_daily_rows(
    sh_json: Dict[str, Any],
    *,
    job_id: str,
    point_id: str,
    lat: float,
    lon: float,
    bbox: List[float],
    start_date: str,
    end_date: str,
    interval: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in sh_json.get("data", []):
        interval_obj = item.get("interval", {})
        outputs = item.get("outputs", {})

        sample, nodata = _get_datamask_counts(outputs)
        denom = sample + nodata
        coverage = (sample / denom) if denom > 0 else 0.0

        row: Dict[str, Any] = {
            "job_id": job_id,
            "point_id": point_id,
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
            "p50": {},
        }
        for i, name in enumerate(FEATURE_COLS):
            row["p50"][name] = _get_percentile50(outputs, f"B{i}")
        out.append(row)
    return out


def _median(vals: List[Optional[float]]) -> Optional[float]:
    vv = [v for v in vals if v is not None]
    if not vv:
        return None
    return float(pd.Series(vv).median())


def _aggregate_one_row(
    daily_rows: List[Dict[str, Any]],
    *,
    coverage_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    total = len(daily_rows)
    kept = [r for r in daily_rows if (r.get("coverage") is not None and r["coverage"] >= coverage_threshold)]

    base = daily_rows[0] if total else {}
    agg: Dict[str, Any] = {
        "job_id": base.get("job_id"),
        "point_id": base.get("point_id"),
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
        "p50_aggregated": {},
    }

    for name in FEATURE_COLS:
        agg["p50_aggregated"][name] = _median([r.get("p50", {}).get(name) for r in kept]) if kept else None

    return kept, agg


# ---------------------- Job runner ----------------------
@dataclass(frozen=True)
class Job:
    job_id: str
    point_id: str
    lat: float
    lon: float
    start_date: str
    end_date: str
    bbox: List[float]


def _run_job(job: Job, *, evalscript: str, interval: str, res: int, coverage_threshold: float, out_raw_dir: str) -> Dict[str, Any]:
    """
    Runs one Statistical API request and returns:
      {
        status: "SUCCESS",
        raw_path: "...",
        daily_rows: [...],
        kept_rows: [...],
        aggregated: {...}
      }
    or status FAILED.
    """
    client_id = os.getenv("SH_CLIENT_ID")
    client_secret = os.getenv("SH_CLIENT_SECRET")
    if not client_id or not client_secret:
        return {
            "status": "FAILED",
            "job_id": job.job_id,
            "point_id": job.point_id,
            "error": "Missing env vars SH_CLIENT_ID / SH_CLIENT_SECRET",
        }

    try:
        oauth = _oauth_session(client_id, client_secret)
        req = _build_stats_request(
            bbox=job.bbox,
            start_date=job.start_date,
            end_date=job.end_date,
            interval=interval,
            evalscript=evalscript,
            res=res,
        )
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        resp = oauth.request("POST", STATS_URL, headers=headers, json=req)
        sh_json = resp.json()

        raw_dir = Path(out_raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{job.point_id}__{job.job_id}.json"
        raw_path.write_text(json.dumps(sh_json, indent=2), encoding="utf-8")

        daily_rows = _parse_daily_rows(
            sh_json,
            job_id=job.job_id,
            point_id=job.point_id,
            lat=job.lat,
            lon=job.lon,
            bbox=job.bbox,
            start_date=job.start_date,
            end_date=job.end_date,
            interval=interval,
        )
        kept_rows, agg_row = _aggregate_one_row(daily_rows, coverage_threshold=coverage_threshold)

        return {
            "status": "SUCCESS",
            "job_id": job.job_id,
            "point_id": job.point_id,
            "raw_path": str(raw_path),
            "daily_rows": daily_rows,
            "kept_rows": kept_rows,
            "aggregated": agg_row,
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "job_id": job.job_id,
            "point_id": job.point_id,
            "error": str(e),
        }


# ---------------------- Main ----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=str, required=True)
    ap.add_argument("--sheet", type=str, default="0", help='Sheet name or index as string (e.g. "0" or "Sheet1")')
    ap.add_argument("--limit", type=int, default=-1, help="If >0, process only first N rows")
    ap.add_argument("--workers", type=int, default=3, help="Use 2 or 3")
    ap.add_argument("--evalscript_path", type=str, required=True, help="Evalscript file with features bands=18")
    ap.add_argument("--window_days", type=int, default=15, help="+/- days around SURVEY_DATE")
    ap.add_argument("--bbox_size_m", type=float, default=30.0, help="Square bbox side length in meters")
    ap.add_argument("--interval", type=str, default="P1D")
    ap.add_argument("--res", type=int, default=20, help="Use 20 to support SWIR bands consistently")
    ap.add_argument("--coverage_threshold", type=float, default=0.8)
    ap.add_argument("--out_dir", type=str, default="out_batch_json")
    args = ap.parse_args()

    # Optional: dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv("vm.env")
    except Exception:
        pass

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise SystemExit(f"XLSX not found: {xlsx_path}")

    # sheet parsing: allow "0" as index
    sheet: Any
    if args.sheet.isdigit():
        sheet = int(args.sheet)
    else:
        sheet = args.sheet

    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in XLSX: {missing}")

    df = df.copy()
    df["POINT_ID"] = df["POINT_ID"].apply(_normalize_point_id)
    df = df[df["POINT_ID"] != ""]
    df = df.dropna(subset=["TH_LAT", "TH_LONG", "SURVEY_DATE"])

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    evalscript = Path(args.evalscript_path).read_text(encoding="utf-8")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_raw_dir = out_dir / "raw_response"
    out_daily_parsed = out_dir / "daily_parsed.jsonl"
    out_daily_kept = out_dir / "daily_kept.jsonl"
    out_agg = out_dir / "aggregated_one_row.jsonl"
    out_err = out_dir / "errors.jsonl"

    # clear old outputs (simple + predictable)
    for p in [out_daily_parsed, out_daily_kept, out_agg, out_err]:
        if p.exists():
            p.unlink()

    # Build jobs
    jobs: List[Job] = []
    for _, r in df.iterrows():
        point_id = _normalize_point_id(r["POINT_ID"])
        lat = float(r["TH_LAT"])
        lon = float(r["TH_LONG"])

        survey_dt = pd.to_datetime(r["SURVEY_DATE"], errors="coerce")
        if pd.isna(survey_dt):
            # write error immediately
            with open(out_err, "a", encoding="utf-8") as f:
                f.write(json.dumps({"status": "FAILED", "point_id": point_id, "error": f"Invalid SURVEY_DATE: {r['SURVEY_DATE']}"},
                                   ensure_ascii=False) + "\n")
            continue

        center = pd.to_datetime(survey_dt).normalize()
        start_date = (center - pd.Timedelta(days=args.window_days)).strftime("%Y-%m-%d")
        end_date = (center + pd.Timedelta(days=args.window_days)).strftime("%Y-%m-%d")

        job_id = uuid.uuid4().hex[:10]
        bbox = _bbox_around_point_m(lat, lon, args.bbox_size_m)

        jobs.append(Job(
            job_id=job_id,
            point_id=point_id,
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            bbox=bbox,
        ))

    print(f"Prepared {len(jobs)} jobs from {xlsx_path}. workers={args.workers}")

    # Run jobs
    if args.workers <= 1:
        for i, job in enumerate(jobs, start=1):
            print(f"[{i}/{len(jobs)}] point_id={job.point_id}")
            res = _run_job(
                job,
                evalscript=evalscript,
                interval=args.interval,
                res=args.res,
                coverage_threshold=args.coverage_threshold,
                out_raw_dir=str(out_raw_dir),
            )
            _append_outputs(res, out_daily_parsed, out_daily_kept, out_agg, out_err)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(
                    _run_job,
                    job,
                    evalscript=evalscript,
                    interval=args.interval,
                    res=args.res,
                    coverage_threshold=args.coverage_threshold,
                    out_raw_dir=str(out_raw_dir),
                ): job
                for job in jobs
            }
            done = 0
            for fut in as_completed(futs):
                done += 1
                job = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"status": "FAILED", "job_id": job.job_id, "point_id": job.point_id, "error": str(e)}

                print(f"[{done}/{len(jobs)}] {res.get('status')} point_id={job.point_id}")
                _append_outputs(res, out_daily_parsed, out_daily_kept, out_agg, out_err)

    print(f"Raw per-point API JSON: {out_raw_dir}")
    print(f"Daily parsed JSONL:     {out_daily_parsed}")
    print(f"Daily kept JSONL:       {out_daily_kept}")
    print(f"Aggregated JSONL:       {out_agg}")
    print(f"Errors JSONL:           {out_err}")


def _append_outputs(
    res: Dict[str, Any],
    out_daily_parsed: Path,
    out_daily_kept: Path,
    out_agg: Path,
    out_err: Path
) -> None:
    status = res.get("status")
    if status != "SUCCESS":
        with open(out_err, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        return

    # write daily parsed (jsonl: one record per interval)
    for row in res.get("daily_rows", []):
        with open(out_daily_parsed, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # write kept daily
    for row in res.get("kept_rows", []):
        with open(out_daily_kept, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # write aggregated (one record per point)
    with open(out_agg, "a", encoding="utf-8") as f:
        f.write(json.dumps(res.get("aggregated", {}), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
