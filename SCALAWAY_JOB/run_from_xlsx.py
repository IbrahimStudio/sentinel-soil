from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

import pandas as pd

from pipeline.worker import run_one_job  # adjust if your module path differs


# ----------------- EDIT THESE (debug-friendly) -----------------
XLSX_PATH = "gabri_filters.xlsx"
SHEET_NAME = 0               # or "Sheet1"
LIMIT: Optional[int] = 1  # set 5 for a quick test
N_WORKERS = 1               # set 1 for easiest debugging

CONFIG_PATH = "configs/dev.yaml"

# Job defaults (match your pipeline assumptions)
WINDOW_DAYS = 15             # +/- 15 days around survey_date
WINDOW_W = 3
WINDOW_H = 3
RES_M = 10

NDVI_THRESHOLD = 0.2
MIN_OBS = 2
MAX_CLOUD_COVERAGE = 80
MOSAICKING_ORDER = "mostRecent"

OUT_RESULTS_JSONL = "run_results.jsonl"
# ---------------------------------------------------------------

def _normalize_point_id(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _load_env() -> None:
    # Optional: load env vars from vm.env (good for Object Storage creds, SentinelHub creds, etc.)
    try:
        from dotenv import load_dotenv
        load_dotenv("vm.env")
        print("Loaded env from vm.env")
    except Exception:
        print("dotenv not available or vm.env missing; continuing without it")


def _build_payload(row: pd.Series) -> Dict[str, Any]:
    point_id = _normalize_point_id(row["POINT_ID"])
    lat = float(row["TH_LAT"])
    lon = float(row["TH_LONG"])

    survey_dt = pd.to_datetime(row["SURVEY_DATE"], errors="coerce")
    if pd.isna(survey_dt):
        raise ValueError(f"Invalid SURVEY_DATE for point_id={point_id}: {row['SURVEY_DATE']}")
    survey_date = survey_dt.date().isoformat()

    # Unique job id per run (lets you re-run same point without collisions)
    job_id = f"{point_id}__{uuid.uuid4().hex[:8]}"

    payload: Dict[str, Any] = {
        "job_id": job_id,
        "point_id": point_id,
        "lat": lat,
        "lon": lon,
        "survey_date": survey_date,

        # time window around survey
        "window_days": WINDOW_DAYS,

        # spatial window
        "spatial_window": {"w": WINDOW_W, "h": WINDOW_H},
        "res_m": RES_M,

        # filters / aggregation
        "ndvi_threshold": NDVI_THRESHOLD,
        "min_obs": MIN_OBS,

        # SentinelHub request settings
        "max_cloud_coverage": MAX_CLOUD_COVERAGE,
        "mosaicking_order": MOSAICKING_ORDER,
    }

    return payload


def _run_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Run a single job; returns manifest dict
    return run_one_job(payload, config_path=CONFIG_PATH)


def main() -> None:
    _load_env()

    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)

    required = ["POINT_ID", "TH_LAT", "TH_LONG", "SURVEY_DATE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in XLSX: {missing}")

    # Clean rows
    df = df.copy()
    df["POINT_ID"] = df["POINT_ID"].apply(_normalize_point_id)
    df = df[df["POINT_ID"] != ""]
    df = df.dropna(subset=["TH_LAT", "TH_LONG", "SURVEY_DATE"])

    if LIMIT is not None:
        df = df.head(LIMIT)

    payloads: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        payloads.append(_build_payload(r))

    print(f"Prepared {len(payloads)} jobs from {XLSX_PATH}. N_WORKERS={N_WORKERS}")

    # Run
    results: List[Dict[str, Any]] = []
    t0 = datetime.now()

    if N_WORKERS == 1:
        for i, payload in enumerate(payloads, start=1):
            print(f"[{i}/{len(payloads)}] Running point_id={payload['point_id']} job_id={payload['job_id']}")
            manifest = _run_payload(payload)
            results.append(manifest)
            with open(OUT_RESULTS_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(manifest, ensure_ascii=False) + "\n")
    else:
        # Use process-based parallelism (safe for GDAL/rasterio; avoids GIL limits)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(_run_payload, p): p for p in payloads}
            done = 0
            for fut in as_completed(futs):
                payload = futs[fut]
                done += 1
                try:
                    manifest = fut.result()
                except Exception as e:
                    manifest = {
                        "job_id": payload["job_id"],
                        "point_id": payload["point_id"],
                        "status": "FAILED",
                        "error": str(e),
                    }

                print(f"[{done}/{len(payloads)}] {manifest.get('status')} point_id={payload['point_id']}")
                results.append(manifest)
                with open(OUT_RESULTS_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(manifest, ensure_ascii=False) + "\n")

    dt = datetime.now() - t0
    ok = sum(1 for r in results if r.get("status") == "SUCCESS")
    fail = len(results) - ok
    print(f"Done in {dt}. SUCCESS={ok} FAILED={fail}")
    print(f"Results appended to: {OUT_RESULTS_JSONL}")


if __name__ == "__main__":
    main()
