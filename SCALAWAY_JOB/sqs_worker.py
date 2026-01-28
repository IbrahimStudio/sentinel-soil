from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import boto3

from pipeline.worker import run_one_job  # <-- adjust import to your module path


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def build_job_payload_from_seed(seed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a minimal SQS 'seed' message into the full payload expected by parse_job().
    """
    point_id = str(seed["point_id"]).strip()
    lat = float(seed["lat"])
    lon = float(seed["lon"])
    survey_date = str(seed["survey_date"])  # ISO date string

    payload: Dict[str, Any] = {
        "job_id": seed.get("job_id") or f"{point_id}__{uuid.uuid4().hex[:8]}",
        "point_id": point_id,
        "lat": lat,
        "lon": lon,
        "survey_date": survey_date,

        # 30-day window around SURVEY_DATE
        "window_days": int(os.getenv("WINDOW_DAYS", "15")),  # +/- 15 => 30 total

        # 3x3 pixels @ 10m
        "window": {
            "w": int(os.getenv("WINDOW_W", "3")),
            "h": int(os.getenv("WINDOW_H", "3")),
        },
        "res_m": float(os.getenv("RES_M", "10")),

        # Bare soil filtering / aggregation knobs
        "ndvi_threshold": float(os.getenv("NDVI_THRESHOLD", "0.2")),
        "min_obs": int(os.getenv("MIN_OBS", "2")),

        # SentinelHub request knobs (keep defaults here)
        "max_cloud_coverage": float(os.getenv("MAX_CLOUD_COVERAGE", "80")),
        "mosaicking_order": os.getenv("MOSAICKING_ORDER", "mostRecent"),
    }
    return payload


def main() -> None:
    queue_url = _env("SQS_QUEUE_URL")
    sqs_endpoint = _env("SQS_ENDPOINT")
    region = os.getenv("AWS_DEFAULT_REGION", "fr-par")

    wait_time = int(os.getenv("SQS_WAIT_TIME", "20"))         # long poll
    visibility_timeout = int(os.getenv("SQS_VISIBILITY", "900"))  # seconds
    idle_sleep = float(os.getenv("IDLE_SLEEP", "1.0"))

    config_path = os.getenv("CONFIG_PATH", "configs/dev.yaml")

    sqs = boto3.client(
        "sqs",
        endpoint_url=sqs_endpoint,
        region_name=region,
        aws_access_key_id=_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("AWS_SECRET_ACCESS_KEY"),
    )

    while True:
        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=wait_time,
            VisibilityTimeout=visibility_timeout,
            MessageAttributeNames=["All"],
            AttributeNames=["All"],
        )

        msgs = resp.get("Messages", [])
        if not msgs:
            time.sleep(idle_sleep)
            continue

        msg = msgs[0]
        receipt = msg["ReceiptHandle"]

        try:
            seed = json.loads(msg["Body"])
            payload = build_job_payload_from_seed(seed)

            manifest = run_one_job(payload, config_path=config_path)

            if manifest.get("status") == "SUCCESS":
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
            else:
                # Let it reappear after visibility timeout (or add DLQ later)
                pass

        except Exception:
            # also let it reappear; you can add backoff if needed
            pass


if __name__ == "__main__":
    main()
