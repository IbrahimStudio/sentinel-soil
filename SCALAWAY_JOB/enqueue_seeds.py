from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd

try:
    # Optional: makes local debugging easier if you have vm.env
    from dotenv import load_dotenv
    load_dotenv("vm.env")
except Exception:
    pass


# --------- EDIT THESE FOR YOUR DEBUG SESSION ----------
XLSX_PATH = "gabri_filters.xlsx"   # <- your seed file
SHEET_NAME = 0                     # or "Sheet1"
LIMIT: Optional[int] = 5           # set None to send all
# ------------------------------------------------------

REQUIRED_COLS = ["POINT_ID", "TH_LAT", "TH_LONG", "SURVEY_DATE"]


def must_env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing env var: {name}")
    return v


def normalize_point_id(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Excel sometimes turns IDs into floats like "123.0"
    if s.endswith(".0"):
        s = s[:-2]
    return s


def build_seed_messages(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df = df.copy()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in XLSX: {missing}")

    df["POINT_ID"] = df["POINT_ID"].apply(normalize_point_id)
    df["SURVEY_DATE"] = pd.to_datetime(df["SURVEY_DATE"], errors="coerce").dt.date

    df = df.dropna(subset=["SURVEY_DATE", "TH_LAT", "TH_LONG"])
    df = df[df["POINT_ID"] != ""]

    msgs: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        msgs.append(
            {
                "point_id": r["POINT_ID"],
                "lat": float(r["TH_LAT"]),
                "lon": float(r["TH_LONG"]),
                "survey_date": r["SURVEY_DATE"].isoformat(),
                # optional metadata for traceability (harmless if unused)
                "depth": str(r["Depth"]) if "Depth" in df.columns and not pd.isna(r.get("Depth")) else None,
                "nuts_0": r.get("NUTS_0", None),
                "nuts_1": r.get("NUTS_1", None),
                "nuts_2": r.get("NUTS_2", None),
                "nuts_3": r.get("NUTS_3", None),
                "lc": r.get("LC", None),
                "lu": r.get("LU", None),
            }
        )
    return msgs


def main() -> None:
    sqs_endpoint = must_env("SQS_ENDPOINT", "https://sqs.mnq.fr-par.scaleway.com")
    queue_url = must_env("SQS_QUEUE_URL")
    region = os.getenv("AWS_DEFAULT_REGION", "fr-par")

    sqs = boto3.client(
        "sqs",
        endpoint_url=sqs_endpoint,
        region_name=region,
        aws_access_key_id=must_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=must_env("AWS_SECRET_ACCESS_KEY"),
    )

    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)
    msgs = build_seed_messages(df)

    if LIMIT is not None:
        msgs = msgs[:LIMIT]

    print(f"Prepared {len(msgs)} messages from {XLSX_PATH}")

    sent = 0
    for i, m in enumerate(msgs, start=1):
        body = json.dumps(m, ensure_ascii=False)
        try:
            sqs.send_message(QueueUrl=queue_url, MessageBody=body)
            sent += 1
            if i % 25 == 0 or i == len(msgs):
                print(f"Sent {sent}/{len(msgs)}")
        except Exception as e:
            print(f"[ERROR] Failed sending message #{i} point_id={m.get('point_id')}: {e}")
            # For debugging: stop immediately on first failure
            raise

    print("Done.")


if __name__ == "__main__":
    main()
