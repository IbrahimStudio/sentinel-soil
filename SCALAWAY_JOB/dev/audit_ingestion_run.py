#!/usr/bin/env python3
"""
Audit which points have been successfully ingested in S3.

By default only objects written within the last 3 days are counted as
belonging to the current run. Use --max-age-days to adjust, or 0 to
disable the filter entirely.

Usage:
    cd SCALAWAY_JOB
    source .env  # or set env vars manually
    uv run python dev/audit_ingestion_run.py \\
        --prefix batch_results/20260416T103000 \\
        --xlsx gabri_filters.xlsx \\
        --max-age-days 3

Output:
    - Count of succeeded / failed / missing points (current run only)
    - List of succeeded point_ids
    - List of failed point_ids with error messages
    - CSV uploaded to S3: {prefix}/audit/audit_results.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# sh_pipeline lives under SCALAWAY_JOB/ingestion/
_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo / "ingestion"))

from sh_pipeline.storage import storage_from_env


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit ingestion results in S3")
    ap.add_argument("--prefix", default="batch_results",
                    help="Storage prefix used during the run (default: batch_results)")
    ap.add_argument("--xlsx", default=None,
                    help="Optional: gabri_filters.xlsx to cross-reference expected point_ids")
    ap.add_argument("--out", default="audit_results.csv",
                    help="Output CSV filename (uploaded to S3 under {prefix}/audit/, default: audit_results.csv)")
    ap.add_argument("--max-age-days", type=float, default=3.0,
                    help="Ignore S3 objects older than this many days (0 = no filter, default: 3)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)

    cutoff: datetime | None = None
    if args.max_age_days > 0:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=args.max_age_days)
        print(f"Filtering: only objects written after {cutoff.strftime('%Y-%m-%d %H:%M UTC')} "
              f"(--max-age-days {args.max_age_days})")

    print(f"Connecting to Scaleway S3 (bucket: {os.environ.get('SCALEWAY_S3_BUCKET', '?')}) ...")
    storage = storage_from_env()

    def fresh_keys(prefix_path: str) -> tuple[set[str], dict[str, datetime]]:
        """
        Return (set of point_ids, mapping point_id -> last_modified)
        for objects that pass the age filter.
        """
        objects = storage.list_objects_with_metadata(prefix_path)
        ids: set[str] = set()
        ts_map: dict[str, datetime] = {}
        stale = 0
        for obj in objects:
            lm: datetime = obj["last_modified"]
            if cutoff and lm < cutoff:
                stale += 1
                continue
            pid = Path(obj["key"]).stem
            ids.add(pid)
            ts_map[pid] = lm
        if stale:
            print(f"  [{prefix_path}] skipped {stale} stale object(s) older than cutoff")
        return ids, ts_map

    print(f"Listing objects under '{args.prefix}/' ...")
    succeeded_ids, agg_ts   = fresh_keys(f"{args.prefix}/aggregated/")
    failed_ids,   err_ts    = fresh_keys(f"{args.prefix}/errors/")
    has_parsed,   _         = fresh_keys(f"{args.prefix}/daily_parsed/")
    has_kept,     _         = fresh_keys(f"{args.prefix}/daily_kept/")

    # All keys regardless of age (for stale detection)
    all_agg_objects = storage.list_objects_with_metadata(f"{args.prefix}/aggregated/")
    stale_ids = {Path(o["key"]).stem for o in all_agg_objects} - succeeded_ids

    print(f"\n{'='*62}")
    print(f"  INGESTION AUDIT — prefix: {args.prefix}/")
    if cutoff:
        print(f"  Run window:  last {args.max_age_days} day(s) (after {cutoff.strftime('%Y-%m-%d %H:%M UTC')})")
    print(f"{'='*62}")
    print(f"  Succeeded  (aggregated/*.json):      {len(succeeded_ids):>5}")
    print(f"  Failed     (errors/*.json):           {len(failed_ids):>5}")
    print(f"  daily_parsed present:                 {len(has_parsed):>5}")
    print(f"  daily_kept present:                   {len(has_kept):>5}")
    if stale_ids:
        print(f"  Stale (from previous run, excluded): {len(stale_ids):>5}")

    # Cross-reference with XLSX if provided
    expected_ids: set[str] = set()
    if args.xlsx:
        try:
            import pandas as pd
            df = pd.read_excel(args.xlsx)
            if "POINT_ID" not in df.columns:
                print(f"  WARNING: POINT_ID column not found in {args.xlsx}")
            else:
                expected_ids = {str(pid).strip() for pid in df["POINT_ID"].dropna()}
                missing_ids = expected_ids - succeeded_ids - failed_ids
                print(f"\n  Expected (from XLSX):                 {len(expected_ids):>5}")
                print(f"  Missing (pipeline crashed before):    {len(missing_ids):>5}")
                if missing_ids:
                    print(f"\n  Missing point_ids:")
                    for pid in sorted(missing_ids)[:20]:
                        print(f"    {pid}")
                    if len(missing_ids) > 20:
                        print(f"    ... and {len(missing_ids) - 20} more (see {args.out})")
        except ImportError:
            print("  (Install pandas to use --xlsx cross-reference)")

    # Show failed points with error messages
    if failed_ids:
        # Re-fetch error keys respecting age filter
        all_err_objects = storage.list_objects_with_metadata(f"{args.prefix}/errors/")
        fresh_err_keys = [
            o["key"] for o in all_err_objects
            if (not cutoff or o["last_modified"] >= cutoff)
        ]
        print(f"\n{'='*62}")
        print(f"  FAILED POINTS ({len(failed_ids)} total)")
        print(f"{'='*62}")
        for key in fresh_err_keys[:10]:
            try:
                text = storage.get_text(key)
                err_data = json.loads(text)
                pid = err_data.get("point_id", Path(key).stem)
                err_msg = err_data.get("error", "unknown")
                print(f"  {pid}: {err_msg[:100]}")
            except Exception as e:
                print(f"  (could not read {key}: {e})")
        if len(failed_ids) > 10:
            print(f"  ... and {len(failed_ids) - 10} more (see {args.out})")

    # Write CSV to S3
    all_ids = succeeded_ids | failed_ids | expected_ids | stale_ids
    if not all_ids:
        print("\n  No objects found. Check --prefix and S3 credentials.")
        return

    rows = []
    for pid in sorted(all_ids):
        if pid in succeeded_ids:
            status = "SUCCESS"
        elif pid in failed_ids:
            status = "FAILED"
        elif pid in stale_ids:
            status = "STALE"
        else:
            status = "MISSING"
        ts = agg_ts.get(pid) or err_ts.get(pid)
        rows.append({
            "point_id": pid,
            "status": status,
            "last_modified_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ") if ts else "",
            "has_aggregated": pid in succeeded_ids,
            "has_daily_parsed": pid in has_parsed,
            "has_daily_kept": pid in has_kept,
            "has_error": pid in failed_ids,
        })

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

    s3_key = f"{args.prefix}/audit/{args.out}"
    storage.put_text(s3_key, buf.getvalue(), content_type="text/csv")
    print(f"\n  Full audit uploaded to: s3://{os.environ.get('SCALEWAY_S3_BUCKET', '?')}/{s3_key}")
    if succeeded_ids:
        print(f"\n  SUCCESS point_ids (usable for training):")
        for pid in sorted(succeeded_ids)[:30]:
            ts = agg_ts.get(pid)
            ts_str = f"  [{ts.strftime('%Y-%m-%d %H:%M UTC')}]" if ts else ""
            print(f"    {pid}{ts_str}")
        if len(succeeded_ids) > 30:
            print(f"    ... and {len(succeeded_ids) - 30} more (see {args.out})")


if __name__ == "__main__":
    main()
