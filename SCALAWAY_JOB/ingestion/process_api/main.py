"""
Job 1v2 – Process API Raster Ingestion entry point.

For each LUCAS point:
  1. Check S3 for existing raw raster — skip if present
  2. Query Catalog API for acquisitions within the time window
  3. Fetch 9×9 raster per date via Process API
  4. Store .npz + _meta.json to S3

Environment variables (forwarded by Airflow/docker-compose):
    SH_CLIENT_ID, SH_CLIENT_SECRET
    SCALEWAY_S3_ENDPOINT, SCALEWAY_S3_BUCKET, SCALEWAY_ACCESS_KEY, SCALEWAY_SECRET_KEY

Runtime mounts:
    /data/input.xlsx          — gabri_filters.xlsx (read-only)
    /data/filter_config.json  — filter_config.json (read-only)
    /data/evalscript.js       — evalscript_process_api.js (read-only)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from raster_fetcher import fetch_all_lucas_rasters
from sh_clients import clients_from_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process API raster ingestion")
    p.add_argument("--xlsx",            default="/data/input.xlsx",         help="Path to gabri_filters.xlsx")
    p.add_argument("--filter-config",   default="/data/filter_config.json", help="Path to filter_config.json")
    p.add_argument("--evalscript",      default="/data/evalscript.js",      help="Path to evalscript_process_api.js")
    p.add_argument("--time-window-days",type=int,  default=365,  help="Half-width temporal window around survey date (days). Ignored when --start-date/--end-date are set.")
    p.add_argument("--start-date",      default=None, help="Fixed start date for all points (YYYY-MM-DD). Overrides --time-window-days.")
    p.add_argument("--end-date",        default=None, help="Fixed end date for all points (YYYY-MM-DD). Overrides --time-window-days.")
    p.add_argument("--workers",         type=int,  default=4,    help="Parallel fetch threads (I/O bound)")
    p.add_argument("--raster-prefix",   default="raw_rasters/",             help="S3 prefix for storing raw rasters")
    p.add_argument("--limit",           type=int,  default=-1,   help="Process only first N rows (-1 = all). Use small values for testing.")
    return p.parse_args()


def main() -> None:
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    args = parse_args()

    # docker-compose renders unset env vars as empty strings — normalise to None
    if not args.start_date:
        args.start_date = None
    if not args.end_date:
        args.end_date = None

    fixed_dates = bool(args.start_date and args.end_date)
    if not fixed_dates and not (90 <= args.time_window_days <= 3650):
        sys.exit("--time-window-days must be in [90, 3650]")

    # --- Load filter config ---
    import json
    with open(args.filter_config) as f:
        filter_config = json.load(f)
    log.info("filter_config version=%s", filter_config.get("version"))

    # --- Load evalscript ---
    evalscript = open(args.evalscript).read()

    # --- Load LUCAS points ---
    lucas_df = pd.read_excel(args.xlsx)
    if args.limit > 0:
        lucas_df = lucas_df.head(args.limit)
        log.info("Limited to %d rows (--limit %d)", len(lucas_df), args.limit)

    missing = {"POINT_ID", "TH_LAT", "TH_LONG", "SURVEY_DATE"} - set(lucas_df.columns)
    if missing:
        sys.exit(f"Missing required columns in xlsx: {missing}")

    log.info("Loaded %d LUCAS points.", len(lucas_df))

    # --- Build API clients ---
    catalog, process = clients_from_env(evalscript)

    # --- Fetch rasters ---
    fetch_all_lucas_rasters(
        lucas_df,
        filter_config,
        catalog,
        process,
        s3_endpoint=   os.environ["SCALEWAY_S3_ENDPOINT"],
        s3_bucket=     os.environ["SCALEWAY_S3_BUCKET"],
        s3_access_key= os.environ["SCALEWAY_ACCESS_KEY"],
        s3_secret_key= os.environ["SCALEWAY_SECRET_KEY"],
        raster_prefix= args.raster_prefix,
        time_window_days=args.time_window_days,
        start_date=args.start_date,
        end_date=args.end_date,
        workers=args.workers,
    )

    log.info("Ingestion complete.")


if __name__ == "__main__":
    main()
