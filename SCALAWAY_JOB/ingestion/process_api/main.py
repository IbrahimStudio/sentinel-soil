"""
Job 1v2 – Process API Raster Ingestion entry point.

For each LUCAS point:
  1. Check S3 for existing raw raster — skip if present
  2. Fetch ALL orbit passes in one multi-temporal Process API request
  3. Apply seasonal month filter client-side
  4. Store .npz + _meta.json to S3

Environment variables (forwarded by Airflow/docker-compose):
    SH_CLIENT_ID, SH_CLIENT_SECRET
    SCALEWAY_S3_ENDPOINT, SCALEWAY_S3_BUCKET, SCALEWAY_ACCESS_KEY, SCALEWAY_SECRET_KEY
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
    p = argparse.ArgumentParser(description="Process API raster ingestion (multi-temporal)")
    p.add_argument("--xlsx",             default="/data/input.xlsx",         help="Path to gabri_filters.xlsx")
    p.add_argument("--evalscript",       default="/data/evalscript.js",      help="Per-date evalscript (legacy fallback)")
    p.add_argument("--evalscript-mt",    default="/data/evalscript_mt.js",   help="Multi-temporal evalscript (evalscript_multitemporal.js)")
    p.add_argument("--time-window-days", type=int, default=365, help="Half-width window around survey date (days). Ignored when --start-date/--end-date are set.")
    p.add_argument("--start-date",       default=None, help="Fixed start date for all points (YYYY-MM-DD).")
    p.add_argument("--end-date",         default=None, help="Fixed end date for all points (YYYY-MM-DD).")
    p.add_argument("--workers",          type=int, default=4, help="Parallel fetch threads")
    p.add_argument("--raster-prefix",    default="raw_rasters/", help="S3 prefix for raw rasters")
    p.add_argument("--limit",            type=int, default=-1, help="Process only first N rows (-1 = all)")
    p.add_argument("--season-months",    type=str, default=None, metavar="M,M,...",
                   help="Comma-separated months to keep after fetch (1=Jan … 12=Dec). "
                        "Example for bare-soil window: --season-months 10,11,12,1,2,3,4")
    return p.parse_args()


def _parse_season_months(raw: str | None) -> list[int] | None:
    if not raw or not raw.strip():
        return None
    try:
        months = [int(m.strip()) for m in raw.split(",") if m.strip()]
        invalid = [m for m in months if not 1 <= m <= 12]
        if invalid:
            raise ValueError(f"Month values out of range 1–12: {invalid}")
        return months
    except ValueError as e:
        raise SystemExit(f"Invalid --season-months value '{raw}': {e}") from e


def main() -> None:
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    args = parse_args()

    if not args.start_date:
        args.start_date = None
    if not args.end_date:
        args.end_date = None

    fixed_dates = bool(args.start_date and args.end_date)
    if not fixed_dates and not (90 <= args.time_window_days <= 3650):
        sys.exit("--time-window-days must be in [90, 3650]")

    # Load evalscripts
    evalscript    = open(args.evalscript).read()
    evalscript_mt = open(args.evalscript_mt).read()

    # Load LUCAS points
    lucas_df = pd.read_excel(args.xlsx)
    if args.limit > 0:
        lucas_df = lucas_df.head(args.limit)
        log.info("Limited to %d rows (--limit %d)", len(lucas_df), args.limit)

    missing = {"POINT_ID", "TH_LAT", "TH_LONG", "SURVEY_DATE"} - set(lucas_df.columns)
    if missing:
        sys.exit(f"Missing required columns in xlsx: {missing}")

    log.info("Loaded %d LUCAS points.", len(lucas_df))

    # Build API clients
    _, process = clients_from_env(evalscript, evalscript_mt)

    # Fetch rasters
    fetch_all_lucas_rasters(
        lucas_df,
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
        season_months=_parse_season_months(args.season_months),
    )

    log.info("Ingestion complete.")


if __name__ == "__main__":
    main()
