"""
Job 2 – Feature Store entry point.

Two modes selected via --source:

  statistics  (default) — reads aggregated JSON results from S3, joins with labels.
  process_api           — loads raw .npz rasters from S3, applies filter_config,
                          extracts center-pixel temporal medians + spectral indices.

Environment variables:
    SCALEWAY_S3_ENDPOINT, SCALEWAY_S3_BUCKET, SCALEWAY_ACCESS_KEY, SCALEWAY_SECRET_KEY
    FEATURE_STORE_PREFIX   S3 prefix for aggregated JSON files (default: soil-sentinel/only_scl/aggregated/)
    LABELS_PATH            Path to gabri_filters.xlsx (default: /data/gabri_filters.xlsx)
    OUTPUT_PATH            Local output path for feature parquet (default: /output/features.parquet)
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from aggregator import join_features_with_gabri_filters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate S3 results into feature table")
    p.add_argument("--source", choices=["statistics", "process_api"], default="statistics",
                   help="Data source: 'statistics' (Statistics API JSON) or 'process_api' (raw rasters)")
    p.add_argument("--prefix",        default=os.getenv("FEATURE_STORE_PREFIX", "soil-sentinel/only_scl/aggregated/"))
    p.add_argument("--labels",        default=os.getenv("LABELS_PATH", "/data/gabri_filters.xlsx"))
    p.add_argument("--output",        default=os.getenv("OUTPUT_PATH", "/output/features.parquet"))
    # process_api mode only
    p.add_argument("--filter-config", default="/data/filter_config.json",
                   help="Path to filter_config.json (process_api mode only)")
    p.add_argument("--raster-prefix", default="raw_rasters/",
                   help="S3 prefix for raw .npz rasters (process_api mode only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.source == "process_api":
        from process_api_extractor import extract_features

        print(f"[process_api] Loading rasters from s3://{os.environ.get('SCALEWAY_S3_BUCKET')}/{args.raster_prefix}")
        print(f"[process_api] Filter config: {args.filter_config}")
        print(f"[process_api] Labels: {args.labels}")

        lucas_df = pd.read_excel(args.labels)
        with open(args.filter_config) as f:
            filter_config = json.load(f)

        df = extract_features(
            lucas_df,
            filter_config,
            s3_endpoint=   os.environ["SCALEWAY_S3_ENDPOINT"],
            s3_bucket=     os.environ["SCALEWAY_S3_BUCKET"],
            s3_access_key= os.environ["SCALEWAY_ACCESS_KEY"],
            s3_secret_key= os.environ["SCALEWAY_SECRET_KEY"],
            raster_prefix= args.raster_prefix,
        )
        print(f"  → {len(df)} records extracted")

    else:
        print(f"[statistics] Reading aggregated results from prefix: {args.prefix}, joining with labels from: {args.labels}")
        df = join_features_with_gabri_filters(args.prefix, gabri_filters_path=args.labels, output_path="/tmp/texture_scl_features.xlsx")
        print(f"  → {len(df)} records after join")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Feature table written to: {args.output}")


if __name__ == "__main__":
    main()
