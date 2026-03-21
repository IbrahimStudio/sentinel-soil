"""
Job 2 – Feature Store entry point.

Reads aggregated JSON results from Scaleway S3, joins with ground-truth labels
(gabri_filters.xlsx), and writes a consolidated feature table.

Environment variables:
    SCALEWAY_S3_ENDPOINT, SCALEWAY_S3_BUCKET, SCALEWAY_ACCESS_KEY, SCALEWAY_SECRET_KEY
    FEATURE_STORE_PREFIX   S3 prefix for aggregated JSON files (default: soil-sentinel/only_scl/aggregated/)
    LABELS_PATH            Path to gabri_filters.xlsx (default: /data/gabri_filters.xlsx)
    OUTPUT_PATH            Local output path for feature parquet (default: /output/features.parquet)
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from aggregator import join_features_with_gabri_filters, read_bucket_and_aggregate_to_dataframe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate S3 results into feature table")
    p.add_argument("--prefix", default=os.getenv("FEATURE_STORE_PREFIX", "soil-sentinel/only_scl/aggregated/"))
    p.add_argument("--labels", default=os.getenv("LABELS_PATH", "/data/gabri_filters.xlsx"))
    p.add_argument("--output", default=os.getenv("OUTPUT_PATH", "/output/features.parquet"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading aggregated results from prefix: {args.prefix}")
    df = read_bucket_and_aggregate_to_dataframe(bucket_prefix=args.prefix)
    print(f"  → {len(df)} records loaded")

    if os.path.exists(args.labels):
        print(f"Joining with labels from: {args.labels}")
        df = join_features_with_gabri_filters(df, gabri_path=args.labels)
        print(f"  → {len(df)} records after join")
    else:
        print(f"Labels file not found at {args.labels}, skipping join")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Feature table written to: {args.output}")


if __name__ == "__main__":
    main()
