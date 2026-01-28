from __future__ import annotations

import os
from io import BytesIO
from typing import List

import boto3
import pandas as pd
from botocore.config import Config

# Optional: load env vars from vm.env
try:
    from dotenv import load_dotenv
    load_dotenv("vm.env")
except Exception:
    pass


# ------------------ CONFIG ------------------
BUCKET = os.environ["S3_BUCKET"]  # e.g. "soil-sentinel"

# Base prefix where features live
# You said: soil-sentinel/soil-sentinel/features/
FEATURES_PREFIX = "soil-sentinel/features/"

TARGET_FILENAME = "seed_summary.parquet"

# Optional: save merged output locally
OUT_PARQUET = "seed_summary_all.parquet"
OUT_CSV = "seed_summary_all.xlsx"
# -------------------------------------------


def make_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],  # e.g. https://s3.fr-par.scw.cloud
        region_name=os.environ.get("AWS_DEFAULT_REGION", "fr-par"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def list_feature_folders(s3) -> List[str]:
    """
    Returns all 'folder' prefixes directly under FEATURES_PREFIX.
    """
    resp = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=FEATURES_PREFIX,
        Delimiter="/",
        MaxKeys=1000,
    )

    prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
    return prefixes


def read_parquet_from_s3(s3, key: str) -> pd.DataFrame:
    """
    Download a parquet file from S3 into a pandas DataFrame.
    """
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = obj["Body"].read()
    return pd.read_parquet(BytesIO(data))


def main():
    s3 = make_s3_client()

    print(f"Listing feature folders under: {FEATURES_PREFIX}")
    folders = list_feature_folders(s3)

    print(f"Found {len(folders)} folders")

    dfs = []

    for i, folder in enumerate(sorted(folders), start=1):
        parquet_key = folder + TARGET_FILENAME
        print(f"[{i}/{len(folders)}] Reading {parquet_key}")

        try:
            df = read_parquet_from_s3(s3, parquet_key)

            # Optional: keep track of origin folder / seed
            df["features_prefix"] = folder

            dfs.append(df)

        except s3.exceptions.NoSuchKey:
            print(f"  -> WARNING: {TARGET_FILENAME} not found in {folder}")
        except Exception as e:
            print(f"  -> ERROR reading {parquet_key}: {e}")

    if not dfs:
        raise RuntimeError("No seed_summary.parquet files were read.")

    final_df = pd.concat(dfs, ignore_index=True)

    print("\nFinal DataFrame")
    print("Shape:", final_df.shape)
    print("Columns:", list(final_df.columns))
    print("\nHead:")
    print(final_df.head())

    # Optional: save locally
    final_df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved merged parquet to: {OUT_PARQUET}")
    final_df.to_excel(OUT_CSV, index=False)
    print(f"\nSaved merged xlsx to: {OUT_CSV}")

    return final_df


if __name__ == "__main__":
    df = main()
