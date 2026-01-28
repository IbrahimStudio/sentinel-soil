import os
import boto3
from botocore.config import Config
import botocore

try:
    from dotenv import load_dotenv
    load_dotenv("vm.env")
except Exception:
    pass

S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
BUCKET = os.environ.get("S3_BUCKET")
REGION = os.environ.get("AWS_DEFAULT_REGION", "fr-par")

print("S3_ENDPOINT =", S3_ENDPOINT)
print("S3_BUCKET   =", BUCKET)
print("REGION      =", REGION)

if not S3_ENDPOINT or not BUCKET:
    raise RuntimeError("Missing S3_ENDPOINT or S3_BUCKET in env/vm.env")

# Force path-style addressing (important for some S3-compatible providers)
cfg = Config(
    signature_version="s3v4",
    s3={"addressing_style": "path"},
)

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    region_name=REGION,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    config=cfg,
)

def show_err(e: botocore.exceptions.ClientError):
    err = e.response.get("Error", {})
    print("ERROR Code   :", err.get("Code"))
    print("ERROR Msg    :", err.get("Message"))
    print("HTTP Status  :", e.response.get("ResponseMetadata", {}).get("HTTPStatusCode"))

print("\n--- Test 1: list_buckets() ---")
try:
    resp = s3.list_buckets()
    print("OK. Buckets visible:", [b["Name"] for b in resp.get("Buckets", [])])
except botocore.exceptions.ClientError as e:
    print("list_buckets FAILED")
    show_err(e)
    raise

print("\n--- Test 2: head_bucket(BUCKET) ---")
try:
    s3.head_bucket(Bucket=BUCKET)
    print("OK. Bucket exists and is accessible.")
except botocore.exceptions.ClientError as e:
    print("head_bucket FAILED")
    show_err(e)
    raise

print("\n--- Test 3: list_objects_v2(BUCKET, MaxKeys=20) ---")
try:
    resp = s3.list_objects_v2(Bucket=BUCKET, MaxKeys=20)
    objs = resp.get("Contents", [])
    print("OK. Objects returned:", len(objs))
    for o in objs:
        print(" -", o["Key"])
except botocore.exceptions.ClientError as e:
    print("list_objects_v2 FAILED")
    show_err(e)
    raise
