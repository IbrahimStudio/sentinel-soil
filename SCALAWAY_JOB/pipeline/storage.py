# sentinel_soil/pipeline/storage.py
from __future__ import annotations

import os
import time
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

import logging


@dataclass(frozen=True)
class StorageConfig:
    endpoint_url: str
    region: str
    bucket: str
    access_key: str
    secret_key: str


def _norm_key(key: str) -> str:
    """S3 object keys must not start with '/', and should use '/' separators."""
    k = key.strip().lstrip("/")
    # convert Windows backslashes if any
    return k.replace("\\", "/")


def _guess_content_type(path: Path) -> str:
    # common types we use a lot (be explicit)
    ext = path.suffix.lower()
    if ext == ".parquet":
        return "application/octet-stream"
    if ext in (".tif", ".tiff"):
        return "image/tiff"
    if ext == ".json":
        return "application/json"
    if ext in (".log", ".txt"):
        return "text/plain"
    if ext == ".png":
        return "image/png"

    # fallback to mimetypes
    ct, _ = mimetypes.guess_type(str(path))
    return ct or "application/octet-stream"


class StorageClient:
    def upload_file(self, local_path: Path, key: str, *, content_type: Optional[str] = None) -> None:
        raise NotImplementedError

    def upload_tree(
        self,
        local_dir: Path,
        prefix: str,
        *,
        include_globs: Optional[list[str]] = None,
        exclude_globs: Optional[list[str]] = None,
    ) -> int:
        raise NotImplementedError

    def put_text(self, key: str, text: str, *, content_type: str = "text/plain") -> None:
        raise NotImplementedError


class S3StorageClient(StorageClient):
    def __init__(self, cfg: StorageConfig, *, logger: Optional[logging.Logger] = None):
        self.bucket = cfg.bucket
        self.log = logger or logging.getLogger(__name__)

        # retries + timeouts for job robustness
        boto_cfg = BotoConfig(
            retries={"max_attempts": 8, "mode": "standard"},
            connect_timeout=15,
            read_timeout=120,
        )

        self.s3 = boto3.client(
            "s3",
            endpoint_url=cfg.endpoint_url,
            region_name=cfg.region,
            aws_access_key_id=cfg.access_key,
            aws_secret_access_key=cfg.secret_key,
            config=boto_cfg,
        )

    def upload_file(self, local_path: Path, key: str, *, content_type: Optional[str] = None) -> None:
        local_path = Path(local_path)
        if not local_path.exists() or not local_path.is_file():
            raise FileNotFoundError(f"upload_file: not a file: {local_path}")

        key = _norm_key(key)
        ct = content_type or _guess_content_type(local_path)

        extra_args = {"ContentType": ct}

        self._upload_with_retry(local_path, key, extra_args=extra_args)

    def _upload_with_retry(self, local_path: Path, key: str, *, extra_args: dict) -> None:
        # boto has internal retries; this is a second layer for transient edge cases
        max_tries = 3
        base_sleep = 1.0
        last_err: Optional[Exception] = None

        for attempt in range(1, max_tries + 1):
            try:
                self.s3.upload_file(
                    Filename=str(local_path),
                    Bucket=self.bucket,
                    Key=key,
                    ExtraArgs=extra_args,
                )
                self.log.info(f"Uploaded: s3://{self.bucket}/{key} ({local_path.name})")
                return
            except (BotoCoreError, ClientError) as e:
                last_err = e
                self.log.warning(f"Upload attempt {attempt}/{max_tries} failed for {key}: {e}")
                if attempt < max_tries:
                    time.sleep(base_sleep * attempt)

        raise RuntimeError(f"Failed to upload {local_path} to {key}") from last_err

    def upload_tree(
        self,
        local_dir: Path,
        prefix: str,
        *,
        include_globs: Optional[list[str]] = None,
        exclude_globs: Optional[list[str]] = None,
    ) -> int:
        local_dir = Path(local_dir)
        if not local_dir.exists():
            self.log.warning(f"upload_tree: local_dir does not exist: {local_dir}")
            return 0
        if not local_dir.is_dir():
            raise ValueError(f"upload_tree: not a dir: {local_dir}")

        prefix = _norm_key(prefix)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        # gather candidate files
        if include_globs:
            candidates: set[Path] = set()
            for g in include_globs:
                candidates.update([p for p in local_dir.rglob(g) if p.is_file()])
            files = sorted(candidates)
        else:
            files = sorted([p for p in local_dir.rglob("*") if p.is_file()])

        # apply excludes
        if exclude_globs:
            excluded: set[Path] = set()
            for g in exclude_globs:
                excluded.update([p for p in local_dir.rglob(g) if p.is_file()])
            files = [p for p in files if p not in excluded]

        count = 0
        for p in files:
            rel = p.relative_to(local_dir).as_posix()
            key = f"{prefix}{rel}"
            self.upload_file(p, key)
            count += 1

        self.log.info(f"upload_tree complete: {count} files -> s3://{self.bucket}/{prefix}")
        return count

    def put_text(self, key: str, text: str, *, content_type: str = "text/plain") -> None:
        key = _norm_key(key)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType=content_type,
        )
        self.log.info(f"Put object: s3://{self.bucket}/{key}")

    def list_objects(self, prefix: str) -> list[str]:
        """
        List objects in S3 bucket with given prefix

        Args:
            prefix: S3 prefix to filter objects

        Returns:
            List of object keys
        """
        prefix = _norm_key(prefix)
        self.log.info(f"Listing objects with prefix: s3://{self.bucket}/{prefix}")

        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket,
                Prefix=prefix
            )

            objects = []
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append(obj['Key'])

            self.log.info(f"Found {len(objects)} objects with prefix: {prefix}")
            return objects

        except Exception as e:
            self.log.error(f"Failed to list objects: {e}")
            raise

    def get_text(self, key: str) -> str:
        """
        Get text content of an S3 object

        Args:
            key: S3 object key

        Returns:
            Text content of the object
        """
        key = _norm_key(key)
        self.log.info(f"Getting text content: s3://{self.bucket}/{key}")

        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=key
            )
            return response['Body'].read().decode('utf-8')

        except Exception as e:
            self.log.error(f"Failed to get object {key}: {e}")
            raise


def storage_from_env(*, logger: Optional[logging.Logger] = None) -> S3StorageClient:
    """
    Expected env vars for Scaleway Object Storage (S3-compatible):
      SCALEWAY_S3_ENDPOINT   e.g. https://s3.fr-par.scw.cloud
      SCALEWAY_S3_REGION     e.g. fr-par
      SCALEWAY_S3_BUCKET
      SCALEWAY_ACCESS_KEY
      SCALEWAY_SECRET_KEY
    """
    missing = [k for k in [
        "SCALEWAY_S3_ENDPOINT",
        "SCALEWAY_S3_BUCKET",
        "SCALEWAY_ACCESS_KEY",
        "SCALEWAY_SECRET_KEY",
    ] if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")

    cfg = StorageConfig(
        endpoint_url=os.environ["SCALEWAY_S3_ENDPOINT"],
        region=os.environ.get("SCALEWAY_S3_REGION", "fr-par"),
        bucket=os.environ["SCALEWAY_S3_BUCKET"],
        access_key=os.environ["SCALEWAY_ACCESS_KEY"],
        secret_key=os.environ["SCALEWAY_SECRET_KEY"],
    )
    return S3StorageClient(cfg, logger=logger)
