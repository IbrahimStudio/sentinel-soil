from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Absolute path so the SQLite DB always lands next to this file,
# regardless of which directory the training script is run from.
_HERE = Path(__file__).parent


@dataclass
class EvaluationConfig:
    tracking_uri: str
    artifact_root: str
    experiment_name: str
    s3_endpoint_url: str
    s3_bucket: str
    placement_log_prefix: str


def from_env() -> EvaluationConfig:
    default_db = f"sqlite:///{_HERE / 'mlflow.db'}"
    return EvaluationConfig(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", default_db),
        artifact_root=os.environ.get(
            "MLFLOW_ARTIFACT_ROOT",
            "s3://soil-sentinel/evaluation/mlflow-artifacts",
        ),
        experiment_name=os.environ.get(
            "MLFLOW_EXPERIMENT_NAME",
            "sentinel-soil-texture",
        ),
        # MLFLOW_S3_ENDPOINT_URL takes precedence; fall back to the main S3 endpoint
        s3_endpoint_url=os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL",
            os.environ.get("SCALEWAY_S3_ENDPOINT", ""),
        ),
        s3_bucket=os.environ.get("SCALEWAY_S3_BUCKET", "soil-sentinel"),
        placement_log_prefix=os.environ.get(
            "PLACEMENT_LOG_PREFIX",
            "evaluation/placement-logs",
        ),
    )
