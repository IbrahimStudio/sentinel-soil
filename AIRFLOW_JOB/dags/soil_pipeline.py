"""
Sentinel Soil Pipeline DAG
==========================

Three tasks run in sequence:
    ingestion → feature_store → training

Each task is a DockerOperator — it spins up the corresponding Docker image,
runs main.py with the parameters you set in the UI, then tears the container down.

All parameters are exposed as DAG Params: when you click "Trigger DAG w/ config"
in the Airflow UI you get a form where you can change any of these values before
the run starts.

Concepts for reference:
  - DAG       : the pipeline definition (this file). Airflow reads it on startup.
  - Task      : one unit of work (here: one Docker container run).
  - Operator  : the class that defines *how* a task executes (DockerOperator).
  - Param     : a typed, documented input exposed in the trigger UI.
  - Template  : {{ params.X }} is replaced at runtime with the value you entered.
"""

from __future__ import annotations

import os
from datetime import datetime

from airflow.decorators import dag
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


# ---------------------------------------------------------------------------
# Host-side paths
# DockerOperator talks to the HOST Docker daemon, so volume source paths must
# be absolute paths on the host machine — not paths inside the Airflow container.
# HOST_REPO_PATH is set in .env and forwarded into the Airflow containers.
# ---------------------------------------------------------------------------
_REPO = os.environ.get("HOST_REPO_PATH", "")
_SCALAWAY_JOB = f"{_REPO}/SCALAWAY_JOB"


def _s3_env() -> dict:
    """
    Collect S3 + SH credentials from the Airflow container's environment and
    forward them to every job container.  The values come from .env via the
    docker-compose env_file directive.
    """
    keys = [
        "SH_CLIENT_ID", "SH_CLIENT_SECRET",
        "SCALEWAY_S3_ENDPOINT", "SCALEWAY_S3_REGION",
        "SCALEWAY_S3_BUCKET", "SCALEWAY_ACCESS_KEY", "SCALEWAY_SECRET_KEY",
    ]
    return {k: os.environ.get(k, "") for k in keys}


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
@dag(
    dag_id="soil_pipeline",
    description="Sentinel-2 → bare-soil features → soil texture ML",
    schedule=None,          # manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["soil", "sentinel-2"],
    # ------------------------------------------------------------------
    # Params are the inputs shown in the "Trigger DAG" form in the UI.
    # Each Param has a type (validated before the run starts) and an
    # optional description shown as a tooltip.
    # ------------------------------------------------------------------
    params={
        # ── Ingestion ─────────────────────────────────────────────────
        "EVALSCRIPT": Param(
            "sh_statistics/evalscripts/only_scl.js",
            type="string",
            description="Path to evalscript, relative to the ingestion container working dir",
        ),
        "TIME_WINDOW": Param(
            60,
            type=["integer", "null"],
            description=(
                "Total days around each survey date to query (e.g. 60 → ±30 days). "
                "Set to null/empty to use START_DATE + END_DATE instead."
            ),
        ),
        "START_DATE": Param(
            "",
            type="string",
            description=(
                "Fixed start date for all points (YYYY-MM-DD). "
                "Used when TIME_WINDOW is null (e.g. '2015-07-01' for full S2 archive)."
            ),
        ),
        "END_DATE": Param(
            "",
            type="string",
            description=(
                "Fixed end date for all points (YYYY-MM-DD). "
                "Used together with START_DATE when TIME_WINDOW is null."
            ),
        ),
        "WORKERS": Param(
            8,
            type="integer",
            minimum=1,
            maximum=32,
            description="Number of parallel Sentinel Hub API workers",
        ),
        "NDVI_THRESHOLD": Param(
            0.2,
            type="number",
            minimum=0.0,
            maximum=1.0,
            description="Max NDVI to consider a day as bare soil",
        ),
        "COVERAGE_THRESHOLD": Param(
            0.8,
            type="number",
            minimum=0.0,
            maximum=1.0,
            description="Min valid-pixel fraction to keep an acquisition day",
        ),
        "LIMIT": Param(
            -1,
            type="integer",
            minimum=-1,
            description="Process only first N rows (-1 = all points). Use a small number for testing.",
        ),
        # ── Training ──────────────────────────────────────────────────
        "TARGET": Param(
            "all",
            type="string",
            enum=["all", "Clay", "Silt", "Sand", "Coarse"],
            description="Soil texture fraction(s) to train. 'all' trains Clay, Silt, Sand and Coarse in sequence.",
        ),
        "PIPELINE_VERSION": Param(
            "v2",
            type="string",
            enum=["v1", "v2"],
            description="v1 = RF/GBM/ElasticNet; v2 = adds SHAP + drops collinear fractions",
        ),
    },
)
def soil_pipeline() -> None:

    # ------------------------------------------------------------------
    # Shared DockerOperator kwargs
    #
    # network_mode: attach job containers to the same Docker network as
    #   MinIO so they can reach http://minio:9000 by hostname.
    # auto_remove:  "success" → container is deleted after a successful
    #   run; kept on failure so you can inspect logs with `docker logs`.
    # mount_tmp_dir: Airflow mounts a tmp dir by default — disable it
    #   to keep containers clean.
    # ------------------------------------------------------------------
    _common = dict(
        docker_url="unix://var/run/docker.sock",
        network_mode="soil-pipeline-net",
        auto_remove="success",
        mount_tmp_dir=False,
        environment=_s3_env(),
    )

    # ------------------------------------------------------------------
    # Task 1 — Ingestion
    #
    # Reads the Excel file, calls the Sentinel Hub Statistics API for
    # each LUCAS point, and stores aggregated JSON results to S3.
    #
    # Volume mount: the Excel file on the host is bind-mounted read-only
    # into the container at /data/input.xlsx.
    # ------------------------------------------------------------------
    ingestion = DockerOperator(
        task_id="ingestion",
        image="rg.fr-par.scw.cloud/soil-sentinel/ingestion:latest",
        command=[
            "python", "main.py",
            "--xlsx",               "/data/input.xlsx",
            "--evalscript_path",    "{{ params.EVALSCRIPT }}",
            "--time_window",        "{{ params.TIME_WINDOW }}",
            "--start_date",         "{{ params.START_DATE }}",
            "--end_date",           "{{ params.END_DATE }}",
            "--workers",            "{{ params.WORKERS }}",
            "--ndvi_threshold",     "{{ params.NDVI_THRESHOLD }}",
            "--coverage_threshold", "{{ params.COVERAGE_THRESHOLD }}",
            "--limit",              "{{ params.LIMIT }}",
        ],
        mounts=[
            Mount(
                source=f"{_SCALAWAY_JOB}/gabri_filters.xlsx",
                target="/data/input.xlsx",
                type="bind",
                read_only=True,
            )
        ],
        **_common,
    )

    # ------------------------------------------------------------------
    # Task 2 — Feature Store
    #
    # Reads the aggregated JSONs from S3, joins with ground-truth labels
    # from the Excel file, and writes a consolidated feature parquet.
    # ------------------------------------------------------------------
    feature_store = DockerOperator(
        task_id="feature_store",
        image="rg.fr-par.scw.cloud/soil-sentinel/feature-store:latest",
        command=[
            "python", "main.py",
            "--prefix",  "batch_results/aggregated/",
            "--labels",  "/data/input.xlsx",
            "--output",  "/output/features.parquet",
        ],
        mounts=[
            Mount(
                source=f"{_SCALAWAY_JOB}/gabri_filters.xlsx",
                target="/data/input.xlsx",
                type="bind",
                read_only=True,
            ),
            Mount(
                source=f"{_SCALAWAY_JOB}/feature_store/output",
                target="/output",
                type="bind",
            ),
        ],
        **_common,
    )

    # ------------------------------------------------------------------
    # Task 3 — Training
    #
    # Loads the feature parquet produced by job 2, trains ML models,
    # and writes reports to SCALAWAY_JOB/training/reports/.
    # ------------------------------------------------------------------
    training = DockerOperator(
        task_id="training",
        image="rg.fr-par.scw.cloud/soil-sentinel/training:latest",
        command=[
            "python", "main.py",
            "--features",     "/data/features.parquet",
            "--target",       "{{ params.TARGET }}",
            "--pipeline",     "{{ params.PIPELINE_VERSION }}",
            "--reports-dir",  "/reports",
        ],
        mounts=[
            Mount(
                source=f"{_SCALAWAY_JOB}/feature_store/output",
                target="/data",
                type="bind",
                read_only=True,
            ),
            Mount(
                source=f"{_SCALAWAY_JOB}/training/reports",
                target="/reports",
                type="bind",
            ),
        ],
        **_common,
    )

    # ------------------------------------------------------------------
    # Dependency declaration — this is what makes it a DAG.
    # >> means "must complete successfully before the next task starts".
    # If ingestion fails, feature_store and training are skipped.
    # ------------------------------------------------------------------
    ingestion >> feature_store >> training


# Airflow discovers the DAG by calling the decorated function at import time.
soil_pipeline()
