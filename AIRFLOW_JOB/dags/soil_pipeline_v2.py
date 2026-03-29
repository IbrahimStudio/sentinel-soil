"""
Soil Pipeline v2 — Unified DAG (Statistics API + Process API)
=============================================================

A single DAG that runs either the Statistics API pipeline (existing) or the
Process API pipeline (v2), selected at trigger time via the PIPELINE_TYPE param.

Route:
    trigger → branch_on_pipeline_type
                 ├─ [stats_api]   ingestion_stats_api   → feature_store_stats_api   → training_stats_api
                 └─ [process_api] ingestion_process_api → feature_store_process_api → training_process_api

Docker images:
    rg.fr-par.scw.cloud/soil-sentinel/ingestion:stats-api     ← existing ingestion
    rg.fr-par.scw.cloud/soil-sentinel/ingestion:process-api   ← new (no sentinelhub)
    rg.fr-par.scw.cloud/soil-sentinel/feature-store:latest    ← shared, --source flag selects mode
    rg.fr-par.scw.cloud/soil-sentinel/training:stats-api      ← existing (one target at a time)
    rg.fr-par.scw.cloud/soil-sentinel/training:process-api    ← new (all 4 targets + clean .pkl export)

Build commands (from SCALAWAY_JOB/):
    docker build -f ingestion/Dockerfile            -t rg.fr-par.scw.cloud/soil-sentinel/ingestion:stats-api    .
    docker build -f ingestion/process_api/Dockerfile -t rg.fr-par.scw.cloud/soil-sentinel/ingestion:process-api .
    docker build -f feature_store/Dockerfile        -t rg.fr-par.scw.cloud/soil-sentinel/feature-store:latest   .
    docker build -f training/Dockerfile             -t rg.fr-par.scw.cloud/soil-sentinel/training:stats-api     .
    docker build -f training/Dockerfile.process-api -t rg.fr-par.scw.cloud/soil-sentinel/training:process-api   .
"""

from __future__ import annotations

import os
from datetime import datetime

from airflow.decorators import dag
from airflow.models.param import Param
from airflow.operators.python import BranchPythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

_REPO        = os.environ.get("HOST_REPO_PATH", "")
_SCALAWAY    = f"{_REPO}/SCALAWAY_JOB"
_DSM_WEBAPP  = f"{_REPO}/DSM_WEBAPP"
_REGISTRY    = "rg.fr-par.scw.cloud/soil-sentinel"


def _s3_env() -> dict:
    keys = [
        "SH_CLIENT_ID", "SH_CLIENT_SECRET",
        "SCALEWAY_S3_ENDPOINT", "SCALEWAY_S3_REGION",
        "SCALEWAY_S3_BUCKET", "SCALEWAY_ACCESS_KEY", "SCALEWAY_SECRET_KEY",
    ]
    return {k: os.environ.get(k, "") for k in keys}


@dag(
    dag_id="soil_pipeline_v2",
    description="Unified pipeline: Statistics API or Process API, selected at trigger",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["soil", "sentinel-2", "v2"],
    params={
        # ── Route selection ────────────────────────────────────────────
        "PIPELINE_TYPE": Param(
            "stats_api",
            type="string",
            enum=["stats_api", "process_api"],
            description="stats_api: Statistics API (existing). process_api: Process API (v2, trains all 4 targets).",
        ),
        # ── Statistics API params (ignored when PIPELINE_TYPE=process_api) ──
        "EVALSCRIPT": Param(
            "sh_statistics/evalscripts/only_scl.js",
            type="string",
            description="Evalscript path inside the ingestion container (stats_api only)",
        ),
        "TIME_WINDOW": Param(
            60, type="integer", minimum=1,
            description="Total days around survey date (stats_api only, e.g. 60 → ±30 days)",
        ),
        "NDVI_THRESHOLD": Param(
            0.2, type="number", minimum=0.0, maximum=1.0,
            description="Max NDVI for bare soil (stats_api only)",
        ),
        "COVERAGE_THRESHOLD": Param(
            0.8, type="number", minimum=0.0, maximum=1.0,
            description="Min valid-pixel fraction (stats_api only)",
        ),
        "TARGET": Param(
            "Clay", type="string", enum=["Clay", "Silt", "Sand", "Coarse"],
            description="Soil texture fraction to predict (stats_api only — process_api trains all 4)",
        ),
        "PIPELINE_VERSION": Param(
            "v2", type="string", enum=["v1", "v2"],
            description="Training pipeline version (stats_api only)",
        ),
        # ── Process API params (ignored when PIPELINE_TYPE=stats_api) ──
        "TIME_WINDOW_DAYS": Param(
            365, type="integer", minimum=90, maximum=1825,
            description="Half-width of temporal window around survey date (process_api only). "
                        "More days = more bare-soil observations. Sentinel-2 archive from ~2017.",
        ),
        "RASTER_PREFIX": Param(
            "raw_rasters/", type="string",
            description="S3 prefix for raw raster storage (process_api only)",
        ),
        # ── Shared ────────────────────────────────────────────────────
        "WORKERS": Param(
            8, type="integer", minimum=1, maximum=32,
            description="Parallel API workers",
        ),
        "LIMIT": Param(
            -1, type="integer", minimum=-1,
            description="Process only first N rows (-1 = all). Small values for testing.",
        ),
    },
)
def soil_pipeline_v2() -> None:

    _common = dict(
        docker_url="unix://var/run/docker.sock",
        network_mode="soil-pipeline-net",
        auto_remove="success",
        mount_tmp_dir=False,
        environment=_s3_env(),
    )

    # ------------------------------------------------------------------
    # Branch
    # ------------------------------------------------------------------
    def _pick_branch(**ctx) -> str:
        pt = ctx["params"]["PIPELINE_TYPE"]
        return f"ingestion_{pt}"

    branch = BranchPythonOperator(
        task_id="branch_on_pipeline_type",
        python_callable=_pick_branch,
    )

    # ==================================================================
    # STATISTICS API branch
    # ==================================================================

    ingestion_stats = DockerOperator(
        task_id="ingestion_stats_api",
        image=f"{_REGISTRY}/ingestion:stats-api",
        command=[
            "python", "main.py",
            "--xlsx",               "/data/input.xlsx",
            "--evalscript_path",    "{{ params.EVALSCRIPT }}",
            "--time_window",        "{{ params.TIME_WINDOW }}",
            "--workers",            "{{ params.WORKERS }}",
            "--ndvi_threshold",     "{{ params.NDVI_THRESHOLD }}",
            "--coverage_threshold", "{{ params.COVERAGE_THRESHOLD }}",
            "--limit",              "{{ params.LIMIT }}",
        ],
        mounts=[
            Mount(source=f"{_SCALAWAY}/gabri_filters.xlsx", target="/data/input.xlsx", type="bind", read_only=True),
        ],
        **_common,
    )

    feature_store_stats = DockerOperator(
        task_id="feature_store_stats_api",
        image=f"{_REGISTRY}/feature-store:latest",
        command=[
            "python", "main.py",
            "--source",  "statistics",
            "--prefix",  "batch_results/aggregated/",
            "--labels",  "/data/input.xlsx",
            "--output",  "/output/features.parquet",
        ],
        mounts=[
            Mount(source=f"{_SCALAWAY}/gabri_filters.xlsx",    target="/data/input.xlsx", type="bind", read_only=True),
            Mount(source=f"{_SCALAWAY}/feature_store/output",  target="/output",          type="bind"),
        ],
        **_common,
    )

    training_stats = DockerOperator(
        task_id="training_stats_api",
        image=f"{_REGISTRY}/training:stats-api",
        command=[
            "python", "main.py",
            "--features",    "/data/features.parquet",
            "--target",      "{{ params.TARGET }}",
            "--pipeline",    "{{ params.PIPELINE_VERSION }}",
            "--reports-dir", "/reports",
        ],
        mounts=[
            Mount(source=f"{_SCALAWAY}/feature_store/output", target="/data",    type="bind", read_only=True),
            Mount(source=f"{_SCALAWAY}/training/reports",     target="/reports", type="bind"),
        ],
        **_common,
    )

    # ==================================================================
    # PROCESS API branch
    # ==================================================================

    ingestion_process = DockerOperator(
        task_id="ingestion_process_api",
        image=f"{_REGISTRY}/ingestion:process-api",
        command=[
            "python", "main.py",
            "--xlsx",              "/data/input.xlsx",
            "--filter-config",     "/data/filter_config.json",
            "--evalscript",        "/data/evalscript.js",
            "--time-window-days",  "{{ params.TIME_WINDOW_DAYS }}",
            "--workers",           "{{ params.WORKERS }}",
            "--raster-prefix",     "{{ params.RASTER_PREFIX }}",
            "--limit",             "{{ params.LIMIT }}",
        ],
        mounts=[
            Mount(source=f"{_SCALAWAY}/gabri_filters.xlsx",            target="/data/input.xlsx",          type="bind", read_only=True),
            Mount(source=f"{_DSM_WEBAPP}/filter_config.json",          target="/data/filter_config.json",  type="bind", read_only=True),
            Mount(source=f"{_DSM_WEBAPP}/evalscript_process_api.js",   target="/data/evalscript.js",       type="bind", read_only=True),
        ],
        **_common,
    )

    coverage_report = DockerOperator(
        task_id="coverage_report_process_api",
        image=f"{_REGISTRY}/feature-store:latest",
        command=[
            "python", "coverage_report.py",
            "--xlsx",          "/data/input.xlsx",
            "--filter-config", "/config/filter_config.json",
            "--raster-prefix", "{{ params.RASTER_PREFIX }}",
            "--output-dir",    "/reports/coverage",
            "--limit",         "{{ params.LIMIT }}",
        ],
        mounts=[
            Mount(source=f"{_SCALAWAY}/gabri_filters.xlsx",         target="/data/input.xlsx",         type="bind", read_only=True),
            Mount(source=f"{_DSM_WEBAPP}/filter_config.json",        target="/config/filter_config.json", type="bind", read_only=True),
            Mount(source=f"{_SCALAWAY}/training/reports",            target="/reports",                 type="bind"),
        ],
        **_common,
    )

    feature_store_process = DockerOperator(
        task_id="feature_store_process_api",
        image=f"{_REGISTRY}/feature-store:latest",
        command=[
            "python", "main.py",
            "--source",         "process_api",
            "--labels",         "/data/input.xlsx",
            "--filter-config",  "/data/filter_config.json",
            "--raster-prefix",  "{{ params.RASTER_PREFIX }}",
            "--output",         "/output/features_v2.parquet",
        ],
        mounts=[
            Mount(source=f"{_SCALAWAY}/gabri_filters.xlsx",          target="/data/input.xlsx",          type="bind", read_only=True),
            Mount(source=f"{_DSM_WEBAPP}/filter_config.json",        target="/data/filter_config.json",  type="bind", read_only=True),
            Mount(source=f"{_SCALAWAY}/feature_store/output",        target="/output",                   type="bind"),
        ],
        **_common,
    )

    training_process = DockerOperator(
        task_id="training_process_api",
        image=f"{_REGISTRY}/training:process-api",
        command=[
            "python", "main_v2.py",
            "--features",        "/data/features_v2.parquet",
            "--model-dir",       "/models",
            "--filter-config",   "/config/filter_config.json",
            "--reports-dir",     "/reports",
            "--model-s3-prefix", "models/process_api/",
        ],
        mounts=[
            Mount(source=f"{_SCALAWAY}/feature_store/output",  target="/data",    type="bind", read_only=True),
            Mount(source=f"{_DSM_WEBAPP}/filter_config.json",  target="/config/filter_config.json", type="bind", read_only=True),
            Mount(source=f"{_DSM_WEBAPP}/models",              target="/models",  type="bind"),
            Mount(source=f"{_SCALAWAY}/training/reports",      target="/reports", type="bind"),
        ],
        **_common,
    )

    # ------------------------------------------------------------------
    # Dependency graph
    # ------------------------------------------------------------------
    branch >> [ingestion_stats,   ingestion_process]

    ingestion_stats   >> feature_store_stats   >> training_stats
    ingestion_process >> coverage_report >> feature_store_process >> training_process


soil_pipeline_v2()
