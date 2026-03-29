# Sentinel Soil — Airflow Orchestration

This directory contains the Airflow-based orchestration layer for the pipeline defined in `SCALAWAY_JOB/`. The job code (ingestion, feature_store, training) is unchanged — Airflow is purely a wrapper that runs the existing Docker images in sequence, with a web UI for triggering runs and inspecting results.

Two DAGs are available:

| DAG | Route | Description |
|-----|-------|-------------|
| `soil_pipeline` | Statistics API | Original pipeline — aggregated p50 features, one target at a time |
| `soil_pipeline_v2` | Selectable via `PIPELINE_TYPE` param | Unified DAG: branches to either Statistics API or Process API chain at runtime |

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Parallelism — current approach and future work](#2-parallelism--current-approach-and-future-work)
3. [First-time setup](#3-first-time-setup)
4. [Running the pipeline](#4-running-the-pipeline)
5. [Known Limitations](#5-known-limitations)
6. [Airflow concepts quick reference](#6-airflow-concepts-quick-reference)

---

## 1. Architecture

```
AIRFLOW_JOB/
  Dockerfile              # extends apache/airflow:2.9.3, adds DockerOperator provider
  docker-compose.yaml     # postgres + airflow-webserver + airflow-scheduler + minio (local)
  Makefile                # convenience commands: init, start, stop, build-jobs, build-jobs-v2
  .env                    # credentials and host paths (not committed)
  dags/
    soil_pipeline.py      # original DAG: 3 sequential tasks, Statistics API only
    soil_pipeline_v2.py   # v2 DAG: BranchPythonOperator → stats_api or process_api chain
  logs/                   # written by Airflow at runtime (gitignored)
  plugins/                # empty, required by Airflow conventions
```

**Services started by docker-compose:**

| Service | Role |
|---------|------|
| `postgres` | Airflow metadata database (stores DAG state, run history, task logs) |
| `airflow-webserver` | Web UI at http://localhost:8080 |
| `airflow-scheduler` | Watches DAGs, triggers tasks when a run is started |
| `airflow-init` | One-time DB migration + admin user creation (run via `make init`) |
| `minio` | Local S3 emulator — only started with `make start-local` |

**How tasks execute:**

Each task in the DAG is a `DockerOperator`. When the scheduler triggers a task, it calls the host Docker daemon (via the mounted `/var/run/docker.sock` socket) and spins up the corresponding job image as a sibling container. The container runs `main.py` (or `main_v2.py`) with the parameters you set in the UI, writes results to S3, then is removed on success. On failure it is kept so you can inspect its logs with `docker logs`.

All three job containers join the `soil-pipeline-net` Docker network, which is also where MinIO runs, so they can reach it at `http://minio:9000`.

---

## 2. Parallelism — current approach and future work

### Current approach: intra-container parallelism

The ingestion job handles parallelism internally via Python's `ProcessPoolExecutor` (Statistics API) or `ThreadPoolExecutor` (Process API). When you set `WORKERS=8`, the single ingestion container fans out 8 worker processes/threads that call Sentinel Hub concurrently.

```
Airflow DAG run
  └── Task: ingestion  (1 Docker container)
        ├── worker 1 → point_id A
        ├── worker 2 → point_id B
        ├── ...
        └── worker 8 → point_id H
```

This is appropriate here because the bottleneck is network I/O and API rate limits, not CPU.

### Future improvement: dynamic task mapping

A more Airflow-native approach would use **dynamic task mapping** to spawn one DockerOperator task per chunk of points at runtime. Airflow generates the task graph dynamically and runs all chunks in parallel.

```
Airflow DAG run
  ├── Task: ingestion_chunk_0   (points 0–49)
  ├── Task: ingestion_chunk_1   (points 50–99)
  └── ...
       └── (all join) → feature_store → training
```

**When to implement it:** if the dataset grows significantly, if per-point retry becomes important, or if the pipeline moves to a multi-node Airflow deployment.

---

## 3. First-time setup

### Prerequisites

- Docker and Docker Compose installed
- Job images built (choose one or both):

```bash
# Statistics API images (ingestion:stats-api, feature-store:latest, training:stats-api)
make build-jobs

# Process API images (ingestion:process-api, training:process-api)
# Also rebuilds the shared feature-store:latest
make build-jobs-v2
```

### Create your `.env` file

```bash
cp .env.example .env
```

| Variable | What to set |
|----------|-------------|
| `HOST_REPO_PATH` | Absolute path to this repo on your machine (e.g. `/Users/you/Projects/sentinel-soil`) |
| `AIRFLOW_FERNET_KEY` | Generate with: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `AIRFLOW_SECRET_KEY` | Any random string |
| `SH_CLIENT_ID` / `SH_CLIENT_SECRET` | Sentinel Hub credentials |
| `SCALEWAY_S3_*` | For local runs: leave as MinIO defaults. For Scaleway: your actual credentials |

`HOST_REPO_PATH` is required because `DockerOperator` talks to the **host** Docker daemon. Volume mounts must reference host-absolute paths, not paths inside the Airflow container.

### Initialise the database

```bash
make init
```

Builds the custom Airflow image and runs DB migrations + creates the `admin` user. Only needed once (or after `make clean`).

---

## 4. Running the pipeline

### Start Airflow

```bash
make start-local   # Airflow + MinIO (local runs)
make start         # Airflow only (Scaleway S3)
```

### `soil_pipeline` DAG — Statistics API

1. Open http://localhost:8080 and log in (`admin` / `admin`)
2. Find `soil_pipeline` and click **Trigger DAG w/ config**
3. Set parameters in the form:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EVALSCRIPT` | `sh_statistics/evalscripts/only_scl.js` | Evalscript path (relative to ingestion container) |
| `TIME_WINDOW` | `60` | Days around survey date to query (±30) |
| `WORKERS` | `8` | Parallel API workers |
| `NDVI_THRESHOLD` | `0.2` | Max NDVI for bare-soil classification |
| `COVERAGE_THRESHOLD` | `0.8` | Min valid-pixel fraction to keep an acquisition |
| `LIMIT` | `-1` | First N rows only (−1 = all) |
| `TARGET` | `Clay` | Soil texture target (`Clay`, `Silt`, `Sand`, `Coarse`) |
| `PIPELINE_VERSION` | `v2` | ML pipeline (`v1` = basic, `v2` = SHAP + collinearity handling) |

4. Click **Trigger** — three tasks run in sequence: `ingestion → feature_store → training`.

> **Note — input file:** The ingestion and feature_store tasks always use `SCALAWAY_JOB/gabri_filters.xlsx` as input. See Known Limitations below.

### `soil_pipeline_v2` DAG — Statistics API or Process API

1. Find `soil_pipeline_v2` and click **Trigger DAG w/ config**
2. Set `PIPELINE_TYPE` to choose the route:

| `PIPELINE_TYPE` | Route taken |
|-----------------|-------------|
| `stats_api` | `ingestion_stats_api → feature_store_stats_api → training_stats_api` |
| `process_api` | `ingestion_process_api → feature_store_process_api → training_process_api` |

**Statistics API parameters** (same as `soil_pipeline`): `EVALSCRIPT`, `TIME_WINDOW`, `NDVI_THRESHOLD`, `COVERAGE_THRESHOLD`, `TARGET`, `PIPELINE_VERSION`

**Process API parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TIME_WINDOW_DAYS` | `365` | Total date range around each survey date (90–3650) |
| `RASTER_PREFIX` | `raw_rasters/` | S3 prefix for raw `.npz` raster cache |

**Shared parameters:** `WORKERS`, `LIMIT`

When `PIPELINE_TYPE=process_api`, the training task trains all four targets (Clay, Silt, Sand, Coarse) in a single run and writes the `.pkl` models directly to `DSM_WEBAPP/models/`.

### Other useful commands

```bash
make stop       # stop all services (data preserved)
make restart    # stop then start
make logs       # tail logs from all services
make clean      # full reset — removes DB and MinIO data
make help       # list all targets
```

---

## 5. Known Limitations

### Dynamic input file selection not supported

The `DockerOperator` templates the `command` field but **does not template the `mounts` field**. The `Mount` source path cannot be set dynamically from a DAG param.

**Current behaviour:** `gabri_filters.xlsx` is hardcoded as the bind-mount source. To use a different file, replace it on disk before triggering.

**Planned fix (two options):**

1. **Mount the directory** — mount all of `SCALAWAY_JOB/` at `/data/` and pass `--xlsx /data/{{ params.XLSX }}` in the command (which *is* templated).
2. **Custom operator subclass** — subclass `DockerOperator`, add `mounts` to `template_fields`, build the `Mount` object inside `execute()` after rendering.

---

## 6. Airflow concepts quick reference

| Concept | What it is |
|---------|-----------|
| **DAG** | A Python file describing the pipeline — which tasks exist, in what order, with what parameters |
| **Task** | One unit of work. Here: one Docker container run |
| **Operator** | The class that defines *how* a task executes. We use `DockerOperator` and `BranchPythonOperator` |
| **DAG Run** | One execution of the DAG — created each time you click Trigger |
| **Param** | A typed input shown in the Trigger form, validated before the run starts |
| **Jinja template** | `{{ params.X }}` inside a task definition — replaced at runtime with the form value |
| **BranchPythonOperator** | Reads a param at runtime and returns the `task_id` of the next task to execute, skipping the other branch |
| **LocalExecutor** | Tasks run as subprocesses on the same machine as the scheduler. Right choice for a single VM |
| **DockerOperator** | Spawns a Docker container, streams its logs into Airflow's task log, removes it on success |
