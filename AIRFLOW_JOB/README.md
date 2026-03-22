# Sentinel Soil — Airflow Orchestration

This directory contains the Airflow-based orchestration layer for the same three-job pipeline defined in `SCALAWAY_JOB/`. The job code (ingestion, feature_store, training) is unchanged — Airflow is purely a wrapper that runs the existing Docker images in sequence, with a web UI for triggering runs and inspecting results.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Parallelism — current approach and future work](#2-parallelism--current-approach-and-future-work)
3. [First-time setup](#3-first-time-setup)
4. [Running the pipeline](#4-running-the-pipeline)
5. [Airflow concepts quick reference](#5-airflow-concepts-quick-reference)

---

## 1. Architecture

```
AIRFLOW_JOB/
  Dockerfile              # extends apache/airflow:2.9.3, adds DockerOperator provider
  docker-compose.yaml     # postgres + airflow-webserver + airflow-scheduler + minio (local)
  Makefile                # convenience commands: init, start, stop, logs
  .env                    # credentials and host paths (not committed)
  dags/
    soil_pipeline.py      # the DAG: 3 tasks, manual trigger, UI params
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

Each task in the DAG is a `DockerOperator`. When the scheduler triggers a task, it calls the host Docker daemon (via the mounted `/var/run/docker.sock` socket) and spins up the corresponding job image as a sibling container. The container runs `main.py` with the parameters you set in the UI, writes results to S3, then is removed on success. On failure it is kept so you can inspect its logs with `docker logs`.

All three job containers join the `soil-pipeline-net` Docker network, which is also where MinIO runs, so they can reach it at `http://minio:9000`.

---

## 2. Parallelism — current approach and future work

### Current approach: intra-container parallelism

The ingestion job handles parallelism internally via Python's `ProcessPoolExecutor`. When you set `WORKERS=8` in the trigger form, the single ingestion container fans out 8 worker processes that call the Sentinel Hub Statistics API concurrently.

```
Airflow DAG run
  └── Task: ingestion  (1 Docker container)
        ├── worker process 1 → point_id A
        ├── worker process 2 → point_id B
        ├── ...
        └── worker process 8 → point_id H
```

This is appropriate here because the bottleneck is network I/O and API rate limits, not CPU. Running 8 processes saturates what the Sentinel Hub API will allow without additional effort.

### Future improvement: dynamic task mapping

A more Airflow-native approach would use **dynamic task mapping** to spawn one DockerOperator task per point (or per chunk of points) at runtime. Airflow generates the task graph dynamically based on the contents of the Excel file and runs all tasks in parallel up to a configurable concurrency limit.

```
Airflow DAG run
  ├── Task: ingestion_chunk_0   (Docker container, points 0–49)
  ├── Task: ingestion_chunk_1   (Docker container, points 50–99)
  ├── Task: ingestion_chunk_2   (Docker container, points 100–149)
  └── ...
       └── (all join) → feature_store → training
```

**Advantages over current approach:**
- Failed chunks can be retried individually without rerunning the full ingestion
- Per-chunk progress and logs are visible as separate tasks in the UI
- Work can be spread across multiple machines (with CeleryExecutor or KubernetesExecutor)

**Why we deferred it:**
- Requires reading the Excel file at DAG parse time or using a sensor/pre-task to generate the chunk list
- Higher container startup overhead (one container per chunk vs one container total)
- The current approach already achieves good throughput for the dataset size

**When to implement it:** if the dataset grows significantly, if per-point retry becomes important, or if the pipeline is moved to a multi-node Airflow deployment.

---

## 3. First-time setup

### Prerequisites

- Docker and Docker Compose installed
- The three job images built (from `SCALAWAY_JOB/`):
  ```bash
  make build-jobs
  ```

### Create your `.env` file

```bash
cp .env.example .env
```

Then fill in:

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

This builds the custom Airflow image and runs DB migrations + creates the `admin` user. Only needed once (or after `make clean`).

---

## 4. Running the pipeline

### Start Airflow

```bash
make start-local   # Airflow + MinIO (for local runs)
make start         # Airflow only (when using Scaleway S3)
```

### Trigger a run

1. Open http://localhost:8080 and log in (`admin` / `admin`)
2. Find the `soil_pipeline` DAG and click **Trigger DAG w/ config**
3. A form appears with all parameters pre-filled with defaults — edit what you need:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `XLSX` | `gabri_filters.xlsx` | Input Excel filename inside `SCALAWAY_JOB/` |
| `EVALSCRIPT` | `sh_statistics/evalscripts/only_scl.js` | Evalscript path (relative to ingestion container) |
| `TIME_WINDOW` | `60` | Days around survey date to query (±30) |
| `WORKERS` | `8` | Parallel API workers inside the ingestion container |
| `NDVI_THRESHOLD` | `0.2` | Max NDVI for bare-soil classification |
| `COVERAGE_THRESHOLD` | `0.8` | Min valid-pixel fraction to keep an acquisition |
| `LIMIT` | `-1` | First N rows only (-1 = all). Set to e.g. `10` for a quick test. |
| `TARGET` | `Clay` | Soil texture target (`Clay`, `Silt`, `Sand`, `Coarse`) |
| `PIPELINE_VERSION` | `v2` | ML pipeline (`v1` = basic, `v2` = SHAP + collinearity handling) |

4. Click **Trigger** — the three tasks run in sequence and you can watch progress in the Graph view.

### Other useful commands

```bash
make stop       # stop all services (data preserved)
make restart    # stop then start
make logs       # tail logs from all services
make clean      # full reset — removes DB and MinIO data
make help       # list all targets
```

---

## 5. Airflow concepts quick reference

| Concept | What it is |
|---------|-----------|
| **DAG** | A Python file that describes the pipeline — which tasks exist, in what order, and with what parameters. Airflow reads it at startup and on every file change. |
| **Task** | One unit of work within a DAG. Here: one Docker container run. |
| **Operator** | The class that defines *how* a task executes. We use `DockerOperator`. |
| **DAG Run** | One execution of the DAG — created each time you click Trigger. |
| **Param** | A typed input shown in the Trigger form. Validated before the run starts. |
| **Jinja template** | `{{ params.X }}` inside a task definition — replaced at runtime with the value from the form. |
| **LocalExecutor** | Tasks run as subprocesses on the same machine as the scheduler. No extra workers needed. Right choice for a single VM. |
| **DockerOperator** | Spawns a Docker container, runs it, streams its logs into Airflow's task log, then removes it on success. |
