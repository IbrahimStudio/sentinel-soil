.PHONY: build build-v2 local local-v2 run-ingestion run-ingestion-detached \
        run-feature-store run-training \
        run-ingestion-v2 run-ingestion-v2-detached run-feature-store-v2 run-training-v2 \
        minio clean clean-all logs logs-ingestion status help

COMPOSE        = docker compose
COMPOSE_LOCAL  = docker compose --profile local
COMPOSE_V2     = docker compose --profile v2

# ── S3 destination (shared by all pipelines) ─────────────────────────────────
# S3_BUCKET: overrides SCALEWAY_S3_BUCKET from .env (leave empty to use .env value)
# STORAGE_PREFIX: top-level S3 key namespace for stats-api results.
#   Defaults to "batch_results/{ts}" — the {ts} placeholder is substituted
#   at runtime by main.py with the pipeline start timestamp (e.g. 20260416T103045).
#   For make targets that chain ingestion → feature_store in one invocation (e.g.
#   `make local`), we pre-expand the timestamp here so both jobs share the same
#   prefix. Override on the command line: make local STORAGE_PREFIX=batch_results/myrun
S3_BUCKET         ?=
RUN_TS            := $(shell date -u +%Y%m%dT%H%M%S)
STORAGE_PREFIX    ?= batch_results/$(RUN_TS)
# Feature store reads from the aggregated sub-prefix — kept in sync automatically
FEATURE_STORE_PREFIX = $(STORAGE_PREFIX)/aggregated/
# Expand to an env assignment only when S3_BUCKET is non-empty
_BUCKET_ENV = $(if $(S3_BUCKET),SCALEWAY_S3_BUCKET=$(S3_BUCKET))

# ── Statistics API parameters (override on the command line) ─────────────────
# Date range: use START_DATE/END_DATE for a fixed window (e.g. full S2 archive),
# or TIME_WINDOW (days around each SURVEY_DATE) for a relative window.
# Exactly one of the two modes must be set; both default to empty so the pipeline
# raises a clear error rather than silently using a stale default.
XLSX               ?= ./gabri_filters.xlsx
EVALSCRIPT         ?= sh_statistics/evalscripts/only_scl.js
TIME_WINDOW        ?=
START_DATE         ?=
END_DATE           ?=
WORKERS            ?= 8
NDVI_THRESHOLD     ?= 0.2
COVERAGE_THRESHOLD ?= 0.8
LIMIT              ?= -1

INGESTION_ENV = \
  $(_BUCKET_ENV) \
  XLSX=$(XLSX) \
  EVALSCRIPT=$(EVALSCRIPT) \
  TIME_WINDOW=$(TIME_WINDOW) \
  START_DATE=$(START_DATE) \
  END_DATE=$(END_DATE) \
  STORAGE_PREFIX=$(STORAGE_PREFIX) \
  FEATURE_STORE_PREFIX=$(FEATURE_STORE_PREFIX) \
  WORKERS=$(WORKERS) \
  NDVI_THRESHOLD=$(NDVI_THRESHOLD) \
  COVERAGE_THRESHOLD=$(COVERAGE_THRESHOLD) \
  LIMIT=$(LIMIT)

# ── Process API parameters (override on the command line) ────────────────────
TIME_WINDOW_DAYS ?= 365
RASTER_PREFIX    ?= raw_rasters/

INGESTION_V2_ENV = \
  $(_BUCKET_ENV) \
  WORKERS=$(WORKERS) \
  TIME_WINDOW_DAYS=$(TIME_WINDOW_DAYS) \
  RASTER_PREFIX=$(RASTER_PREFIX) \
  LIMIT=$(LIMIT)

## build: build Statistics API images (ingestion:stats-api, feature-store:latest, training:stats-api)
build:
	$(COMPOSE) build ingestion feature_store training

## build-v2: build Process API images (ingestion:process-api, training:process-api)
##           feature-store:latest is shared — rebuild it too if changed
build-v2:
	$(COMPOSE) build feature_store
	$(COMPOSE_V2) build ingestion_process_api training_v2

## local: run the Statistics API pipeline locally using MinIO as S3
local: build
	@echo "Starting MinIO..."
	$(COMPOSE_LOCAL) up -d minio
	@echo "Waiting for MinIO and creating bucket..."
	$(COMPOSE_LOCAL) run --rm minio_init
	@echo "Running Job 1 – ingestion (stats-api)..."
	$(INGESTION_ENV) $(COMPOSE) run --rm ingestion
	@echo "Running Job 2 – feature store..."
	@mkdir -p feature_store/output && chmod 777 feature_store/output
	$(_BUCKET_ENV) FEATURE_STORE_PREFIX=$(FEATURE_STORE_PREFIX) $(COMPOSE) run --rm feature_store
	@echo "Running Job 3 – training (stats-api)..."
	@mkdir -p training/reports && chmod 777 training/reports
	$(_BUCKET_ENV) $(COMPOSE) run --rm training
	@echo "Done. Check training/reports/ for results."

## local-v2: run the Process API pipeline locally using MinIO as S3
local-v2: build-v2
	@echo "Starting MinIO..."
	$(COMPOSE_LOCAL) up -d minio
	@echo "Waiting for MinIO and creating bucket..."
	$(COMPOSE_LOCAL) run --rm minio_init
	@echo "Running Job 1 – ingestion (process-api)..."
	$(INGESTION_V2_ENV) $(COMPOSE_V2) run --rm ingestion_process_api
	@echo "Running Job 1b – coverage report..."
	$(INGESTION_V2_ENV) $(COMPOSE_V2) run --rm coverage_report
	@echo "Running Job 2 – feature store (process_api mode)..."
	@mkdir -p feature_store/output && chmod 777 feature_store/output
	$(INGESTION_V2_ENV) $(COMPOSE_V2) run --rm feature_store_v2
	@echo "Running Job 3 – training (process-api, all 4 targets)..."
	@mkdir -p training/reports && chmod 777 training/reports
	$(_BUCKET_ENV) $(COMPOSE_V2) run --rm training_v2
	@echo "Done. Models → ../DSM_WEBAPP/models/  Reports → training/reports/"

## minio: start MinIO only (useful to inspect bucket mid-run)
minio:
	$(COMPOSE_LOCAL) up -d minio minio_init

## run-ingestion: run only Job 1 (stats-api)
run-ingestion:
	$(INGESTION_ENV) $(COMPOSE) run --rm ingestion

## run-ingestion-detached: run Job 1 (stats-api) detached — survives terminal/SSH disconnect
##   Follow logs:  make logs-ingestion
##   Check status: make status
run-ingestion-detached:
	$(INGESTION_ENV) $(COMPOSE) up -d --no-deps ingestion
	@echo "Ingestion running in background. Follow with: make logs-ingestion"

## run-feature-store: run only Job 2 (statistics mode)
run-feature-store:
	$(_BUCKET_ENV) FEATURE_STORE_PREFIX=$(FEATURE_STORE_PREFIX) $(COMPOSE) run --rm feature_store

## run-training: run only Job 3 (stats-api)
run-training:
	$(_BUCKET_ENV) $(COMPOSE) run --rm training

## run-ingestion-v2: run only Job 1 (process-api)
run-ingestion-v2:
	$(INGESTION_V2_ENV) $(COMPOSE_V2) run --rm ingestion_process_api

## run-ingestion-v2-detached: run Job 1 (process-api) detached — survives terminal/SSH disconnect
##   Follow logs:  make logs-ingestion
##   Check status: make status
run-ingestion-v2-detached:
	$(INGESTION_V2_ENV) $(COMPOSE_V2) up -d --no-deps ingestion_process_api
	@echo "Ingestion (process-api) running in background. Follow with: make logs-ingestion"

## run-feature-store-v2: run only Job 2 (process_api mode)
run-feature-store-v2:
	$(INGESTION_V2_ENV) $(COMPOSE_V2) run --rm feature_store_v2

## run-training-v2: run only Job 3 (process-api, all 4 targets)
run-training-v2:
	$(_BUCKET_ENV) $(COMPOSE_V2) run --rm training_v2

## logs: tail logs for all running services
logs:
	$(COMPOSE) logs -f

## logs-ingestion: tail logs for the ingestion container only
logs-ingestion:
	$(COMPOSE) logs -f ingestion

## status: show running/exited state of all containers
status:
	$(COMPOSE) ps -a

## clean: remove generated outputs (keeps MinIO data intact)
clean:
	rm -rf feature_store/output/*
	find training/reports -mindepth 1 -not -name '.gitkeep' -delete

## clean-all: remove outputs AND MinIO local storage (full reset)
clean-all: clean
	rm -rf local_storage/

## help: show this help
help:
	@grep -E '^## ' Makefile | sed 's/## /  /'
