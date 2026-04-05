.PHONY: build build-v2 local local-v2 run-ingestion run-feature-store run-training \
        run-ingestion-v2 run-feature-store-v2 run-training-v2 minio clean clean-all logs help

COMPOSE        = docker compose
COMPOSE_LOCAL  = docker compose --profile local
COMPOSE_V2     = docker compose --profile v2

# ── Statistics API parameters (override on the command line) ─────────────────
XLSX               ?= ./gabri_filters.xlsx
EVALSCRIPT         ?= sh_statistics/evalscripts/only_scl.js
TIME_WINDOW        ?= 60
WORKERS            ?= 8
NDVI_THRESHOLD     ?= 0.2
COVERAGE_THRESHOLD ?= 0.8
LIMIT              ?= -1

INGESTION_ENV = \
  XLSX=$(XLSX) \
  EVALSCRIPT=$(EVALSCRIPT) \
  TIME_WINDOW=$(TIME_WINDOW) \
  WORKERS=$(WORKERS) \
  NDVI_THRESHOLD=$(NDVI_THRESHOLD) \
  COVERAGE_THRESHOLD=$(COVERAGE_THRESHOLD) \
  LIMIT=$(LIMIT)

# ── Process API parameters (override on the command line) ────────────────────
TIME_WINDOW_DAYS ?= 365
RASTER_PREFIX    ?= raw_rasters/

INGESTION_V2_ENV = \
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
	$(COMPOSE) run --rm feature_store
	@echo "Running Job 3 – training (stats-api)..."
	$(COMPOSE) run --rm training
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
	$(INGESTION_V2_ENV) $(COMPOSE_V2) run --rm feature_store_v2
	@echo "Running Job 3 – training (process-api, all 4 targets)..."
	$(COMPOSE_V2) run --rm training_v2
	@echo "Done. Models → ../DSM_WEBAPP/models/  Reports → training/reports/"

## minio: start MinIO only (useful to inspect bucket mid-run)
minio:
	$(COMPOSE_LOCAL) up -d minio minio_init

## run-ingestion: run only Job 1 (stats-api)
run-ingestion:
	$(INGESTION_ENV) $(COMPOSE) run --rm ingestion

## run-feature-store: run only Job 2 (statistics mode)
run-feature-store:
	$(COMPOSE) run --rm feature_store

## run-training: run only Job 3 (stats-api)
run-training:
	$(COMPOSE) run --rm training

## run-ingestion-v2: run only Job 1 (process-api)
run-ingestion-v2:
	$(INGESTION_V2_ENV) $(COMPOSE_V2) run --rm ingestion_process_api

## run-feature-store-v2: run only Job 2 (process_api mode)
run-feature-store-v2:
	$(INGESTION_V2_ENV) $(COMPOSE_V2) run --rm feature_store_v2

## run-training-v2: run only Job 3 (process-api, all 4 targets)
run-training-v2:
	$(COMPOSE_V2) run --rm training_v2

## logs: tail logs for all running services
logs:
	$(COMPOSE) logs -f

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
