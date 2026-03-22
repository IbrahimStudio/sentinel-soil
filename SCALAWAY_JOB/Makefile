.PHONY: build local run-ingestion run-feature-store run-training clean logs help

COMPOSE       = docker compose
COMPOSE_LOCAL = docker compose --profile local

# ── Ingestion parameters (override on the command line) ──────────────────────
XLSX               ?= ./gabri_filters.xlsx
EVALSCRIPT         ?= sh_statistics/evalscripts/only_scl.js
TIME_WINDOW        ?= 60
WORKERS            ?= 8
NDVI_THRESHOLD     ?= 0.2
COVERAGE_THRESHOLD ?= 0.8
LIMIT              ?= -1

# Exported so docker-compose substitution picks them up
INGESTION_ENV = \
  XLSX=$(XLSX) \
  EVALSCRIPT=$(EVALSCRIPT) \
  TIME_WINDOW=$(TIME_WINDOW) \
  WORKERS=$(WORKERS) \
  NDVI_THRESHOLD=$(NDVI_THRESHOLD) \
  COVERAGE_THRESHOLD=$(COVERAGE_THRESHOLD) \
  LIMIT=$(LIMIT)

## build: build all 3 Docker images
build:
	$(COMPOSE) build

## local: run the full pipeline locally using MinIO as S3
##        data persists in ./local_storage/ between runs
local: build
	@echo "Starting MinIO..."
	$(COMPOSE_LOCAL) up -d minio
	@echo "Waiting for MinIO and creating bucket..."
	$(COMPOSE_LOCAL) run --rm minio_init
	@echo "Running Job 1 – ingestion..."
	$(INGESTION_ENV) $(COMPOSE) run --rm ingestion
	@echo "Running Job 2 – feature store..."
	$(COMPOSE) run --rm feature_store
	@echo "Running Job 3 – training..."
	$(COMPOSE) run --rm training
	@echo "Done. Check training/reports/ for results."

## minio: start MinIO only (useful to inspect bucket mid-run)
minio:
	$(COMPOSE_LOCAL) up -d minio minio_init

## run-ingestion: run only Job 1
run-ingestion:
	$(INGESTION_ENV) $(COMPOSE) run --rm ingestion

## run-feature-store: run only Job 2
run-feature-store:
	$(COMPOSE) run --rm feature_store

## run-training: run only Job 3
run-training:
	$(COMPOSE) run --rm training

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
