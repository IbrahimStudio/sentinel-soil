.PHONY: init start start-local stop restart logs build-jobs clean help

COMPOSE       = docker compose
COMPOSE_LOCAL = docker compose --profile local

## init: first-time setup — builds the Airflow image and initialises the DB
##       run this once before the first `make start`
init: build
	$(COMPOSE) run --rm airflow-init

## start: start Airflow (webserver + scheduler) — UI at http://localhost:8080
start:
	$(COMPOSE) up -d airflow-webserver airflow-scheduler
	@echo "Airflow UI → http://localhost:8080  (admin / admin)"

## start-local: start Airflow + MinIO (local S3 emulator)
start-local:
	$(COMPOSE_LOCAL) up -d minio
	$(COMPOSE_LOCAL) run --rm minio_init
	$(COMPOSE) up -d airflow-webserver airflow-scheduler
	@echo "Airflow UI  → http://localhost:8080  (admin / admin)"
	@echo "MinIO UI    → http://localhost:9001  (minioadmin / minioadmin)"

## stop: stop all services (data is preserved in Docker volumes)
stop:
	$(COMPOSE) down

## restart: stop then start
restart: stop start

## logs: tail logs from all running services
logs:
	$(COMPOSE) logs -f

## build: build the custom Airflow Docker image
build:
	$(COMPOSE) build

## build-jobs: build the 3 job images from SCALAWAY_JOB
##             required before triggering the DAG for the first time
build-jobs:
	docker compose -f ../SCALAWAY_JOB/docker-compose.yaml build

## clean: remove all containers and volumes (wipes Airflow DB and MinIO data)
clean:
	$(COMPOSE) down -v
	rm -rf local_storage/

## help: show this help
help:
	@grep -E '^## ' Makefile | sed 's/## /  /'
