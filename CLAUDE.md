# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Batch-processes Sentinel-2 Statistics API data for LUCAS soil survey points, filters for bare-soil observations, aggregates spectral features, and feeds them into ML models to predict soil texture (Clay, Silt, Sand, Coarse).

All active code lives inside `SCALAWAY_JOB/`.

## Dev Commands

```bash
cd SCALAWAY_JOB

# Run the main statistics pipeline
uv run python scaleway_batch_stats_from_xlsx.py \
  --xlsx gabri_filters.xlsx \
  --evalscript sh_statistics/evalscripts/only_scl.js \
  --time_window 60 --workers 8

# Run coverage analysis
uv run python -m sh_statistics.coverage.run_coverage

# Run ML training
uv run python Algorithms/soil_texture_pipeline.py

# Collect seed summaries from S3
uv run python dev/collect_seed_summaries.py

# Inspect S3 bucket
uv run python dev/inspect_parque_s3.py
```

## Two Separate Pipelines

There are **two distinct pipelines** in this repo — they do not share code beyond `sh_pipeline/storage.py`:

### Pipeline A — Statistics API (active, main)
`scaleway_batch_stats_from_xlsx.py` → `sh_statistics/` → Scaleway S3

Uses the **Sentinel Hub Statistics API** to get pre-aggregated p50 percentile statistics per time interval. Does not download GeoTIFFs. Faster and lighter.

### Pipeline B — Pixel Time Series (alternative, used by sh_pipeline/)
`sh_pipeline/worker.py` → `domain/` → `utils/sentinelhub_client.py` → Scaleway S3

Downloads per-date GeoTIFFs via the **Process API**, extracts time series, computes bare-soil features locally. Slower, richer output.

## Architecture — SCALAWAY_JOB Module Map

### Pipeline A modules

**`scaleway_batch_stats_from_xlsx.py`** — entry point: reads Excel, builds `JobSpec` list, dispatches workers, uploads to S3

**`sh_statistics/models.py`** — all shared dataclasses:
- `JobSpec`: per-point job config (lat/lon, date range, filtering thresholds)
- `DailyStatsRecord`: one row per acquisition date (p50 values for 18 features)
- `AggregatedStatsRecord`: temporally aggregated across all bare-soil days
- `JobResult`: SUCCESS/FAILED wrapper around a list of `AggregatedStatsRecord`
- `FEATURE_COLS`: the 18 spectral features — B02–B12, NDVI, NDWI, MNDWI, NDMI, BSI, BRIGHT, ALBEDO_PROXY, RED, SWIR1, SWIR2, RED_SWIR1_RATIO, SWIR1_SWIR2_RATIO

**`sh_statistics/client.py`** — OAuth2 client for the Statistics API; two request methods:
- `request_statistics()` — degree-based bbox (EPSG:4326) — **avoid for production**: SH interprets resolution in CRS units so 10 means 10 degrees, not meters
- `request_statistics_meter_based()` — proper EPSG:3857 meter-based request; **use this one**

**`sh_statistics/batch/xlsx_processor.py`** — Excel → `JobSpec`; required columns: `POINT_ID`, `TH_LAT`, `TH_LONG`, `SURVEY_DATE`; handles survey_date ± time_window logic

**`sh_statistics/batch/scaleway_workers.py`** — `ScalewayStatisticsWorker`: `ProcessPoolExecutor` over jobs; bbox size hardcoded as `resolution * 3` (minimum 3×3 pixels); evalscript type detected by checking `"only_scl" in evalscript` string (fragile)

**`sh_statistics/processing/parsers.py`** — `parse_daily_records()` + `aggregate_records()`; extracts p50 percentiles; two code paths depending on evalscript type (`features` vs `only_scl`)

**`sh_statistics/analysis/results_analyzer.py`** — post-run: reads aggregated JSONs from S3, caches in memory, joins with source Excel, writes summary report

**`sh_statistics/evalscripts/`** — JavaScript evalscripts for SH Statistics API:
- `only_scl.js` — **active**: 18 feature bands + `valid` mask using SCL filter only (excludes classes 3,6,9,10,11 — note: allows class 8)
- `coverage_analysis.js` — 3 masks: raw / SCL-filtered / SCL+NDVI-filtered (excludes 3,6,8,9,10,11)
- `no_filters.js` — baseline with no filtering

**`sh_statistics/coverage/`** — standalone coverage analysis sub-system:
- `analysis.py` — `run_coverage_analysis()`: computes retention fractions per point/date
- `plots.py` — histogram, ECDF, time-series, scatter plots
- `run_coverage.py` — CLI entrypoint
- `storage.py` — persist results locally and to S3

### Pipeline B modules

**`sh_pipeline/worker.py`** — `run_one_job()`: extract → bare-soil features → upload manifest. Note: cleanup logic has a bug (`failed` variable never set to True)

**`sh_pipeline/storage.py`** — `S3StorageClient`: boto3 wrapper with retry, content-type detection, glob-filtered tree upload; `storage_from_env()` reads from env

**`sh_pipeline/paths.py`** — `JobPaths` dataclass; deterministic S3 + local key layout under `/tmp/sentinel_soil/<job_id>/`

**`sh_pipeline/models.py`** — `JobSpec`, `PixelWindow`; `parse_job()` validates payload (lat/lon ranges, odd pixel dimensions, NDVI range, etc.)

**`domain/extract_time_series.py`** — downloads orbit-mosaicked TIFFs, splits per-date, saves `bands.json`; bands hardcoded to B02,B03,B04,B08,B11,B12

**`domain/bare_soil_features.py`** — reads per-date GeoTIFF stack → long DataFrame → NDVI/NDWI/NDMI indices → bare-soil filter → pixel-level aggregation → outputs `pixel_timeseries.parquet`, `bare_soil_pixel_features.parquet`, `seed_summary.parquet`

### Shared utilities

**`utils/bucket_data_aggregator.py`** — loads aggregated JSONs from S3 with pickle+XLSX caching (MD5 hash of prefix as cache key); `join_features_with_gabri_filters()` merges features with ground-truth labels

**`utils/scaleway_data_loader.py`** — simpler JSON/JSONL loader from S3 (no caching)

**`utils/config.py`** — YAML → dataclasses loader (`AppConfig` with CDSEConfig, SentinelHubConfig, BatchConfig, etc.)

**`utils/evalscript_builder.py`** — generates orbit-mosaicking evalscripts dynamically (used by Pipeline B)

**`utils/geometry.py`** — `utm_epsg_from_latlon()`, `bbox_for_grid_around_point()` via pyproj (Pipeline B only)

**`utils/sentinelhub_client.py`** — thin wrapper around `sentinelhub` SDK `SentinelHubRequest` (Pipeline B only)

### ML

**`Algorithms/soil_texture_pipeline.py`** — main ML script: RF + HistGradBoost + ElasticNet; two CV strategies: random KFold and spatial block CV (GroupKFold on 20 km EPSG:3857 grid); outputs `summary_metrics.csv`, per-fold scores, feature importance, scatter plots

**`Algorithms/soil_texture_pipeline_v2.py`** — enhanced: drops co-linear texture fractions (e.g. drops Clay/Sand when predicting Silt), adds SHAP interpretation, exports correlation matrix

**`Algorithms/compare_best_models_single_target.py`** — compares scenarios A/B/C per target/metric, generates bar charts

**`Algorithms/corr_predictors.py`** — predictor correlation analysis

**`Algorithms/pca_linear.py`** — PCA + linear model experiments

### Other

**`dev/collect_seed_summaries.py`** — collects `seed_summary.parquet` from all S3 feature folders into one file

**`dev/inspect_parque_s3.py`** — S3 bucket inspection / debugging

**`configs/dev.yaml`** — CDSE auth, SH API endpoints, data paths, AOI, temporal range, batch params

**`gabri_filters.xlsx`** — primary input: LUCAS points with coordinates, survey dates, soil texture labels

## Known Issues (to address in cleanup)

| Severity | Location | Issue |
|----------|----------|-------|
| High | `sh_statistics/processing/parsers.py` | `_as_float_or_none()` and `_median()` each defined twice |
| Medium | `AIRFLOW_JOB/dags/soil_pipeline.py` | `XLSX` DAG param is non-functional: Airflow's `DockerOperator` does not template `mounts`, so `{{ params.XLSX }}` is never rendered in the `Mount` source path. Current workaround: `gabri_filters.xlsx` is hardcoded. Fix options: (1) mount the whole `SCALAWAY_JOB/` dir and use `--xlsx /data/{{ params.XLSX }}` in the command, (2) subclass `DockerOperator` and add `mounts` to `template_fields`. |
| Medium | `sh_statistics/batch/scaleway_workers.py:91` | Evalscript type detected via `"only_scl" in evalscript` string — fragile |
| Medium | `sh_statistics/batch/scaleway_workers.py:74` | Bbox size hardcoded as `resolution * 3` |
| Medium | `sh_pipeline/worker.py:149` | `failed` variable never set to True — cleanup on failure broken |
| Low | `sh_statistics/batch/scaleway_workers.py:216` | `{"timestamp": "TODO"}` placeholder in manifest |
| Low | `sh_statistics/config_validation.py:157` | `any` used instead of `Any` (type hint) |
| Note | `only_scl.js` vs `coverage_analysis.js` | SCL exclusion differs: `only_scl` allows class 8, `coverage_analysis` excludes it |

## Key Filtering Parameters

| Parameter | Default | Set via |
|-----------|---------|---------|
| NDVI threshold (bare soil) | 0.2 | `--ndvi_threshold` CLI |
| MNDWI threshold (water) | 0.0 | `models.py` |
| Sun zenith max | 70° | `models.py` |
| Min pixel coverage | 0.8 | `--coverage_threshold` CLI |
| SCL exclude classes | 3,6,9,10,11 | `only_scl.js` (note: 8 not excluded) |
| Bbox size | `resolution × 3` m | `scaleway_workers.py` |
| Resolution | 10 m | `configs/dev.yaml` |

## Required Environment Variables

```
CDSE_CLIENT_ID          # Copernicus Data Space auth
CDSE_CLIENT_SECRET
SH_CLIENT_ID            # Sentinel Hub Statistics API
SH_CLIENT_SECRET
SCALEWAY_S3_ENDPOINT
SCALEWAY_S3_BUCKET
SCALEWAY_ACCESS_KEY
SCALEWAY_SECRET_KEY
```
