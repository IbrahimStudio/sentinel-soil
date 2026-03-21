# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Batch-processes Sentinel-2 Statistics API data for LUCAS soil survey points, filters for bare-soil observations, aggregates spectral features, and feeds them into ML models for soil texture prediction (Clay, Silt, Sand, Coarse).

**All active code lives inside `SCALAWAY_JOB/`.** The root-level `sentinel_soil/`, `ALGORITHMS/`, and top-level scripts are legacy and will be cleaned up.

## Dev Commands

```bash
# All work happens inside SCALAWAY_JOB (its own .venv)
cd SCALAWAY_JOB

# Run the main batch pipeline
uv run python scaleway_batch_stats_from_xlsx.py \
  --xlsx gabri_filters.xlsx \
  --evalscript sh_statistics/evalscripts/only_scl.js \
  --time_window 60 --workers 8

# Run ML training
uv run python Algorithms/soil_texture_pipeline.py
```

## Architecture — SCALAWAY_JOB

### Data Flow

```
gabri_filters.xlsx  (LUCAS points: POINT_ID, TH_LAT, TH_LONG, SURVEY_DATE + labels)
  → scaleway_batch_stats_from_xlsx.py
      → sh_statistics/batch/scaleway_workers.py   (ProcessPoolExecutor, N workers)
          → Sentinel Hub Statistics API  (CDSE OAuth2)
          → sh_statistics/processing/parsers.py   (p50 extraction, SCL+NDVI filter, coverage check)
          → aggregated JSON per point → Scaleway S3
  → utils/bucket_data_aggregator.py               (load + pickle/XLSX cache + join gabri_filters)
  → Algorithms/soil_texture_pipeline.py           (Random KFold CV + Spatial Block CV)
      → summary_metrics.csv, feature_importance.csv, scatter plots
```

### Module Map

**Entry point**
- `scaleway_batch_stats_from_xlsx.py` — reads Excel, builds `JobSpec` list, dispatches to workers, stores results to S3

**`sh_statistics/`** — Sentinel Hub Statistics API layer
- `models.py` — all shared dataclasses: `JobSpec`, `DailyStatsRecord`, `AggregatedStatsRecord`, `JobResult`, `FEATURE_COLS` (18 spectral features: B02–B12, NDVI, NDWI, MNDWI, NDMI, BSI, brightness, SWIR ratios)
- `client.py` — OAuth2 auth + Statistics API requests; meter-based bbox builder
- `config_validation.py` — preset named configs (`bare_soil_analysis`, `vegetation_monitoring`, `high_resolution`, `medium_resolution`)
- `batch/xlsx_processor.py` — Excel → `JobSpec`; survey_date ± time_window logic; European decimal comma handling
- `batch/scaleway_workers.py` — `ScalewayStatisticsWorker`: parallel execution, retry, S3 upload per job
- `processing/parsers.py` — `parse_daily_records()` + `aggregate_records()`; two evalscript modes: `features` (with dataMask band) vs `only_scl` (SCL-only)
- `analysis/results_analyzer.py` — post-run loader: reads S3 JSONs, caches, joins with source Excel, writes summary
- `evalscripts/` — JavaScript for SH API: `only_scl.js` (active), `coverage_analysis.js`, `no_filters.js`

**`sh_pipeline/`** — Alternative pixel-level time series pipeline (not Statistics API)
- `worker.py` — `run_one_job()`: extract → bare-soil features → upload manifest to S3
- `storage.py` — `S3StorageClient` (boto3 with retry config); `storage_from_env()` reads credentials from env
- `paths.py` — S3 key builders for intermediate / features / logs / manifest
- `models.py` — `JobSpec`, `PixelWindow`; `parse_job()` validates payloads

**`utils/`** — Shared helpers
- `bucket_data_aggregator.py` — S3 JSON loader with pickle+XLSX caching (MD5 prefix key); `join_features_with_gabri_filters()` merges features with `gabri_filters.xlsx`
- `scaleway_data_loader.py` — simpler S3 JSON loader (no caching)

**`Algorithms/`** — ML training
- `soil_texture_pipeline.py` — main script: RF + HistGradBoost + ElasticNet; two CV strategies: random KFold and spatial block CV (GroupKFold on 20 km grid cells via EPSG:3857); outputs `summary_metrics.csv`, per-fold scores, feature importance, plots
- `soil_texture_pipeline_v2.py` — enhanced version
- `compare_best_models_single_target.py` — compares best model across scenarios A/B/C per target/metric
- `corr_predictors.py` — predictor correlation analysis

**`configs/dev.yaml`** — CDSE auth endpoints, SH API URLs, data paths, AOI, temporal range, batch params (30 m bbox, 3×3 pixels, NDVI ≤ 0.2)

**`gabri_filters.xlsx`** — primary input: LUCAS points with coordinates, survey dates, and soil texture ground-truth labels (Clay, Silt, Sand, Coarse)

**`coverage_out/`** — artefacts from coverage analysis runs (histograms, per-point CSV, monthly retention CSV/PNG)

**`analysis_scl_ndvi.csv`** — large (~213k rows) daily statistics result file; feeds into `Algorithms/`

### Key Filtering Parameters

| Parameter | Default | Set via |
|-----------|---------|---------|
| NDVI threshold (bare soil) | 0.2 | `--ndvi_threshold` CLI arg |
| MNDWI threshold (water) | 0.0 | `models.py` |
| Sun zenith max | 70° | `models.py` |
| Min pixel coverage | 0.8 | `--coverage_threshold` CLI arg |
| SCL exclude classes | 3,6,8,9,10,11 | evalscripts |
| Bbox size around point | 30 m | `configs/dev.yaml` |
| Resolution | 10 m | `configs/dev.yaml` |

### Required Environment Variables

```
# Copernicus Data Space (Sentinel Hub auth)
CDSE_CLIENT_ID
CDSE_CLIENT_SECRET

# Scaleway S3 storage
SCALEWAY_S3_ENDPOINT
SCALEWAY_S3_BUCKET
SCALEWAY_ACCESS_KEY
SCALEWAY_SECRET_KEY
```
