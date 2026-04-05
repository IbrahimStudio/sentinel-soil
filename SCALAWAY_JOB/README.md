# Sentinel Soil — Pipeline Guide

Batch-processes Sentinel-2 data for LUCAS soil survey points, filters for bare-soil observations, aggregates spectral features, and trains ML models to predict soil texture (Clay, Silt, Sand, Coarse).

Two parallel pipelines are supported:

| | Statistics API (v1) | Process API (v2) |
|---|---|---|
| **Data source** | SH Statistics API — pre-aggregated p50 per date | SH Process API — raw per-pixel rasters |
| **Filtering** | Server-side SCL + NDVI thresholds in evalscript | Client-side via `DSM_WEBAPP/filter_config.json` |
| **Features** | 18 spectral features, temporal median | 12 spectral features, center-pixel median |
| **Training** | One target at a time (`TARGET=Clay`) | All 4 targets in one run, exports `.pkl` models |
| **Output** | `features.parquet`, `training/reports/` | `features_v2.parquet`, `DSM_WEBAPP/models/*.pkl` |
| **Make target** | `make local` | `make local-v2` |

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Statistics API pipeline (v1)](#2-statistics-api-pipeline-v1)
3. [Process API pipeline (v2)](#3-process-api-pipeline-v2)
4. [Utility targets](#4-utility-targets)
5. [How to modify the Statistics API evalscript](#5-how-to-modify-the-statistics-api-evalscript)
6. [How to modify the Process API filter config](#6-how-to-modify-the-process-api-filter-config)

---

## 1. Prerequisites

- Docker and Docker Compose installed
- A `.env` file in `SCALAWAY_JOB/` with:

```env
# Sentinel Hub credentials (used by both pipelines)
SH_CLIENT_ID=...
SH_CLIENT_SECRET=...

# Scaleway object storage (production)
SCALEWAY_S3_ENDPOINT=...
SCALEWAY_S3_REGION=fr-par
SCALEWAY_S3_BUCKET=...
SCALEWAY_ACCESS_KEY=...
SCALEWAY_SECRET_KEY=...

# Statistics API optional overrides (defaults shown)
FEATURE_STORE_PREFIX=batch_results/aggregated/
TARGET=Clay
PIPELINE_VERSION=v2
```

For local runs, S3 credentials are automatically overridden to point to the local MinIO instance. You still need the Sentinel Hub credentials.

---

## 2. Statistics API pipeline (v1)

### Run locally

```bash
make local
```

This will:
1. Start MinIO at `http://localhost:9000` (console at `http://localhost:9001`)
2. Create the `soil-sentinel` bucket
3. Run Job 1 — ingestion (calls Statistics API, stores aggregated JSONs to S3)
4. Run Job 2 — feature store (reads S3 JSONs, writes `feature_store/output/features.parquet`)
5. Run Job 3 — training (trains RF/HGB/ElasticNet for one target, writes `training/reports/`)

### Run individual jobs

```bash
make run-ingestion      # Job 1 only
make run-feature-store  # Job 2 only
make run-training       # Job 3 only
```

### Ingestion parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `XLSX` | `./gabri_filters.xlsx` | Input Excel file (must have `POINT_ID`, `TH_LAT`, `TH_LONG`, `SURVEY_DATE`) |
| `EVALSCRIPT` | `sh_statistics/evalscripts/only_scl.js` | Path to the SH evalscript |
| `TIME_WINDOW` | `60` | Days around each survey date to query (±30 days) |
| `WORKERS` | `8` | Number of parallel API workers |
| `NDVI_THRESHOLD` | `0.2` | Max NDVI to keep a day as bare soil |
| `COVERAGE_THRESHOLD` | `0.8` | Min valid pixel fraction to keep a day |
| `LIMIT` | `-1` | Process only first N rows (−1 = all); useful for testing |

```bash
make run-ingestion LIMIT=10                        # quick test on 10 points
make run-ingestion TIME_WINDOW=90 WORKERS=4
make local TIME_WINDOW=120 WORKERS=12 LIMIT=50
```

### Training parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET` | `Clay` | Target to train (`Clay`, `Silt`, `Sand`, `Coarse`) |
| `PIPELINE_VERSION` | `v2` | `v1` = basic RF/GBM/ElasticNet, `v2` = adds SHAP + drops collinear fractions |

```bash
TARGET=Sand make run-training
TARGET=Silt PIPELINE_VERSION=v1 make run-training
```

---

## 3. Process API pipeline (v2)

The v2 pipeline fetches raw 9×9 pixel rasters (90×90 m) centred on each LUCAS point, stores them as `.npz` on S3 (cache-once, reuse), applies per-pixel filtering locally, and trains all four texture models in a single run. Trained models are exported to `DSM_WEBAPP/models/` for use by the inference webapp.

### Configuration files

Before running, ensure these files exist in `DSM_WEBAPP/`:

- `filter_config.json` — SCL classes, NDVI/NBR2 thresholds, temporal aggregation rules
- `evalscript_process_api.js` — JS evalscript returning 11 bands (B02–B12 + SCL) as FLOAT32

Both files are mounted read-only into the containers at runtime.

### Build images

```bash
make build-v2
```

Builds `ingestion:process-api` and `training:process-api`. Also rebuilds the shared `feature-store:latest` image.

### Run locally

```bash
make local-v2
```

This will:
1. Start MinIO
2. Run Job 1 — ingestion (fetches rasters via Process API, caches `.npz` + `_meta.json` on S3)
3. Run Job 2 — feature store (`--source process_api`: loads rasters, applies filters, writes `features_v2.parquet`)
4. Run Job 3 — training (trains all 4 targets, exports `clay.pkl`, `silt.pkl`, `sand.pkl`, `coarse.pkl`, `features.json` to `DSM_WEBAPP/models/`)

### Run individual jobs

```bash
make run-ingestion-v2       # Job 1 only
make run-feature-store-v2   # Job 2 only
make run-training-v2        # Job 3 only
```

### Process API ingestion parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TIME_WINDOW_DAYS` | `365` | Total date range to query around each survey date (90–3650) |
| `RASTER_PREFIX` | `raw_rasters/` | S3 prefix for storing/reading cached rasters |
| `WORKERS` | `8` | Parallel fetcher threads |
| `LIMIT` | `-1` | Process only first N rows (−1 = all) |

```bash
make run-ingestion-v2 LIMIT=20 TIME_WINDOW_DAYS=730
make local-v2 TIME_WINDOW_DAYS=1825 WORKERS=4
```

The raster cache is content-addressed by point ID. Re-running ingestion with a longer `TIME_WINDOW_DAYS` only fetches the newly-covered dates — existing `.npz` files on S3 are **not** overwritten (to force a refresh, delete them from S3 first).

### Process API training outputs

| File | Description |
|------|-------------|
| `DSM_WEBAPP/models/clay.pkl` | Best model for Clay |
| `DSM_WEBAPP/models/silt.pkl` | Best model for Silt |
| `DSM_WEBAPP/models/sand.pkl` | Best model for Sand |
| `DSM_WEBAPP/models/coarse.pkl` | Best model for Coarse |
| `DSM_WEBAPP/models/features.json` | Feature contract: 12 feature names + `filter_config_hash` |
| `DSM_WEBAPP/models/VERSION` | Pipeline version string |
| `DSM_WEBAPP/models/training_report.json` | Per-target metrics (random CV + spatial block CV) |
| `training/reports/` | Scatter plots, per-fold scores, feature importance |

The `features.json` file embeds a hash of `filter_config.json`. The inference worker checks this hash at startup to detect mismatches between training and inference filter configs.

---

## 4. Utility targets

```bash
make build          # build Statistics API images
make build-v2       # build Process API images
make minio          # start MinIO only (useful to inspect the bucket mid-run)
make logs           # tail logs for all running containers
make clean          # remove feature_store output and training reports (keeps MinIO data)
make clean-all      # full reset including MinIO local storage
make help           # list all available targets
```

---

## 5. How to modify the Statistics API evalscript

The evalscript is a JavaScript file passed to the Sentinel Hub Statistics API. It defines which spectral bands and indices are computed per acquisition date. Changing it affects five downstream layers.

### The cascade

```
1. evalscript JS         — defines output band names, count, and order
         ↓
2. scaleway_workers.py   — detects evalscript "type" to select a parsing branch
         ↓
3. parsers.py            — parses the raw SH API JSON into DailyStatsRecord
         ↓
4. models.py             — FEATURE_COLS maps band positions to feature names
         ↓
5. feature_store         — reads p50_aggregated keys as parquet column names
         ↓
6. training              — expects specific column names in the feature table
```

### Layer 2 — evalscript type detection (`scaleway_workers.py:92`)

```python
evalscript_type = "only_scl" if "only_scl" in evalscript.lower() else "features"
```

This is a fragile string check. If you introduce a new evalscript with a different filename, it will silently fall through to the `features` branch. Either include `only_scl` in the filename or add a new branch.

### Layer 3 — Response parsing (`parsers.py`)

- **`only_scl` branch**: coverage computed from `sampleCount / noDataCount` across all 18 feature bands. The number `18` is hardcoded — update it if you add/remove bands.
- **`features` branch**: reads coverage from a dedicated `valid` output band. If your evalscript lacks a `valid` output, coverage returns 0.0 and all days are dropped.

### Layer 4 — Feature column mapping (`models.py:FEATURE_COLS`)

```python
FEATURE_COLS = [
    "B02", "B03", "B04", "B08", "B11", "B12",
    "NDVI", "NDWI", "MNDWI", "NDMI", "BSI",
    "BRIGHT", "ALBEDO_PROXY",
    "RED", "SWIR1", "SWIR2",
    "RED_SWIR1_RATIO", "SWIR1_SWIR2_RATIO"
]
```

The parser maps `B0 → FEATURE_COLS[0]` by position. If the evalscript changes band order or count, feature names will be silently mis-assigned. **Always update `FEATURE_COLS` to match the evalscript — same names, same order.**

### Checklist when changing the evalscript

- [ ] Update the JS file
- [ ] Update `FEATURE_COLS` in `ingestion/sh_statistics/models.py`
- [ ] If band count changed: update `range(18)` in `parsers.py`
- [ ] If filename does not contain `"only_scl"`: update type detection in `scaleway_workers.py:92`
- [ ] Check that training column references still align

---

## 6. How to modify the Process API filter config

All filtering logic for the v2 pipeline lives in `DSM_WEBAPP/filter_config.json`. Changes here propagate to both training (via `main_v2.py`) and inference (via the webapp worker) — no code changes needed.

```json
{
  "pixel_filter": {
    "scl_keep_classes": [4, 5],
    "ndvi_max": 0.25,
    "nbr2_max": 0.125
  },
  "temporal_aggregation": {
    "min_valid_observations_per_pixel": 3
  }
}
```

**After changing `filter_config.json` you must retrain** — the `features.json` embeds a hash of the config, and the inference worker will refuse to serve predictions if the hash mismatches. Rerun `make local-v2` (or just `make run-feature-store-v2 && make run-training-v2` if rasters are already cached).

Key thresholds and their effect:

| Field | Effect |
|-------|--------|
| `scl_keep_classes` | Only pixels with these SCL values are kept. `[4, 5]` = vegetation-free land + bare soil. Relaxing to include `[4, 5, 6]` adds water-adjacent pixels. |
| `ndvi_max` | Pixels above this NDVI are discarded. Increase to 0.30 to keep sparse-vegetation observations; decrease to 0.20 for stricter bare-soil only. |
| `nbr2_max` | Removes dry crop residues and wet soils. Threshold 0.125 works well; relax to 0.175 if clay predictions appear truncated. |
| `min_valid_observations_per_pixel` | Minimum number of valid acquisition dates to compute a reliable temporal median. Increase for higher-quality features, at the cost of fewer points surviving. |
