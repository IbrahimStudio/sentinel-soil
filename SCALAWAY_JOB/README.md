# Sentinel Soil — Pipeline Guide

Batch-processes Sentinel-2 Statistics API data for LUCAS soil survey points, filters for bare-soil observations, aggregates spectral features, and trains ML models to predict soil texture (Clay, Silt, Sand, Coarse).

---

## Table of Contents

1. [How to use the pipeline](#1-how-to-use-the-pipeline)
2. [How to modify parts of the pipeline](#2-how-to-modify-parts-of-the-pipeline)

---

## 1. How to use the pipeline

### Prerequisites

- Docker and Docker Compose installed
- A `.env` file in `SCALAWAY_JOB/` with the following variables:

```env
# Sentinel Hub Statistics API credentials
SH_CLIENT_ID=...
SH_CLIENT_SECRET=...

# Scaleway object storage (used in production)
SCALEWAY_S3_ENDPOINT=...
SCALEWAY_S3_REGION=fr-par
SCALEWAY_S3_BUCKET=...
SCALEWAY_ACCESS_KEY=...
SCALEWAY_SECRET_KEY=...

# Optional overrides (defaults shown)
FEATURE_STORE_PREFIX=batch_results/aggregated/
TARGET=Clay
PIPELINE_VERSION=v2
```

For local runs, the S3 credentials are overridden automatically to point to MinIO (see below). You still need the Sentinel Hub credentials.

---

### Running the full pipeline locally

```bash
make local
```

This will:
1. Start a local MinIO instance (S3 emulator) at `http://localhost:9000`
2. Create the `soil-sentinel` bucket
3. Run Job 1 — ingestion (calls Sentinel Hub, stores results to MinIO)
4. Run Job 2 — feature store (aggregates S3 JSONs into a parquet feature table)
5. Run Job 3 — training (trains ML models, writes reports to `training/reports/`)

MinIO data persists in `./local_storage/` between runs. The MinIO console UI is available at `http://localhost:9001` (user: `minioadmin` / `minioadmin`).

---

### Running individual jobs

```bash
make run-ingestion      # Job 1 only
make run-feature-store  # Job 2 only
make run-training       # Job 3 only
```

---

### Ingestion parameters

The ingestion job accepts several parameters that can be overridden directly on the `make` command line. All parameters have defaults so you only need to specify what you want to change.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `XLSX` | `./gabri_filters.xlsx` | Path to the input Excel file with LUCAS points |
| `EVALSCRIPT` | `sh_statistics/evalscripts/only_scl.js` | Path to the Sentinel Hub evalscript |
| `TIME_WINDOW` | `60` | Days around each survey date to query (±30 days) |
| `WORKERS` | `8` | Number of parallel API workers |
| `NDVI_THRESHOLD` | `0.2` | Max NDVI to keep a day as bare soil |
| `COVERAGE_THRESHOLD` | `0.8` | Min valid pixel fraction to keep a day |
| `LIMIT` | `-1` | Process only first N rows (-1 = all); useful for testing |

**Examples:**

```bash
# Quick test on 10 points only
make run-ingestion LIMIT=10

# Different time window and fewer workers
make run-ingestion TIME_WINDOW=90 WORKERS=4

# Stricter bare-soil filtering
make run-ingestion NDVI_THRESHOLD=0.15 COVERAGE_THRESHOLD=0.9

# Different input file and evalscript
make run-ingestion XLSX=./my_points.xlsx EVALSCRIPT=sh_statistics/evalscripts/no_filters.js

# Full pipeline with custom ingestion params
make local TIME_WINDOW=120 WORKERS=12 LIMIT=50
```

The input Excel file must contain the columns: `POINT_ID`, `TH_LAT`, `TH_LONG`, `SURVEY_DATE`.

---

### Training parameters

Training parameters are controlled via environment variables (in `.env` or on the command line):

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET` | `Clay` | Soil texture target (`Clay`, `Silt`, `Sand`, `Coarse`) |
| `PIPELINE_VERSION` | `v2` | ML pipeline version (`v1` = basic RF/GBM/ElasticNet, `v2` = adds SHAP + drops collinear fractions) |

```bash
TARGET=Sand make run-training
TARGET=Silt PIPELINE_VERSION=v1 make run-training
```

---

### Utility targets

```bash
make build      # (Re)build all 3 Docker images
make minio      # Start MinIO only, useful to inspect the bucket mid-run
make logs       # Tail logs for all running containers
make clean      # Remove feature_store output and training reports (keeps MinIO data)
make clean-all  # Full reset including MinIO local storage
make help       # List all available targets
```

---

## 2. How to modify parts of the pipeline

### Modifying the evalscript

The evalscript is a JavaScript file passed to the Sentinel Hub Statistics API. It defines which spectral bands and indices are computed and returned per acquisition date. Changing it is the most impactful modification you can make, because the evalscript output shape propagates through the entire pipeline.

#### The cascade

When you change the evalscript, you are potentially affecting five layers in sequence:

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

#### Layer 1 — The evalscript JS

Evalscripts live in `ingestion/sh_statistics/evalscripts/`. The active one is `only_scl.js`.

The evalscript must return a `features` output containing all spectral bands as separate outputs. The band order in the JS `return` statement defines the positional index (B0, B1, B2, ...) that the Statistics API will use in its response. **This order must match `FEATURE_COLS` in `models.py`.**

The evalscript also controls which pixels are filtered out (e.g. via SCL classes). Note that `only_scl.js` excludes SCL classes 3, 6, 9, 10, 11 but **allows class 8** (cloud shadows). The alternative `coverage_analysis.js` excludes class 8 as well. This difference is invisible to the Python code but affects which days are considered valid.

#### Layer 2 — evalscript type detection (`scaleway_workers.py:92`)

```python
evalscript_type = "only_scl" if "only_scl" in evalscript.lower() else "features"
```

This is a fragile string check. There are only two recognised types: `only_scl` and `features`. The type controls how coverage is calculated in the parser (layer 3). **If you introduce a new evalscript with a different name, it will silently fall through to the `features` branch**, which may compute coverage incorrectly.

If you add a new evalscript:
- Either include the string `only_scl` in the filename if it follows the same output structure, or
- Add a new branch to this detection logic and to the parser.

#### Layer 3 — Response parsing (`parsers.py`)

`parse_daily_records()` has two code paths based on `evalscript_type`:

- **`only_scl` branch**: computes coverage by accumulating `sampleCount` / `noDataCount` across all 18 feature bands. The number `18` is **hardcoded** on line 344 (`for i in range(18)`). If you add or remove bands, update this number.
- **`features` branch**: reads coverage from a dedicated `valid` output band in the response. If your new evalscript does not expose a `valid` output, this branch will return coverage = 0.0 for all days, effectively dropping everything.

#### Layer 4 — Feature column mapping (`models.py:FEATURE_COLS`)

```python
FEATURE_COLS = [
    "B02", "B03", "B04", "B08", "B11", "B12",
    "NDVI", "NDWI", "MNDWI", "NDMI", "BSI",
    "BRIGHT", "ALBEDO_PROXY",
    "RED", "SWIR1", "SWIR2",
    "RED_SWIR1_RATIO", "SWIR1_SWIR2_RATIO"
]
```

This list is the contract between the evalscript and the Python code. The parser maps `B0 → FEATURE_COLS[0]`, `B1 → FEATURE_COLS[1]`, and so on, **purely by position**. If the evalscript returns bands in a different order, or a different number of bands, the feature names will be silently mis-assigned or the parser will return `None` for missing positions.

**When you change the evalscript band set, update `FEATURE_COLS` to match — same names, same order.**

#### Layer 5 — Feature store (`feature_store/`)

The feature store reads the aggregated JSON files from S3 and extracts the `p50_aggregated` dictionary, whose keys come directly from `FEATURE_COLS`. These keys become column names in the output `features.parquet`. No column list is hardcoded here — it adapts to whatever keys are present. However, if the column names change, the training job will break.

#### Layer 6 — Training (`training/`)

The training scripts reference specific feature column names when building the feature matrix. If you rename or remove features, update the column selection in `Algorithms/soil_texture_pipeline.py` (or `_v2.py`).

---

#### Summary checklist when changing the evalscript

- [ ] Update the JS file to return the desired bands in the desired order
- [ ] Update `FEATURE_COLS` in `ingestion/sh_statistics/models.py` to match (same order)
- [ ] If band count changed: update `range(18)` in `parsers.py:parse_daily_records()` (the `only_scl` coverage branch)
- [ ] If the evalscript filename does not contain `"only_scl"`: update the type detection in `scaleway_workers.py:92` or adjust coverage logic accordingly
- [ ] If the evalscript does not expose a `valid` output and you are using the `features` branch: add coverage handling
- [ ] Check that training column references still align with the new `FEATURE_COLS`
