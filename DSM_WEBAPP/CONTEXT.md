# DSM Webapp — Project Context for Claude Code

## What this project is

A cloud-native web application for Digital Soil Mapping (DSM) inference.
Users draw a field polygon on a map, submit a job, and receive a per-pixel
soil property map (clay, silt, sand, coarse fraction) overlaid on their field.

This project has two distinct components that must be built in order:

1. **Training pipeline rewrite** — retrain the 4 soil property models using
   Sentinel Hub Process API (not Statistical API) so that training and inference
   use identical data sources and feature engineering.

2. **Inference webapp** — FastAPI backend + inference worker + React frontend
   that serves predictions from the retrained models.

The training pipeline rewrite must be completed and validated before any
webapp inference code is trusted.

---

## Existing codebase

The researcher has already built and validated:
- `soil_texture_pipeline_v2.py` — full training pipeline (scikit-learn, Statistical API)
- `scaleway_workers.py` — Sentinel Hub Statistical API client (`StatisticsApiClient`)
- `only_scl.js` — existing evalscript for Statistical API (SCL filtering)
- Trained sklearn pipeline objects (`.pkl`) for 4 targets: clay, silt, sand, coarse
- Ground truth: `gabri_filters.xlsx` — LUCAS soil observation points with coordinates

These files are the authoritative reference. Do not rewrite logic from scratch —
adapt and extend what exists.

---

## Critical architectural decision: feature parity by construction

The old pipeline used **Statistical API**, which returns aggregated statistics
(mean, p50, stdev) over a bbox. The new pipeline uses **Process API**, which
returns raw per-pixel reflectance arrays.

These are fundamentally different data sources. Attempting to harmonize inference
against Statistical API-trained models would introduce systematic, hard-to-detect
errors. The correct solution is:

**Retrain the models end-to-end using Process API data.**

This guarantees feature parity by construction: training and inference use the
same evalscript, the same filter logic, and the same temporal aggregation code.

---

## Pre-development: lock the canonical filter config

Before writing any code, open `only_scl.js` and `soil_texture_pipeline_v2.py`
side by side and write `filter_config.json` at the project root.

### Filter design rationale (literature-backed)

The filter pipeline operates at two levels: scene-level (Catalog API, free)
and pixel-level (applied after Process API fetch, authoritative).

**Why two spectral indices, not one:**
The existing pipeline uses only NDVI < 0.25. This is insufficient. NDVI
removes green vegetation but passes two major reflectance contaminants:
- Dry crop residues / straw (post-harvest fields look bare to NDVI)
- Wet/moist soils (moisture systematically shifts SWIR reflectance)

Both contaminants corrupt soil property predictions. The literature has
converged on a second index — NBR2 — to catch what NDVI misses.

**NBR2** = (B11 - B12) / (B11 + B12). Sensitive to dry vegetation residues
and soil moisture via the SWIR bands. Standard in European cropland DSM
since Castaldi et al. (2019). Threshold 0.125 is the most cited value
across recent multi-site studies (Wetterlind et al. 2025, 34 European sites;
Silvero et al. 2021; Dvorakova et al. 2022).

**Known bias:** NBR2 is correlated with clay content. Strict NBR2 filtering
can systematically exclude high-clay pixels, truncating the prediction range.
Monitor clay prediction distribution after retraining. If the upper tail is
missing (e.g. no predictions > 40%), relax NBR2 threshold to 0.175.

**SCL class selection:** Keep only SCL 4 (vegetation) and SCL 5 (bare/not-vegetated).
All other classes are noise, misclassification, or physically invalid for soil
reflectance. SCL 2 (dark areas/shadows) deserves explicit exclusion — it is
frequently misclassified cloud shadow and passes naive filters.

**Catalog API filters:** Scene-level metadata is tile-wide (~100×100 km), not
field-specific. A tile with s2:vegetation_percentage = 90% may still have your
50ha field fully bare. Use catalog filters as cheap pre-filters only — be
generous. The pixel-level filter is authoritative.

### `filter_config.json`

```json
{
  "version": "2.0",
  "catalog_prefilter": {
    "eo:cloud_cover_lt": 80,
    "s2:snow_ice_percentage_lt": 5,
    "s2:cloud_shadow_percentage_lt": 40,
    "s2:not_vegetated_percentage_gt": 1,
    "rationale": {
      "cloud_cover": "Generous — per-pixel SCL handles residual cloud. Strict scene-level cloud filter would discard many usable partial-cloud scenes.",
      "snow_ice": "Hard exclude. Snow physically corrupts SWIR reflectance. No recovery via per-pixel filter.",
      "cloud_shadow": "40% threshold catches heavily shadowed scenes. Per-pixel SCL class 3 handles the rest.",
      "not_vegetated": "Eliminates fully vegetated scenes (dense summer canopy). Even 1% not-vegetated tile coverage is enough to proceed — the field may be fully bare."
    }
  },
  "pixel_filter": {
    "scl_keep_classes": [4, 5],
    "scl_exclude_classes": [0, 1, 2, 3, 6, 7, 8, 9, 10, 11],
    "ndvi_max": 0.25,
    "nbr2_max": 0.125,
    "rationale": {
      "scl": "Keep only SCL 4 (vegetation) and SCL 5 (bare/not-vegetated). NDVI then separates these two. All other classes are clouds, shadow, water, snow, or unclassified noise.",
      "ndvi": "NDVI < 0.25 removes active green vegetation. Consistent with Silvero et al. (2021), Dvorakova et al. (2022), and the existing training pipeline. More conservative than Castaldi (0.35) — prioritizes purity over coverage.",
      "nbr2": "NBR2 < 0.125 removes dry crop residues and wet soils missed by NDVI. Castaldi et al. (2019), Wetterlind et al. (2025, 34 European sites). If clay predictions are truncated at high values, relax to 0.175."
    }
  },
  "temporal_aggregation": {
    "method": "median",
    "min_valid_observations_per_pixel": 3,
    "rationale": "Median (P50) suppresses noise and outliers. P90 (driest soil) is an alternative for moisture-sensitive targets but requires more acquisitions and is not the documented default for texture mapping."
  },
  "job_filters": {
    "min_bare_soil_pixel_pct_per_date": 0.10,
    "min_bare_soil_acquisitions": 3,
    "rationale": "A date is kept if ≥10% of field pixels pass all pixel filters. Job fails if fewer than 3 such dates exist over the time window."
  }
}
```

This file is the single source of truth for all filtering logic. Both training
and inference load it at runtime via `FILTER_CONFIG_PATH` env var. Never
hardcode SCL classes, index thresholds, or count thresholds in Python.

**Verification step before committing:** For a handful of LUCAS points you
have ground truth for, run both filter configurations (old: NDVI-only; new:
NDVI+NBR2) and compare retention rates. Expect ~20-40% reduction in valid
observations. If retention drops below the minimum threshold for many points,
the NBR2 threshold may need relaxing to 0.175 for European temperate climates.

---

## Shared evalscript

One evalscript file, used by both the training pipeline and the inference worker:

**`evalscript_process_api.js`**

Requirements:
- Returns bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, SCL
- Output dtype: FLOAT32
- Band values: BOA reflectance, scaled to [0, 1] range
- SCL returned as raw integer class value (not normalized)
- No server-side filtering — all filtering applied client-side in Python
  using `filter_config.json`
- B11 and B12 are mandatory — they are required for NBR2 computation
  (NBR2 = (B11 - B12) / (B11 + B12)), the dry-residue/moisture filter

This file lives at the project root. Both pipelines reference the same file —
they do not have separate evalscripts.

---

## Part 1: Training pipeline rewrite

### Goal

Produce retrained model artifacts (`.pkl` files) trained on Process API data,
with a `features.json` that is the authoritative feature contract for inference.

### Raw raster storage strategy

**Store raw rasters to S3 before any filtering or aggregation.**

This is the most important architectural decision in the training pipeline.
Storing raw rasters means retraining with different filters, thresholds, or
model architectures costs zero additional Sentinel Hub API calls.

#### Raster spatial extent: 9×9 pixels (90×90m)

Fetch a 90×90m bbox centered on each LUCAS point, producing a 9×9 pixel raster
at 10m resolution. This is larger than strictly needed for the current tabular
models (which use the center pixel only), but enables future spatial model
experiments without re-fetching.

The current tabular pipeline and all retrained models use only the center pixel.
The surrounding pixels are stored but not used yet. This preserves options for
future spatial models (CNN, ViT) without requiring a full re-fetch of the
Sentinel Hub data.

#### Center pixel identification

Do not assume the center pixel is always `[4, 4]`. Compute it explicitly:
1. Reproject the LUCAS point coordinates (EPSG:4326) to EPSG:3857
2. Build the 90×90m bbox centered on the reprojected point
3. Identify which pixel's centroid is closest to the LUCAS point
4. Store as `center_pixel: [row, col]` in sidecar metadata

In practice this will be `[4, 4]` for a correctly centered bbox, but computing
it explicitly makes the metadata trustworthy and handles floating point edge cases.

The training pipeline reads `center_pixel` from metadata to extract the correct
pixel — it does not hardcode `[4, 4]`.

#### Raw raster file format

One `.npz` file per LUCAS point stored on S3:
`s3://{BUCKET}/raw_rasters/{lucas_point_id}.npz`

```python
np.savez(
    path,
    rasters=arr,       # shape: (N_dates, 9, 9, 11) — float32
                       # band order: B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12,SCL
    dates=dates,       # shape: (N_dates,) — string "YYYY-MM-DD"
    cloud_pct=cloud_pct,  # shape: (N_dates,) — float from Catalog API
)
```

Sidecar at `s3://{BUCKET}/raw_rasters/{lucas_point_id}_meta.json`:

```json
{
  "lucas_point_id": "...",
  "lucas_lat": 45.123,
  "lucas_lon": 10.456,
  "bbox_epsg4326": [xmin, ymin, xmax, ymax],
  "bbox_epsg3857": [xmin, ymin, xmax, ymax],
  "resolution_m": 10,
  "raster_shape": [9, 9, 11],
  "center_pixel": [4, 4],
  "bands": ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"],
  "fetch_date": "2024-01-15",
  "time_range": ["2022-01-01", "2023-01-01"],
  "n_dates_fetched": 47
}
```

### Training pipeline steps

```
gabri_filters.xlsx
       ↓
1. For each LUCAS point:
   a. Check if raw raster already exists on S3 — skip fetch if present
   b. Catalog API → enumerate acquisitions (cloud cover ≤ 80%)
   c. Process API → fetch 9×9 raster per date using evalscript_process_api.js
   d. Compute center_pixel from LUCAS coordinates + bbox
   e. Save .npz + _meta.json to S3
       ↓
2. For each LUCAS point (load from S3, no re-fetch):
   a. Load filter_config.json
   b. Per-date, per-pixel: apply SCL keep-classes filter (keep SCL 4 and 5 only)
   c. Compute NDVI = (B08 - B04) / (B08 + B04); mask pixels where NDVI >= ndvi_max
   d. Compute NBR2 = (B11 - B12) / (B11 + B12); mask pixels where NBR2 >= nbr2_max
   e. Extract center pixel using center_pixel from metadata
   f. Compute temporal median across valid dates at center pixel
   g. Compute indices from median bands: NDVI, NDMI
   h. Build feature vector — one row per LUCAS point
       ↓
3. Join features with soil property ground truth from gabri_filters.xlsx
       ↓
4. Train 4 sklearn pipelines (clay, silt, sand, coarse)
   — same model architecture as soil_texture_pipeline_v2.py
   — same spatial CV strategy (SpatialGroupKFold)
       ↓
5. Export model artifacts to $MODEL_DIR
```

### Model artifacts layout

```
$MODEL_DIR/
├── clay.pkl
├── silt.pkl
├── sand.pkl
├── coarse.pkl
├── features.json          ← authoritative feature order + metadata
├── VERSION                ← plain text, e.g. "v2.0-processapi"
└── training_report.json   ← R2/RMSE/MAE per target, n_points, date, filter hash
```

### `features.json` format

```json
{
  "feature_names": [
    "B02_median", "B03_median", "B04_median", "B05_median",
    "B06_median", "B07_median", "B08_median", "B8A_median",
    "B11_median", "B12_median",
    "NDVI", "NDMI"
  ],
  "n_features": 12,
  "source": "process_api",
  "spatial_aggregation": "center_pixel",
  "temporal_aggregation": "median",
  "filter_config_hash": "<sha256 of filter_config.json at training time>"
}
```

The `filter_config_hash` lets the inference worker detect if it is running
with a different filter config than the model was trained with.

### Training checkpoint test

```python
import pickle, json, numpy as np, hashlib

# 1. Verify feature contract
features = json.load(open(f"{MODEL_DIR}/features.json"))
assert len(features["feature_names"]) == features["n_features"]

# 2. Verify filter config hash matches
cfg_hash = hashlib.sha256(open("filter_config.json","rb").read()).hexdigest()
assert features["filter_config_hash"] == cfg_hash, "Hash mismatch"

# 3. Verify model accepts correct feature shape
for target in ["clay", "silt", "sand", "coarse"]:
    model = pickle.load(open(f"{MODEL_DIR}/{target}.pkl", "rb"))
    X = np.random.rand(5, features["n_features"])
    preds = model.predict(X)
    assert preds.shape == (5,)
    assert (preds >= 0).all() and (preds <= 100).all()

# 4. Check predictions sum near 100 for a single sample
models = {t: pickle.load(open(f"{MODEL_DIR}/{t}.pkl","rb"))
          for t in ["clay","silt","sand","coarse"]}
X1 = np.random.rand(1, features["n_features"])
total = sum(m.predict(X1)[0] for m in models.values())
assert abs(total - 100) < 25, f"Targets sum to {total}, expected ~100"

print("Training checkpoint: PASSED")
```

---

## Part 2: Inference webapp

### Technical stack

| Layer | Technology |
|---|---|
| Backend API | Python, FastAPI |
| Inference worker | Python, scikit-learn, numpy, rasterio |
| Satellite data | Sentinel Hub Process API + Catalog API |
| Database | PostgreSQL (Supabase) |
| Object storage | Scaleway S3-compatible |
| Queue | PostgreSQL-backed (jobs table, SKIP LOCKED) |
| Frontend | React, MapLibre GL JS |
| Infrastructure | Scaleway VM (2 vCPU, 4GB RAM), systemd, nginx |

### Services on the VM

Two long-running processes managed by systemd:

1. **`dsm-api`** — FastAPI app, port 8000, behind nginx on 443
2. **`dsm-worker`** — inference worker, polls PostgreSQL for queued jobs

No external queue service. Worker polls:
```sql
SELECT * FROM jobs WHERE status = 'QUEUED'
ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
```

### Data model

```sql
CREATE TABLE jobs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status              TEXT NOT NULL DEFAULT 'QUEUED'
                        CHECK (status IN ('QUEUED','RUNNING','DONE','FAILED')),
    polygon             JSONB NOT NULL,
    bbox                FLOAT[] NOT NULL,      -- [xmin, ymin, xmax, ymax] EPSG:4326
    time_window_days    INT NOT NULL DEFAULT 365,
    resolution_m        INT NOT NULL DEFAULT 10,
    n_acquisitions      INT,
    n_bare_soil         INT,
    model_version       TEXT,                  -- from VERSION file
    filter_config_hash  TEXT,                  -- sha256 of filter_config.json used
    result_keys         JSONB,                 -- S3 keys per target once DONE
    error_msg           TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX jobs_status_idx ON jobs(status) WHERE status = 'QUEUED';
```

`filter_config_hash` records which filter config produced each job's results.
Enables auditing when filter parameters are tuned between model versions.

### API contract

#### `POST /jobs`

Request:
```json
{
  "polygon": { "type": "Feature", "geometry": { ... } },
  "time_window_days": 365
}
```

Validation (return 422 with descriptive message on failure):
- Polygon area ≤ 50 ha
- Valid geometry (no self-intersections)
- Single exterior ring, no holes (v1)
- `time_window_days` in [90, 3650]
  — Process API has no time limit. Sentinel-2 Level-2A archive starts ~April 2017.
    More acquisitions → more stable temporal median, especially for fields with low
    bare-soil frequency. 3650 days (≈10 years) covers the full archive. Default of
    365 days is a sensible starting point; users with vegetated or cloudy fields
    should be encouraged to widen the window.

Response `202 Accepted`:
```json
{ "job_id": "uuid" }
```

#### `GET /jobs/:id`

```json
{
  "id": "uuid",
  "status": "QUEUED|RUNNING|DONE|FAILED",
  "created_at": "iso8601",
  "updated_at": "iso8601",
  "n_acquisitions": 24,
  "n_bare_soil": 8,
  "error_msg": null
}
```

#### `GET /jobs/:id/result`

Returns `404` if status is not `DONE`.

```json
{
  "bbox": [xmin, ymin, xmax, ymax],
  "targets": {
    "clay":   "https://s3.../presigned...",
    "silt":   "https://s3.../presigned...",
    "sand":   "https://s3.../presigned...",
    "coarse": "https://s3.../presigned..."
  },
  "coverage": "https://s3.../presigned...",
  "model_version": "v2.0-processapi",
  "n_bare_soil": 8
}
```

Presigned URLs: 1h TTL. `coverage` PNG shows valid observation count per pixel,
transparent where count = 0.

### Inference worker — step-by-step

#### Step 1 — Claim job
```sql
UPDATE jobs SET status='RUNNING', updated_at=now()
WHERE id = (
  SELECT id FROM jobs WHERE status='QUEUED'
  ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
)
RETURNING *;
```

#### Step 2 — Validate filter/model consistency
- Load `filter_config.json`, compute sha256 hash
- Load `features.json`, check `filter_config_hash` matches
- If mismatch: fail job with `error_msg = 'model_filter_config_mismatch'`
- Store hash in `jobs.filter_config_hash`

#### Step 3 — Catalog API query
- AOI: job bbox, time range: `[now - time_window_days, now]`
- Apply scene-level pre-filter (CQL2-json, `filter-lang: "cql2-json"`):
  ```json
  {
    "op": "and",
    "args": [
      {"op": "<",  "args": [{"property": "eo:cloud_cover"}, 80]},
      {"op": "<",  "args": [{"property": "s2:snow_ice_percentage"}, 5]},
      {"op": "<",  "args": [{"property": "s2:cloud_shadow_percentage"}, 40]},
      {"op": ">",  "args": [{"property": "s2:not_vegetated_percentage"}, 1]}
    ]
  }
  ```
- These are tile-wide percentages, not field-specific. The per-pixel filter
  is authoritative. Catalog filters exist only to skip obviously useless scenes.
- Output: list of `(date, scene_id, cloud_pct)` sorted ascending
- Update `jobs.n_acquisitions`

#### Step 4 — Fetch and filter all candidate dates
For fields ≤ 50 ha at 10 m resolution the raster per date is tiny (~71×71 pixels
maximum). Fetching all 11 bands in one request costs the same as a 5-band request
(same Process API processing unit area). Do not split this into a lightweight
pre-screen + full fetch — that doubles HTTP requests for no bandwidth saving.

For each candidate date:
- Fetch full raster using shared `evalscript_process_api.js`
- bbox in EPSG:4326, resolution 10m
- Returns numpy array `(H, W, 11)` — band order per evalscript
- Apply pixel filter from filter_config.json:
  1. SCL in scl_keep_classes [4, 5]
  2. NDVI = (B08 - B04) / (B08 + B04) < ndvi_max
  3. NBR2 = (B11 - B12) / (B11 + B12) < nbr2_max
- Keep date if fraction of passing pixels ≥ `min_bare_soil_pixel_pct_per_date`
- If kept dates < `min_bare_soil_acquisitions`:
  → `status = 'FAILED'`, `error_msg = 'insufficient_bare_soil'`
- Update `jobs.n_bare_soil`

#### Step 5 — (merged into Step 4 — see above)

#### Step 5 — Per-pixel bare soil masking
For each kept date's full raster:
- Apply filter_config.json pixel_filter:
  1. SCL in scl_keep_classes [4, 5] — all other classes → NaN
  2. NDVI = (B08 - B04) / (B08 + B04) ≥ ndvi_max → NaN
  3. NBR2 = (B11 - B12) / (B11 + B12) ≥ nbr2_max → NaN (dry residues + wet soil)
- Log per-date counts: n_total, n_scl_pass, n_ndvi_pass, n_nbr2_pass
- Set all-NaN pixels across spectral bands

#### Step 6 — Temporal aggregation
- Stack valid arrays: `(N_dates, H, W, 10_spectral_bands)` — SCL excluded
- Per-pixel nanmedian across N_dates → `(H, W, 10)`
- Compute from median bands:
  - NDVI = (B08_med - B04_med) / (B08_med + B04_med)
  - NDMI = (B08_med - B11_med) / (B08_med + B11_med)
- Track `coverage_array (H, W)` = count of non-NaN dates per pixel

#### Step 7 — Feature matrix construction
- Concatenate spectral medians + NDVI + NDMI → `(H, W, 12)`
- Reshape to `(H*W, 12)`
- Validate column order against `features.json["feature_names"]` — raise on mismatch
- Identify valid rows: non-NaN AND inside polygon

#### Step 8 — Polygon clip mask
- Rasterize job polygon onto `(H, W)` grid (EPSG:4326, same resolution/bbox)
- Out-of-polygon pixels excluded from inference, set to NaN in output

#### Step 9 — Model inference
Models loaded at **worker startup**, not per-job.

For each target [clay, silt, sand, coarse]:
- `predictions = model.predict(valid_pixel_features)`
- Reconstruct `(H, W)`: valid pixels → predictions, rest → NaN

#### Step 10 — Render and upload
For each target:
- Fixed colormap (sequential, perceptually uniform — e.g. YlOrBr for clay)
- NaN and out-of-polygon → alpha=0
- Output: RGBA PNG
- Upload: `s3://{BUCKET}/jobs/{job_id}/{target}.png`

For coverage:
- Sequential blue colormap, normalize to [0, N_max]
- Out-of-polygon → transparent
- Upload: `s3://{BUCKET}/jobs/{job_id}/coverage.png`

Update `jobs.result_keys`, `jobs.status = 'DONE'`, `jobs.updated_at`.

#### Step 11 — Coordinate system (critical for overlay accuracy)
- All PNGs produced in EPSG:4326
- Pixel `(0, 0)` = top-left = `(xmin, ymax)` of bbox
- Pixel `(W-1, H-1)` = bottom-right = `(xmax, ymin)` of bbox
- MapLibre `addSource` type `image` corners: `[[xmin,ymax],[xmax,ymax],[xmax,ymin],[xmin,ymin]]`

### Watchdog

Thread within the worker process, runs every 5 minutes:
```sql
UPDATE jobs SET status='FAILED', error_msg='timeout', updated_at=now()
WHERE status = 'RUNNING'
AND updated_at < now() - INTERVAL '30 minutes';
```

Worker calls `update_job(conn, job_id, updated_at=now())` at every step.

### Configuration — environment variables

```bash
# Sentinel Hub
SH_CLIENT_ID=
SH_CLIENT_SECRET=
SH_TOKEN_URL=https://services.sentinel-hub.com/oauth/token
SH_PROCESS_URL=https://services.sentinel-hub.com/api/v1/process
SH_CATALOG_URL=https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search

# Storage
S3_BUCKET=
S3_ENDPOINT_URL=
S3_ACCESS_KEY=
S3_SECRET_KEY=
S3_RESULT_PREFIX=jobs/
S3_RASTER_PREFIX=raw_rasters/
S3_PRESIGN_TTL_SECONDS=3600

# Database
DATABASE_URL=

# Models + config
MODEL_DIR=
FILTER_CONFIG_PATH=
EVALSCRIPT_PATH=

# Worker tuning
WORKER_POLL_INTERVAL_SECONDS=10
JOB_TIMEOUT_MINUTES=30
MAX_POLYGON_AREA_HA=50
```

---

## Development sequence

```
Step A  Lock filter_config.json                     ← prerequisite for everything
Step B  Write evalscript_process_api.js             ← prerequisite for training + inference
Step C  Training pipeline rewrite                   ← prerequisite for inference worker
Step D  Phase 0 + Phase 1 (infra + API skeleton)   ← can run in parallel with C
Step E  Phase 2 (SH client + Process API)
Step F  Phase 3–4 (worker core + render/upload)
Step G  Phase 5 (frontend)
Step H  Phase 6 (hardening)
```

Steps A–C solve feature parity by construction. Do not skip or reorder.

---

## Phase checkpoints

### Phase 0 — Infrastructure
**Goal:** VM reachable, DB schema applied, all credentials verified.

```bash
python -c "import boto3; print('S3 ok')"
python -c "import psycopg2; psycopg2.connect(DATABASE_URL); print('DB ok')"
curl -s -X POST $SH_TOKEN_URL \
  -d "grant_type=client_credentials&client_id=$SH_CLIENT_ID&client_secret=$SH_CLIENT_SECRET" \
  | python -m json.tool | grep access_token
# All three succeed without error
```

---

### Phase 1 — API skeleton
**Goal:** All endpoints return correct shapes. No worker yet.

```bash
# POST valid polygon (< 50ha, valid geometry) → 202 + job_id
# POST polygon > 50ha → 422 with message
# POST self-intersecting polygon → 422 with message
# GET /jobs/:id → status JSON (status = QUEUED)
# GET /jobs/:id/result on QUEUED job → 404
# Manually UPDATE jobs SET status='DONE', result_keys='{"clay":"jobs/x/clay.png",...}'
# GET /jobs/:id/result → 200 with 5 presigned URLs
```

---

### Phase 2 — Sentinel Hub Process API client
**Goal:** Fetch correct per-pixel rasters for a known location and date.

```python
arr = process_client.fetch(bbox, date, evalscript_path="evalscript_process_api.js")
assert arr.shape == (H, W, 11)
assert 0.0 <= arr[..., :10].max() <= 1.5   # reflectance range
assert 0 <= arr[..., 10].max() <= 11        # SCL class range

# For a known LUCAS point:
# Compare center pixel B04, B08 values to Statistical API p50 for same point/date
# Document the comparison — any systematic offset must be understood and resolved
```

---

### Phase 3 — Training pipeline rewrite
**Goal:** Retrained models, features.json, and filter_config_hash all consistent.

```python
# Run training checkpoint test (see Part 1 section above)
# Additionally for 3 held-out LUCAS points:
# 1. Fetch Process API rasters
# 2. Apply filter_config, extract center pixel, compute features
# 3. model.predict() → predictions in [0, 100]
# 4. clay + silt + sand + coarse within 20% of 100
# 5. Predictions differ across the 3 points (model is discriminative)
```

---

### Phase 4 — Inference worker core
**Goal:** Worker claims a job and produces correct prediction arrays end-to-end.

```python
# Submit job for a known agricultural location (use a LUCAS point)
# Worker runs to DONE without error
# Verify:
#   jobs.n_bare_soil >= MIN_BARE_SOIL_ACQUISITIONS
#   prediction arrays shape == (H, W)
#   all non-NaN predictions in [0, 100]
#   out-of-polygon pixels are NaN
#   clay + silt + sand + coarse ≈ 100 per pixel (within model error)
#   coverage_array has integer values, non-zero inside polygon
```

---

### Phase 5 — Render and upload
**Goal:** Job reaches DONE, PNGs downloadable and spatially correct.

```bash
# GET /jobs/:id/result → 5 presigned URLs
# Download each PNG:
#   Dimensions match bbox at 10m resolution
#   Transparent pixels only outside polygon boundary
#   Visible spatial variation (not uniform color)
#   Coverage PNG non-zero inside polygon
#   Load PNG into QGIS using bbox → confirm it aligns with polygon geometry
```

---

### Phase 6 — Frontend MVP
**Goal:** Full user flow works end-to-end in browser.

- Draw polygon → Analyze → status transitions (QUEUED → RUNNING → DONE)
- Result overlay aligns with drawn polygon (no visible drift)
- Layer toggle works for all 5 layers (clay, silt, sand, coarse, coverage)
- FAILED job shows human-readable error:
  - `insufficient_bare_soil` → "Not enough cloud-free bare soil images were
    found for this field in the past year. This field may be permanently
    vegetated, or you may need to try a wider time window."
  - `model_filter_config_mismatch` → "Internal configuration error — please
    contact the administrator."
- Polygon > 50ha blocked client-side before submission with clear message

---

### Phase 7 — Hardening
**Goal:** Failure modes handled, system observable, deployment scripted.

- Kill worker mid-job → watchdog marks FAILED within 30 minutes
- Simulate SH API timeout → retry fires (3 attempts, exponential backoff)
- `/health` endpoint → 200 if DB reachable, 503 if not
- Deploy new model version: replace pkl + VERSION + features.json, restart worker,
  new jobs carry new `model_version` in result
- `.env.example` documents every variable with a one-line description
- Runbook covers: check job status, manually retry a FAILED job, deploy new model

---

## Open questions (need resolution before or during implementation)

1. **Multi-tile mosaic behaviour** — Does the Sentinel Hub Process API seamlessly
   mosaic when a bbox spans two S2 tiles? Verify empirically at Phase 2 with a
   bbox that straddles a known tile boundary. If not seamless, decide: fail the
   job with `error_msg = 'tile_boundary'` or handle explicitly.

2. **Seasonality filtering** — With time windows up to 3650 days, the temporal
   median mixes observations from different seasons and soil moisture states. For
   texture mapping this is probably acceptable (texture is intrinsic), but it is
   an untested assumption. Check whether training points with multi-year windows
   show systematically higher or lower residuals. If yes, consider restricting
   acquisition dates to a seasonal window (e.g. March–May or October–November for
   European temperate cropland). This would be a new field in `filter_config.json`.

3. **NBR2 threshold validation** — The 0.125 threshold is literature-derived.
   Before committing to it, run both NDVI-only and NDVI+NBR2 filters on a sample
   of LUCAS points and compare retention rates. If many European points drop below
   `min_bare_soil_acquisitions = 3`, relax to 0.175 and document in filter_config.

---

## Out of scope for v1

- Authentication / multi-user
- Uncertainty maps (v2: per-acquisition inference, stddev per pixel)
- Seasonality-aware temporal filtering (v2 — see open questions)
- Airflow orchestration of inference jobs
- Docker / containerization
- Multiple concurrent workers
- DEM, weather, or hyperspectral inputs
- LUCAS 2022 data integration
- Spatial model experiments — raw rasters stored at 9×9 to enable this later

---

## Implementation notes

**Reuse, don't rewrite:**
- `StatisticsApiClient` OAuth2 and retry logic → base class for `ProcessAPIClient`
- `soil_texture_pipeline_v2.py` feature formulas → copy exactly into training
  rewrite and inference worker. Do not rephrase index formulas.

**filter_config.json is the single source of truth:**
- Both pipelines load it at startup via `json.load(open(FILTER_CONFIG_PATH))`
- Never hardcode SCL classes, index thresholds, or coverage fractions in Python
- The filter `"version"` field must be logged with every job and training run

**NBR2 is new — implement carefully:**
- NBR2 = (B11 - B12) / (B11 + B12)
- Apply AFTER NDVI filter: NDVI removes active vegetation first; NBR2 then
  removes dry crop residues and wet soils from what remains
- NBR2 is correlated with clay content. After retraining, inspect the clay
  prediction distribution on validation points. If no predictions exceed ~40%
  clay, the filter is truncating high-clay samples — relax nbr2_max to 0.175
  and retrain. Document the change in filter_config.json version field.

**Catalog API filter uses CQL2-json:**
- Pass as `"filter-lang": "cql2-json"` in the POST body
- Only `"op": "and"` is supported as a logical operator
- `s2:snow_ice_percentage`, `s2:cloud_shadow_percentage`, `s2:not_vegetated_percentage`
  are tile-wide ESA metadata, not field-specific. They are pre-filters only.
- Never use Catalog filter results as authoritative bare soil classification.

**evalscript_process_api.js is shared:**
- Both pipelines read it from `EVALSCRIPT_PATH` env var and send it in requests
- Never duplicate or inline the evalscript string in Python code

**`jobs.bbox` is a denormalization of `jobs.polygon`:**
- bbox is computable from the polygon — storing it separately is for query convenience.
- Keep them in sync: compute bbox from polygon in the `POST /jobs` handler, never
  accept bbox from the client directly.
- Consider a DB check constraint: `bbox[0] < bbox[2] AND bbox[1] < bbox[3]` to
  catch accidental NaN/null at insert time.

**Multi-tile handling — NEEDS THOUGHT (see open questions below):**
- A field near a Sentinel-2 tile boundary (every ~100 km) may have overlapping
  coverage from two tiles on the same date with different cloud conditions.
- The Process API can mosaic across tiles automatically when bbox spans two tiles.
  Verify this behaviour empirically at Phase 2 — fetch a bbox that straddles a
  known tile boundary and confirm the raster is complete and seamless.
- If the API returns partial coverage, the worker must detect this (NaN strip along
  one edge) and either fail gracefully or request both tiles explicitly.

**jobs.updated_at must be updated at every worker step:**
- Implement a helper: `update_job(conn, job_id, **fields)` that always sets
  `updated_at = now()` regardless of what other fields are passed
- Call this after every step in the worker

**Log pixel counts at every filtering step:**
- Per date: `n_total_pixels`, `n_scl_pass`, `n_ndvi_pass`, `n_nbr2_pass`, `n_kept`
- Use structured JSON logging (one dict per line) → captured by systemd journal
- These counts are essential for debugging, NBR2 threshold tuning, and future
  model improvement decisions

**Feature validation before predict() — mandatory, runs every job:**
```python
expected = features_json["feature_names"]
actual = list(feature_df.columns)
if expected != actual:
    raise ValueError(f"Feature mismatch: expected {expected}, got {actual}")
```

**Raw raster re-fetch guard:**
- Training pipeline checks S3 for existing `.npz` before fetching
- Only fetch if the file does not exist
- This makes the training pipeline safely re-runnable after partial failures