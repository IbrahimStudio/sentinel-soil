# DEVLOG — Sentinel Soil

Developer log tracking architectural decisions, pipeline iterations, and key findings.
Entries are in reverse chronological order (newest first).

---

## 2026-04-18 | Feature aggregation + training run (1 100 points, all 4 targets)

### Context

Ingestion rerun (see entry below) completed successfully. 1 110 points landed in `batch_results/aggregated/`; 1 100 joined with ground-truth labels after the feature-store step (10 points had no matching POINT_ID in `gabri_filters.xlsx`).

### Feature store (Job 2)

```
make run-feature-store STORAGE_PREFIX=batch_results
```

Reads all JSONs under `batch_results/aggregated/`, joins with `gabri_filters.xlsx`, writes `feature_store/output/features.parquet` (1 100 rows × 39 features). No changes to the aggregator.

### Training (Job 3)

```
make run-training   # ran 4 times, one target per container via TARGET env var
```

Pipeline v2, 5-fold CV (random KFold + spatial GroupKFold on 20 km grid). Best results per target:

| Target | CV | Model | R² | RMSE | MAE |
|--------|----|-------|----|------|-----|
| Clay | random | RF n1200_leaf1_sqrt | 0.561 | 7.70 | 5.98 |
| Clay | spatial | RF n1200_leaf1_sqrt | 0.547 | 7.86 | 6.11 |
| Silt | random | RF n1200_leaf1_sqrt | 0.501 | 8.61 | 6.71 |
| Silt | spatial | RF n1200_leaf1_sqrt | 0.498 | 8.65 | 6.71 |
| Sand | random | HGB lr0.03_iter1200 | 0.408 | 13.44 | 10.59 |
| Sand | spatial | RF n800_leaf2_sqrt | 0.388 | 13.73 | 11.00 |
| Coarse | random | RF n1200_leaf1_sqrt | 0.252 | 11.10 | 8.52 |
| Coarse | spatial | RF n1200_leaf1_sqrt | 0.235 | 11.25 | 8.61 |

Random Forest dominates all targets. Clay/Silt are the best-behaved targets (~0.50–0.56 R²). Sand moderate (0.39–0.41). Coarse weakest (0.24–0.25) — likely a combination of sparse ground truth and high natural variability. The random vs spatial CV gap is small for Clay/Silt (≤0.01–0.02), slightly larger for Sand/Coarse, suggesting reasonable geographic generalization. Reports + fitted pipelines in `training/reports/run_20260418_21*/`.

### Fixes in this session

**`sh_statistics/batch/scaleway_workers.py` — year-chunked requests**

Added `_year_chunks(start_date, end_date)` which splits any date range into calendar-year sub-requests. A full 10-year window (~3 650 daily intervals) was hitting SH Statistics API server timeouts; annual chunks (~365 intervals each) are reassembled in memory before parsing. The chunking is transparent to callers.

**`ingestion/process_api/sh_clients.py` — correct API base URLs**

Fixed two wrong endpoint URLs introduced in a previous session:
- `https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search` → `https://services.sentinel-hub.com/catalog/v1/search`
- `https://services.sentinel-hub.com/api/v1/process` → `https://services.sentinel-hub.com/process/v1`

The old paths returned 404/400; the corrected paths match the current SH API spec.

**`sh_pipeline/storage.py` — `put_bytes` + `list_objects_with_metadata`**

- `put_bytes(key, data, content_type)` added as a primitive; `put_text` now delegates to it.
- `list_objects` refactored to call a new `list_objects_with_metadata` which returns `[{"key", "last_modified", "size"}]` dicts. The audit script and results analyzer both needed `last_modified` for recency filtering.

**`sh_statistics/analysis/results_analyzer.py` — S3-only output**

`ResultsAnalyzer.join_with_source_xlsx` and `generate_summary_report` previously wrote to local disk. Rewritten to upload directly to S3 under `{storage_prefix}/analysis/` using `put_bytes`/`put_text`. Removed the `output_dir` attribute and the `mkdir` calls that assumed a writable local filesystem.

---

## 2026-04-18 | Pipeline rerun + OAuth/client optimisation

### Context

Previous ingestion run (2026-04-15, full archive `2015-07-01 → 2026-04-15`) completed 0/1100 points in the Statistics API pipeline. All 1100 points landed in `batch_results/errors/` with `401 Unauthorized` against `services.sentinel-hub.com`. The 593 points already in `batch_results/aggregated/` were from an earlier partial run and were unaffected.

Root cause: SH credentials were temporarily invalid at run time. Credentials are now restored and verified.

### Rerun

Identified failed points via `dev/audit_ingestion_run.py` (new script added this session — see below). Generated `failed_points_rerun.xlsx` (1100 rows, filtered from `gabri_filters.xlsx`) and reran with:

- Date window: `2015-07-01 → 2026-04-18` (full Sentinel-2 archive, same as original run)
- Workers: 3 (reduced from 8 to stay within SH rate limits)
- Storage prefix: `batch_results/` (same as original run, so downstream feature-eng and training see the complete 1100-point dataset without any merge step)

### New tooling: `dev/audit_ingestion_run.py`

CLI script to audit any ingestion run directly from S3:

```
uv run python dev/audit_ingestion_run.py \
  --prefix batch_results \
  --xlsx gabri_filters.xlsx \
  --max-age-days 0
```

Reports succeeded / failed / missing counts, shows error messages for failed points, cross-references against the source XLSX, and uploads a full `audit_results.csv` to `{prefix}/audit/`. `--max-age-days 0` disables the recency filter to show all-time results.

### OAuth2 / client lifecycle optimisation

**Problem:** `create_client_from_env()` was called at the top of `_execute_single_job_static` — once per job. With 1100 jobs and 3 workers this meant up to 1100 OAuth2 token fetches instead of 3. The internal token-refresh logic inside `StatisticsApiClient._get_oauth_session()` was effectively dead because the client was thrown away after every job.

**Fix — `sh_statistics/client.py`:**

`create_client_from_env()` now returns a **process-level singleton**. First call in a process does the OAuth handshake; all subsequent calls return the cached instance. This benefits both pipelines:
- Pipeline A: each worker process pays the handshake cost once, regardless of how many jobs it processes
- Pipeline B: `run_one_job()` is called per SQS message in a consumer loop — previously created a new client per message, now reuses the singleton

**Fix — `sh_statistics/batch/scaleway_workers.py`:**

Replaced the `ProcessPoolExecutor` approach with an explicit `initializer`:

```python
ProcessPoolExecutor(
    max_workers=n,
    initializer=_worker_process_init,
    initargs=(evalscript, interval, resolution),
)
```

`_worker_process_init` runs once per worker process and stores the SH client plus the run-level config (evalscript, interval, resolution) as module-level globals. The per-job function `_execute_job_in_worker` reads from those globals — no per-job client creation, no per-job pickling of the evalscript string.

Removed the now-redundant `_execute_single_job_static` and `_execute_single_job` instance methods. The sequential path (`max_workers <= 1`) also calls `_worker_process_init` + `_execute_job_in_worker` for consistency.

**Net result:** OAuth handshakes reduced from N_jobs → N_workers (1100 → 3 for the current run).

---

## 2026-04-05 | Process API pipeline complete

### What was built

Full v2 pipeline implementing the architectural vision in `DSM_WEBAPP/CONTEXT.md`:

| Component | Files |
|-----------|-------|
| Raw raster ingestion | `SCALAWAY_JOB/ingestion/process_api/` |
| Post-ingestion quality check | `SCALAWAY_JOB/feature_store/coverage_report.py` |
| Feature extraction | `SCALAWAY_JOB/feature_store/process_api_extractor.py` |
| Training (all 4 targets) | `SCALAWAY_JOB/training/main_v2.py` |
| Airflow DAG (unified) | `AIRFLOW_JOB/dags/soil_pipeline_v2.py` |
| Filter config (source of truth) | `DSM_WEBAPP/filter_config.json` |

### Filter chain design — key decision

**Decision:** Remove all catalog-level (scene-level) pre-filters except extreme snow coverage (≥90%).

**Reasoning:** Sentinel Hub catalog metadata is tile-wide (~110×110 km). Our field patches are 90×90 m. A tile with 80% cloud cover may still have a perfectly clear 90m patch. Applying strict cloud/shadow/vegetation filters at tile level creates:
- False negatives: valid patches discarded because the tile is globally cloudy
- False passes: tiles that pass the filter but whose specific 90m area is bad

The pixel-level filter chain (SCL allowlist → NDVI min → NDVI max → NBR2 max) is authoritative — it operates at the correct spatial scale. The only tile-level filter that makes physical sense is extreme snow: if the entire 110km tile is snow-covered, there is genuinely no bare soil anywhere.

**Practical outcome:** `sh_clients.py` was simplified — removed 4-condition CQL2-json filter block, replaced with single Python check on `s2:snow_ice_percentage >= 90`.

**Note:** The initial CQL2-json implementation also caused a 400 Bad Request from the SH Catalog API — turns out the endpoint does not support CQL2-json or the STAC query extension. The current implementation does client-side filtering on the returned scene list.

### Coverage report

`coverage_report.py` generates 7 diagnostic plots and structured JSON alerts:
- `hist_dates.png` — fetched vs valid dates per point
- `ecdf_valid_dates.png` — ECDF with min_obs threshold line
- `scl_breakdown.png` — which SCL classes dominate (reveals filter aggression)
- `ndvi_distribution.png` — before/after filter NDVI histogram
- `band_ranges.png` — reflectance box plots (sanity: expect [0, 1])
- `monthly_heatmap.png` — acquisition frequency by month (reveals seasonal gaps)
- `map_coverage.png` — geographic scatter coloured by valid dates (reveals spatial bias)

Alerts fire if: fetch rate < 90%, survival rate < 50%, P25 valid dates < min_obs, bands outside [0, 1].

### Feature sanity checks

`process_api_extractor.py` runs `_sanity_check()` before returning the DataFrame:
- Hard errors (raises): < 10 rows, band_max > 10.0, all-NaN column
- Warnings (logged): < 50 rows, band_max > 1.5, constant feature, label sum outside [85, 115], > 5% NaN

### S3 versioning pattern

Both model artifacts and coverage reports follow dual-path upload:
- `models/process_api/{YYYYMMDD_HHMMSS}/clay.pkl` — immutable versioned copy
- `models/process_api/latest/clay.pkl` — always overwritten for inference worker

### Docker architecture

Two separate image families prevent dependency bloat:
- `ingestion:stats-api` + `training:stats-api` — Statistics API pipeline (v1)
- `ingestion:process-api` + `training:process-api` — Process API pipeline (v2)
- `feature-store:latest` — shared between both (selects mode via `--source` arg)

v2 services run under Docker Compose profile `v2`. `make local-v2` starts MinIO + runs all 4 v2 jobs in sequence.

---

## 2026-03 | Statistics API pipeline (v1) — established baseline

### What was built

End-to-end containerized pipeline on Scaleway:

1. **Ingestion** — reads `gabri_filters.xlsx`, calls SH Statistics API with `only_scl.js` evalscript, stores per-point aggregated JSONs to S3
2. **Feature store** — joins aggregated JSONs with ground-truth labels, writes `features.parquet`
3. **Training** — loads parquet, trains RF/HistGradBoost/ElasticNet for one target at a time, writes plots and metrics to `training/reports/`
4. **Airflow** (`soil_pipeline` DAG) — orchestrates the 3 jobs via `DockerOperator`

### Known limitations identified

- Statistics API `only_scl.js` allows SCL class 8 (unclassified); `coverage_analysis.js` excludes it — inconsistency
- Evalscript type detected by checking `"only_scl" in evalscript` string — fragile
- `XLSX` DAG param non-functional: `DockerOperator` does not template `mounts`, so input file is hardcoded to `gabri_filters.xlsx`
- `_as_float_or_none()` and `_median()` each defined twice in `sh_statistics/processing/parsers.py`
- `sh_pipeline/worker.py:149` — `failed` variable never set to True (cleanup on failure broken)

### Feature set

18 features: B02–B12, NDVI, NDWI, MNDWI, NDMI, BSI, BRIGHT, ALBEDO_PROXY, RED, SWIR1, SWIR2, RED_SWIR1_RATIO, SWIR1_SWIR2_RATIO

### Why v2 was needed

The Statistics API returns pre-aggregated tile statistics. Training and inference cannot use the same data path unless the inference worker also calls the Statistics API per request — which is too slow and tightly coupled to Sentinel Hub availability. The Process API approach stores raw rasters on S3, enabling fully offline inference after training, and guarantees feature parity by construction.

---

## Filter config quick reference

`DSM_WEBAPP/filter_config.json` is the single source of truth for all v2 filtering.
**After any change, retrain:** `features.json` embeds a `filter_config_hash` and the inference worker rejects mismatches.

| Parameter | Value | Effect |
|-----------|-------|--------|
| `catalog_prefilter.s2:snow_ice_percentage_lt` | 90 | Skip entirely snow-covered tiles |
| `pixel_filter.scl_keep_classes` | [4, 5] | Keep only vegetation-free land + bare soil |
| `pixel_filter.ndvi_min` | -0.1 | Reject dark pixels (water, deep shadow) |
| `pixel_filter.ndvi_max` | 0.25 | Reject active green vegetation |
| `pixel_filter.nbr2_max` | 0.125 | Reject dry residues and wet soils |
| `temporal_aggregation.min_valid_observations_per_pixel` | 3 | Minimum dates for reliable median |
| `job_filters.min_bare_soil_pixel_pct_per_date` | 0.10 | Min fraction of field passing filters per date |
| `job_filters.min_bare_soil_acquisitions` | 3 | Min valid dates before point is dropped |
