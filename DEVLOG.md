# DEVLOG — Sentinel Soil

Developer log tracking architectural decisions, pipeline iterations, and key findings.
Entries are in reverse chronological order (newest first).

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
