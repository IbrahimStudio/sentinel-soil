# ml-soil — INTEGRATION.md

## Purpose

This document specifies the contracts between `ml-soil` and the three other repositories it integrates with: `terraOS`, `sentinel-soil`, and `soil-moisture`. Each contract is a boundary that crosses repository ownership; getting them wrong creates the kind of coupling that's expensive to undo later. The discipline at these boundaries is more important than the discipline inside any single repo.

For purely internal concerns of `ml-soil` (architecture, build sequence, key principles), see `CONTEXT.md`.

## The four repositories

- **`terraOS`** — the application. Owns user state, polygons (in its own Postgres), sensor data, jobs. Consumes `ml-soil` via HTTP from one of its internal modules.
- **`sentinel-soil`** — training pipeline for soil texture models. Outputs versioned model artifact bundles. Does not serve.
- **`soil-moisture`** — development environment for moisture inference (OPTRAM, trapezoid). Publishes a distilled subpackage `soil-moisture-inference` consumed by `ml-soil`.
- **`ml-soil`** — the unified inference service (deployed as the `MlSoil` container). Composes texture and moisture pipelines under a uniform API and adds zoning logic.

The deployment unit at runtime is `MlSoil`, a single container. `sentinel-soil` and `soil-moisture` are not running services — they are source repositories whose outputs (artifacts and a Python subpackage, respectively) `MlSoil` consumes at build / startup time.

## Contract 1: terraOS ↔ ml-soil (HTTP API)

### Transport

HTTP/JSON. Internal API key in v1 (header `X-Internal-API-Key`). All requests are stateless and idempotent. Field polygons are passed inline as GeoJSON; `ml-soil` does not have access to terraOS's Postgres.

### Endpoints

**Property prediction.**

`POST /predict/texture`

Request:
```json
{
  "field_id": "<opaque string>",
  "polygon": { /* GeoJSON Polygon, EPSG:4326 */ },
  "time_window": ["YYYY-MM-DD", "YYYY-MM-DD"],
  "pipeline": "texture_v1",
  "config": { /* pipeline-specific, optional */ }
}
```

Response:
```json
{
  "features": [
    {
      "tile_id": "string",
      "coords": [lat, lon],
      "features": {"clay": 25.3, "silt": 41.2, "sand": 33.5, "coarse": 12.1},
      "metadata": {"n_obs": 8, "coverage": 0.75}
    }
  ],
  "provenance": { /* see Provenance section */ }
}
```

`POST /predict/moisture` — same shape, different pipelines (`moisture_v1`, future variants). Feature names differ; the schema is determined by the pipeline.

**Zoning.**

`POST /zoning/heterogeneity`

Request:
```json
{
  "fields": [
    {"field_id": "...", "polygon": { /* GeoJSON */ }}
  ],
  "time_window": [...],
  "pipeline": "texture_v1",
  "config": {...}
}
```

Response:
```json
{
  "per_field_metrics": {
    "<field_id>": {
      "cv": {"clay": 0.18, "silt": 0.12, ...},
      "eigengap_k": 3,
      "morans_i": {"clay": 0.42, ...},
      "summary": {...}
    }
  },
  "ranking": [
    {"field_id": "...", "composite_score": 0.78, "justification": "..."}
  ],
  "provenance": {...}
}
```

`POST /zoning/placement`

Request:
```json
{
  "field_id": "...",
  "polygon": {...},
  "time_window": [...],
  "n_sensors": 5,
  "feature_pipeline": "texture_v1",
  "methods": ["representative", "maximin", "stratified"],
  "config": {...}
}
```

Response:
```json
{
  "per_method_results": {
    "representative": {
      "selected": [{"coords": [lat, lon], "score": 0.87, "rank": 1, "rationale": {...}}, ...],
      "all_candidates": [...],
      "objective": "representative",
      "config": {...}
    },
    "maximin": {...},
    "stratified": {...}
  },
  "zone_map_geojson": {...},
  "provenance": {...}
}
```

**Future endpoint (specified, not built):**

`POST /zoning/score_placement` — given a placement and sensor observations, compute representation metrics (within-vs-across-zone correlation, rank agreement, representativeness scores per Vachaud temporal-stability framework). Phase 4.

### Operational endpoints

`GET /health` — liveness + loaded pipeline IDs + model versions + cache backend status.

### Error responses

Standard HTTP status codes. Error body shape:
```json
{
  "error": "<machine-readable code>",
  "message": "<human-readable detail>",
  "request_id": "<uuid>"
}
```

Codes `ml-soil` owns:

- `400` — invalid input (malformed GeoJSON, inverted time window, polygon area exceeds defensive cap)
- `404` — unknown `pipeline_id`, unknown placement method
- `422` — input shape valid but inconsistent with the pipeline's expected feature schema
- `500` — internal error (artifact load failure, unexpected exception)
- `503` — pipeline temporarily unavailable (Sentinel Hub down, model artifact corrupted)

Pipeline failures from `texture_v1` or `moisture_v1` are wrapped, not leaked. Callers never see a stack trace from sklearn or from `soil-moisture-inference`.

### Async pattern (deferred)

Endpoints that opt into async return `202 Accepted` with `{"job_id": "...", "poll_url": "/jobs/{id}"}`. `GET /jobs/{id}` returns the same response body as the synchronous version once complete. Not built in v1; specified here so the API design accommodates it without breaking changes.

### Polygon size limits

`terraOS` enforces ≤50 ha at its layer. `ml-soil` enforces a defensive cap (proposed: 100 ha) and returns `400` if exceeded. The cap exists to bound worst-case Sentinel Hub costs in case `terraOS`'s validation is bypassed.

## Contract 2: sentinel-soil → ml-soil (Model artifact handoff)

This is the most fragile contract in the system. `sentinel-soil` trains models against features it computes; `ml-soil` computes features independently at inference time. If the two diverge — a feature renamed, an aggregation changed, a filter threshold drifted — the model silently produces wrong predictions. Manifest validation is what prevents this.

### Artifact bundle structure

```
artifacts/texture_v1/
  clay.pkl
  silt.pkl
  sand.pkl
  coarse.pkl
  manifest.json
  evalscript.js
```

Storage: a versioned location in S3 — proposed `s3://ml-models/texture_v1/`. `sentinel-soil` writes; `ml-soil` reads. Versioning is by directory: `texture_v1`, `texture_v2`, etc. Never overwrite.

### manifest.json schema

```json
{
  "pipeline_id": "texture_v1",
  "model_version": "1.0.3",
  "trained_at": "2025-04-12T08:00:00Z",
  "training_data_hash": "sha256:...",
  "sklearn_version": "1.5.2",
  "python_version": "3.11.4",
  "feature_schema": [
    "B02_mean", "B02_stddev", "B02_p10", "B02_p90",
    "B03_mean", "...",
    "NDVI_mean", "NDMI_mean"
  ],
  "aoi_geometry": {"shape": "square", "size_m": 30},
  "bare_soil_filter": {
    "scl_classes": [4, 5, 6, 7],
    "ndvi_threshold": 0.3
  },
  "temporal_aggregations": ["mean", "stddev", "p10", "p50", "p90"],
  "targets": ["clay", "silt", "sand", "coarse"],
  "training_cv": {
    "strategy": "spatial_group_k_fold",
    "r2": {"clay": 0.41, "silt": 0.34, "sand": 0.39, "coarse": 0.28}
  }
}
```

`feature_schema` is the canonical ordered list of feature names the trained models expect. Order matters — it corresponds to the column order of the training matrix at fit time.

### Startup validation

When `ml-soil` loads `texture_v1`, in order:

1. Read `manifest.json`. Refuse to register the pipeline if missing or malformed.
2. Load each sklearn model (`clay.pkl`, etc.).
3. Verify `len(model.feature_names_in_) == len(manifest.feature_schema)` for each model where sklearn exposes this attribute. Where not exposed (e.g., older Pipeline wrappings), warn but proceed.
4. Verify `manifest.aoi_geometry` matches `ml-soil`'s tile-grid generator's output configuration.
5. Verify the evalscript hash matches what `ml-soil` would send to Sentinel Hub for this pipeline.
6. If any required check fails: do not register the pipeline. Log the inconsistency loudly. Health check returns `503` for that `pipeline_id` until resolved.

The `evalscript.js` shipped in the bundle is used **verbatim** by `ml-soil`. There is no second source of truth — what `sentinel-soil` trained against is what `ml-soil` queries with.

### Adding a new texture version

`sentinel-soil` retrains and publishes a new bundle at `s3://ml-models/texture_v2/` with appropriate manifest. `ml-soil` adds one line to its pipeline registry pointing at `texture_v2`. Restart container. Done.

### What `sentinel-soil` must not do

- Publish to a directory that already exists (overwriting models in place).
- Ship a bundle without a manifest, or with an incomplete manifest.
- Change the feature schema or evalscript without bumping the version.

## Contract 3: soil-moisture → ml-soil (Distilled Python subpackage)

### The distillation discipline

`soil-moisture` (the repo) contains research code: training, calibration, experimental variants, notebooks, scripts. Most of this is *not* what `ml-soil` needs. When a moisture inference path reaches validated maturity, the relevant code is extracted into a clean subpackage.

Concretely: a subdirectory of `soil-moisture` — proposed `soil-moisture/packages/soil_moisture_inference/` — that:

- Is independently pip-installable (`pip install -e soil-moisture/packages/soil_moisture_inference`).
- Has its own `pyproject.toml`, version, tests.
- Exposes a minimal public API.
- Has no runtime dependencies on the rest of `soil-moisture` (the research / training scaffolding).

`ml-soil` pins to a specific version in its own `pyproject.toml`: `soil-moisture-inference==0.1.0`.

### Public API

The minimum interface `soil-moisture-inference` exposes:

```python
def predict_moisture(
    polygon: Polygon,
    time_window: tuple[date, date],
    config: dict,
) -> list[FeatureVectorDict]: ...

PIPELINE_ID: str = "moisture_v1"
FEATURE_SCHEMA: list[str] = [...]
```

Where `FeatureVectorDict` is a documented dict shape (or a Pydantic model defined locally to the subpackage):

```python
{
    "tile_id": str,
    "coords": (lat, lon),
    "features": dict[str, float],
    "metadata": dict[str, Any],
}
```

The shape mirrors `ml-soil`'s `FeatureVector` but is defined locally to the subpackage to avoid `soil-moisture-inference` having to depend on `ml-soil`. `ml-soil` wraps this with a thin adapter class that satisfies the internal `Pipeline` protocol and registers it.

### Versioning and promotion

`soil-moisture-inference` follows semver:

- *Patch* (`0.1.0` → `0.1.1`) — bug fixes, no schema or output semantics change.
- *Minor* (`0.1.0` → `0.2.0`) — new optional config, additional metadata, no breaking change.
- *Major* (`0.1.0` → `1.0.0`) — breaking change: schema modified, semantics shifted, required config added.

When `soil-moisture` validates a new moisture pipeline (e.g., a refined OPTRAM variant), the decision is:

- Same semantics, just better fit? → patch or minor bump.
- Different schema or output? → either major bump, *or* publish as a new `pipeline_id` (`moisture_v2`).

The latter is preferred for research purposes: it lets `ml-soil` register both `moisture_v1` and `moisture_v2` so users can compare them. Major-version replacement only when the older version is truly obsolete and no longer being compared against.

### What `soil-moisture` must not do

- Change `predict_moisture`'s signature without bumping major.
- Change the meaning of features in `FEATURE_SCHEMA` without bumping major.
- Have `soil_moisture_inference` import from elsewhere in the `soil-moisture` repo.

## Contract 4: Shared types

### The `FeatureVector` type

`FeatureVector` is the boundary type that crosses repositories. For v1:

- It lives in `ml-soil` as a Pydantic model.
- `soil-moisture-inference` defines its own local `FeatureVectorDict` (a TypedDict or local Pydantic model) with the same shape.
- `ml-soil`'s adapter converts between the two.

This avoids a circular dependency — `soil-moisture-inference` doesn't need `ml-soil` as a dependency. The cost is the duplicated definition.

When more than two repositories need shared types, extract them into a standalone `soil-types` (or similar) package that all consumers depend on. That refactor is deferred; trivial when needed.

### Pipeline IDs and method IDs

- **Pipeline IDs** are strings; the canonical list lives in `ml-soil`'s registry. New IDs are added there.
- **Method IDs** (placement methods) live in `ml-soil`'s placement registry.
- **Provenance fields** are owned by `ml-soil`; source repos don't compute them.

## Versioning across pipelines

`ml-soil` simultaneously hosts multiple pipelines, each potentially at a different version:

```
texture_v1     (Statistical API)   — model_version 1.0.3
texture_v2     (Process API)       — model_version 0.9.0 (preview)
moisture_v1    (OPTRAM)            — soil-moisture-inference 0.1.0
```

Caller-side rules:

- The `pipeline` field in API requests is the `pipeline_id` (e.g., `texture_v1`), not a specific model version.
- The latest registered version of that pipeline is used by default.
- To pin: include `model_version` in the request `config`. `ml-soil` returns `404` if the pinned version isn't loaded.

Response-side: provenance always includes the exact version that ran. The caller can re-run with the same version by pinning.

## Provenance

Every response carries a `provenance` block:

```json
{
  "request_id": "<uuid>",
  "timestamp": "<ISO-8601>",
  "pipeline_id": "texture_v1",
  "pipeline_version": "1.0.3",
  "evalscript_hash": "sha256:...",
  "filter_config_hash": "sha256:...",
  "time_window": ["YYYY-MM-DD", "YYYY-MM-DD"],
  "ml_soil_version": "0.4.2",
  "downstream_versions": {
    "soil-moisture-inference": "0.1.0"
  }
}
```

This is the audit log of how a result was produced. Sufficient to reproduce the result given the same inputs.

## Failure modes and how they're caught

Four classes of cross-repo failure to defend against:

**(a) Silent feature schema drift.** `sentinel-soil` retrains with a renamed feature; manifest is updated but `ml-soil`'s feature computation lags. Caught by: feature-schema check at artifact load time. The manifest's `feature_schema` is the authoritative list; `ml-soil`'s feature computation is validated against it at startup. Mismatch → pipeline refuses to register.

**(b) Silent evalscript drift.** `sentinel-soil` retrains with a tweaked evalscript; the bundle ships the new script; `ml-soil` keeps using an old local copy. Caught by: the evalscript ships *in the bundle* (`evalscript.js`) and `ml-soil` uses that file, not a local copy. Hash recorded in provenance.

**(c) `soil-moisture-inference` semantic change without version bump.** Caught by: regression tests in `ml-soil` against fixture inputs. CI fails when a moisture pipeline's outputs shift unexpectedly on the same fixtures. Also caught by: provenance recording the exact pinned version, so the caller can detect post-hoc.

**(d) `ml-soil` adapter assumes a field that the soil-moisture package no longer produces.** Caught by: the adapter validates the returned dict against the expected feature schema before constructing FeatureVectors. Loud failure, not silent transformation.

What's *not* caught automatically: scientific drift in the underlying methods (e.g., OPTRAM calibration becomes systematically biased). That requires Phase 4 sensor data and is out of scope for the infrastructure layer.

## Open contract questions

To resolve with the relevant repo owners:

- **Artifact storage location.** S3 path `s3://ml-models/texture_v1/` proposed — confirm with whoever owns `sentinel-soil` deployment.
- **`soil-moisture` versioning discipline.** If one exists, align `soil-moisture-inference` to it; if not, this document proposes one.
- **Internal API key mechanism.** If `terraOS` already has one for inter-service auth, use it; otherwise, propose one.
- **Polygon size cap.** Defensive limit in `ml-soil` (proposed 100 ha) above `terraOS`'s 50 ha policy.
- **Manifest validation strictness.** Should `ml-soil` refuse to start if any pipeline's manifest is missing, or just degrade gracefully (registering only the working pipelines)? Proposed: degrade gracefully, surface state via `/health`.