# ml-soil — CONTEXT.md

## Purpose & Context

`ml-soil` is the unified ML inference service for the terraOS Digital Soil Mapping framework. It is deployed as a single container — `MlSoil` — that terraOS consumes via HTTP. It composes three internal modules into a uniform API: **texture property prediction**, **moisture property prediction**, and **zoning** (heterogeneity scoring + sensor placement).

The texture and moisture inference paths are owned by separate repositories (`sentinel-soil` and `soil-moisture`). When a pipeline reaches validated maturity, its **full inference path** — feature extraction from Sentinel Hub, model application, post-processing — is distilled into a stable, pip-installable subpackage: `sentinel-soil-inference` and `soil-moisture-inference`. `ml-soil` depends on these subpackages by pinned version. `ml-soil` itself owns only the composition / registry layer, the zoning logic (clustering, placement, heterogeneity), the FastAPI surface, and the internal feature cache. It does not call Sentinel Hub directly, does not load model artifacts, and does not compute features.

The immediate research goals: produce per-tile texture and moisture predictions over a field polygon (Phase 0 heterogeneity scoring, Phase 1 sensor placement), serving an experimental deployment of 5 IoT moisture sensors on a ~10 ha irrigated field. The longer-term goal: closed-loop validation comparing placement strategies from both pipelines against measured sensor data.

## Scope & Non-Scope

**In scope:**
- Stateless inference API: given a polygon, return per-tile predictions or zoning outputs.
- Composition of texture and moisture inference packages under a uniform `Pipeline` protocol.
- Plug-in registry that allows new pipelines (e.g., `texture_v2`, future moisture variants) to be added in days, not weeks.
- Property prediction endpoints (texture and moisture features per tile).
- Zoning endpoints (heterogeneity, placement) parameterized by feature pipeline.
- Internal feature cache to make zoning re-runs interactive.

**Out of scope:**
- Training. Lives in `sentinel-soil` (texture) and `soil-moisture` (moisture).
- Feature extraction. Lives in the respective inference subpackages.
- Sentinel Hub interaction. Lives in the inference subpackages (or a shared client package — see Open Questions).
- Model artifact management. Internal to the inference subpackages.
- Research / experimental code. Lives in source repositories until distilled.
- User authentication, polygon storage, sensor data, job persistence. `terraOS` owns these.
- Job queue / worker pool. Synchronous API for typical field sizes; async only if measured to be needed.
- Decision-quality validation (Phase 4, requires sensor data).

## Key Architectural Decisions

**Inference packages own end-to-end inference.** Each pipeline's source repository publishes a pip-installable subpackage that owns the full inference path: feature extraction, Sentinel Hub access, model loading, model application, post-processing. `ml-soil` does not duplicate any of this. It calls one function per pipeline.

**Composition over reimplementation.** `ml-soil` does not duplicate logic that lives in `sentinel-soil` or `soil-moisture`. It composes the validated inference packages they publish.

**Distillation discipline.** When a pipeline matures in its source repo, the validated inference path — and only that path, including feature extraction — is published as a versioned subpackage. `ml-soil` pins to specific versions. Research churn in source repositories does not propagate to `ml-soil`.

**Plug-in is the primary engineering requirement.** Every architectural decision is evaluated against: *can a new pipeline be added in roughly one week without touching anything outside `core/pipelines/`?* If the answer is no, the abstraction is wrong.

**Stateless API, polygon inline.** `ml-soil` has no database connection. The polygon is passed inline as GeoJSON in every request, with `field_id` as an opaque identifier for cache keys, provenance, and logs. This decouples `ml-soil` from `terraOS`'s Postgres entirely.

**Internal feature cache.** Heavy work (the full inference call into a subpackage) is cached so re-running zoning with different parameters is sub-second. Cache key: `(field_id, pipeline_id, config_hash, time_window_hash)`. The cache stores `list[FeatureVector]`, not raw Sentinel Hub responses (those are an internal concern of the subpackage).

**Synchronous v1, async only when needed.** No SKIP LOCKED queue. Async returns `job_id` only if measured to be needed.

**Pipeline-agnostic downstream.** The zoning, placement, clustering, and heterogeneity layers operate exclusively on `list[FeatureVector]`. They have no knowledge of which pipeline produced their input.

**Versioned pipelines coexist.** `texture_v1` and `texture_v2` are registered side-by-side. The caller picks via API parameter. Pipelines are versioned, never replaced in place.

## The Plug-in Mechanism

The single most important interface in the codebase. Two pieces.

**The `Pipeline` protocol.** Every feature-generating pipeline implements this:

```python
class FeatureVector(BaseModel):
    tile_id: str
    coords: tuple[float, float]      # centroid (lat, lon), EPSG:4326
    features: dict[str, float]       # named features
    metadata: dict[str, Any]         # n_obs, coverage, uncertainty, etc.

class Pipeline(Protocol):
    pipeline_id: str                 # e.g. "texture_v1", "moisture_v1"
    feature_schema: list[str]        # which keys appear in features

    def generate(
        self,
        polygon: Polygon,
        time_window: tuple[date, date],
        config: dict,
    ) -> list[FeatureVector]: ...
```

**The registry.** A module-level dict, populated at import time:

```python
# ml_soil/core/pipelines/__init__.py
REGISTRY: dict[str, Pipeline] = {
    "texture_v1": TextureV1Pipeline(),
    "moisture_v1": MoistureV1Pipeline(),
}

def get_pipeline(pipeline_id: str) -> Pipeline:
    if pipeline_id not in REGISTRY:
        raise UnknownPipelineError(pipeline_id)
    return REGISTRY[pipeline_id]
```

A concrete pipeline class is now a thin adapter around a published subpackage:

```python
# ml_soil/core/pipelines/texture_v1.py
from sentinel_soil_inference import predict_texture_v1, PIPELINE_ID, FEATURE_SCHEMA

class TextureV1Pipeline:
    pipeline_id = PIPELINE_ID
    feature_schema = FEATURE_SCHEMA

    def generate(self, polygon, time_window, config):
        raw = predict_texture_v1(polygon, time_window, config)
        return [FeatureVector(**r) for r in raw]
```

Adding a new pipeline:
1. The source repo publishes (or extends) its inference subpackage with a new entry point.
2. `ml-soil` bumps the package pin in `pyproject.toml` and writes a thin adapter (5–10 lines).
3. Register the adapter in the pipeline registry.

**The hard invariant:** no code outside `core/pipelines/` references any specific `pipeline_id` by name. If you find an `if pipeline_id == "texture_v1":` anywhere else, the abstraction has leaked and must be refactored before continuing.

The same registry pattern applies to placement methods and clustering algorithms — three small registries, same shape.

## Pipelines

### texture_v1 (build now)

- **Origin:** trained by `sentinel-soil`; inference path packaged as `sentinel-soil-inference` (pip-installable, pinned by `ml-soil`).
- **Integration:** thin adapter in `ml-soil` that satisfies the `Pipeline` protocol by calling `sentinel_soil_inference.predict_texture_v1(...)`.
- **What lives in the package:** tiling logic, Sentinel Hub client and evalscript, SCL+NDVI filter, temporal aggregation, model loading from artifact bundle, sklearn application, compositional closure, output shaping.
- **What lives in ml-soil:** the adapter (5–10 lines), the registry entry. Nothing else.

Feature parity with training, evalscript management, artifact loading, and manifest validation are all internal concerns of `sentinel-soil-inference`. `ml-soil` is not aware of them.

### moisture_v1 (integrate)

- **Origin:** `soil-moisture` repository, near Phase 4 maturity. OPTRAM + trapezoid method.
- **Integration:** packaged as `soil-moisture-inference` (pip-installable, pinned by `ml-soil`); thin adapter in `ml-soil`.
- **What lives in the package:** wetting-event detection (if applicable), Sentinel Hub queries, OPTRAM / trapezoid computation, decay fitting, feature aggregation.
- **What lives in ml-soil:** the adapter, the registry entry.

The exact feature schema is owned by `soil-moisture-inference`. `ml-soil` does not enforce a schema; it consumes whatever the package declares in `FEATURE_SCHEMA`. The zoning layer is feature-name-agnostic.

### texture_v2 (planned, ~1 week to integrate when ready)

- **Origin:** sklearn models trained by `sentinel-soil` on Process API raster data. In development.
- **Integration:** same shape as `texture_v1`. `sentinel-soil-inference` extends to export a `predict_texture_v2` entry point (Process API internally — raster fetch, per-pixel prediction, aggregation to tiles), or publishes a new major version, depending on the discipline established in `sentinel-soil`'s versioning.
- **Expected effort once the package ships v2:** write the adapter (5–10 lines), register, ship. Roughly one week including integration tests.

## Zoning Layer

The placement and heterogeneity logic consumes `list[FeatureVector]` and is unaware of which pipeline produced it.

### Heterogeneity metrics (for `/zoning/heterogeneity`)

Per-feature dispersion (variance, CV), effective number of zones via eigengap on the feature covariance, Moran's I for spatial autocorrelation, summary statistics. Output is a per-field metric breakdown plus a cross-field comparative ranking. Used for experimental site selection (Phase 0). See METHODS.md for the methodology and literature.

### Placement methods (for `/zoning/placement`)

Same registry pattern as pipelines. v1 methods:

- `representative` — Cluster on standardized features (K-means, k ∈ {2,3,4,5}, silhouette-based suggestion). Within each cluster, pick the tile closest to the centroid in standardized feature space. Optional 3×3 modal filter for spatial regularization.
- `maximin` — Farthest-point sampling in standardized feature space. No clustering intermediate.
- `stratified` — K-means clustering, then uniform random selection within each cluster (seeded). Baseline.

**Log every candidate.** `all_candidates` in `PlacementResult` carries the full ranked candidate set, not just the selected N. This is the data asset that enables retrospective method comparison in Phase 4.

GP-based methods (kriging variance, mutual information maximization) are not implemented in v1: they are ill-posed at N = small because the prior dominates the data.

## API Surface (v1, overview)

The full HTTP contract spec is in INTEGRATION.md. Brief overview:

- `POST /predict/texture` — per-tile texture predictions for a polygon.
- `POST /predict/moisture` — per-tile moisture features for a polygon.
- `POST /zoning/heterogeneity` — per-field heterogeneity metrics + cross-field ranking.
- `POST /zoning/placement` — per-method placement results with full candidate logs.
- `GET /health` — liveness + loaded pipeline IDs + package versions.
- `POST /zoning/score_placement` — Phase 4, specified but not built.

All requests carry `polygon` (GeoJSON) inline. All responses carry full provenance.

## Cross-repo Contracts (Overview)

`ml-soil` interfaces with three other repositories. Full specifications in INTEGRATION.md.

- **terraOS ↔ ml-soil — HTTP API.** terraOS holds polygons; ml-soil receives them inline. Internal API key in v1.
- **sentinel-soil → ml-soil — Distilled Python subpackage** (`sentinel-soil-inference`). ml-soil pins to a specific version. Identical pattern to soil-moisture-inference.
- **soil-moisture → ml-soil — Distilled Python subpackage** (`soil-moisture-inference`). ml-soil pins to a specific version.

## Phasing

- **Phase 0** — Cross-field heterogeneity scoring across candidate polygons. Independent value; ships first.
- **Phase 1** — Placement on the instrumented field with multiple methods returned for comparison. Builds directly on Phase 0 components.
- **Phase 2** — Sensor deployment + irrigation log + weather data ingest. Owned by terraOS; ml-soil provides `/zoning/score_placement`.
- **Phase 3** — `moisture_v1` integrated via `soil-moisture-inference`. Retrospective placement on the instrumented field. Method comparison.
- **Phase 4** — Closed-loop validation using measured moisture. Representation metrics first; decision metrics after a season of crop data.

## Module Layout

```
ml-soil/
  api/
    routes/
      predict.py            # /predict/texture, /predict/moisture
      zoning.py             # /zoning/heterogeneity, /zoning/placement
      health.py
    schemas.py              # Pydantic request/response models
  core/
    pipelines/
      __init__.py           # REGISTRY, get_pipeline
      base.py               # Pipeline protocol, FeatureVector
      mock.py               # MockPipeline for end-to-end testing
      texture_v1.py         # thin adapter around sentinel-soil-inference
      texture_v2.py         # stub until package ships v2
      moisture_v1.py        # thin adapter around soil-moisture-inference
    placement/
      __init__.py           # REGISTRY of methods
      base.py
      representative.py
      maximin.py
      stratified.py
    clustering/
      kmeans.py
      smoothing.py
    heterogeneity/
      metrics.py
      report.py
    features/
      cache.py
      tiling.py              # ONLY if ml-soil ever needs to tile (probably not — packages own this)
  config/
    settings.py
  tests/
  Dockerfile
  pyproject.toml             # pins sentinel-soil-inference, soil-moisture-inference
  README.md
  CONTEXT.md
  INTEGRATION.md
```

Notably absent compared to earlier drafts: no `sentinel_hub/` directory, no `artifacts/` directory. Both are concerns of the inference subpackages, not `ml-soil`.

**Library, CLI, and API are layered.** `core/` is independently usable from a notebook. `api/` is a thin wrapper over `core/`. The HTTP layer is the last thing built.

## Build Sequence (v1)

Front-load abstractions. The reason: if the abstractions don't hold, every concrete pipeline pays for it. If they hold, plugging in `texture_v2` and any future moisture variant is mostly mechanical.

1. **Repo skeleton + the abstractions.** `Pipeline` protocol, `FeatureVector` type, pipeline registry, placement-method registry, cache interface. A `MockPipeline` that returns synthetic feature vectors for any polygon. No external packages, no Sentinel Hub, no models. Tests against the mock.

2. **Wire everything end-to-end against the mock.** Heterogeneity metrics, K-means clustering, three placement methods, response assembly. The mock-driven integration test produces a complete `PlacementResult` and a complete `HeterogeneityReport`. *Invariant check:* no code outside `core/pipelines/mock.py` references the string `"mock"`. If it does, the abstraction has leaked.

3. **`TextureV1Pipeline` adapter** (once `sentinel-soil-inference` is publishable). Pin the package in `pyproject.toml`. Write the 5–10 line adapter. Register. Re-run the end-to-end test with `pipeline="texture_v1"`. Nothing outside the new adapter file should need to change.

4. **`MoistureV1Pipeline` adapter** (once `soil-moisture-inference` is publishable). Pin, adapt, register, re-run.

5. **FastAPI wrap.** Pydantic schemas, endpoint handlers, basic auth. Synchronous responses.

6. **Docker + integration.** Containerize, end-to-end test against a real field via terraOS.

Steps 1–2 verify the architecture before any external dependencies are wired — this is the part that actually pays off the "1 week to plug in" claim, and it can be completed before `sentinel-soil` or `soil-moisture` have published their subpackages. Steps 3–4 are gated on those packages being ready; both should take days rather than weeks given the thin-adapter pattern.

## Key Principles

- **Inference packages own end-to-end inference.** `ml-soil` is a composition + zoning service, not an inference engine.
- **Composition over reimplementation.** `ml-soil` does not duplicate logic that lives in `sentinel-soil` or `soil-moisture`.
- **Distillation over direct import.** Only validated inference paths cross repo boundaries, as versioned subpackages.
- **Plug-in is non-negotiable.** No code outside `core/pipelines/` knows the name of any specific pipeline. Same for placement methods.
- **Pipeline contract is the most important interface.** Adding a new pipeline must not require touching the placement, clustering, or API layers.
- **Log every candidate, not just the selected.** Retrospective method comparison requires the full candidate set with scores.
- **Named features over positional vectors.** Self-documenting, mix-friendly, cache-friendly.
- **Versioned pipelines coexist.** Never replace in place.
- **Provenance on every response.** Pipeline ID, package versions, config hash, timestamp.
- **Polygon inline, not server-side lookup.** `ml-soil` has no database connection.
- **Library + CLI + API, in that order of primacy.** Research workflows need notebook access to intermediate outputs.
- **No premature scaling.** No queue, no workers, no horizontal scaling until measured.

## Tools & Resources

- **Language:** Python 3.11+
- **Web framework:** FastAPI + Pydantic
- **Geospatial:** `shapely`, `pyproj` (input handling; tile-grid generation is internal to inference packages)
- **Storage (cache):** Scaleway S3 in deployed environments; local filesystem in dev
- **Containerization:** Docker
- **Testing:** `pytest`
- **Pinned subpackage dependencies:** `sentinel-soil-inference`, `soil-moisture-inference`

Note: `ml-soil` does not directly depend on `scikit-learn`, `sentinelhub-py`, or any other ML / RS library. Those are concerns of the inference subpackages.

## Open Questions / Deferred Decisions

- **Sentinel Hub client ownership.** Both inference packages need a Sentinel Hub client. Three options: each brings its own (bad), `ml-soil` owns and injects (awkward), or a shared `sentinel-hub-client` package both depend on (cleanest). The current de facto situation is that `soil-moisture` reaches into `sentinel-soil` for the client; formalizing this into a separate package is the proposed direction.
- **Inference package structure.** Subdirectory of the source repo (`sentinel-soil/packages/sentinel_soil_inference/`) or separately published package? Subdirectory is simpler in early phases; separate package becomes attractive when consumers other than `ml-soil` appear.
- **`FeatureVector` location.** Currently in `ml-soil`. If shared types proliferate, extract to a `soil-types` package that the inference packages and `ml-soil` all depend on.
- **Cache backend abstraction.** Local filesystem in dev; S3 once deployed. Interface abstracts this.
- **Async pattern.** Build only when measured to be needed.
- **Polygon area hard limit.** Defensive cap in `ml-soil` (e.g., 100 ha) on top of `terraOS`'s 50 ha policy.
- **Composite heterogeneity score weights.** Default weights are a guess; tunable in config.

## What This Service Does Not Do

For clarity when collaborators approach this:

- It does not own feature extraction. Inference packages do.
- It does not call Sentinel Hub directly. Inference packages do.
- It does not load model artifacts. Inference packages do.
- It does not store user data. Polygons arrive in requests.
- It does not authenticate users. terraOS does.
- It does not store sensor data. terraOS does.
- It does not own the UI. Outputs are JSON; visualization is terraOS's responsibility.
- It does not train models. Training lives in `sentinel-soil` and `soil-moisture`.
- It does not import research / development code from source repositories. Only validated, versioned subpackages.
- It does not orchestrate long-running jobs in v1. If ever needed, returns a `job_id` and exposes a poll endpoint; terraOS handles user-facing polling.