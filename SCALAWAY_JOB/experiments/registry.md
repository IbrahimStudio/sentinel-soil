# Experiment Registry — Process API Pipeline (v2)

Each row is a reproducible run. Add a row when you start a new experiment; fill in results when done.

---

## Quick reference

| ID | Name | Status | Clay R² | Silt R² | Sand R² | Coarse R² |
|----|------|--------|---------|---------|---------|-----------|
| **v1-A** | Stats API · spectral only | ✅ done | 0.278 | 0.295 | 0.072 | 0.045 |
| **v1-B** | Stats API · spectral + coords | ✅ done | 0.406 | 0.392 | 0.240 | 0.239 |
| **exp-001** | Process API · center pixel · temporal median | ✅ done | 0.162 | 0.146 | 0.027 | 0.033 |
| **exp-002** | Process API · 3×3 patch median | planned | — | — | — | — |
| **exp-003** | exp-001 + season filter Oct–Apr | planned | — | — | — | — |
| **exp-004** | exp-001 + seasonal split medians | planned | — | — | — | — |
| **exp-005** | exp-001 + lat/lon | planned | — | — | — | — |
| **exp-006** | exp-001 + n_valid_dates as feature | planned | — | — | — | — |
| **exp-007** | exp-002 + coords (best of 001–006) | planned | — | — | — | — |

CV scores shown are **spatial GroupKFold** (20 km blocks) — the conservative estimate.  
v1-A/v1-B used Statistics API (18 p50 features over per-survey ±365-day windows); exp-001+ use Process API (9-year full archive, center-pixel medians).

---

## exp-001 · Process API baseline — center pixel, temporal median

**Goal:** Establish a clean v2 spectral-only baseline comparable to v1-A.  
**Decision refs:** D-001, D-002, D-003, D-004

| Parameter | Value |
|-----------|-------|
| Raster source | `soil-sentinel/raw_rasters_v2/` (788 points) |
| Spatial extraction | Center pixel `[4,4]` of 9×9 grid |
| Temporal window | 2017-01-01 – 2026-05-31 (full archive, no season filter) |
| Pixel filter | SCL ∈ {4,5}, NDVI ∈ (−0.1, 0.25), NBR2 < 0.125 |
| Min valid observations | 3 |
| Aggregation | Temporal median per band |
| Features | B02–B12 medians (10) + NDVI + NDMI = **12 features** |
| Geographic features | None |
| Models | RF (3 configs), HGB (2 configs), ElasticNet (2 configs) |
| CV | 5-fold RandomKFold + 5-fold SpatialGroupKFold (20 km blocks) |
| Training targets | Clay, Silt, Sand, Coarse |
| filter_config hash | _(filled by training job: see `features.json`)_ |
| Run timestamp | _(filled when done)_ |
| Report path | `training/reports/run_<ts>/summary_metrics.csv` |

**Results (spatial GroupKFold CV):**

| Target | Best model | R² | RMSE | MAE |
|--------|-----------|-----|------|-----|
| Clay | enet/a0.01_l10.2 | 0.162 | 11.08 | 9.05 |
| Silt | rf/n800_leaf4_log2 | 0.146 | 10.54 | 8.33 |
| Sand | enet/a0.05_l10.5 | 0.027 | 17.69 | 14.80 |
| Coarse | enet/a0.01_l10.2 | 0.033 | 12.50 | 9.79 |

Random KFold: Clay 0.180 / Silt 0.149 / Sand 0.030 / Coarse 0.033.

**Notes:**
- **Worse than v1-A across all targets** — surprising given the 9-year archive.
- **ElasticNet wins Clay/Sand/Coarse** (RF only wins Silt). This is the opposite of v1 where RF dominated everything. Likely caused by: (a) only 554 points (vs ~1100 in v1), (b) full 9-year median averages across very different temporal soil conditions, reducing non-linear structure that RF could exploit.
- **554 of 788 ingested points extracted** — 234 dropped for < 3 valid bare-soil dates at the center pixel. These are likely persistently vegetated or cloudy locations.
- Sanity check flag: the 4-fraction sum check was incorrect (Coarse is gravel, not fine earth) — fixed in `process_api_extractor.py`.
- Models written to `DSM_WEBAPP/models/` (clay.pkl, silt.pkl, sand.pkl, coarse.pkl). Report at `training/reports/run_20260602_154443/`.

---

## exp-002 · 3×3 patch median (planned)

**Goal:** Test whether averaging over the inner 3×3 pixels reduces center-pixel noise and improves R², especially for Sand/Coarse.

**Change from exp-001:** In `process_api_extractor.py`, replace center-pixel extraction with a spatial median over pixels `[3:6, 3:6]` (the inner 3×3 of the 9×9 grid) before the temporal median. All filter thresholds unchanged.

**Hypothesis:** LUCAS GPS error (~10 m) can shift the true sample location by up to one pixel. Averaging the inner 3×3 should be robust to that. Downside: field boundaries may introduce mixed pixels at the edge of the 3×3.

**Expected effort:** ~20 min change in `process_api_extractor.py` + re-run feature store + training. No new ingestion needed.

---

## exp-003 · Season filter Oct–Apr (planned)

**Goal:** Test whether restricting to the bare-soil season (Oct–Apr) improves signal quality by eliminating summer observations (post-harvest stubble, green cover).

**Change from exp-001:** Pass `--season-months 10,11,12,1,2,3,4` to ingestion. But since rasters are already fetched with all months, apply a date mask in feature extraction instead: filter `dates` array to months ∈ {10,11,12,1,2,3,4} before computing the median.

**Risk:** Some points (southern Europe, dryland) may drop below `min_valid_observations = 3` with the seasonal restriction. Log how many points are lost.

---

## exp-004 · Seasonal split medians (planned)

**Goal:** Test whether separating autumn (Sep–Nov), winter (Dec–Feb), and spring (Mar–May) medians provides complementary spectral information — e.g. moisture differences between seasons that correlate with texture.

**Change from exp-001:** Compute three separate per-band medians (autumn/winter/spring), concatenate → 3 × 12 = 36 features. Points with < 2 valid observations in any season are assigned NaN for that season; the pipeline's `SimpleImputer(strategy="median")` fills these at training time.

**Hypothesis:** Winter bare-soil spectra reflect soil moisture regime (clay soils stay wetter longer), while spring spectra capture surface crust formation (silty soils). The three-season representation should lift Clay and Silt R².

---

## exp-005 · Add lat/lon (planned)

**Goal:** Replicate the v1-B result (spectral + coords) on v2 features to check whether geographic signal is consistent across pipelines.

**Change from exp-001:** Add `TH_LAT` and `TH_LONG` to the feature vector in `main_v2.py` (one line: extend `FEATURE_NAMES` and ensure the columns flow through from the parquet).

**Expected outcome:** R² lift of ~0.10–0.15 for Clay/Silt (based on v1 precedent). If the lift is smaller it suggests the v2 spectral features already capture more geographic structure than v1.

---

## exp-006 · n_valid_dates as feature (planned)

**Goal:** Test whether the number of valid bare-soil observations is a useful predictor. Points with more observations (e.g. dryland, sparse vegetation) may be in arid regions with characteristic soil texture.

**Change from exp-001:** `n_valid_dates` is already written to the parquet by the feature extractor. Add it to `FEATURE_NAMES` in `main_v2.py`.

**Risk:** `n_valid_dates` encodes geography (climate, land use) and may act as a proxy for coordinates — inflating R² without adding interpretable spectral signal. Check SHAP values to verify.

---

## exp-007 · Best combination (planned)

**Goal:** After exp-002 through exp-006, assemble the best-performing combination of spatial extraction strategy, seasonal features, and geographic features.

**Design:** TBD after reviewing exp-001 through exp-006 results.
