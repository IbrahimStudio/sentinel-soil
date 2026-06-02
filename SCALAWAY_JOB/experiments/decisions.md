# Design Decisions Log

Decisions that shaped the v2 (Process API) pipeline. Recorded so future experiments know what was intentional vs. accidental, and what the tradeoffs were.

---

## D-001 · Fetch 9×9 raster, extract only center pixel for baseline

**Status:** Active (exp-001 baseline)  
**Date:** 2026-06-02

**Decision:** The ingestion fetches a 9×9 pixel patch (90×90 m at 10 m resolution) centered on each LUCAS point. For the baseline feature extraction, only the single center pixel `[4, 4]` is used.

**Rationale:**
- The LUCAS field protocol measures soil at a point, not over an area. The center pixel is the closest spatial match to what was sampled.
- Fetching a 9×9 patch costs the same API quota as 3×3. Having the surrounding pixels stored now means we can run neighbourhood experiments (D-002) without re-fetching.
- The center pixel may be noisy due to sub-pixel misregistration (~10 m GPS error is common in LUCAS). Future experiments will test whether aggregating over the inner 3×3 or full 9×9 improves robustness.

**Consequences:**
- Single-pixel medians are noisier than patch-level aggregations. Expect lower R² for Sand/Coarse which have high spatial heterogeneity.
- If a LUCAS point falls on a field boundary, the center pixel may be mixed. The 3×3 neighborhood experiment (exp-002) will test sensitivity to this.

---

## D-002 · No seasonal filtering in baseline

**Status:** Active (exp-001 baseline)  
**Date:** 2026-06-02

**Decision:** The baseline uses all orbit passes from 2017-01-01 to 2026-05-31 (no `--season-months` filter), then applies pixel-level bare-soil filtering via SCL+NDVI+NBR2 at the feature extraction step.

**Rationale:**
- The full-archive rasters are already fetched with no seasonal restriction. Re-running ingestion just for seasonally filtered data wastes quota.
- The pixel filter (NDVI < 0.25, NBR2 < 0.125) acts as an implicit seasonal filter: bare soil is most visible in autumn/winter in temperate Europe, so valid observations cluster naturally in those months.
- The `filter_config.json` `min_valid_observations_per_pixel: 3` ensures a point is only used if it has at least 3 clear bare-soil observations.
- An explicit Oct–Apr seasonal mask (exp-003) will test whether restricting to known bare-soil months improves signal at the cost of coverage (some points may drop below min_obs).

**Consequences:**
- A point surveyed in July could have very few bare-soil observations if its surrounding dates are vegetated. The `n_valid_dates` column in the feature table will diagnose this.
- Summer observations of (temporarily) bare fields (post-harvest) will be included. These may be spectrally different from winter bare soil and add noise.

---

## D-003 · Temporal median as aggregation

**Status:** Active (exp-001 baseline)  
**Date:** 2026-06-02

**Decision:** All valid bare-soil observations at the center pixel are collapsed to a single per-band median. This produces 10 band medians + NDVI + NDMI = 12 scalar features per point.

**Rationale:**
- Median is robust to remaining outliers (thin cloud, haze) that pass the SCL/NDVI/NBR2 filter.
- Consistent with the v1 Statistics API pipeline (which used P50 per-band), making the two pipelines directly comparable.
- A single feature vector per point keeps the training dataset simple and avoids temporal alignment problems.

**Consequences:**
- All temporal information is discarded. Seasonal variation patterns (spring vs. autumn spectra) that may correlate with soil type are lost. This is a hypothesis to test: exp-004 (seasonal split medians) will check whether separate summer/winter medians add predictive power.
- The median across N dates has lower variance than any individual observation, but points with few valid dates (n_valid_dates = 3–5) are still noisy.
- **exp-001 finding:** the 9-year archive median performed *worse* than v1's ±365-day survey-date window (Clay R² 0.18 vs 0.28). Averaging over 9 years of changing land use/management smooths away the soil signal. A survey-date-centred window (or seasonal filter) is likely necessary to recover it.

---

## D-004 · No geographic features in baseline

**Status:** Active (exp-001 baseline)  
**Date:** 2026-06-02

**Decision:** The baseline feature vector contains only spectral features (no lat/lon/elevation).

**Rationale:**
- The v1 spectral-only baseline (Statistics API, R² Clay 0.278) is the appropriate comparison point: same information content, different data source (Process API vs Statistics API, 9-year full archive vs per-survey-date window).
- Including coordinates would mask whether the spectral signal improved, since v1 already showed coords add ~0.13 R² for Clay.
- Once the spectral-only v2 baseline is established, adding coords is a one-line change in training and will be exp-005.

**Consequences:**
- Expected R² will be in the 0.25–0.35 range based on v1 precedent. Sand and Coarse will likely remain weak without geographic context.
