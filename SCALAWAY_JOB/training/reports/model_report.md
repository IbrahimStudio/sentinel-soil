# Soil Texture Model Report
**Generated:** 2026-05-03  
**Pipeline version:** v2 (`soil_texture_pipeline_v2.py`)  
**Dataset:** ~1 100 LUCAS survey points, 18 spectral features (`p50_*`)  
**CV strategies:** RandomKFold (5-fold) and SpatialGroupKFold (5-fold, 20 km blocks)

---

## Run history

| Run ID | Features used | Chemistry dropped | Coords/Elev |
|--------|--------------|-------------------|-------------|
| `run_20260418_*` | spectral + chemistry + coords | ✗ | ✓ (leaked) |
| `run_20260503_125342` | spectral only | ✓ | ✗ |
| `run_20260503_130423` | spectral + coords/elev | ✓ | ✓ |

> **Note on Apr 18 runs:** those results are inflated. Chemistry columns (`pH`, `OC`, `CaCO3`, `N`, `K`, etc.) were included as predictors — they are lab measurements from the same samples and directly leak ground-truth information. The May 03 runs are the correct baselines.

---

## Summary of best models

### Spectral-only — run_20260503_125342
18 `p50_*` spectral features. No coordinates, no chemistry, no coverage metadata.

| Target | Best model | CV strategy | R² (mean) | RMSE (mean) | Weights |
|--------|-----------|-------------|-----------|-------------|---------|
| **Clay** | `rf / n1200_leaf1_sqrt` | RandomKFold | **0.278** | 9.90 | `run_20260503_125342/pipeline__Clay__rf__n1200_leaf1_sqrt__RandomKFold.joblib` |
| **Clay** | `rf / n800_leaf2_sqrt` | SpatialGroupKFold | **0.276** | 9.94 | `run_20260503_125342/pipeline__Clay__rf__n800_leaf2_sqrt__SpatialGroupKFold.joblib` |
| **Silt** | `rf / n800_leaf2_sqrt` | RandomKFold | **0.295** | 10.24 | `run_20260503_125342/pipeline__Silt__rf__n800_leaf2_sqrt__RandomKFold.joblib` |
| **Silt** | `rf / n800_leaf2_sqrt` | SpatialGroupKFold | **0.290** | 10.29 | `run_20260503_125342/pipeline__Silt__rf__n800_leaf2_sqrt__SpatialGroupKFold.joblib` |
| **Sand** | `rf / n800_leaf2_sqrt` | RandomKFold | **0.072** | 16.82 | `run_20260503_125342/pipeline__Sand__rf__n800_leaf2_sqrt__RandomKFold.joblib` |
| **Sand** | `rf / n1200_leaf1_sqrt` | SpatialGroupKFold | **0.070** | 16.90 | `run_20260503_125342/pipeline__Sand__rf__n1200_leaf1_sqrt__SpatialGroupKFold.joblib` |
| **Coarse** | `rf / n800_leaf4_log2` | RandomKFold | **0.045** | 12.54 | `run_20260503_125342/pipeline__Coarse__rf__n800_leaf4_log2__RandomKFold.joblib` |
| **Coarse** | `rf / n800_leaf4_log2` | SpatialGroupKFold | **0.067** | 12.42 | `run_20260503_125342/pipeline__Coarse__rf__n800_leaf4_log2__SpatialGroupKFold.joblib` |

### Spectral + Coords/Elev — run_20260503_130423
18 `p50_*` spectral features + `lat`, `lon`, `TH_LAT`, `TH_LONG`, `Elev`. Chemistry and coverage metadata excluded.

| Target | Best model | CV strategy | R² (mean) | RMSE (mean) | Weights |
|--------|-----------|-------------|-----------|-------------|---------|
| **Clay** | `rf / n1200_leaf1_sqrt` | RandomKFold | **0.406** | 8.96 | `run_20260503_130423/pipeline__Clay__rf__n1200_leaf1_sqrt__RandomKFold.joblib` |
| **Clay** | `rf / n1200_leaf1_sqrt` | SpatialGroupKFold | **0.391** | 9.11 | `run_20260503_130423/pipeline__Clay__rf__n1200_leaf1_sqrt__SpatialGroupKFold.joblib` |
| **Silt** | `rf / n1200_leaf1_sqrt` | RandomKFold | **0.392** | 9.51 | `run_20260503_130423/pipeline__Silt__rf__n1200_leaf1_sqrt__RandomKFold.joblib` |
| **Silt** | `rf / n1200_leaf1_sqrt` | SpatialGroupKFold | **0.388** | 9.55 | `run_20260503_130423/pipeline__Silt__rf__n1200_leaf1_sqrt__SpatialGroupKFold.joblib` |
| **Sand** | `rf / n800_leaf2_sqrt` | RandomKFold | **0.240** | 15.23 | `run_20260503_130423/pipeline__Sand__rf__n800_leaf2_sqrt__RandomKFold.joblib` |
| **Sand** | `rf / n1200_leaf1_sqrt` | SpatialGroupKFold | **0.229** | 15.38 | `run_20260503_130423/pipeline__Sand__rf__n1200_leaf1_sqrt__SpatialGroupKFold.joblib` |
| **Coarse** | `rf / n800_leaf2_sqrt` | RandomKFold | **0.239** | 11.20 | `run_20260503_130423/pipeline__Coarse__rf__n800_leaf2_sqrt__RandomKFold.joblib` |
| **Coarse** | `rf / n800_leaf2_sqrt` | SpatialGroupKFold | **0.223** | 11.34 | `run_20260503_130423/pipeline__Coarse__rf__n800_leaf2_sqrt__SpatialGroupKFold.joblib` |

All paths are relative to `SCALAWAY_JOB/training/reports/`.

---

## Three-way comparison (best R² per target/CV)

| Target | CV | Spectral-only | Spectral+Coords | Apr 18 (leaked) |
|--------|-----|:---:|:---:|:---:|
| Clay | Random | 0.278 | **0.406** | ~~0.561~~ |
| Clay | Spatial | 0.276 | **0.391** | ~~0.547~~ |
| Silt | Random | 0.295 | **0.392** | ~~0.501~~ |
| Silt | Spatial | 0.290 | **0.388** | ~~0.498~~ |
| Sand | Random | 0.072 | **0.240** | ~~0.408~~ |
| Sand | Spatial | 0.070 | **0.229** | ~~0.388~~ |
| Coarse | Random | 0.045 | **0.239** | ~~0.252~~ |
| Coarse | Spatial | 0.067 | **0.223** | ~~0.235~~ |

---

## Full results by target

### Clay

#### Spectral-only (run_20260503_125342)

| Model | Config | CV | R² | RMSE |
|-------|--------|----|----|------|
| rf | n1200_leaf1_sqrt | random | 0.278 | 9.90 |
| rf | n800_leaf2_sqrt | random | 0.277 | 9.90 |
| rf | n800_leaf4_log2 | random | 0.277 | 9.90 |
| enet | a0.01_l10.2 | random | 0.207 | 10.38 |
| enet | a0.05_l10.5 | random | 0.189 | 10.50 |
| hgb | lr0.05_iter800 | random | 0.147 | 10.75 |
| hgb | lr0.03_iter1200 | random | 0.138 | 10.80 |
| rf | n800_leaf2_sqrt | spatial | 0.276 | 9.94 |
| rf | n800_leaf4_log2 | spatial | 0.276 | 9.94 |
| rf | n1200_leaf1_sqrt | spatial | 0.275 | 9.95 |
| enet | a0.01_l10.2 | spatial | 0.210 | 10.38 |
| enet | a0.05_l10.5 | spatial | 0.193 | 10.49 |
| hgb | lr0.05_iter800 | spatial | 0.131 | 10.87 |
| hgb | lr0.03_iter1200 | spatial | 0.125 | 10.91 |

#### Spectral + Coords/Elev (run_20260503_130423)

| Model | Config | CV | R² | RMSE |
|-------|--------|----|----|------|
| rf | n1200_leaf1_sqrt | random | 0.406 | 8.96 |
| rf | n800_leaf2_sqrt | random | 0.404 | 8.98 |
| rf | n800_leaf4_log2 | random | 0.394 | 9.06 |
| hgb | lr0.05_iter800 | random | 0.329 | 9.51 |
| hgb | lr0.03_iter1200 | random | 0.323 | 9.56 |
| enet | a0.01_l10.2 | random | 0.241 | 10.16 |
| enet | a0.05_l10.5 | random | 0.231 | 10.22 |
| rf | n1200_leaf1_sqrt | spatial | 0.391 | 9.11 |
| rf | n800_leaf2_sqrt | spatial | 0.390 | 9.12 |
| rf | n800_leaf4_log2 | spatial | 0.379 | 9.20 |
| hgb | lr0.03_iter1200 | spatial | 0.343 | 9.46 |
| hgb | lr0.05_iter800 | spatial | 0.335 | 9.52 |
| enet | a0.01_l10.2 | spatial | 0.241 | 10.17 |
| enet | a0.05_l10.5 | spatial | 0.232 | 10.23 |

### Silt

#### Spectral-only (run_20260503_125342)

| Model | Config | CV | R² | RMSE |
|-------|--------|----|----|------|
| rf | n800_leaf2_sqrt | random | 0.295 | 10.24 |
| rf | n800_leaf4_log2 | random | 0.293 | 10.26 |
| rf | n1200_leaf1_sqrt | random | 0.293 | 10.26 |
| enet | a0.01_l10.2 | random | 0.261 | 10.49 |
| enet | a0.05_l10.5 | random | 0.249 | 10.58 |
| hgb | lr0.03_iter1200 | random | 0.192 | 10.95 |
| hgb | lr0.05_iter800 | random | 0.186 | 10.99 |
| rf | n800_leaf2_sqrt | spatial | 0.290 | 10.29 |
| rf | n800_leaf4_log2 | spatial | 0.289 | 10.30 |
| rf | n1200_leaf1_sqrt | spatial | 0.287 | 10.32 |
| enet | a0.01_l10.2 | spatial | 0.265 | 10.48 |
| enet | a0.05_l10.5 | spatial | 0.252 | 10.57 |
| hgb | lr0.03_iter1200 | spatial | 0.164 | 11.16 |
| hgb | lr0.05_iter800 | spatial | 0.150 | 11.26 |

#### Spectral + Coords/Elev (run_20260503_130423)

| Model | Config | CV | R² | RMSE |
|-------|--------|----|----|------|
| rf | n1200_leaf1_sqrt | random | 0.392 | 9.51 |
| rf | n800_leaf2_sqrt | random | 0.391 | 9.51 |
| rf | n800_leaf4_log2 | random | 0.384 | 9.57 |
| hgb | lr0.03_iter1200 | random | 0.341 | 9.88 |
| hgb | lr0.05_iter800 | random | 0.338 | 9.91 |
| enet | a0.01_l10.2 | random | 0.254 | 10.52 |
| enet | a0.05_l10.5 | random | 0.243 | 10.61 |
| rf | n1200_leaf1_sqrt | spatial | 0.388 | 9.55 |
| rf | n800_leaf2_sqrt | spatial | 0.388 | 9.56 |
| rf | n800_leaf4_log2 | spatial | 0.385 | 9.58 |
| hgb | lr0.03_iter1200 | spatial | 0.327 | 10.02 |
| hgb | lr0.05_iter800 | spatial | 0.320 | 10.07 |
| enet | a0.01_l10.2 | spatial | 0.257 | 10.53 |
| enet | a0.05_l10.5 | spatial | 0.246 | 10.61 |

### Sand

#### Spectral-only (run_20260503_125342)

| Model | Config | CV | R² | RMSE |
|-------|--------|----|----|------|
| rf | n800_leaf2_sqrt | random | 0.072 | 16.82 |
| rf | n1200_leaf1_sqrt | random | 0.072 | 16.82 |
| rf | n800_leaf4_log2 | random | 0.068 | 16.86 |
| enet | a0.01_l10.2 | random | 0.028 | 17.24 |
| enet | a0.05_l10.5 | random | 0.011 | 17.39 |
| hgb | lr0.05_iter800 | random | −0.072 | 18.08 |
| hgb | lr0.03_iter1200 | random | −0.082 | 18.17 |
| rf | n1200_leaf1_sqrt | spatial | 0.070 | 16.90 |
| rf | n800_leaf2_sqrt | spatial | 0.068 | 16.92 |
| rf | n800_leaf4_log2 | spatial | 0.068 | 16.92 |
| enet | a0.01_l10.2 | spatial | 0.037 | 17.21 |
| enet | a0.05_l10.5 | spatial | 0.022 | 17.35 |
| hgb | lr0.03_iter1200 | spatial | −0.078 | 18.19 |
| hgb | lr0.05_iter800 | spatial | −0.083 | 18.23 |

#### Spectral + Coords/Elev (run_20260503_130423)

| Model | Config | CV | R² | RMSE |
|-------|--------|----|----|------|
| rf | n800_leaf2_sqrt | random | 0.240 | 15.23 |
| rf | n1200_leaf1_sqrt | random | 0.236 | 15.26 |
| rf | n800_leaf4_log2 | random | 0.226 | 15.37 |
| hgb | lr0.05_iter800 | random | 0.146 | 16.11 |
| hgb | lr0.03_iter1200 | random | 0.142 | 16.16 |
| enet | a0.01_l10.2 | random | 0.045 | 17.08 |
| enet | a0.05_l10.5 | random | 0.032 | 17.21 |
| rf | n1200_leaf1_sqrt | spatial | 0.229 | 15.38 |
| rf | n800_leaf2_sqrt | spatial | 0.227 | 15.40 |
| rf | n800_leaf4_log2 | spatial | 0.221 | 15.46 |
| hgb | lr0.05_iter800 | spatial | 0.135 | 16.29 |
| hgb | lr0.03_iter1200 | spatial | 0.134 | 16.30 |
| enet | a0.01_l10.2 | spatial | 0.051 | 17.08 |
| enet | a0.05_l10.5 | spatial | 0.039 | 17.19 |

### Coarse

#### Spectral-only (run_20260503_125342)

| Model | Config | CV | R² | RMSE |
|-------|--------|----|----|------|
| rf | n800_leaf4_log2 | random | 0.045 | 12.54 |
| rf | n800_leaf2_sqrt | random | 0.037 | 12.59 |
| rf | n1200_leaf1_sqrt | random | 0.033 | 12.61 |
| enet | a0.01_l10.2 | random | 0.033 | 12.63 |
| enet | a0.05_l10.5 | random | 0.031 | 12.64 |
| hgb | lr0.03_iter1200 | random | −0.149 | 13.74 |
| hgb | lr0.05_iter800 | random | −0.153 | 13.77 |
| rf | n800_leaf4_log2 | spatial | 0.067 | 12.42 |
| rf | n800_leaf2_sqrt | spatial | 0.061 | 12.46 |
| rf | n1200_leaf1_sqrt | spatial | 0.057 | 12.48 |
| enet | a0.05_l10.5 | spatial | 0.030 | 12.66 |
| enet | a0.01_l10.2 | spatial | 0.029 | 12.66 |
| hgb | lr0.03_iter1200 | spatial | −0.148 | 13.76 |
| hgb | lr0.05_iter800 | spatial | −0.148 | 13.76 |

#### Spectral + Coords/Elev (run_20260503_130423)

| Model | Config | CV | R² | RMSE |
|-------|--------|----|----|------|
| rf | n800_leaf2_sqrt | random | 0.239 | 11.20 |
| rf | n1200_leaf1_sqrt | random | 0.238 | 11.21 |
| rf | n800_leaf4_log2 | random | 0.235 | 11.24 |
| hgb | lr0.03_iter1200 | random | 0.139 | 11.92 |
| hgb | lr0.05_iter800 | random | 0.130 | 11.98 |
| enet | a0.01_l10.2 | random | 0.105 | 12.15 |
| enet | a0.05_l10.5 | random | 0.104 | 12.15 |
| rf | n800_leaf2_sqrt | spatial | 0.223 | 11.34 |
| rf | n1200_leaf1_sqrt | spatial | 0.222 | 11.34 |
| rf | n800_leaf4_log2 | spatial | 0.220 | 11.36 |
| enet | a0.05_l10.5 | spatial | 0.109 | 12.14 |
| enet | a0.01_l10.2 | spatial | 0.109 | 12.14 |
| hgb | lr0.03_iter1200 | spatial | 0.106 | 12.16 |
| hgb | lr0.05_iter800 | spatial | 0.105 | 12.16 |

---

## Key insights

### 1. Apr 18 results were inflated by chemistry leakage
The Apr 18 runs included lab chemistry measurements (`pH`, `OC`, `CaCO3`, `N`, `K`, etc.) from the same field samples as the texture labels. These are correlated with texture by definition and gave artificially high R². The May 03 runs are the correct baselines.

### 2. Spectral features alone carry limited texture signal
Pure spectral-only R²: Clay 0.28, Silt 0.29, Sand 0.07, Coarse 0.05. Sand and Coarse are near-zero — bare-soil spectra from the Statistics API (18 `p50_*` features) capture almost none of their variance.

### 3. Coords/Elev add real, geographically-grounded signal
Adding `lat`, `lon`, `TH_LAT`, `TH_LONG`, `Elev` lifts R² substantially: Clay +0.13, Silt +0.10, Sand +0.16, Coarse +0.19. The random↔spatial CV gap is small (≤0.015 for Clay/Silt), indicating the models generalise within the spatial block structure rather than memorising exact point locations.

### 4. Sand and Coarse remain the weakest targets
Even with coordinates, Sand (0.24) and Coarse (0.24) are well below Clay (0.41) and Silt (0.39). Their distributions are likely driven by geological heterogeneity that neither spectral features nor coarse geographic position can explain from this dataset.

### 5. Random Forest dominates both feature sets
RF is the top performer for every target under both CV strategies. HGB gains more from the coord features (gap to RF narrows) but never overtakes RF. ElasticNet is a distant baseline in all scenarios.

### 6. HGB is unreliable for spectral-only Sand and Coarse
HGB produces negative R² for Sand and Coarse in the spectral-only run, suggesting it overfits badly on noisy targets with weak spectral signal. RF's implicit bagging makes it more robust in this regime.

### 7. Spatial CV gap is minimal for the May 03 runs
The random↔spatial R² gap is ≤0.015 across all targets and feature sets, suggesting the remaining signal in the models generalises to held-out spatial blocks. The Apr 18 spatial CV gap was larger partly because chemistry features inflated random CV scores.

---

## Which weights to use

**For spectral-only inference** (no geographic metadata required at inference time):

| Target | Recommended weights (spatial CV) |
|--------|----------------------------------|
| Clay | `run_20260503_125342/pipeline__Clay__rf__n800_leaf2_sqrt__SpatialGroupKFold.joblib` |
| Silt | `run_20260503_125342/pipeline__Silt__rf__n800_leaf2_sqrt__SpatialGroupKFold.joblib` |
| Sand | `run_20260503_125342/pipeline__Sand__rf__n1200_leaf1_sqrt__SpatialGroupKFold.joblib` |
| Coarse | `run_20260503_125342/pipeline__Coarse__rf__n800_leaf4_log2__SpatialGroupKFold.joblib` |

**For spectral + geographic inference** (lat/lon/Elev available at inference time):

| Target | Recommended weights (spatial CV) |
|--------|----------------------------------|
| Clay | `run_20260503_130423/pipeline__Clay__rf__n1200_leaf1_sqrt__SpatialGroupKFold.joblib` |
| Silt | `run_20260503_130423/pipeline__Silt__rf__n1200_leaf1_sqrt__SpatialGroupKFold.joblib` |
| Sand | `run_20260503_130423/pipeline__Sand__rf__n1200_leaf1_sqrt__SpatialGroupKFold.joblib` |
| Coarse | `run_20260503_130423/pipeline__Coarse__rf__n800_leaf2_sqrt__SpatialGroupKFold.joblib` |

All paths relative to `SCALAWAY_JOB/training/reports/`.
