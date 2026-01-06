# Sentinel-2 & LUCAS Soil Modelling Pipeline – Condensed Summary

## 1. Objective
The goal of this work is to predict soil properties (texture, chemistry, organic carbon) by integrating **Sentinel-2 optical imagery** with **LUCAS soil observations**. The main challenge is transforming noisy, temporally dense satellite data into **stable, soil-representative features** aligned with the spatial support of ground measurements.

---

## 2. Conceptual Design

The pipeline follows best practices in **Digital Soil Mapping (DSM)**:

1. Use **bare-soil reflectance** rather than vegetated observations  
2. Preserve temporal information before aggregation  
3. Match the spatial support of satellite features to soil sampling  
4. Produce interpretable, reproducible features for modelling  

Each LUCAS point is treated as a **seed location**, around which Sentinel-2 data are extracted and summarized.

---

## 3. Spatial Unit: From Pixels to Seed Windows

Initial consideration of using individual Sentinel-2 pixels as samples was discarded. A single pixel (10 m) does not represent the support area of a LUCAS soil observation and would introduce spatial pseudo-replication.

**Final choice:**
- One data point per LUCAS location  
- Sentinel-2 data aggregated over a **100 × 100 m window** around each seed  

This improves robustness and ensures physical consistency with soil measurements.

---

## 4. Sentinel-2 Data Extraction

Data are accessed via the **Copernicus Data Space Ecosystem** using the **Sentinel Hub Python SDK**.

For each seed window:
- All Sentinel-2 acquisitions in a given time interval are retrieved
- Bands used:
  - **10 m**: B02, B03, B04, B08  
  - **20 m (resampled)**: B11, B12
- Data are extracted using **ORBIT mosaicking** to preserve all acquisitions  

No temporal aggregation is applied at this stage.

---

## 5. Per-Pixel Time Series

Each pixel in the window has a time series of reflectance values:

(date, pixel, B02, B03, B04, B08, B11, B12)


This structure allows pixel-level filtering and avoids early information loss. Intermediate results are stored as **Parquet files** for efficiency and reproducibility.

---

## 6. Spectral Indices

Spectral indices are computed **per pixel per date**, before aggregation:

- NDVI (vegetation filter)
- NDMI (moisture proxy)
- NDWI (water sensitivity)
- NBR (dry/burn signal)
- MSAVI (soil-adjusted vegetation)
- EVI (robust vegetation index)

Computing indices at this stage avoids bias from non-linear transformations applied after aggregation.

---

## 7. Bare-Soil Filtering

To isolate soil reflectance:
- NDVI is used as a vegetation proxy
- Observations are retained only if:

NDVI < 0.2


Filtering is applied **before temporal aggregation**, ensuring vegetated observations do not influence soil features. This step reflects standard practice in DSM literature.

---

## 8. Temporal Aggregation (Per Pixel)

For each pixel, using only bare-soil observations:
- Median, mean, and standard deviation are computed for all bands and indices
- The number of valid observations (`n_obs_baresoil`) is stored as a quality indicator  

This reduces noise while preserving physically meaningful variability.

---

## 9. Spatial Aggregation (Per Seed)

Pixel-level features are aggregated across the window to produce **one feature vector per seed**.

For each feature:
- Spatial median  
- Spatial mean  
- Spatial standard deviation  

These statistics capture both the typical soil signal and its local heterogeneity.

---

## 10. Final Sentinel Feature Set

Each LUCAS point is enriched with a single row of Sentinel-derived features, including:

- Aggregated reflectance bands  
- Aggregated spectral indices  
- Quality indicators (number of valid bare-soil observations, pixel coverage)  

The result is a **training-ready satellite feature table**.

---

## 11. Ground Truth: LUCAS Dataset

The LUCAS dataset provides soil properties such as:
- Texture: Clay, Sand, Silt  
- Chemistry: pH, EC, OC, nutrients  
- Ancillary information: Elevation, Land Cover/Use, NUTS regions  

Each LUCAS observation is joined one-to-one with the Sentinel feature vector derived from its seed window.

---

## 12. Batch Processing

A batch workflow processes multiple LUCAS points automatically:
- Extract Sentinel-2 data per seed
- Build time series and features
- Save diagnostics and plots
- Produce a consolidated Parquet dataset  

This enables scalable experimentation and reproducibility.

---

## 13. Readiness for Modelling

At the end of this pipeline:
- Temporal noise is controlled  
- Vegetation and clouds are minimized  
- Spatial support matches soil sampling  
- Features are interpretable and physically grounded  
- Ground truth labels are available  

The dataset is therefore **ready for supervised soil property modelling**.

---

## 14. Next Steps

Remaining work focuses on modelling rather than data preparation:
- Spatially aware cross-validation
- Baseline regressors (Random Forest, Gradient Boosting)
- Error analysis and feature importance
- Comparison with literature benchmarks

---

## One-sentence summary

This pipeline transforms raw Sentinel-2 time series into bare-soil, spatially aggregated features aligned with LUCAS soil observations, producing a robust and defensible dataset for soil property prediction.
