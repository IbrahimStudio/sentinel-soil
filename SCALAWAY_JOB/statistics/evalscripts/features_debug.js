//VERSION=3
/**
 * Sentinel Hub Evalscript for Soil Feature Computation with Debugging
 *
 * Extended version that tracks pixel-level filtering stages for debugging data loss.
 * Computes 18 features + 4 debug bands tracking filter stages.
 *
 * Features:
 * - 6 raw bands (B02, B03, B04, B08, B11, B12)
 * - 10 spectral indices and features
 * - 2 band ratios
 * - 4 debug bands (filter stage tracking)
 */

function setup() {
  return {
    input: [{
      bands: [
        "B02","B03","B04","B08","B11","B12",
        "SCL",
        "sunZenithAngles",
        "dataMask"
      ]
    }],
    output: [
      // 18 features + 4 debug bands = 22 total
      { id: "features", bands: 22, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function isBadSCL(scl) {
  // Exclude: cloud shadow, water, med/high clouds, cirrus, snow/ice
  return (scl === 3) || (scl === 6) || (scl === 8) || (scl === 9) || (scl === 10) || (scl === 11);
}

function safeDiv(num, den) {
  var eps = 1e-6;
  return num / (Math.abs(den) < eps ? (den >= 0 ? eps : -eps) : den);
}

function nd(a, b) {
  // normalized difference (a-b)/(a+b)
  return safeDiv(a - b, a + b);
}

function evaluatePixel(s) {
  // --- Debug tracking: initialize all stages as valid (1.0) ---
  var debug_stage_scl = 1.0;
  var debug_stage_sza = 1.0;
  var debug_stage_ndvi = 1.0;
  var debug_stage_mndwi = 1.0;

  // --- Base pixel-level validity ---
  var valid = (s.dataMask === 1);

  // Stage 1: SCL filtering
  if (valid && isBadSCL(s.SCL)) {
    valid = false;
    debug_stage_scl = 0.0;  // Mark SCL as cause of rejection
  }

  // Stage 2: Solar Zenith Angle filtering
  if (valid && (s.sunZenithAngles >= 70)) {
    valid = false;
    debug_stage_sza = 0.0;  // Mark SZA as cause of rejection
  }

  // If already invalid, early return with dataMask=0 and debug info
  if (!valid) {
    return {
      features: [
        // 18 feature bands (all 0 since invalid)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // 4 debug bands showing which stages failed
        debug_stage_scl, debug_stage_sza, debug_stage_ndvi, debug_stage_mndwi
      ],
      dataMask: [0]
    };
  }

  // --- Raw bands (reflectance) ---
  var B02 = s.B02; // Blue
  var B03 = s.B03; // Green
  var B04 = s.B04; // Red
  var B08 = s.B08; // NIR
  var B11 = s.B11; // SWIR1
  var B12 = s.B12; // SWIR2

  // --- Spectral Indices ---
  var NDVI  = nd(B08, B04);          // vegetation residual
  var NDWI  = nd(B03, B08);          // (McFeeters) surface water / moisture proxy
  var MNDWI = nd(B03, B11);          // Xu (better for open water)
  var NDMI  = nd(B08, B11);          // water content (veg/soil moisture proxy)
  var BSI   = safeDiv((B11 + B04) - (B08 + B02), (B11 + B04) + (B08 + B02)); // bare soil index

  // --- Brightness / "Albedo" proxy ---
  var BRIGHT = (B02 + B03 + B04 + B08 + B11 + B12) / 6.0;
  var ALBEDO_PROXY = BRIGHT;

  // --- Red/SWIR features (SOC/texture proxies) ---
  var RED   = B04;
  var SWIR1 = B11;
  var SWIR2 = B12;
  var RED_SWIR1_RATIO = safeDiv(RED, SWIR1);
  var SWIR1_SWIR2_RATIO = safeDiv(SWIR1, SWIR2);

  // Stage 3: NDVI threshold filtering
  var ok_ndvi = (NDVI < 0.2);
  if (valid && !ok_ndvi) {
    valid = false;
    debug_stage_ndvi = 0.0;  // Mark NDVI as cause of rejection
  }

  // Stage 4: MNDWI threshold filtering
  var ok_mndwi = (MNDWI < 0.0);
  if (valid && !ok_mndwi) {
    valid = false;
    debug_stage_mndwi = 0.0;  // Mark MNDWI as cause of rejection
  }

  if (!valid) {
    return {
      features: [
        // 18 feature bands (all 0 since invalid)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // 4 debug bands showing which stages failed
        debug_stage_scl, debug_stage_sza, debug_stage_ndvi, debug_stage_mndwi
      ],
      dataMask: [0]
    };
  }

  // Output vector layout:
  // [0..5] raw bands: B02,B03,B04,B08,B11,B12
  // [6..10] indices: NDVI, NDWI, MNDWI, NDMI, BSI
  // [11..12] brightness: BRIGHT, ALBEDO_PROXY
  // [13..15] raw bands: RED, SWIR1, SWIR2
  // [16..17] ratios: RED/SWIR1, SWIR1/SWIR2
  // [18..21] debug: SCL_pass, SZA_pass, NDVI_pass, MNDWI_pass
  return {
    features: [
      B02, B03, B04, B08, B11, B12,
      NDVI, NDWI, MNDWI, NDMI, BSI,
      BRIGHT, ALBEDO_PROXY,
      RED, SWIR1, SWIR2,
      RED_SWIR1_RATIO, SWIR1_SWIR2_RATIO,
      debug_stage_scl, debug_stage_sza, debug_stage_ndvi, debug_stage_mndwi
    ],
    dataMask: [1]
  };
}