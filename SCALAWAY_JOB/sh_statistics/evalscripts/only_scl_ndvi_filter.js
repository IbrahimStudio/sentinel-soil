//VERSION=3
/**
 * Sentinel Hub Evalscript for Soil Feature Computation (SCL-Only + NDVI Filtering)
 *
 * Computes 18 features from Sentinel-2 L2A data:
 * - 6 raw bands (B02, B03, B04, B08, B11, B12)
 * - 10 spectral indices and features
 * - 2 band ratios
 *
 * Includes filtering for:
 * - Clouds, shadows, water, snow/ice (via SCL)
 * - NDVI < 0.2 (to filter out vegetated areas)
 * - NO sun zenith angle filtering
 * - NO water filtering
 *
 * This filtering should increase data availability for bare soil analysis
 * while excluding vegetated pixels and obviously problematic pixels.
 */

function setup() {
  return {
    input: [{
      bands: [
        "B02","B03","B04","B08","B11","B12",
        "SCL",
        "dataMask"
      ]
    }],
    output: [
      // 6 raw bands + 10 features + 2 ratios = 18 total
      { id: "features", bands: 18, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function isBadSCL(scl) {
  // Exclude: cloud shadow, water, med/high clouds, cirrus, snow/ice
  // But be more permissive for soil analysis - allow some cloudy pixels
  return (scl === 3) || (scl === 6) || (scl === 9) || (scl === 10) || (scl === 11);
  // Allow SCL 8 (cloud medium probability) and SCL 7 (cloud low probability)
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
  // --- Base pixel-level validity --- SCL + NDVI filtering ---
  var valid = (s.dataMask === 1) && !isBadSCL(s.SCL);

  // If already invalid, early return with dataMask=0
  // (features content doesn't matter when masked)
  if (!valid) {
    return {
      features: new Array(18).fill(0),
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

  // --- Calculate NDVI for filtering ---
  var NDVI = nd(B08, B04); // vegetation residual

  // --- Apply NDVI filter: only keep pixels with NDVI < 0.2 ---
  if (NDVI >= 0.2) {
    return {
      features: new Array(18).fill(0),
      dataMask: [0]
    };
  }

  // --- Spectral Indices ---
  var NDWI  = nd(B03, B08);          // (McFeeters) surface water / moisture proxy
  var MNDWI = nd(B03, B11);          // Xu (better for open water)
  var NDMI  = nd(B08, B11);          // water content (veg/soil moisture proxy)
  var BSI   = safeDiv((B11 + B04) - (B08 + B02), (B11 + B04) + (B08 + B02)); // bare soil index

  // --- Brightness / "Albedo" proxy ---
  // Simple brightness: mean reflectance across selected bands
  // (If later you want a physically-based broadband albedo, we can swap to a weighted formula.)
  var BRIGHT = (B02 + B03 + B04 + B08 + B11 + B12) / 6.0;
  var ALBEDO_PROXY = BRIGHT; // keeping separate name for clarity

  // --- Red/SWIR features (SOC/texture proxies) ---
  var RED   = B04;
  var SWIR1 = B11;
  var SWIR2 = B12;
  var RED_SWIR1_RATIO = safeDiv(RED, SWIR1);
  var SWIR1_SWIR2_RATIO = safeDiv(SWIR1, SWIR2);

  // Output vector layout:
  // [0..5] raw bands: B02,B03,B04,B08,B11,B12
  // [6..10] indices: NDVI, NDWI, MNDWI, NDMI, BSI
  // [11..12] brightness: BRIGHT, ALBEDO_PROXY
  // [13..15] raw bands: RED, SWIR1, SWIR2
  // [16..17] ratios: RED/SWIR1, SWIR1/SWIR2
  return {
    features: [
      B02, B03, B04, B08, B11, B12,
      NDVI, NDWI, MNDWI, NDMI, BSI,
      BRIGHT, ALBEDO_PROXY,
      RED, SWIR1, SWIR2,
      RED_SWIR1_RATIO, SWIR1_SWIR2_RATIO
    ],
    dataMask: [1]
  };
}