//VERSION=3
/**
 * Sentinel Hub Evalscript for Soil Features + Coverage (Statistical API-ready)
 *
 * Outputs:
 *  - features[18]  (FLOAT32)
 *  - valid[1]      (UINT8)  -> coverage = mean(valid); coverage_pct = 100*mean(valid)
 *  - dataMask[1]   (UINT8)  (optional debugging)
 *
 * Filters (pixel-level):
 *  - Must have dataMask==1
 *  - SCL must NOT be in bad classes (cloud/shadow/cirrus/snow/ice + water optional)
 *  - NDVI < NDVI_THRESHOLD (bare-ish soil constraint)
 */

var NDVI_THRESHOLD = 0.25;   // change to 0.20 if you want stricter

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
      { id: "features", bands: 18, sampleType: "FLOAT32" },
      { id: "valid", bands: 1, sampleType: "UINT8" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}

function isBadSCL(scl) {
  // Sentinel-2 L2A SCL classes (common ones):
  // 3 = Cloud shadow
  // 6 = Water
  // 8 = Cloud medium probability
  // 9 = Cloud high probability
  // 10 = Thin cirrus
  // 11 = Snow or ice
  //
  // For soil work, you can decide whether to allow 7/8.
  // Here we are moderately permissive: exclude 3,8,9,10,11 and ALSO exclude water (6).
  return (scl === 3) || (scl === 6) || (scl === 8) || (scl === 9) || (scl === 10) || (scl === 11);
}

function safeDiv(num, den) {
  var eps = 1e-6;
  return num / (Math.abs(den) < eps ? (den >= 0 ? eps : -eps) : den);
}

function nd(a, b) {
  return safeDiv(a - b, a + b);
}

function evaluatePixel(s) {
  // --- Base validity: must have data and pass SCL mask ---
  var ok = (s.dataMask === 1) && !isBadSCL(s.SCL);

  // Early reject
  if (!ok) {
    return {
      features: new Array(18).fill(0),
      valid: [0],
      dataMask: [0]
    };
  }

  // --- Raw bands (Sentinel Hub defaults to reflectance units for S2 optical bands) ---
  var B02 = s.B02; // Blue
  var B03 = s.B03; // Green
  var B04 = s.B04; // Red
  var B08 = s.B08; // NIR
  var B11 = s.B11; // SWIR1
  var B12 = s.B12; // SWIR2

  // --- NDVI filter (bare-ish soil constraint) ---
  var NDVI = nd(B08, B04);
  if (NDVI >= NDVI_THRESHOLD) {
    return {
      features: new Array(18).fill(0),
      valid: [0],
      dataMask: [0]
    };
  }

  // --- Indices ---
  var NDWI  = nd(B03, B08);
  var MNDWI = nd(B03, B11);
  var NDMI  = nd(B08, B11);
  var BSI   = safeDiv((B11 + B04) - (B08 + B02), (B11 + B04) + (B08 + B02));

  // --- Brightness / albedo proxy ---
  var BRIGHT = (B02 + B03 + B04 + B08 + B11 + B12) / 6.0;
  var ALBEDO_PROXY = BRIGHT;

  // --- Ratios / band proxies ---
  var RED   = B04;
  var SWIR1 = B11;
  var SWIR2 = B12;
  var RED_SWIR1_RATIO   = safeDiv(RED, SWIR1);
  var SWIR1_SWIR2_RATIO = safeDiv(SWIR1, SWIR2);

  return {
    features: [
      B02, B03, B04, B08, B11, B12,
      NDVI, NDWI, MNDWI, NDMI, BSI,
      BRIGHT, ALBEDO_PROXY,
      RED, SWIR1, SWIR2,
      RED_SWIR1_RATIO, SWIR1_SWIR2_RATIO
    ],
    valid: [1],
    dataMask: [1]
  };
}
