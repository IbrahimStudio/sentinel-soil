//VERSION=3
/**
 * Sentinel Hub Evalscript for Soil Feature Computation
 * + Validity flag for coverage computation via Statistical API
 *
 * Coverage per day (for a 3×3 window) will be:
 *   coverage = mean(valid)      // [0..1]
 *   coverage_pct = 100*mean(valid)
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
      { id: "features", bands: 18, sampleType: "FLOAT32" },
      { id: "valid", bands: 1, sampleType: "UINT8" },     // 0/1 flag (use mean as coverage)
      { id: "dataMask", bands: 1, sampleType: "UINT8" }   // optional: keep for debugging
    ]
  };
}

function isBadSCL(scl) {
  // Exclude: cloud shadow, water, cirrus, snow/ice, high clouds
  // (still permissive: allows SCL 7 and 8)
  return (scl === 3) || (scl === 6) || (scl === 9) || (scl === 10) || (scl === 11);
}

function safeDiv(num, den) {
  var eps = 1e-6;
  return num / (Math.abs(den) < eps ? (den >= 0 ? eps : -eps) : den);
}

function nd(a, b) {
  return safeDiv(a - b, a + b);
}

function evaluatePixel(s) {
  // Base validity: must have data + pass SCL
  var ok = (s.dataMask === 1) && !isBadSCL(s.SCL);

  // If base invalid: emit valid=0 and mask=0
  if (!ok) {
    return {
      features: new Array(18).fill(0),
      valid: [0],
      dataMask: [0]
    };
  }

  // Raw bands
  var B02 = s.B02, B03 = s.B03, B04 = s.B04, B08 = s.B08, B11 = s.B11, B12 = s.B12;

  // NDVI filter
  var NDVI = nd(B08, B04);
  if (NDVI >= 0.2) {
    return {
      features: new Array(18).fill(0),
      valid: [0],
      dataMask: [0]
    };
  }

  // Indices
  var NDWI  = nd(B03, B08);
  var MNDWI = nd(B03, B11);
  var NDMI  = nd(B08, B11);
  var BSI   = safeDiv((B11 + B04) - (B08 + B02), (B11 + B04) + (B08 + B02));

  // Brightness / albedo proxy
  var BRIGHT = (B02 + B03 + B04 + B08 + B11 + B12) / 6.0;
  var ALBEDO_PROXY = BRIGHT;

  // Ratios / band proxies
  var RED   = B04;
  var SWIR1 = B11;
  var SWIR2 = B12;
  var RED_SWIR1_RATIO   = safeDiv(RED, SWIR1);
  var SWIR1_SWIR2_RATIO = safeDiv(SWIR1, SWIR2);

  // Pixel is valid for soil features
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
