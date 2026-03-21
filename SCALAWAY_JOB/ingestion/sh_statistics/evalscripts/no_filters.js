//VERSION=3
/**
 * LIGHT baseline (NO FILTERS) — parser compatible
 * Outputs:
 *  - features: 18 float32 (same layout as full pipeline)
 *  - valid: 0/1 (used as coverage via mean)
 *  - dataMask: 0/1 (debug)
 *
 * Policy:
 *  valid = (input dataMask == 1)
 *
 * Features:
 *  - Compute only a small subset (most significant) and keep rest 0:
 *    B04, B08, B11, NDVI, NDMI, BRIGHT
 */

function setup() {
  return {
    input: [{
      bands: ["B02","B03","B04","B08","B11","B12","dataMask"]
    }],
    output: [
      { id: "features", bands: 18, sampleType: "FLOAT32" },
      { id: "valid", bands: 1, sampleType: "UINT8" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}

function safeDiv(num, den) {
  var eps = 1e-6;
  return num / (Math.abs(den) < eps ? (den >= 0 ? eps : -eps) : den);
}

function nd(a, b) {
  return safeDiv(a - b, a + b);
}

function evaluatePixel(s) {
  if (s.dataMask !== 1) {
    return {
      features: new Array(18).fill(0),
      valid: [0],
      dataMask: [0]
    };
  }

  // Always allocate full 18-length vector
  var f = new Array(18).fill(0);

  // Raw bands (we keep only the most informative ones non-zero)
  // Indices: keep reflectance scaling consistent with your pipeline (as provided by SH)
  var B02 = s.B02, B03 = s.B03, B04 = s.B04, B08 = s.B08, B11 = s.B11, B12 = s.B12;

  // Put selected raw bands (others remain 0 if you want, but writing them costs nothing)
  // If you prefer "some bands", keep these three:
  f[2] = B04; // B04
  f[3] = B08; // B08
  f[4] = B11; // B11

  // Indices (cheap)
  var NDVI = nd(B08, B04);
  var NDMI = nd(B08, B11);

  f[6] = NDVI; // NDVI
  f[9] = NDMI; // NDMI

  // Brightness proxy (cheap)
  var BRIGHT = (B02 + B03 + B04 + B08 + B11 + B12) / 6.0;
  f[11] = BRIGHT; // BRIGHT
  f[12] = BRIGHT; // ALBEDO_PROXY (same proxy)

  return {
    features: f,
    valid: [1],
    dataMask: [1]
  };
}