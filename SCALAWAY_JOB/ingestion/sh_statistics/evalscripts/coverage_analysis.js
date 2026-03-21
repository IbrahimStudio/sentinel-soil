//VERSION=3
/**
 * Sentinel Hub Evalscript for Coverage Analysis
 *
 * Outputs three masks for quantifying data availability:
 * 1. dataMask: All available pixels (no filtering)
 * 2. scl_ok: Pixels passing SCL filtering only
 * 3. scl_ok_ndvi: Pixels passing both SCL and NDVI filtering
 *
 * Coverage metrics computed from Statistical API:
 * - observable_fraction = mean(dataMask)
 * - kept_scl_abs = mean(scl_ok)
 * - kept_scl_ndvi_abs = mean(scl_ok_ndvi)
 * - saved_scl = kept_scl_abs / observable_fraction
 * - saved_scl_ndvi = kept_scl_ndvi_abs / observable_fraction
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
      { id: "dataMask", bands: 1, sampleType: "UINT8" },   // All available pixels
      { id: "scl_ok", bands: 1, sampleType: "UINT8" },     // SCL-filtered pixels
      { id: "scl_ok_ndvi", bands: 1, sampleType: "UINT8" }  // SCL+NDVI filtered pixels
    ]
  };
}

function isBadSCL(scl) {
  // Exclude: cloud shadow, water, cirrus, snow/ice, high clouds
  // Same exclusion classes as only_scl_ndvi_filter.js
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
  // Base validity: must have data
  var has_data = (s.dataMask === 1);

  // SCL filtering
  var scl_ok = has_data && !isBadSCL(s.SCL);

  // NDVI filtering (only applied if SCL passes)
  var ndvi_ok = false;
  if (scl_ok) {
    var NDVI = nd(s.B08, s.B04);
    ndvi_ok = (NDVI < 0.2);  // NDVI threshold of 0.2
  }

  // Output the three masks
  return {
    dataMask: [has_data ? 1 : 0],
    scl_ok: [scl_ok ? 1 : 0],
    scl_ok_ndvi: [ndvi_ok ? 1 : 0]
  };
}