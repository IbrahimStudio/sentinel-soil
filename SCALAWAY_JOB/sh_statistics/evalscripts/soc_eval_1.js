//VERSION=3
/**
 * Sentinel Hub Evalscript for SOC Mapping
 *
 * Computes 8 features for soil organic carbon prediction:
 * - 6 raw bands (B02, B03, B04, B08, B11, B12) - normalized to reflectance
 * - 2 spectral indices (NDVI, NBR2)
 *
 * Includes pixel-level filtering for bare soil:
 * - SCL mask (keep only BARE_SOILS = 5)
 * - NDVI < 0.25 (exclude vegetation)
 * - NBR2 < 0.0075 (exclude high moisture)
 */

function setup() {
  return {
    input: [{
      bands: ["B02","B03","B04","B08","B11","B12","SCL","dataMask"]
    }],
    output: [
      // 6 raw bands + 2 indices = 8 total features
      { id: "features", bands: 8, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}

function evaluatePixel(s) {
  // --- Base pixel-level validity ---
  var valid = s.dataMask === 1;

  // --- SCL-based filtering: keep only BARE_SOILS (5) ---
  var sclOk = (s.SCL === 5);

  // --- Compute indices ---
  var eps = 1e-6;
  var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + eps);
  var nbr2 = (s.B11 - s.B12) / (s.B11 + s.B12 + eps);

  // --- Bare-soil rules ---
  var bareOk = (ndvi < 0.25) && (nbr2 < 0.0075);

  // --- Final keep mask ---
  var keep = valid && sclOk && bareOk;

  // --- Normalize raw bands to reflectance (0-1 range) ---
  // Sentinel-2 L2A DN values need to be divided by 10000 to get reflectance
  var B02 = s.B02 / 10000.0; // Blue
  var B03 = s.B03 / 10000.0; // Green
  var B04 = s.B04 / 10000.0; // Red
  var B08 = s.B08 / 10000.0; // NIR
  var B11 = s.B11 / 10000.0; // SWIR1
  var B12 = s.B12 / 10000.0; // SWIR2

  // --- Output ---
  // If pixel should be kept, return normalized features
  // Otherwise return zeros (will be filtered by dataMask)
  if (keep) {
    return {
      features: [B02, B03, B04, B08, B11, B12, ndvi, nbr2],
      dataMask: [1]
    };
  } else {
    return {
      features: [0, 0, 0, 0, 0, 0, 0, 0],
      dataMask: [0]
    };
  }
}
