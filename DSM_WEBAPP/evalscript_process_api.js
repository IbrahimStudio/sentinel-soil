//VERSION=3
/**
 * evalscript_process_api.js
 *
 * Shared evalscript for training pipeline and inference worker.
 * Used via EVALSCRIPT_PATH env var — never inlined in Python code.
 *
 * Returns 11 bands in this exact order:
 *   [0] B02, [1] B03, [2] B04, [3] B05, [4] B06, [5] B07,
 *   [6] B08, [7] B8A, [8] B11, [9] B12, [10] SCL
 *
 * All spectral bands are BOA reflectance in [0, 1] range, FLOAT32.
 * SCL is the raw integer class value stored as FLOAT32 (cast to int in Python).
 *
 * NO server-side filtering is applied. All bare-soil filtering is performed
 * client-side in Python using filter_config.json. B11 and B12 are mandatory
 * for NBR2 = (B11 - B12) / (B11 + B12).
 */

function setup() {
  return {
    input: [{
      bands: ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"]
    }],
    output: [{
      id: "default",
      bands: 11,
      sampleType: "FLOAT32"
    }]
  };
}

function evaluatePixel(s) {
  return [
    s.B02, s.B03, s.B04, s.B05, s.B06, s.B07, s.B08, s.B8A, s.B11, s.B12,
    s.SCL   // raw integer class: 0=NoData, 1=Saturated, 2=Dark, 3=Shadow,
            // 4=Vegetation, 5=Bare/NotVegetated, 6=Water, 7=Unclassified,
            // 8=CloudMediumProb, 9=CloudHighProb, 10=ThinCirrus, 11=Snow
  ];
}
