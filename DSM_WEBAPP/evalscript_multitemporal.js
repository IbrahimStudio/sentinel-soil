//VERSION=3
/**
 * evalscript_multitemporal.js
 *
 * Multi-temporal Process API evalscript — fetches ALL orbit passes for a
 * given time window in a single API request instead of one call per date.
 *
 * Mosaicking.ORBIT: evaluatePixel() receives one raw sample per orbit pass
 * with NO server-side aggregation or compositing. All quality filtering
 * (SCL, NDVI, NBR2) continues to happen client-side in Python.
 *
 * Output layout per pixel — flat array of (MAX_SCENES × N_BANDS) float32:
 *   orbit i, band j  →  index [i * 11 + j]
 *   Band order: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, SCL
 *   Unused slots (i >= actual_n_scenes) are filled with -9999.
 *
 * Actual scene count + dates are stored in userdata.json (tar response):
 *   { "n_scenes": <int>, "dates": ["YYYY-MM-DD", ...] }
 *
 * MAX_SCENES must match the Python constant in sh_clients.py.
 */

var N_BANDS = 11;
var MAX_SCENES = 400;

function setup() {
    return {
        input: [{
            bands: ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"]
        }],
        output: [{
            id: "default",
            bands: N_BANDS * MAX_SCENES,
            sampleType: "FLOAT32"
        }],
        mosaicking: Mosaicking.ORBIT
    };
}

function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
    var dates = [];
    for (var i = 0; i < scenes.orbits.length && i < MAX_SCENES; i++) {
        dates.push(scenes.orbits[i].dateFrom.substring(0, 10));
    }
    outputMetadata.userData = {
        n_scenes: dates.length,
        dates: dates
    };
}

function evaluatePixel(samples) {
    var n = Math.min(samples.length, MAX_SCENES);
    var out = new Array(N_BANDS * MAX_SCENES);
    for (var k = 0; k < out.length; k++) out[k] = -9999;

    for (var i = 0; i < n; i++) {
        var s = samples[i];
        var b = i * N_BANDS;
        out[b]    = s.B02;
        out[b+1]  = s.B03;
        out[b+2]  = s.B04;
        out[b+3]  = s.B05;
        out[b+4]  = s.B06;
        out[b+5]  = s.B07;
        out[b+6]  = s.B08;
        out[b+7]  = s.B8A;
        out[b+8]  = s.B11;
        out[b+9]  = s.B12;
        out[b+10] = s.SCL;
    }
    return out;
}
