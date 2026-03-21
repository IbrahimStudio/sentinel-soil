//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02","B03","B04","B08","B11","B12","dataMask"] }],
    output: [{ id:"features", bands: 6, sampleType:"FLOAT32" }, { id:"dataMask", bands: 1 }]
  };
}
function evaluatePixel(s){
  return { features:[s.B02,s.B03,s.B04,s.B08,s.B11,s.B12], dataMask:[s.dataMask] };
}