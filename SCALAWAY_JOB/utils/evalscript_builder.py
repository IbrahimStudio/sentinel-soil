from typing import List


def build_raw_bands_evalscript(bands: List[str]) -> str:
    bands_js = ", ".join(f'"{b}"' for b in bands)
    values_js = ", ".join(f"sample.{b}" for b in bands)

    return f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: ["B02", "B03", "B04", "B08"],
      units: "REFLECTANCE"
    }}],
    output: {{
      bands: 4,
      sampleType: SampleType.FLOAT32
    }},
    mosaicking: Mosaicking.MEDIAN
  }};
}}

function evaluatePixel(sample) {{
  return [
    sample.B02,
    sample.B03,
    sample.B04,
    sample.B08
  ];
}}
"""
def build_orbit_timeseries_evalscript(bands: List[str], units: str = "REFLECTANCE") -> str:
    """
    ORBIT mosaicking: returns all acquisitions within time interval.
    Output bands = n_observations * len(bands).
    acquisition dates are stored in userdata.
    """
    bands_js = "[" + ", ".join(f'"{b}"' for b in bands) + "]"
    per_obs_return = ", ".join(f"sample.{b}" for b in bands)
    n_bands = len(bands)

    return f"""//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: {bands_js},
      units: "{units}"
    }}],
    output: {{
      bands: 1,
      sampleType: SampleType.FLOAT32
    }},
    mosaicking: Mosaicking.ORBIT
  }};
}}

function updateOutput(outputs, collection) {{
  Object.values(outputs).forEach((output) => {{
    output.bands = collection.scenes.length * {n_bands};
  }});
}}

function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {{
  var dds = [];
  for (var i = 0; i < scenes.length; i++) {{
    dds.push(scenes[i].date);
  }}
  outputMetadata.userData = {{
    "acquisition_dates": JSON.stringify(dds)
  }};
}}

function evaluatePixel(samples) {{
  var out = [];
  for (var i = 0; i < samples.length; i++) {{
    var sample = samples[i];
    out = out.concat([{per_obs_return}]);
  }}
  return out;
}}
"""