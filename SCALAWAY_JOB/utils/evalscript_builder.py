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

def build_orbit_timeseries_evalscript_filtered(
    bands: List[str],
    *,
    units: str = "REFLECTANCE",
    ndvi_max: float = 0.25,
    use_ndwi_guard: bool = True,
    use_bsi_guard: bool = False,
) -> str:
    # Spectral bands we want to output (reflectance)
    out_bands = bands[:]  # e.g. ["B02","B03","B04","B08","B11","B12"]

    # Ensure needed spectral bands exist for indices
    required_spec = set(out_bands)
    required_spec.update({"B04", "B08"})  # NDVI
    if use_ndwi_guard:
        required_spec.add("B03")
    if use_bsi_guard:
        required_spec.update({"B02", "B11"})

    # Inputs: reflectance spectral bands + DN classification bands
    spec_input_bands = sorted(required_spec)
    spec_bands_js = "[" + ", ".join(f'"{b}"' for b in spec_input_bands) + "]"

    out_return = ", ".join(f"sample.{b}" for b in out_bands)
    n_out = len(out_bands)
    per_obs_out_len = n_out + 1  # + VALID

    ndwi_guard_js = (
        "var ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);\n"
        "if (!(ndwi < 0.0)) { valid = false; }\n"
        if use_ndwi_guard
        else ""
    )

    bsi_guard_js = (
        "var bsi = ((sample.B11 + sample.B04) - (sample.B08 + sample.B02)) / "
        "((sample.B11 + sample.B04) + (sample.B08 + sample.B02));\n"
        "if (!(bsi > 0.0)) { valid = false; }\n"
        if use_bsi_guard
        else ""
    )

    return f"""//VERSION=3
function setup() {{
  return {{
    input: [
      {{
        bands: {spec_bands_js},
        units: "{units}"
      }},
      {{
        bands: ["SCL", "dataMask"],
        units: "DN"
      }}
    ],
    output: {{
      bands: 1,
      sampleType: SampleType.FLOAT32
    }},
    mosaicking: Mosaicking.ORBIT
  }};
}}

function updateOutput(outputs, collection) {{
  Object.values(outputs).forEach((output) => {{
    output.bands = collection.scenes.length * {per_obs_out_len};
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

    var valid = true;

    // dataMask guard
    if (sample.dataMask === 0) {{
      valid = false;
    }}

    // SCL guard: bare soil only
    if (sample.SCL !== 5) {{
      valid = false;
    }}

    // NDVI guard
    var ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
    if (!(ndvi < {ndvi_max})) {{
      valid = false;
    }}

    {ndwi_guard_js}
    {bsi_guard_js}

    if (valid) {{
      out = out.concat([{out_return}, 1.0]);
    }} else {{
      var nan = NaN;
      var arr = [];
      for (var k = 0; k < {n_out}; k++) arr.push(nan);
      arr.push(0.0);
      out = out.concat(arr);
    }}
  }}
  return out;
}}
"""