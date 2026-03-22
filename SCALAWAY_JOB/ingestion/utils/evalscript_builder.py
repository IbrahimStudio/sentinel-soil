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
    """
    ORBIT mosaicking with per-pixel filtering using SCL + index thresholds.
    Output layout per observation:
      [band1, band2, ..., bandN, validMask]
    Invalid observations return NaN for bands and 0 for validMask.
    """

    # Ensure needed bands exist for indices
    required = set(bands)
    required.add("SCL")
    required.add("dataMask")
    required.add("B04")  # red for NDVI
    required.add("B08")  # nir for NDVI
    if use_ndwi_guard:
        required.add("B03")  # green for NDWI
    if use_bsi_guard:
        required.add("B11")  # SWIR1 for BSI

    # Keep user-specified order for output bands, but inputs must include extras
    input_bands = sorted(required)  # deterministic

    bands_js = "[" + ", ".join(f'"{b}"' for b in input_bands) + "]"
    out_return = ", ".join(f"sample.{b}" for b in bands)  # only the requested bands in output

    n_out = len(bands)
    per_obs_out_len = n_out + 1  # + mask

    # JS snippets for optional guards
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

    // --- base validity ---
    var valid = true;

    // no-data guard
    if (sample.dataMask === 0) {{
      valid = false;
    }}

    // SCL guard: keep ONLY bare soil (5)
    // Excludes water (6), vegetation (4), shadows (3), clouds (7-10), snow (11), etc.
    if (sample.SCL !== 5) {{
      valid = false;
    }}

    // NDVI guard: remove sparse vegetation / residues
    var ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
    if (!(ndvi < {ndvi_max})) {{
      valid = false;
    }}

    // Optional extra guards
    {ndwi_guard_js}
    {bsi_guard_js}

    if (valid) {{
      out = out.concat([{out_return}, 1.0]);
    }} else {{
      // return NaNs for bands + 0 mask
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
