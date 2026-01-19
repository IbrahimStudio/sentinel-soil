import pandas as pd
from pathlib import Path

# -------------------------
# Paths
# -------------------------
LUCAS_2018_CSV = Path(r"data\LUCAS_DS\LUCAS-SOIL-2018.csv")          # <-- update filename if different
LUCAS_2015_CSV = Path(r"data\LUCAS_DS\LUCAS_Topsoil_2015_20200323.csv")

OUT_CSV = Path(r"data\LUCAS_DS\lucas_2018_with_2015_texture.csv")
OUT_PARQUET = Path(r"data\LUCAS_DS\lucas_2018_with_2015_texture.parquet")
OUT_EXCEL = Path(r"data\LUCAS_DS\lucas_2018_with_2015_texture.xlsx")

# -------------------------
# Load
# -------------------------
df_2018 = pd.read_csv(LUCAS_2018_CSV)
df_2015 = pd.read_csv(LUCAS_2015_CSV)

print(f"LUCAS 2022 rows: {len(df_2018)}")
print(f"LUCAS 2015 rows: {len(df_2015)}")

# -------------------------
# Normalize join key names + types
# -------------------------
# LUCAS 2022 key might be POINT_ID or Point_id; normalize to POINT_ID
if "POINT_ID" not in df_2018.columns:
    if "Point_id" in df_2018.columns:
        df_2018 = df_2018.rename(columns={"Point_id": "POINT_ID"})
    elif "POINTID" in df_2018.columns:
        df_2018 = df_2018.rename(columns={"POINTID": "POINT_ID"})
    else:
        raise KeyError("Couldn't find a POINT_ID-like column in LUCAS 2022 (expected POINT_ID / Point_id / POINTID).")

# LUCAS 2015 key sometimes is Point_id or POINTID; normalize to POINT_ID
if "POINT_ID" not in df_2015.columns:
    if "Point_ID" in df_2015.columns:
        df_2015 = df_2015.rename(columns={"Point_ID": "POINT_ID"})
    else:
        raise KeyError("Couldn't find a POINT_ID-like column in LUCAS 2015 (expected POINT_ID / Point_id / POINTID).")

# Force join keys to string on both sides
df_2018["POINT_ID"] = df_2018["POINT_ID"].astype(str).str.strip()
df_2015["POINT_ID"] = df_2015["POINT_ID"].astype(str).str.strip()

# -------------------------
# Select only texture columns from 2015 (robust to naming variants)
# -------------------------
# Try common variants seen across LUCAS files
texture_candidates = [
    "Clay", "Silt", "Sand", "Coarse",
    "CLAY", "SILT", "SAND", "COARSE",
    "clay", "silt", "sand", "coarse"
]

available_textures = [c for c in texture_candidates if c in df_2015.columns]
if not available_textures:
    # If your 2015 file uses different names, print columns to inspect
    raise KeyError(
        "Could not find texture columns in LUCAS 2015. "
        "Expected something like Clay/Silt/Sand/Coarse. "
        f"Columns available: {list(df_2015.columns)[:50]} ..."
    )

# Keep only POINT_ID + textures, and de-duplicate per point
df_2015_texture = (
    df_2015[["POINT_ID"] + available_textures]
    .drop_duplicates(subset="POINT_ID")
)

print("Texture columns taken from 2015:", available_textures)
print(f"Unique texture rows (2015): {len(df_2015_texture)}")

# -------------------------
# LEFT JOIN (2022 ← 2015 texture)
# -------------------------
df_joined = df_2018.merge(
    df_2015_texture,
    on="POINT_ID",
    how="left",
    suffixes=("", "_2015")
)

# -------------------------
# Diagnostics
# -------------------------
# How many 2022 rows did NOT get textures?
missing_any = df_joined[available_textures].isna().all(axis=1).sum()
print(f"Rows in 2022 without any texture values from 2015: {missing_any} / {len(df_joined)}")

# -------------------------
# Save outputs
# -------------------------
df_joined.to_csv(OUT_CSV, index=False, sep=",", decimal=".")
df_joined.to_parquet(OUT_PARQUET, index=False)
df_joined.to_excel(OUT_EXCEL, index=False)

print("Saved outputs:")
print(f" - {OUT_CSV}")
print(f" - {OUT_PARQUET}")
print(f" - {OUT_EXCEL}")
