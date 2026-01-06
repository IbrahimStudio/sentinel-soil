import pandas as pd
from pathlib import Path

# -------------------------
# Paths
# -------------------------
LUCAS_2015_CSV = Path("data\LUCAS_DS\LUCAS_Topsoil_2015_20200323.csv")
LUCAS_2009_CSV = Path("data\LUCAS_DS\LUCAS_TOPSOIL_v1.xlsx")

OUT_CSV = Path("data\LUCAS_DS\lucas_2015_with_coords.csv")
OUT_PARQUET = Path("data\LUCAS_DS\lucas_2015_with_coords.parquet")

# -------------------------
# Load
# -------------------------
df_2015 = pd.read_csv(LUCAS_2015_CSV, dtype={"POINT_ID": str})
df_2009 = pd.read_excel(LUCAS_2009_CSV)

print(f"LUCAS 2015 rows: {len(df_2015)}")
print(f"LUCAS 2009 rows: {len(df_2009)}")

# -------------------------
# Clean POINT_ID in 2009
# -------------------------

# Convert to string
df_2009["POINT_ID"] = df_2009["POINT_ID"].astype(str)

# Remove invalid POINT_IDs
invalid_mask = (
    df_2009["POINT_ID"].isna()
    | df_2009["POINT_ID"].str.upper().isin(["NE", "NAN"])
)

df_2009_valid = df_2009.loc[~invalid_mask].copy()

# Normalize float-like IDs (e.g. "1234567.0" → "1234567")
df_2009_valid["POINT_ID"] = (
    df_2009_valid["POINT_ID"]
    .str.replace(r"\.0$", "", regex=True)
)

print(f"Valid LUCAS 2009 points after cleaning: {len(df_2009_valid)}")

# -------------------------
# Build lookup table
# -------------------------
df_2009_lookup = (
    df_2009_valid[["POINT_ID", "GPS_LAT", "GPS_LONG"]]
    .dropna(subset=["GPS_LAT", "GPS_LONG"])
    .drop_duplicates(subset="POINT_ID")
)

print(f"Unique POINT_IDs in 2009 lookup: {len(df_2009_lookup)}")

# -------------------------
# LEFT JOIN (2015 ← 2009)
# -------------------------
df_2015 = df_2015.rename(columns={"Point_ID": "POINT_ID"})
df_2015["POINT_ID"] = df_2015["POINT_ID"].astype(str)
df_2009_lookup["POINT_ID"] = df_2009_lookup["POINT_ID"].astype(str)

df_joined = df_2015.merge(
    df_2009_lookup,
    on="POINT_ID",
    how="left"
)
df_joined = df_joined.rename(columns={"GPS_LAT": "lat", "GPS_LONG": "lon"})

# -------------------------
# Diagnostics
# -------------------------
missing_coords = df_joined["lat"].isna().sum()
print(f"Rows in 2015 without GPS coords: {missing_coords}")

if missing_coords > 0:
    print("Example missing POINT_IDs:")
    print(
        df_joined.loc[df_joined["lat"].isna(), "POINT_ID"]
        .head(10)
    )

# -------------------------
# Save outputs
# -------------------------
df_joined.to_csv(OUT_CSV, index=False)
df_joined.to_parquet(OUT_PARQUET, index=False)

print("Saved outputs:")
print(f" - {OUT_CSV}")
print(f" - {OUT_PARQUET}")
