data directory structure to be

project_root/
  data/
    raw/
      sentinel-2-l2a/
        field_A/                         # aoi.name from config
          sentinel-2-l2a_2024-06-01_2024-06-30_B02-B03-B04-B08-B11-B12_512x512.tif
          sentinel-2-l2a_2024-07-01_2024-07-31_B02-B03-B04-B08-B11-B12_512x512.tif
    indices/
      ndvi/
        field_A/
          sentinel-2-l2a_2024-06-01_2024-06-30_ndvi_512x512.tif
    metadata/
      raw/
        sentinel-2-l2a/
          field_A/
            sentinel-2-l2a_2024-06-01_2024-06-30_B02-B03-B04-B08-B11-B12_512x512.json
