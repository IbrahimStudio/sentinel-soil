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


Index(['row', 'col', 'x', 'y', 'B02_median_bs', 'B02_mean_bs', 'B02_std_bs',
       'B03_median_bs', 'B03_mean_bs', 'B03_std_bs', 'B04_median_bs',
       'B04_mean_bs', 'B04_std_bs', 'B08_median_bs', 'B08_mean_bs',
       'B08_std_bs', 'B11_median_bs', 'B11_mean_bs', 'B11_std_bs',
       'B12_median_bs', 'B12_mean_bs', 'B12_std_bs', 'NDVI_median_bs',
       'NDVI_mean_bs', 'NDVI_std_bs', 'NDMI_median_bs', 'NDMI_mean_bs',
       'NDMI_std_bs', 'NDWI_median_bs', 'NDWI_mean_bs', 'NDWI_std_bs',
       'NBR_median_bs', 'NBR_mean_bs', 'NBR_std_bs', 'MSAVI_median_bs',
       'MSAVI_mean_bs', 'MSAVI_std_bs', 'EVI_median_bs', 'EVI_mean_bs',
       'EVI_std_bs', 'n_obs_baresoil'],
      dtype='object')


citations: https://developers.google.com/earth-engine/datasets/catalog/JRC_LUCAS_HARMO_THLOC_V1?utm_source=chatgpt.com&hl=it#description
LUCAS
