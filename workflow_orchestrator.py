from extract_time_series import main
from bare_soil_aggregate_pixels import main as aggregate_main

def _main(lat, lon , window_m, start_date, end_date):
    main(lat, lon , window_m, start_date, end_date)
    aggregate_main(lat, lon , window_m, start_date, end_date)

if __name__ == "__main__":
    lat = 44.8915307
    lon = 10.01263632
    window_m = 100.0
    start_date = "2024-06-01"
    end_date = "2024-06-30"
    _main(lat, lon , window_m, start_date, end_date)