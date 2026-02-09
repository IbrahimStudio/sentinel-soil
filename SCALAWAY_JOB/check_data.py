import pandas as pd

df = pd.read_csv('scaleway_batch_results_analysis.csv')

print(f'Total records: {len(df)}')
print(f'Records with coverage > 0: {len(df[df["coverage"] > 0])}')
print(f'Unique coverage values: {df["coverage"].unique()}')
print(f'Coverage statistics:')
print(df["coverage"].describe())

print(f'\nSample NDVI values:')
print(df["p50_NDVI"].head(10))

print(f'\nSample MNDWI values:')
print(df["p50_MNDWI"].head(10))

print(f'\nSample records:')
print(df[['lat', 'lon', 'from_time', 'coverage', 'p50_NDVI', 'p50_MNDWI']].head(5))