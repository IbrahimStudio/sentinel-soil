import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("IT_LUCAS_2022.csv")
profile = ProfileReport(df, title="CSV Analysis", explorative=True)
profile.to_file("IT_LUCAS_2022.html")
