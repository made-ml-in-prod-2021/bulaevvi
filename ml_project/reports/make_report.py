import pandas as pd
from pandas_profiling import ProfileReport


INPUT_FILE_NAME = 'data/raw/heart.csv'
OUTPUT_FILE_NAME = 'reports/report.html'


data = pd.read_csv(INPUT_FILE_NAME)
profile = ProfileReport(data)
profile.to_file(output_file = OUTPUT_FILE_NAME)