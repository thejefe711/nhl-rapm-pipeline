import duckdb
import pandas as pd

file_path = 'nhl_pipeline/staging/20232024/2023020001_events.parquet'
conn = duckdb.connect()
df = conn.execute(f"SELECT * FROM read_parquet('{file_path}') LIMIT 1").df()
print("All Columns in Staging Events:")
for col in df.columns.tolist():
    print(col)
