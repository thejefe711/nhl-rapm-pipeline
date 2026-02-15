import pandas as pd
import glob
import os

# Find the file
pattern = "**/*2024020001_event_on_ice.parquet"
files = glob.glob(pattern, recursive=True)

if not files:
    print(f"File not found matching {pattern}")
    # Try searching in nhl_pipeline
    files = glob.glob(f"nhl_pipeline/{pattern}", recursive=True)

if not files:
    print("File still not found. Listing current directory:")
    print(os.listdir('.'))
    print("Listing nhl_pipeline:")
    if os.path.exists('nhl_pipeline'):
        print(os.listdir('nhl_pipeline'))
    exit(1)

file_path = files[0]
print(f"Found file: {file_path}")

# Read parquet
try:
    event_on_ice = pd.read_parquet(file_path)

    print("Columns:")
    print(event_on_ice.columns.tolist())

    print("\nFirst 5 rows:")
    print(event_on_ice.head())

    print("\nSample row with all values:")
    if not event_on_ice.empty:
        print(event_on_ice.iloc[0].to_dict())
    else:
        print("DataFrame is empty")

except Exception as e:
    print(f"Error reading parquet: {e}")
