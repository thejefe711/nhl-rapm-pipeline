
import pandas as pd
import time
from pathlib import Path

path = Path("staging/20242025/shots_with_xg.parquet")
if not path.exists():
    print("File not found")
    exit(1)

start = time.time()
df = pd.read_parquet(path)
end = time.time()
print(f"Read {len(df)} rows in {end - start:.4f} seconds")

# Benchmark repeated reading
start = time.time()
for _ in range(10):
    pd.read_parquet(path)
end = time.time()
print(f"Read 10 times in {end - start:.4f} seconds (avg {(end - start)/10:.4f}s)")
