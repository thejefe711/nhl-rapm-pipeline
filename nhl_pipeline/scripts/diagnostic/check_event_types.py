"""
Check event types in detail
"""
import pandas as pd
from pathlib import Path

canonical_dir = Path("canonical/20252026")
files = list(canonical_dir.glob("*_event_on_ice.parquet"))[:20]

all_events = []
for f in files:
    df = pd.read_parquet(f)
    if "event_type" in df.columns:
        all_events.extend(df["event_type"].tolist())

# Count
from collections import Counter
counts = Counter(all_events)

print("Event type counts (20 sample files):")
print("=" * 50)
for et, count in counts.most_common():
    marker = "<<< TURNOVER" if et in ["TAKEAWAY", "GIVEAWAY"] else ""
    print(f"  {et}: {count} {marker}")
