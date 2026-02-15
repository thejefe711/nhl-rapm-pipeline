#!/usr/bin/env python3
"""Check SPECIAL_TEAMS metrics in database."""

import duckdb
from pathlib import Path

con = duckdb.connect(str(Path(__file__).parent / "nhl_canonical.duckdb"), read_only=True)

print("Metrics in 20252026:")
r = con.execute("""
    SELECT metric_name, COUNT(*) as cnt 
    FROM apm_results 
    WHERE season = '20252026' 
    GROUP BY metric_name 
    ORDER BY metric_name
""").df()
print(r.to_string())

print("\n\nSPECIAL_TEAMS metrics across all seasons:")
r2 = con.execute("""
    SELECT season, COUNT(*) as cnt 
    FROM apm_results 
    WHERE metric_name LIKE '%pp%' OR metric_name LIKE '%pk%'
    GROUP BY season 
    ORDER BY season
""").df()
print(r2.to_string())
