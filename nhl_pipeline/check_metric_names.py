#!/usr/bin/env python3
"""Check PP metrics in database."""

import duckdb
from pathlib import Path

con = duckdb.connect(str(Path(__file__).parent / "nhl_canonical.duckdb"), read_only=True)

print("PP metrics in 20242025 (sample):")
r = con.execute("""
    SELECT DISTINCT metric_name 
    FROM apm_results 
    WHERE season = '20242025'
    AND (metric_name LIKE '%pp%' OR metric_name LIKE '%pk%' OR metric_name NOT LIKE '%5v5%')
    ORDER BY metric_name
    LIMIT 20
""").df()
print(r.to_string())

print("\n\nAll distinct metric names across all seasons:")
r2 = con.execute("""
    SELECT DISTINCT metric_name 
    FROM apm_results 
    ORDER BY metric_name
""").df()
print(r2.to_string())
