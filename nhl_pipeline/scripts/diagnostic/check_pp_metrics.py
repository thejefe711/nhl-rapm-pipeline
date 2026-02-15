#!/usr/bin/env python3
"""Check PP/PK RAPM metrics in database."""

import duckdb
from pathlib import Path

con = duckdb.connect(str(Path(__file__).parent / "nhl_canonical.duckdb"), read_only=True)

print("PP/PK metrics in 20252026:")
r = con.execute("""
    SELECT metric_name, COUNT(*) as cnt 
    FROM apm_results 
    WHERE season = '20252026'
    AND (metric_name LIKE '%pp%' OR metric_name LIKE '%pk%')
    GROUP BY metric_name 
    ORDER BY metric_name
""").df()
print(r.to_string() if not r.empty else "No PP/PK metrics found")

print("\n\nAll metrics in 20252026:")
r2 = con.execute("""
    SELECT metric_name, COUNT(*) as cnt 
    FROM apm_results 
    WHERE season = '20252026'
    GROUP BY metric_name 
    ORDER BY metric_name
""").df()
print(r2.to_string())

print("\n\nPP metrics by season:")
r3 = con.execute("""
    SELECT season, COUNT(*) as cnt 
    FROM apm_results 
    WHERE metric_name LIKE '%pp%' OR metric_name LIKE '%pk%'
    GROUP BY season 
    ORDER BY season
""").df()
print(r3.to_string() if not r3.empty else "No PP/PK metrics in any season")
