import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
q = """
SELECT 
    COUNT(*) as total_rows,
    SUM(CASE WHEN value = 0 THEN 1 ELSE 0 END) as zero_count,
    AVG(value) as avg_value
FROM apm_results
WHERE season = '20252026'
AND metric_name = 'xg_off_rapm_5v5';
"""
print(con.execute(q).df())
con.close()
