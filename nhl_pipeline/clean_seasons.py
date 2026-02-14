import duckdb
con = duckdb.connect('nhl_canonical.duckdb')
# Delete old redundant penalties_taken_rapm_5v5
r1 = con.execute("DELETE FROM apm_results WHERE metric_name='penalties_taken_rapm_5v5'").fetchone()
print(f"Deleted penalties_taken_rapm_5v5: {r1}")
# Delete old penalties_drawn_rapm_5v5 (will be recomputed independently)
r2 = con.execute("DELETE FROM apm_results WHERE metric_name='penalties_drawn_rapm_5v5'").fetchone()
print(f"Deleted penalties_drawn_rapm_5v5: {r2}")
con.close()
print("Done - ready for re-run")
