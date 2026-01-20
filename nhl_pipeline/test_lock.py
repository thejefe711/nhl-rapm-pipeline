import duckdb
try:
    con = duckdb.connect('nhl_canonical.duckdb')
    print("Successfully connected to DuckDB (read-write)")
    con.close()
except Exception as e:
    print(f"Failed to connect: {e}")
