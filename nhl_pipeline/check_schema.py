import duckdb
con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

# Check all table schemas
tables = ['players', 'shifts', 'events', 'games', 'apm_results']
for table in tables:
    print(f"\n=== {table} ===")
    try:
        df = con.execute(f'SELECT * FROM {table} LIMIT 1').df()
        for col in df.columns:
            print(f"  {col}")
    except Exception as e:
        print(f"  Error: {e}")

# Check if there's position in apm_results
print("\n=== Sample apm_results ===")
print(con.execute("SELECT DISTINCT metric_name FROM apm_results WHERE metric_name LIKE '%position%' LIMIT 10").df())

con.close()
