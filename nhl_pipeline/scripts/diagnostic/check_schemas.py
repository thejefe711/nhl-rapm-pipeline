import duckdb
import pandas as pd

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

for table in ['players', 'apm_results', 'events', 'teams']:
    print(f"\n--- {table} ---")
    try:
        print(con.execute(f"PRAGMA table_info('{table}')").df()[['name', 'type']])
    except Exception as e:
        print(f"Error getting info for {table}: {e}")

con.close()
