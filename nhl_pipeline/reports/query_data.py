#!/usr/bin/env python3
"""Query data structure and samples for user."""

import duckdb
import json

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)

# 1. RAPM output for one player (Connor McDavid = 8478402)
print('=== 1. RAPM OUTPUT (Connor McDavid, 2024-2025) ===')
rapm = con.execute('''
    SELECT metric_name, value 
    FROM apm_results 
    WHERE player_id = 8478402 AND season = '20242025'
    ORDER BY metric_name
''').df()
for _, r in rapm.iterrows():
    print(f"  {r['metric_name']:45s}: {r['value']:+.4f}")

# 2. DLM output
print('\n=== 2. DLM OUTPUT ===')
tables = [r[0] for r in con.execute('SHOW TABLES').fetchall()]
dlm_tables = [t for t in tables if 'dlm' in t.lower() or 'forecast' in t.lower()]
print(f'DLM-related tables: {dlm_tables}')

for t in dlm_tables:
    print(f'\nTable: {t}')
    schema = con.execute(f'DESCRIBE {t}').df()
    print(schema[['column_name', 'column_type']].to_string(index=False))
    sample = con.execute(f'SELECT * FROM {t} WHERE player_id = 8478402 LIMIT 5').df()
    if not sample.empty:
        print('Sample for McDavid:')
        print(sample.to_string(index=False))
    else:
        print('(no data for McDavid)')

# 3. Multi-season data
print('\n=== 3. SEASONS AVAILABLE ===')
seasons = con.execute('SELECT DISTINCT season FROM apm_results ORDER BY season').df()
print(f'Seasons in apm_results: {seasons["season"].tolist()}')
print(f'Total: {len(seasons)} seasons')

# 4. Database structure
print('\n=== 4. DATABASE STRUCTURE ===')
print(f'Database: nhl_canonical.duckdb')
print(f'Tables ({len(tables)}):')
for t in sorted(tables):
    count = con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'  {t}: {count:,} rows')

con.close()
