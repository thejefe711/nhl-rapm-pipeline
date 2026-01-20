#!/usr/bin/env python3
"""Query data structure and samples - output to JSON."""

import duckdb
import json
from pathlib import Path

con = duckdb.connect('nhl_canonical.duckdb', read_only=True)
result = {}

# 1. RAPM output for one player (Connor McDavid = 8478402)
rapm = con.execute('''
    SELECT metric_name, value 
    FROM apm_results 
    WHERE player_id = 8478402 AND season = '20242025'
    ORDER BY metric_name
''').df()
result['1_rapm_mcdavid_2024'] = {r['metric_name']: round(r['value'], 4) for _, r in rapm.iterrows()}

# 2. DLM output
tables = [r[0] for r in con.execute('SHOW TABLES').fetchall()]
dlm_tables = [t for t in tables if 'dlm' in t.lower() or 'forecast' in t.lower()]
result['2_dlm_tables'] = dlm_tables

dlm_samples = {}
for t in dlm_tables:
    schema = con.execute(f'DESCRIBE {t}').df()
    dlm_samples[t] = {
        'schema': schema[['column_name', 'column_type']].to_dict(orient='records'),
        'sample_mcdavid': con.execute(f'SELECT * FROM {t} WHERE player_id = 8478402 LIMIT 5').df().to_dict(orient='records')
    }
result['2_dlm_samples'] = dlm_samples

# 3. Multi-season data
seasons = con.execute('SELECT DISTINCT season FROM apm_results ORDER BY season').df()
result['3_seasons'] = {
    'list': seasons['season'].tolist(),
    'count': len(seasons)
}

# 4. Database structure
result['4_database_structure'] = {
    'file': 'nhl_canonical.duckdb',
    'tables': {}
}
for t in sorted(tables):
    count = con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    result['4_database_structure']['tables'][t] = count

con.close()

Path('reports/data_structure.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
print('Written to reports/data_structure.json')
