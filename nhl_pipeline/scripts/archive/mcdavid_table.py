import duckdb
from pathlib import Path

db_path = Path("nhl_pipeline/nhl_canonical.duckdb")
conn = duckdb.connect(str(db_path))

pid = 8478402
query = f"""
SELECT 
    season, 
    metric_name, 
    value
FROM apm_results 
WHERE player_id = {pid}
AND metric_name IN ('corsi_rapm_5v5', 'xg_rapm_5v5', 'goals_rapm_5v5')
ORDER BY season, metric_name
"""
results = conn.execute(query).fetchall()

data = {}
for season, metric, value in results:
    if season not in data:
        data[season] = {}
    data[season][metric] = value

print("| Season | Corsi RAPM 5v5 | xG RAPM 5v5 | Goals RAPM 5v5 |")
print("| :--- | :---: | :---: | :---: |")
for season in sorted(data.keys()):
    corsi = data[season].get('corsi_rapm_5v5', 0)
    xg = data[season].get('xg_rapm_5v5', 0)
    goals = data[season].get('goals_rapm_5v5', 0)
    print(f"| {season} | {corsi:.4f} | {xg:.4f} | {goals:.4f} |")

conn.close()
