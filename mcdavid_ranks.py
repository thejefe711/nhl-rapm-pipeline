import duckdb
import pandas as pd
from pathlib import Path

db_path = Path("nhl_pipeline/nhl_canonical.duckdb")
conn = duckdb.connect(str(db_path))

pid = 8478402 # Connor McDavid

# Get all available metrics from the database
metrics_res = conn.execute("SELECT DISTINCT metric_name FROM apm_results").fetchall()
metrics = [m[0] for m in metrics_res]

# Load position cache
pos_cache_path = Path("nhl_pipeline/profile_data/position_cache.json")
if pos_cache_path.exists():
    import json
    pos_cache = json.loads(pos_cache_path.read_text(encoding="utf-8"))
    pos_cache = {int(k): v for k, v in pos_cache.items()}
else:
    pos_cache = {}

all_ranks = []

# Get all seasons
seasons = conn.execute("SELECT DISTINCT season FROM apm_results ORDER BY season").fetchall()
seasons = [s[0] for s in seasons]

for season in seasons:
    for metric in metrics:
        # Get all players for this season and metric
        query = f"""
        SELECT player_id, value 
        FROM apm_results 
        WHERE season = '{season}' AND metric_name = '{metric}'
        """
        df = conn.execute(query).df()
        
        if df.empty:
            continue
            
        # Add positions
        df['pos'] = df['player_id'].map(pos_cache).fillna('F')
        
        # Calculate ranks (All Players)
        df['rank_all'] = df['value'].rank(ascending=False, method='min')
        df['pct_all'] = df['value'].rank(pct=True) * 100
        
        # Calculate ranks (Same Position)
        mcd_pos = pos_cache.get(pid, 'C')
        # Group by position and rank
        df['rank_pos'] = df.groupby('pos')['value'].rank(ascending=False, method='min')
        df['pct_pos'] = df.groupby('pos')['value'].rank(pct=True) * 100
        
        # Find McDavid
        mcd = df[df['player_id'] == pid]
        if not mcd.empty:
            all_ranks.append({
                'season': season,
                'metric': metric,
                'value': mcd.iloc[0]['value'],
                'rank_all': int(mcd.iloc[0]['rank_all']),
                'total_all': len(df),
                'pct_all': mcd.iloc[0]['pct_all'],
                'rank_pos': int(mcd.iloc[0]['rank_pos']),
                'total_pos': len(df[df['pos'] == mcd_pos]),
                'pct_pos': mcd.iloc[0]['pct_pos'],
                'pos': mcd_pos
            })

conn.close()

# Display results
if all_ranks:
    all_ranks.sort(key=lambda x: (x['season'], x['metric']))
    print(f"Connor McDavid Full RAPM Profile (Position: {all_ranks[0]['pos']}):")
    header = f"{'Season':<10} | {'Metric':<35} | {'Value':<8} | {'Rank(All)':<9} | {'Pct(All)':<8} | {'Rank(Pos)':<9} | {'Pct(Pos)':<8}"
    print(header)
    print("-" * len(header))
    for r in all_ranks:
        print(f"{r['season']:<10} | {r['metric']:<35} | {r['value']:8.4f} | {r['rank_all']:4}/{r['total_all']:<4} | {r['pct_all']:6.1f}% | {r['rank_pos']:4}/{r['total_pos']:<4} | {r['pct_pos']:6.1f}%")
else:
    print("No ranking data found.")
