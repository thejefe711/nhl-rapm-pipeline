import duckdb
import pandas as pd

conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
slavin_id = 8476958
season = '20232024'

metrics = [
    'xg_off_rapm_5v5', 
    'xg_def_rapm_5v5', 
    'corsi_off_rapm_5v5', 
    'corsi_def_rapm_5v5', 
    'blocked_shot_to_xg_swing_rapm_5v5_w10'
]

print(f"Jaccob Slavin RAPM ({season}):")
for m in metrics:
    res = conn.execute(f"""
        SELECT value, 
               (SELECT COUNT(*) + 1 FROM apm_results WHERE season='{season}' AND metric_name='{m}' AND value > a.value AND toi_seconds >= 30000) as rank_desc,
               (SELECT COUNT(*) + 1 FROM apm_results WHERE season='{season}' AND metric_name='{m}' AND value < a.value AND toi_seconds >= 30000) as rank_asc,
               (SELECT COUNT(*) FROM apm_results WHERE season='{season}' AND metric_name='{m}' AND toi_seconds >= 30000) as total
        FROM apm_results a
        WHERE player_id = {slavin_id} AND season = '{season}' AND metric_name = '{m}'
    """).df()
    
    if not res.empty:
        val = res.iloc[0]['value']
        total = res.iloc[0]['total']
        if 'def' in m or 'pk' in m:
            rank = res.iloc[0]['rank_asc']
        else:
            rank = res.iloc[0]['rank_desc']
        print(f"{m}: {val:.4f} (Rank {int(rank)}/{int(total)})")
