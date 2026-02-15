
import duckdb
import pandas as pd
from pathlib import Path

def analyze_results():
    db_path = Path("nhl_canonical.duckdb")
    if not db_path.exists():
        print("Database not found!")
        return

    con = duckdb.connect(str(db_path), read_only=True)
    
    season = "20242025"
    metrics = ["corsi_off_rapm_5v5", "xg_off_rapm_5v5"]
    
    print(f"--- RAPM Analysis for {season} ---")
    
    for metric in metrics:
        print(f"\nTop 10 Players for {metric}:")
        df = con.execute(f"""
            SELECT 
                p.full_name,
                a.value as rapm,
                a.toi_seconds / 60.0 as toi_min
            FROM apm_results a
            LEFT JOIN players p ON a.player_id = p.player_id
            WHERE a.metric_name = '{metric}' 
              AND a.season = '{season}'
            ORDER BY a.value DESC
            LIMIT 10
        """).df()
        print(df.to_string(index=False))
        
        print(f"\nElite Players for {metric}:")
        elite_ids = [8478402, 8477492, 8478483, 8479318] # McDavid, MacKinnon, Kucherov, Draisaitl
        df_elite = con.execute(f"""
            SELECT 
                p.full_name,
                a.value as rapm,
                a.toi_seconds / 60.0 as toi_min
            FROM apm_results a
            LEFT JOIN players p ON a.player_id = p.player_id
            WHERE a.metric_name = '{metric}' 
              AND a.season = '{season}'
              AND a.player_id IN ({','.join(map(str, elite_ids))})
            ORDER BY a.value DESC
        """).df()
        print(df_elite.to_string(index=False))

    con.close()

if __name__ == "__main__":
    analyze_results()
