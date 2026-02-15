import duckdb
import pandas as pd

def audit_rapm():
    db_path = 'nhl_canonical.duckdb'
    conn = duckdb.connect(db_path)
    
    players = ['Connor McDavid', 'Nathan MacKinnon', 'Dylan Strome', 'Blake Lizotte']
    metrics = ['corsi_off_rapm_5v5', 'xg_off_rapm_5v5', 'corsi_pp_off_rapm']
    
    print("=== RAPM Value Audit ===")
    
    for player in players:
        print(f"\n--- {player} ---")
        query = f"""
            SELECT season, metric_name, value 
            FROM apm_results 
            WHERE player_id = (SELECT player_id FROM players WHERE full_name = '{player}')
            AND metric_name IN ({','.join([f"'{m}'" for m in metrics])})
            ORDER BY season, metric_name
        """
        try:
            res = conn.execute(query).df()
            if res.empty:
                print("No data found.")
            else:
                print(res.to_string(index=False))
        except Exception as e:
            print(f"Error querying {player}: {e}")

if __name__ == "__main__":
    audit_rapm()
