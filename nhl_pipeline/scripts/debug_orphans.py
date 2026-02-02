import duckdb
import pandas as pd

def main():
    conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
    
    # Check for orphan games in shifts (game_id starts with 2024)
    query = """
    SELECT DISTINCT game_id 
    FROM shifts s 
    WHERE NOT EXISTS (SELECT 1 FROM games g WHERE g.game_id = s.game_id) 
    AND CAST(game_id AS VARCHAR) LIKE '2024%'
    LIMIT 20
    """
    
    df = conn.execute(query).fetchdf()
    print("Orphan Game IDs:")
    print(df)
    
    if not df.empty:
        # Check if these games exist in raw/staging?
        # For now just list them
        pass

if __name__ == "__main__":
    main()
