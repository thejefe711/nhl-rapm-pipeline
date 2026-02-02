import duckdb
import pandas as pd

def main():
    conn = duckdb.connect('nhl_pipeline/nhl_canonical.duckdb')
    
    print("=== Investigating Orphan Players ===")
    orphans = conn.execute("""
        SELECT DISTINCT e.player_1_id 
        FROM events e 
        LEFT JOIN players p ON e.player_1_id = p.player_id
        WHERE p.player_id IS NULL
        AND CAST(e.game_id AS VARCHAR) LIKE '2024%'
        LIMIT 10
    """).fetchdf()
    print(orphans)
    
    print("\n=== Investigating Game Count ===")
    count = conn.execute("SELECT COUNT(*) FROM games WHERE season = 2024").fetchone()[0]
    print(f"Game count for season=2024: {count}")
    
    # Check if maybe season is stored differently
    sample_games = conn.execute("SELECT * FROM games LIMIT 5").fetchdf()
    print("\nSample games data:")
    print(sample_games)

    print("\n=== Investigating Duplicate Events ===")
    dupes = conn.execute("""
        SELECT event_id, COUNT(*) as cnt
        FROM events 
        WHERE CAST(game_id AS VARCHAR) LIKE '2024%'
        GROUP BY event_id 
        HAVING COUNT(*) > 1
        LIMIT 5
    """).fetchdf()
    print(dupes)

if __name__ == "__main__":
    main()
