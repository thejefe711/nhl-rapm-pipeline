"""
Discover schema and find player IDs
"""
import duckdb

def main():
    con = duckdb.connect("nhl_canonical.duckdb", read_only=True)
    
    print("Tables in database:")
    tables = con.execute("SHOW TABLES").fetchall()
    for t in tables:
        print(f"  - {t[0]}")
        columns = con.execute(f"DESCRIBE {t[0]}").fetchall()
        for col in columns:
            print(f"    * {col[0]} ({col[1]})")
            
    # Search for McDavid in any table that has a name column
    print("\nSearching for McDavid/Slavin...")
    
    # Check if a players table exists
    tables_list = [t[0] for t in tables]
    if "players" in tables_list:
        p = con.execute("SELECT player_id, full_name FROM players WHERE full_name LIKE 'Connor McDavid%' OR full_name LIKE 'Jaccob Slavin%'").fetchall()
        for r in p:
            print(f"Found in players table: {r}")
            
    # Check apm_results if it has full_name (maybe I added it?)
    if "apm_results" in tables_list:
        cols = [c[0] for c in con.execute("DESCRIBE apm_results").fetchall()]
        if "full_name" in cols:
            p = con.execute("SELECT DISTINCT player_id, full_name FROM apm_results WHERE full_name LIKE 'Connor McDavid%' OR full_name LIKE 'Jaccob Slavin%'").fetchall()
            for r in p:
                print(f"Found in apm_results table: {r}")
        else:
            print("apm_results does NOT have full_name column.")

    con.close()

if __name__ == "__main__":
    main()
