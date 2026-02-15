import json
import duckdb
from pathlib import Path

def populate_player_info():
    print("Scanning raw game files for player info...")
    
    # Dictionary to store unique players: id -> {name, pos, first_s, last_s}
    players = {}
    
    # Scan all seasons in raw/
    raw_dir = Path("raw")
    if not raw_dir.exists():
        print("Error: 'raw' directory not found.")
        return

    season_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f"Found seasons: {[d.name for d in season_dirs]}")

    for season_dir in season_dirs:
        season_id = int(season_dir.name)
        game_dirs = [d for d in season_dir.iterdir() if d.is_dir()]
        print(f"  Scanning {season_id}: {len(game_dirs)} games...")
        
        for i, game_dir in enumerate(game_dirs):
            if i % 100 == 0:
                print(f"    Processed {i}/{len(game_dirs)} games...", end='\r')
            pbp_path = game_dir / "play_by_play.json"
            if not pbp_path.exists():
                continue
                
            try:
                # Read raw JSON
                data = json.loads(pbp_path.read_text(encoding="utf-8"))
                
                # Check rosterSpots (most reliable source)
                roster = data.get("rosterSpots", [])
                for p in roster:
                    pid = p.get("playerId")
                    if not pid:
                        continue
                        
                    # Extract info
                    fname = p.get("firstName", {}).get("default", "")
                    lname = p.get("lastName", {}).get("default", "")
                    full_name = f"{fname} {lname}".strip()
                    pos = p.get("positionCode", "UNK")
                    
                    if pid not in players:
                        players[pid] = {
                            "player_name": full_name,
                            "position": pos,
                            "first_season": season_id,
                            "last_season": season_id
                        }
                    else:
                        # Update seasons range
                        players[pid]["last_season"] = max(players[pid]["last_season"], season_id)
                        players[pid]["first_season"] = min(players[pid]["first_season"], season_id)
                        
                        # Update name if it was empty/short (sometimes data quality varies)
                        if len(full_name) > len(players[pid]["player_name"]):
                            players[pid]["player_name"] = full_name
            except Exception as e:
                print(f"Error reading {pbp_path}: {e}")

    print(f"\nFound {len(players):,} unique players.")
    
    # Connect to DuckDB
    con = duckdb.connect("nhl_canonical.duckdb")
    
    # Create table
    con.execute("DROP TABLE IF EXISTS player_info")
    con.execute("""
        CREATE TABLE player_info (
            player_id INTEGER PRIMARY KEY,
            player_name VARCHAR,
            position VARCHAR,
            first_season INTEGER,
            last_season INTEGER
        )
    """)
    
    # Prepare data for insertion
    to_insert = []
    for pid, info in players.items():
        to_insert.append((
            pid,
            info["player_name"],
            info["position"],
            info["first_season"],
            info["last_season"]
        ))
    
    # Bulk insert
    print("Inserting into DuckDB...")
    con.executemany("INSERT INTO player_info VALUES (?, ?, ?, ?, ?)", to_insert)
    
    # Verify coverage
    print("Verifying against apm_results...")
    res = con.execute("""
        SELECT COUNT(DISTINCT ar.player_id) as missing_count
        FROM apm_results ar
        LEFT JOIN player_info pi ON ar.player_id = pi.player_id
        WHERE pi.player_id IS NULL
    """).fetchone()
    
    missing = res[0]
    print(f"Missing player IDs in player_info: {missing}")
    
    # Show sample
    print("\nSample Data:")
    print(con.execute("SELECT * FROM player_info LIMIT 5").fetchdf())
    
    con.close()
    print("\nDone!")

if __name__ == "__main__":
    populate_player_info()
