"""
Verify TOI calculation across data sources.
Compares: Shifts → apm_results → Boxscore
"""
import json
import pandas as pd
import duckdb
from pathlib import Path


def verify_toi_for_player(player_id: int, player_name: str, season: str = "20242025"):
    root = Path(".")
    staging_dir = root / "staging" / season
    raw_dir = root / "raw" / season
    db_path = root / "nhl_canonical.duckdb"
    
    print(f"\n=== TOI Verification: {player_name} ({season}) ===")
    
    # 1. Calculate TOI from Shifts (Ground Truth)
    shift_files = list(staging_dir.glob("*_shifts.parquet"))
    shift_toi = 0
    games_with_player = 0
    
    for sf in shift_files:
        df = pd.read_parquet(sf)
        player_shifts = df[df["player_id"] == player_id]
        if not player_shifts.empty:
            games_with_player += 1
            shift_toi += player_shifts["duration_seconds"].sum()
    
    print(f"1. FROM SHIFTS:")
    print(f"   Games found: {games_with_player}")
    print(f"   Total TOI:   {shift_toi} sec = {shift_toi/60:.1f} min")
    
    # 2. Get TOI from apm_results (Pipeline Output)
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        apm_df = con.execute(f"""
            SELECT metric_name, toi_seconds, games_count
            FROM apm_results
            WHERE player_id = {player_id} AND season = '{season}'
            LIMIT 5
        """).df()
        con.close()
        
        if not apm_df.empty:
            max_toi = apm_df["toi_seconds"].max()
            max_games = apm_df["games_count"].max()
            print(f"2. FROM apm_results:")
            print(f"   Games:     {max_games}")
            print(f"   Max TOI:   {max_toi} sec = {max_toi/60:.1f} min")
        else:
            print(f"2. FROM apm_results: NO DATA FOUND")
    except Exception as e:
        print(f"2. FROM apm_results: ERROR - {e}")
    
    # 3. Get TOI from Boxscores (NHL Official)
    boxscore_toi = 0
    boxscore_games = 0
    
    for game_dir in raw_dir.iterdir():
        if not game_dir.is_dir():
            continue
        boxscore_path = game_dir / "boxscore.json"
        if not boxscore_path.exists():
            continue
        
        with open(boxscore_path, 'r') as f:
            data = json.load(f)
        
        pbgs = data.get("playerByGameStats", {})
        for team_key in ["homeTeam", "awayTeam"]:
            team_data = pbgs.get(team_key, {})
            for group in ["forwards", "defense"]:
                for player in team_data.get(group, []):
                    if player.get("playerId") == player_id:
                        boxscore_games += 1
                        toi_str = player.get("toi", "00:00")
                        parts = toi_str.split(":")
                        boxscore_toi += int(parts[0]) * 60 + int(parts[1])
    
    print(f"3. FROM BOXSCORES:")
    print(f"   Games:     {boxscore_games}")
    print(f"   Total TOI: {boxscore_toi} sec = {boxscore_toi/60:.1f} min")
    
    # Summary
    print(f"\n--- DISCREPANCY CHECK ---")
    if shift_toi > 0:
        apm_pct = (max_toi / shift_toi * 100) if 'max_toi' in dir() and max_toi else 0
        box_pct = (boxscore_toi / shift_toi * 100) if boxscore_toi else 0
        print(f"  apm_results vs Shifts: {apm_pct:.1f}%")
        print(f"  Boxscore vs Shifts:    {box_pct:.1f}%")


if __name__ == "__main__":
    # Test with Sebastian Aho (F) and Connor McDavid
    verify_toi_for_player(8478427, "Sebastian Aho (F)")
    verify_toi_for_player(8478402, "Connor McDavid")
