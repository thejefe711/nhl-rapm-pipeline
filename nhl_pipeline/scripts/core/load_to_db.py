#!/usr/bin/env python3
"""
Load validated data into DuckDB canonical database.

This creates the CANONICAL layer - the only tables models are allowed to read.
Data only enters canonical if it passes validation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import argparse

try:
    import duckdb
except ImportError:
    print("DuckDB not installed. Run: pip install duckdb")
    exit(1)


SCHEMA = """
-- Games table
CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    season VARCHAR NOT NULL,
    game_date DATE,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_team_abbrev VARCHAR,
    away_team_abbrev VARCHAR,
    game_type INTEGER,  -- 2=regular, 3=playoffs
    venue VARCHAR,
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Shifts table (canonical)
CREATE TABLE IF NOT EXISTS shifts (
    game_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    period INTEGER NOT NULL,
    start_seconds INTEGER NOT NULL,
    end_seconds INTEGER NOT NULL,
    duration_seconds INTEGER NOT NULL,
    shift_number INTEGER,
    first_name VARCHAR,
    last_name VARCHAR,
    team_abbrev VARCHAR
);

-- Events table (canonical)
CREATE TABLE IF NOT EXISTS events (
    game_id INTEGER NOT NULL,
    event_id INTEGER NOT NULL,
    event_type VARCHAR NOT NULL,
    period INTEGER NOT NULL,
    period_seconds INTEGER NOT NULL,
    game_seconds INTEGER NOT NULL,
    x_coord DOUBLE,
    y_coord DOUBLE,
    zone_code VARCHAR,
    event_team_id INTEGER,
    player_1_id INTEGER,
    player_2_id INTEGER,
    player_3_id INTEGER,
    goalie_id INTEGER,
    shot_type VARCHAR,
    strength VARCHAR,
    empty_net BOOLEAN,
    PRIMARY KEY (game_id, event_id)
);

-- Players table (aggregated from shifts)
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    first_name VARCHAR,
    last_name VARCHAR,
    full_name VARCHAR,
    first_seen_game_id INTEGER,
    last_seen_game_id INTEGER,
    games_count INTEGER DEFAULT 0
);

-- Teams table (aggregated from shifts)
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    team_abbrev VARCHAR,
    first_seen_game_id INTEGER,
    last_seen_game_id INTEGER,
    games_count INTEGER DEFAULT 0
);

-- Validation log
CREATE TABLE IF NOT EXISTS validation_log (
    game_id INTEGER,
    season VARCHAR,
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tests_passed INTEGER,
    tests_total INTEGER,
    all_passed BOOLEAN,
    details JSON
);

-- APM / RAPM results (metric table)
CREATE TABLE IF NOT EXISTS apm_results (
    season VARCHAR NOT NULL,
    metric_name VARCHAR NOT NULL,      -- e.g. 'corsi_rapm_5v5'
    player_id INTEGER NOT NULL,
    value DOUBLE NOT NULL,
    games_count INTEGER,
    toi_seconds INTEGER,
    events_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (season, metric_name, player_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_shifts_game ON shifts(game_id);
CREATE INDEX IF NOT EXISTS idx_shifts_player ON shifts(player_id);
CREATE INDEX IF NOT EXISTS idx_shifts_period_time ON shifts(game_id, period, start_seconds);
CREATE INDEX IF NOT EXISTS idx_events_game ON events(game_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_period_time ON events(game_id, period, period_seconds);
CREATE INDEX IF NOT EXISTS idx_apm_player_metric ON apm_results(player_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_apm_season_metric ON apm_results(season, metric_name);
"""


def load_game_metadata(boxscore_path: Path) -> Dict[str, Any]:
    """Extract game metadata from boxscore."""
    with open(boxscore_path) as f:
        data = json.load(f)
    
    return {
        "game_id": data.get("id"),
        "game_date": data.get("gameDate"),
        "home_team_id": data.get("homeTeam", {}).get("id"),
        "away_team_id": data.get("awayTeam", {}).get("id"),
        "home_team_abbrev": data.get("homeTeam", {}).get("abbrev"),
        "away_team_abbrev": data.get("awayTeam", {}).get("abbrev"),
        "game_type": data.get("gameType"),
        "venue": data.get("venue", {}).get("default") if data.get("venue") else None,
    }


def load_to_duckdb(
    db_path: Path,
    staging_dir: Path,
    raw_dir: Path,
    validated_games: List[Dict[str, Any]]
):
    """Load all validated games into DuckDB."""
    
    conn = duckdb.connect(str(db_path))
    
    # Migration safety: older versions of this pipeline used an overly-strict primary key on `shifts`
    # (game_id, player_id, period, shift_number). Some games legitimately contain duplicate/zero
    # shift numbers, which causes constraint errors. We treat shifts as an append-only fact table
    # and drop/recreate it on load.
    try:
        conn.execute("DROP TABLE IF EXISTS shifts")
    except Exception:
        pass

    # Create schema
    print("Creating schema...")
    conn.execute(SCHEMA)
    
    for game_info in validated_games:
        season = game_info["season"]
        game_id = game_info["game_id"]
        
        if not game_info.get("all_passed", False):
            print(f"  Skipping {season}/{game_id} - validation failed")
            continue
        
        print(f"\nLoading {season}/{game_id}...")
        
        # Paths
        shifts_path = staging_dir / season / f"{game_id}_shifts.parquet"
        events_path = staging_dir / season / f"{game_id}_events.parquet"
        boxscore_path = raw_dir / season / game_id / "boxscore.json"
        
        # Load game metadata
        metadata = load_game_metadata(boxscore_path)
        metadata["season"] = season
        
        # Insert game
        conn.execute("""
            INSERT OR REPLACE INTO games 
            (game_id, season, game_date, home_team_id, away_team_id, 
             home_team_abbrev, away_team_abbrev, game_type, venue)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            metadata["game_id"], metadata["season"], metadata["game_date"],
            metadata["home_team_id"], metadata["away_team_id"],
            metadata["home_team_abbrev"], metadata["away_team_abbrev"],
            metadata["game_type"], metadata["venue"]
        ])
        print(f"  OK Game metadata loaded")
        
        # Load shifts
        shifts_df = pd.read_parquet(shifts_path)
        shifts_to_load = shifts_df[[
            "game_id", "player_id", "team_id", "period",
            "start_seconds", "end_seconds", "duration_seconds",
            "shift_number", "first_name", "last_name", "team_abbrev"
        ]]
        
        conn.execute("DELETE FROM shifts WHERE game_id = ?", [int(game_id)])
        conn.execute("INSERT INTO shifts SELECT * FROM shifts_to_load")
        print(f"  OK {len(shifts_to_load)} shifts loaded")
        
        # Load events
        events_df = pd.read_parquet(events_path)
        events_to_load = events_df[[
            "game_id", "event_id", "event_type", "period",
            "period_seconds", "game_seconds", "x_coord", "y_coord",
            "zone_code", "event_team_id", "player_1_id", "player_2_id",
            "player_3_id", "goalie_id", "shot_type", "strength", "empty_net"
        ]]
        
        conn.execute("DELETE FROM events WHERE game_id = ?", [int(game_id)])
        conn.execute("INSERT INTO events SELECT * FROM events_to_load")
        print(f"  OK {len(events_to_load)} events loaded")
        
        # Log validation
        conn.execute("""
            INSERT INTO validation_log 
            (game_id, season, tests_passed, tests_total, all_passed, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            int(game_id), season, 
            game_info.get("tests_passed", 0),
            game_info.get("tests_total", 0),
            game_info.get("all_passed", False),
            json.dumps(game_info.get("details", {}))
        ])
    
    # Update players table
    print("\nUpdating players table...")
    conn.execute("""
        INSERT OR REPLACE INTO players (player_id, first_name, last_name, full_name, 
                                         first_seen_game_id, last_seen_game_id, games_count)
        SELECT 
            player_id,
            MAX(first_name) as first_name,
            MAX(last_name) as last_name,
            MAX(first_name) || ' ' || MAX(last_name) as full_name,
            MIN(game_id) as first_seen_game_id,
            MAX(game_id) as last_seen_game_id,
            COUNT(DISTINCT game_id) as games_count
        FROM shifts
        GROUP BY player_id
    """)
    
    # Update teams table
    print("Updating teams table...")
    conn.execute("""
        INSERT OR REPLACE INTO teams (team_id, team_abbrev, first_seen_game_id, 
                                       last_seen_game_id, games_count)
        SELECT 
            team_id,
            MAX(team_abbrev) as team_abbrev,
            MIN(game_id) as first_seen_game_id,
            MAX(game_id) as last_seen_game_id,
            COUNT(DISTINCT game_id) as games_count
        FROM shifts
        GROUP BY team_id
    """)
    
    # Summary
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)
    
    for table in ["games", "shifts", "events", "players", "teams"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count:,} rows")
    
    conn.close()
    print(f"\nOK Database saved to: {db_path}")


def main():
    """Load validated data into DuckDB."""
    raw_dir = Path(__file__).parent.parent.parent / "raw"
    staging_dir = Path(__file__).parent.parent.parent / "staging"
    db_path = Path(__file__).parent.parent.parent / "nhl_canonical.duckdb"
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    print("=" * 60)
    print("NHL Data Pipeline - Load to DuckDB")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Load canonical tables into DuckDB")
    parser.add_argument(
        "--gate",
        type=str,
        choices=["gate1", "gate2"],
        default="gate2",
        help="Which validation gate to use for selecting games to load (default: gate2).",
    )
    args = parser.parse_args()

    validated_games: List[Dict[str, Any]] = []

    if args.gate == "gate2":
        gate2_path = data_dir / "on_ice_validation.json"
        if not gate2_path.exists():
            print(f"WARN Gate 2 file not found at {gate2_path}; falling back to Gate 1 validation.")
            args.gate = "gate1"
        else:
            data = json.loads(gate2_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected Gate 2 format in {gate2_path} (expected list)")
            for r in data:
                if not isinstance(r, dict):
                    continue
                season = str(r.get("season"))
                game_id = str(r.get("game_id"))
                if not season or not game_id:
                    continue
                # Gate 2 result already includes all_passed
                validated_games.append(
                    {
                        "season": season,
                        "game_id": game_id,
                        "all_passed": bool(r.get("all_passed", False)),
                        # Normalize to the existing validation_log schema
                        "tests_passed": 1 if r.get("all_passed") else 0,
                        "tests_total": 1,
                        "details": r,
                    }
                )
            print(f"Loaded {len(validated_games)} games from Gate 2 ({gate2_path})")

    if args.gate == "gate1":
        # Import validation to get results
        from validate_game import validate_game

        # Find all staged games
        staged_games = []
        for shifts_file in staging_dir.glob("*/*_shifts.parquet"):
            game_id = shifts_file.stem.replace("_shifts", "")
            season = shifts_file.parent.name
            staged_games.append((season, game_id))

        print(f"Found {len(staged_games)} staged games")

        # Validate each game
        for season, game_id in sorted(staged_games):
            print(f"\nValidating {season}/{game_id}...")
            validation = validate_game(staging_dir, raw_dir, season, game_id)

            validated_games.append(
                {
                    "season": season,
                    "game_id": game_id,
                    "all_passed": validation.all_passed,
                    "tests_passed": validation.pass_count,
                    "tests_total": len(validation.results),
                    "details": {r.test_name: r.passed for r in validation.results},
                }
            )

            status = "OK" if validation.all_passed else "FAIL"
            print(f"  {status} {validation.pass_count}/{len(validation.results)} tests passed")

    # Load selected games
    load_to_duckdb(db_path, staging_dir, raw_dir, validated_games)


if __name__ == "__main__":
    main()
