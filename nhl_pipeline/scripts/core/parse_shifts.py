#!/usr/bin/env python3
"""
Parse raw shift JSON into typed, validated rows.

This is the STAGING layer - we map columns explicitly and document everything.
No derived fields. No interpretation. Just typed extraction.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class ShiftRow:
    """
    A single player shift as extracted from NHL API.
    
    Source: api.nhle.com/stats/rest/en/shiftcharts
    
    Column mapping:
        game_id       <- gameId (int)
        player_id     <- playerId (int)
        team_id       <- teamId (int)
        period        <- period (int)
        start_time    <- startTime (string "MM:SS" -> int seconds)
        end_time      <- endTime (string "MM:SS" -> int seconds)
        duration      <- duration (string "MM:SS" -> int seconds)
        shift_number  <- shiftNumber (int)
        event_number  <- eventNumber (int, can be null)
        type_code     <- typeCode (int, 517=regular shift)
        
    Derived (computed here):
        start_seconds <- startTime converted to period-relative seconds
        end_seconds   <- endTime converted to period-relative seconds
        duration_seconds <- duration converted to seconds
    """
    game_id: int
    player_id: int
    team_id: int
    period: int
    start_time: str          # Raw "MM:SS" string
    end_time: str            # Raw "MM:SS" string
    duration: str            # Raw "MM:SS" string
    shift_number: int
    event_number: Optional[int]
    type_code: int
    
    # Computed from raw times
    start_seconds: int
    end_seconds: int
    duration_seconds: int
    
    # Player info (denormalized for convenience)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    team_abbrev: Optional[str] = None


def time_to_seconds(time_str: str) -> int:
    """Convert "MM:SS" to seconds."""
    if not time_str or time_str == "":
        return 0
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0
    except (ValueError, AttributeError):
        return 0


def parse_shift(raw: Dict[str, Any]) -> Optional[ShiftRow]:
    """Parse a single shift from raw API response."""
    try:
        # Required fields - fail if missing
        game_id = raw.get("gameId")
        player_id = raw.get("playerId")
        team_id = raw.get("teamId")
        period = raw.get("period")
        
        if any(x is None for x in [game_id, player_id, team_id, period]):
            return None
        
        # Time fields
        start_time = raw.get("startTime", "")
        end_time = raw.get("endTime", "")
        duration = raw.get("duration", "")
        
        return ShiftRow(
            game_id=int(game_id),
            player_id=int(player_id),
            team_id=int(team_id),
            period=int(period),
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            shift_number=int(raw.get("shiftNumber", 0)),
            event_number=raw.get("eventNumber"),  # Can be null
            type_code=int(raw.get("typeCode", 0)),
            start_seconds=time_to_seconds(start_time),
            end_seconds=time_to_seconds(end_time),
            duration_seconds=time_to_seconds(duration),
            first_name=raw.get("firstName"),
            last_name=raw.get("lastName"),
            team_abbrev=raw.get("teamAbbrev"),
        )
    except Exception as e:
        print(f"  Warning: Failed to parse shift: {e}")
        return None


def parse_shifts_file(shifts_path: Path) -> pd.DataFrame:
    """Parse a raw shifts JSON file into a DataFrame."""
    with open(shifts_path) as f:
        raw_data = json.load(f)
    
    # Shifts are in the 'data' key
    raw_shifts = raw_data.get("data", [])
    
    print(f"  Raw shifts count: {len(raw_shifts)}")
    
    shifts = []
    for raw in raw_shifts:
        shift = parse_shift(raw)
        if shift:
            shifts.append(asdict(shift))
    
    print(f"  Parsed shifts count: {len(shifts)}")
    
    if not shifts:
        return pd.DataFrame()
    
    df = pd.DataFrame(shifts)
    
    # Type enforcement
    df["game_id"] = df["game_id"].astype(int)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"] = df["team_id"].astype(int)
    df["period"] = df["period"].astype(int)
    df["shift_number"] = df["shift_number"].astype(int)
    df["start_seconds"] = df["start_seconds"].astype(int)
    df["end_seconds"] = df["end_seconds"].astype(int)
    df["duration_seconds"] = df["duration_seconds"].astype(int)
    
    return df


def analyze_shifts(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary stats for parsed shifts."""
    if df.empty:
        return {"error": "No shifts parsed"}
    
    return {
        "total_shifts": len(df),
        "unique_players": df["player_id"].nunique(),
        "unique_teams": df["team_id"].nunique(),
        "periods": sorted(df["period"].unique().tolist()),
        "total_duration_seconds": df["duration_seconds"].sum(),
        "avg_shift_duration": df["duration_seconds"].mean(),
        "max_shift_duration": df["duration_seconds"].max(),
        "shifts_per_period": df.groupby("period").size().to_dict(),
    }


def main():
    """Parse all raw shift files."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse NHL shifts to staging parquet")
    parser.add_argument("--season", type=str, default=None, help="Only parse a specific season (e.g., 20252026)")
    parser.add_argument("--overwrite", action="store_true", help="Re-parse even if staging parquet already exists")
    args = parser.parse_args()

    raw_dir = Path(__file__).parent.parent.parent / "raw"
    staging_dir = Path(__file__).parent.parent.parent / "staging"
    data_dir = Path(__file__).parent.parent.parent / "data"
    staging_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NHL Data Pipeline - Parse Shifts")
    print("=" * 60)
    
    # Import schema registry
    from schema_registry import SchemaRegistry
    
    registry_path = data_dir / "schema_registry.parquet"
    registry = SchemaRegistry(registry_path)
    
    # Find all shift files
    if args.season:
        shift_files = list((raw_dir / args.season).glob("*/shifts.json"))
    else:
        shift_files = list(raw_dir.glob("*/*/shifts.json"))
    print(f"Found {len(shift_files)} shift files")
    
    all_results = []
    
    for shifts_path in sorted(shift_files):
        game_id = shifts_path.parent.name
        season = shifts_path.parent.parent.parent.name
        
        print(f"\nParsing {season}/{game_id}...")
        
        # Load raw data for schema registry
        with open(shifts_path) as f:
            raw_data = json.load(f)
        
        # Update schema registry
        registry.update_from_shifts(season, raw_data.get("data", []))
        
        # Skip if already parsed (incremental default)
        output_path = staging_dir / season / f"{game_id}_shifts.parquet"
        if output_path.exists() and not args.overwrite:
            print(f"  SKIP {season}/{game_id} (staging exists)")
            all_results.append({"game_id": game_id, "season": season, "success": True, "skipped": True})
            continue

        df = parse_shifts_file(shifts_path)
        
        if df.empty:
            print(f"  FAIL No shifts parsed")
            all_results.append({"game_id": game_id, "season": season, "success": False})
            continue
        
        # Save to parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"  Saved: {output_path}")
        
        # Analyze
        stats = analyze_shifts(df)
        stats["game_id"] = game_id
        stats["season"] = season
        stats["success"] = True
        all_results.append(stats)
        
        print(f"  OK {stats['total_shifts']} shifts, {stats['unique_players']} players")
    
    # Save schema registry
    registry.save()
    print(f"\nOK Schema registry saved to {registry_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PARSE SUMMARY")
    print("=" * 60)
    
    for r in all_results:
        if r.get("success"):
            if r.get("skipped"):
                print(f"  SKIP {r.get('season')}/{r.get('game_id')} (staging exists)")
            else:
                print(f"  OK {r.get('season')}/{r.get('game_id')}: {r.get('total_shifts', 'NA')} shifts")
        else:
            print(f"  FAIL {r.get('season')}/{r.get('game_id')}: Failed")
    
    return all_results


if __name__ == "__main__":
    main()
