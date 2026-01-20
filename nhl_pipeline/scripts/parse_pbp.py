#!/usr/bin/env python3
"""
Parse raw play-by-play JSON into typed, validated rows.

Source: api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play

Column mapping documented for each field.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
import pandas as pd


# Known event types in NHL API
EVENT_TYPES = {
    "faceoff": "FACEOFF",
    "hit": "HIT",
    "giveaway": "GIVEAWAY",
    "goal": "GOAL",
    "shot-on-goal": "SHOT",
    "missed-shot": "MISSED_SHOT",
    "blocked-shot": "BLOCKED_SHOT",
    "penalty": "PENALTY",
    "stoppage": "STOPPAGE",
    "period-start": "PERIOD_START",
    "period-end": "PERIOD_END",
    "game-end": "GAME_END",
    "takeaway": "TAKEAWAY",
    "delayed-penalty": "DELAYED_PENALTY",
    "failed-shot-attempt": "FAILED_SHOT",
}


@dataclass
class EventRow:
    """
    A single play-by-play event.
    
    Column mapping from NHL API:
        game_id          <- from file path
        event_id         <- eventId (int)
        event_type       <- typeDescKey (string, normalized)
        period           <- periodDescriptor.number (int)
        period_type      <- periodDescriptor.periodType (string: REG, OT, SO)
        time_in_period   <- timeInPeriod (string "MM:SS")
        time_remaining   <- timeRemaining (string "MM:SS")
        
    Event details (vary by event type):
        x_coord          <- details.xCoord (float, can be null)
        y_coord          <- details.yCoord (float, can be null)
        zone_code        <- details.zoneCode (string: O, D, N)
        
    Player IDs (vary by event type):
        player_1_id      <- Primary player (shooter, hitter, etc.)
        player_2_id      <- Secondary player (assist, hittee, etc.)
        player_3_id      <- Tertiary player (second assist)
        goalie_id        <- details.goalieInNetId
        
    Team info:
        event_team_id    <- details.eventOwnerTeamId
        
    Computed:
        period_seconds   <- timeInPeriod converted to seconds
        game_seconds     <- Total seconds from game start
    """
    game_id: int
    event_id: int
    event_type: str
    event_type_raw: str      # Original typeDescKey
    period: int
    period_type: str
    time_in_period: str
    time_remaining: str
    
    # Coordinates
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None
    zone_code: Optional[str] = None
    
    # Players
    player_1_id: Optional[int] = None
    player_2_id: Optional[int] = None
    player_3_id: Optional[int] = None
    goalie_id: Optional[int] = None
    
    # Team
    event_team_id: Optional[int] = None
    
    # Computed
    period_seconds: int = 0
    game_seconds: int = 0
    
    # Shot-specific
    shot_type: Optional[str] = None
    
    # Goal-specific
    strength: Optional[str] = None      # EV, PP, SH
    empty_net: bool = False
    
    # Penalty-specific
    penalty_type: Optional[str] = None
    penalty_minutes: Optional[int] = None


def time_to_seconds(time_str: str) -> int:
    """Convert "MM:SS" to seconds."""
    if not time_str:
        return 0
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0
    except (ValueError, AttributeError):
        return 0


def parse_event(raw: Dict[str, Any], game_id: int) -> Optional[EventRow]:
    """Parse a single event from raw API response."""
    try:
        event_id = raw.get("eventId")
        if event_id is None:
            return None
        
        type_desc_key = raw.get("typeDescKey", "")
        period_desc = raw.get("periodDescriptor", {})
        details = raw.get("details", {})
        
        # Normalize event type
        event_type = EVENT_TYPES.get(type_desc_key, type_desc_key.upper())
        
        time_in_period = raw.get("timeInPeriod", "")
        period = period_desc.get("number", 0)
        period_seconds = time_to_seconds(time_in_period)
        
        # Calculate game seconds (20 min periods)
        game_seconds = (period - 1) * 1200 + period_seconds if period > 0 else 0
        
        event = EventRow(
            game_id=game_id,
            event_id=event_id,
            event_type=event_type,
            event_type_raw=type_desc_key,
            period=period,
            period_type=period_desc.get("periodType", ""),
            time_in_period=time_in_period,
            time_remaining=raw.get("timeRemaining", ""),
            x_coord=details.get("xCoord"),
            y_coord=details.get("yCoord"),
            zone_code=details.get("zoneCode"),
            event_team_id=details.get("eventOwnerTeamId"),
            goalie_id=details.get("goalieInNetId"),
            period_seconds=period_seconds,
            game_seconds=game_seconds,
        )
        
        # Extract player IDs based on event type
        if event_type == "SHOT":
            event.player_1_id = details.get("shootingPlayerId")
            event.shot_type = details.get("shotType")
        elif event_type == "GOAL":
            event.player_1_id = details.get("scoringPlayerId")
            event.player_2_id = details.get("assist1PlayerId")
            event.player_3_id = details.get("assist2PlayerId")
            event.shot_type = details.get("shotType")
            event.strength = details.get("strength", {}).get("code") if isinstance(details.get("strength"), dict) else None
            event.empty_net = details.get("emptyNet", False)
        elif event_type == "MISSED_SHOT":
            event.player_1_id = details.get("shootingPlayerId")
            event.shot_type = details.get("shotType")
        elif event_type == "BLOCKED_SHOT":
            event.player_1_id = details.get("shootingPlayerId")  # Shooter
            event.player_2_id = details.get("blockingPlayerId")  # Blocker
        elif event_type == "HIT":
            event.player_1_id = details.get("hittingPlayerId")
            event.player_2_id = details.get("hitteePlayerId")
        elif event_type == "FACEOFF":
            event.player_1_id = details.get("winningPlayerId")
            event.player_2_id = details.get("losingPlayerId")
        elif event_type == "GIVEAWAY":
            event.player_1_id = details.get("playerId")
        elif event_type == "TAKEAWAY":
            event.player_1_id = details.get("playerId")
        elif event_type == "PENALTY":
            event.player_1_id = details.get("committedByPlayerId")
            event.player_2_id = details.get("drawnByPlayerId")
            event.penalty_type = details.get("descKey")
            event.penalty_minutes = details.get("duration")
        
        return event
    except Exception as e:
        print(f"  Warning: Failed to parse event: {e}")
        return None


def parse_pbp_file(pbp_path: Path) -> pd.DataFrame:
    """Parse a raw play-by-play JSON file into a DataFrame."""
    with open(pbp_path) as f:
        raw_data = json.load(f)
    
    # Extract game ID from data
    game_id = raw_data.get("id", 0)
    
    # Events are in the 'plays' key
    raw_events = raw_data.get("plays", [])
    
    print(f"  Raw events count: {len(raw_events)}")
    
    events = []
    for raw in raw_events:
        event = parse_event(raw, game_id)
        if event:
            events.append(asdict(event))
    
    print(f"  Parsed events count: {len(events)}")
    
    if not events:
        return pd.DataFrame()
    
    df = pd.DataFrame(events)
    return df


def analyze_events(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary stats for parsed events."""
    if df.empty:
        return {"error": "No events parsed"}
    
    event_counts = df["event_type"].value_counts().to_dict()
    
    return {
        "total_events": len(df),
        "periods": sorted(df["period"].unique().tolist()),
        "event_types": event_counts,
        "goals": event_counts.get("GOAL", 0),
        "shots": event_counts.get("SHOT", 0),
        "missed_shots": event_counts.get("MISSED_SHOT", 0),
        "blocked_shots": event_counts.get("BLOCKED_SHOT", 0),
        "faceoffs": event_counts.get("FACEOFF", 0),
        "hits": event_counts.get("HIT", 0),
        "has_coordinates": df["x_coord"].notna().sum(),
    }


def main():
    """Parse all raw play-by-play files."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse NHL play-by-play to staging parquet")
    parser.add_argument("--season", type=str, default=None, help="Only parse a specific season (e.g., 20252026)")
    parser.add_argument("--overwrite", action="store_true", help="Re-parse even if staging parquet already exists")
    args = parser.parse_args()

    raw_dir = Path(__file__).parent.parent / "raw"
    staging_dir = Path(__file__).parent.parent / "staging"
    data_dir = Path(__file__).parent.parent / "data"
    staging_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NHL Data Pipeline - Parse Play-by-Play")
    print("=" * 60)
    
    # Import schema registry
    from schema_registry import SchemaRegistry
    
    registry_path = data_dir / "schema_registry.parquet"
    registry = SchemaRegistry(registry_path)
    
    # Find all pbp files
    if args.season:
        pbp_files = list((raw_dir / args.season).glob("*/play_by_play.json"))
    else:
        pbp_files = list(raw_dir.glob("*/*/play_by_play.json"))
    print(f"Found {len(pbp_files)} play-by-play files")
    
    all_results = []
    
    for pbp_path in sorted(pbp_files):
        game_id = pbp_path.parent.name
        season = pbp_path.parent.parent.name
        
        print(f"\nParsing {season}/{game_id}...")
        
        # Load raw data for schema registry
        with open(pbp_path) as f:
            raw_data = json.load(f)
        
        # Update schema registry
        registry.update_from_pbp(season, raw_data.get("plays", []))
        
        # Also update boxscore schema if available
        boxscore_path = pbp_path.parent / "boxscore.json"
        if boxscore_path.exists():
            with open(boxscore_path) as f:
                boxscore_data = json.load(f)
            registry.update_from_boxscore(season, boxscore_data)
        
        # Skip if already parsed (incremental default)
        output_path = staging_dir / season / f"{game_id}_events.parquet"
        if output_path.exists() and not args.overwrite:
            print(f"  SKIP {season}/{game_id} (staging exists)")
            all_results.append({"game_id": game_id, "season": season, "success": True, "skipped": True})
            continue

        df = parse_pbp_file(pbp_path)
        
        if df.empty:
            print(f"  FAIL No events parsed")
            all_results.append({"game_id": game_id, "season": season, "success": False})
            continue
        
        # Save to parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"  Saved: {output_path}")
        
        # Analyze
        stats = analyze_events(df)
        stats["game_id"] = game_id
        stats["season"] = season
        stats["success"] = True
        all_results.append(stats)
        
        print(f"  OK {stats['total_events']} events, {stats['goals']} goals, {stats['shots']} shots")
    
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
                print(f"  OK {r.get('season')}/{r.get('game_id')}: {r.get('total_events', 'NA')} events")
        else:
            print(f"  FAIL {r.get('season')}/{r.get('game_id')}: Failed")
    
    return all_results


if __name__ == "__main__":
    main()
