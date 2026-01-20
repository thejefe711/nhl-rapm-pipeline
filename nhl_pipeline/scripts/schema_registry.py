#!/usr/bin/env python3
"""
Schema Registry - Track field mappings across seasons.

Detects when NHL API structure changes between seasons.
Stores field presence rates to identify problematic fields.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict, field
import pandas as pd


@dataclass
class FieldMapping:
    """A single field mapping from API to canonical schema."""
    endpoint: str           # 'shifts', 'pbp', 'boxscore'
    season: str             # '20242025'
    api_field: str          # 'details.shootingPlayerId'
    canonical_name: str     # 'shooter_id'
    data_type: str          # 'int', 'float', 'str', 'bool'
    nullable: bool          # Can this field be null?
    games_checked: int = 0
    games_present: int = 0
    sample_values: List[Any] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    
    @property
    def presence_rate(self) -> float:
        if self.games_checked == 0:
            return 0.0
        return 100.0 * self.games_present / self.games_checked
    
    @property
    def status(self) -> str:
        rate = self.presence_rate
        if rate >= 99.9:
            return "✓"
        elif rate >= 95.0:
            return "⚠️"
        else:
            return "✗"


# Expected field mappings per endpoint
# This is the "contract" - if the API changes, we detect it here
EXPECTED_SHIFTS_FIELDS = {
    "gameId": ("game_id", "int", False),
    "playerId": ("player_id", "int", False),
    "teamId": ("team_id", "int", False),
    "period": ("period", "int", False),
    "startTime": ("start_time", "str", False),
    "endTime": ("end_time", "str", False),
    "duration": ("duration", "str", False),
    "shiftNumber": ("shift_number", "int", False),
    "firstName": ("first_name", "str", True),
    "lastName": ("last_name", "str", True),
    "teamAbbrev": ("team_abbrev", "str", True),
    "eventNumber": ("event_number", "int", True),
    "typeCode": ("type_code", "int", True),
}

EXPECTED_PBP_FIELDS = {
    "eventId": ("event_id", "int", False),
    "typeDescKey": ("event_type", "str", False),
    "periodDescriptor.number": ("period", "int", False),
    "periodDescriptor.periodType": ("period_type", "str", False),
    "timeInPeriod": ("time_in_period", "str", False),
    "timeRemaining": ("time_remaining", "str", True),
    "details.xCoord": ("x_coord", "float", True),
    "details.yCoord": ("y_coord", "float", True),
    "details.zoneCode": ("zone_code", "str", True),
    "details.eventOwnerTeamId": ("event_team_id", "int", True),
    "details.shootingPlayerId": ("shooter_id", "int", True),
    "details.goalieInNetId": ("goalie_id", "int", True),
    "details.scoringPlayerId": ("scorer_id", "int", True),
    "details.assist1PlayerId": ("assist1_id", "int", True),
    "details.assist2PlayerId": ("assist2_id", "int", True),
}

EXPECTED_BOXSCORE_FIELDS = {
    "id": ("game_id", "int", False),
    "gameDate": ("game_date", "str", False),
    "gameType": ("game_type", "int", False),
    "homeTeam.id": ("home_team_id", "int", False),
    "awayTeam.id": ("away_team_id", "int", False),
    "homeTeam.abbrev": ("home_team_abbrev", "str", False),
    "awayTeam.abbrev": ("away_team_abbrev", "str", False),
    "venue.default": ("venue", "str", True),
}


def get_nested_value(data: Dict, path: str) -> Any:
    """Get a value from nested dict using dot notation."""
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def check_field_presence(data: List[Dict], field_path: str) -> tuple[int, int, List[Any]]:
    """Check how many records have a field present and non-null."""
    present = 0
    samples = []
    
    for record in data:
        value = get_nested_value(record, field_path)
        if value is not None:
            present += 1
            if len(samples) < 5:
                samples.append(value)
    
    return len(data), present, samples


class SchemaRegistry:
    """Track and validate schema across seasons."""
    
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.mappings: Dict[str, FieldMapping] = {}  # key = endpoint:season:field
        self._load()
    
    def _load(self):
        """Load existing registry from disk."""
        if self.registry_path.exists():
            df = pd.read_parquet(self.registry_path)
            for _, row in df.iterrows():
                mapping = FieldMapping(
                    endpoint=row["endpoint"],
                    season=row["season"],
                    api_field=row["api_field"],
                    canonical_name=row["canonical_name"],
                    data_type=row["data_type"],
                    nullable=row["nullable"],
                    games_checked=row["games_checked"],
                    games_present=row["games_present"],
                    sample_values=json.loads(row["sample_values"]) if row["sample_values"] else [],
                    first_seen=row["first_seen"],
                    last_seen=row["last_seen"],
                )
                key = f"{mapping.endpoint}:{mapping.season}:{mapping.api_field}"
                self.mappings[key] = mapping
    
    def _save(self):
        """Save registry to disk."""
        if not self.mappings:
            return
        
        rows = []
        for mapping in self.mappings.values():
            row = asdict(mapping)
            row["sample_values"] = json.dumps(row["sample_values"])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.registry_path, index=False)
    
    def _get_key(self, endpoint: str, season: str, field: str) -> str:
        return f"{endpoint}:{season}:{field}"
    
    def update_from_shifts(self, season: str, shifts_data: List[Dict]):
        """Update registry from shifts data for a game."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        for api_field, (canonical, dtype, nullable) in EXPECTED_SHIFTS_FIELDS.items():
            key = self._get_key("shifts", season, api_field)
            
            checked, present, samples = check_field_presence(shifts_data, api_field)
            
            if key not in self.mappings:
                self.mappings[key] = FieldMapping(
                    endpoint="shifts",
                    season=season,
                    api_field=api_field,
                    canonical_name=canonical,
                    data_type=dtype,
                    nullable=nullable,
                    games_checked=0,
                    games_present=0,
                    first_seen=today,
                    last_seen=today,
                )
            
            mapping = self.mappings[key]
            mapping.games_checked += 1
            if present > 0:
                mapping.games_present += 1
            mapping.last_seen = today
            if samples and len(mapping.sample_values) < 5:
                mapping.sample_values.extend(samples[:5 - len(mapping.sample_values)])
    
    def update_from_pbp(self, season: str, events_data: List[Dict]):
        """Update registry from play-by-play data for a game."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        for api_field, (canonical, dtype, nullable) in EXPECTED_PBP_FIELDS.items():
            key = self._get_key("pbp", season, api_field)
            
            checked, present, samples = check_field_presence(events_data, api_field)
            
            if key not in self.mappings:
                self.mappings[key] = FieldMapping(
                    endpoint="pbp",
                    season=season,
                    api_field=api_field,
                    canonical_name=canonical,
                    data_type=dtype,
                    nullable=nullable,
                    games_checked=0,
                    games_present=0,
                    first_seen=today,
                    last_seen=today,
                )
            
            mapping = self.mappings[key]
            mapping.games_checked += 1
            if present > 0:
                mapping.games_present += 1
            mapping.last_seen = today
            if samples and len(mapping.sample_values) < 5:
                mapping.sample_values.extend(samples[:5 - len(mapping.sample_values)])
    
    def update_from_boxscore(self, season: str, boxscore_data: Dict):
        """Update registry from boxscore data for a game."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        for api_field, (canonical, dtype, nullable) in EXPECTED_BOXSCORE_FIELDS.items():
            key = self._get_key("boxscore", season, api_field)
            
            value = get_nested_value(boxscore_data, api_field)
            present = 1 if value is not None else 0
            
            if key not in self.mappings:
                self.mappings[key] = FieldMapping(
                    endpoint="boxscore",
                    season=season,
                    api_field=api_field,
                    canonical_name=canonical,
                    data_type=dtype,
                    nullable=nullable,
                    games_checked=0,
                    games_present=0,
                    first_seen=today,
                    last_seen=today,
                )
            
            mapping = self.mappings[key]
            mapping.games_checked += 1
            if present > 0:
                mapping.games_present += 1
            mapping.last_seen = today
            if value is not None and len(mapping.sample_values) < 5:
                mapping.sample_values.append(value)
    
    def save(self):
        """Persist registry to disk."""
        self._save()
    
    def get_summary_df(self) -> pd.DataFrame:
        """Get registry as DataFrame for reporting."""
        rows = []
        for mapping in self.mappings.values():
            rows.append({
                "endpoint": mapping.endpoint,
                "season": mapping.season,
                "api_field": mapping.api_field,
                "canonical_name": mapping.canonical_name,
                "data_type": mapping.data_type,
                "nullable": mapping.nullable,
                "games_checked": mapping.games_checked,
                "games_present": mapping.games_present,
                "presence_rate": mapping.presence_rate,
                "status": mapping.status,
                "first_seen": mapping.first_seen,
                "last_seen": mapping.last_seen,
            })
        
        if not rows:
            return pd.DataFrame()
        
        return pd.DataFrame(rows).sort_values(
            ["endpoint", "season", "presence_rate"],
            ascending=[True, False, True]
        )
    
    def get_problems(self) -> pd.DataFrame:
        """Get fields with low presence rates (potential issues)."""
        df = self.get_summary_df()
        if df.empty:
            return df
        
        # Non-nullable fields with <100% presence = problem
        # Nullable fields with <90% presence = warning
        problems = df[
            ((~df["nullable"]) & (df["presence_rate"] < 100)) |
            ((df["nullable"]) & (df["presence_rate"] < 90))
        ]
        
        return problems
    
    def detect_schema_changes(self) -> List[Dict]:
        """Detect fields that changed between seasons."""
        changes = []
        
        # Group by endpoint and field
        df = self.get_summary_df()
        if df.empty:
            return changes
        
        for (endpoint, api_field), group in df.groupby(["endpoint", "api_field"]):
            if len(group) < 2:
                continue
            
            # Compare presence rates across seasons
            group = group.sort_values("season")
            rates = group["presence_rate"].tolist()
            seasons = group["season"].tolist()
            
            for i in range(1, len(rates)):
                diff = rates[i] - rates[i-1]
                if abs(diff) > 5:  # >5% change
                    changes.append({
                        "endpoint": endpoint,
                        "field": api_field,
                        "from_season": seasons[i-1],
                        "to_season": seasons[i],
                        "from_rate": rates[i-1],
                        "to_rate": rates[i],
                        "change": diff,
                    })
        
        return changes


def main():
    """Demo the schema registry."""
    import json
    
    registry_path = Path(__file__).parent.parent / "data" / "schema_registry.parquet"
    registry = SchemaRegistry(registry_path)
    
    # Load some sample data and update registry
    raw_dir = Path(__file__).parent.parent / "raw"
    
    for season_dir in sorted(raw_dir.glob("*")):
        if not season_dir.is_dir():
            continue
        
        season = season_dir.name
        
        for game_dir in season_dir.glob("*"):
            if not game_dir.is_dir():
                continue
            
            shifts_path = game_dir / "shifts.json"
            pbp_path = game_dir / "play_by_play.json"
            boxscore_path = game_dir / "boxscore.json"
            
            if shifts_path.exists():
                with open(shifts_path) as f:
                    data = json.load(f)
                registry.update_from_shifts(season, data.get("data", []))
            
            if pbp_path.exists():
                with open(pbp_path) as f:
                    data = json.load(f)
                registry.update_from_pbp(season, data.get("plays", []))
            
            if boxscore_path.exists():
                with open(boxscore_path) as f:
                    data = json.load(f)
                registry.update_from_boxscore(season, data)
    
    registry.save()
    
    # Print summary
    print("=" * 70)
    print("SCHEMA REGISTRY SUMMARY")
    print("=" * 70)
    
    df = registry.get_summary_df()
    if df.empty:
        print("No data in registry yet. Run the pipeline first.")
        return
    
    # Summary by endpoint and season
    for endpoint in df["endpoint"].unique():
        print(f"\n📋 {endpoint.upper()}")
        print("-" * 50)
        
        endpoint_df = df[df["endpoint"] == endpoint]
        
        for season in sorted(endpoint_df["season"].unique(), reverse=True):
            season_df = endpoint_df[endpoint_df["season"] == season]
            total = len(season_df)
            ok = (season_df["presence_rate"] >= 99).sum()
            warn = ((season_df["presence_rate"] >= 90) & (season_df["presence_rate"] < 99)).sum()
            bad = (season_df["presence_rate"] < 90).sum()
            
            print(f"  {season}: {ok}✓ {warn}⚠️  {bad}✗  ({total} fields)")
    
    # Problems
    problems = registry.get_problems()
    if not problems.empty:
        print("\n" + "=" * 70)
        print("⚠️  PROBLEM FIELDS")
        print("=" * 70)
        for _, row in problems.iterrows():
            print(f"  {row['endpoint']}/{row['season']}: {row['api_field']} = {row['presence_rate']:.1f}%")
    
    # Schema changes
    changes = registry.detect_schema_changes()
    if changes:
        print("\n" + "=" * 70)
        print("🔄 SCHEMA CHANGES DETECTED")
        print("=" * 70)
        for c in changes:
            direction = "📈" if c["change"] > 0 else "📉"
            print(f"  {direction} {c['endpoint']}/{c['field']}: {c['from_season']}→{c['to_season']} ({c['from_rate']:.1f}%→{c['to_rate']:.1f}%)")


if __name__ == "__main__":
    main()
