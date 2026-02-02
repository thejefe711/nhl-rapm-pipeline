import duckdb
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
from .common import ValidationResult, ValidationReport

@dataclass
class IceStateIssue:
    time: int
    expected_home: int
    actual_home: int
    expected_away: int
    actual_away: int
    strength_state: str

@dataclass
class GameShiftReport:
    game_id: int
    coverage: float
    issues: List[IceStateIssue]

class ShiftValidator:
    def __init__(self, db_connection: duckdb.DuckDBPyConnection):
        self.db = db_connection
        self.results: List[ValidationResult] = []

    def validate_game_shifts(self, game_id: int) -> GameShiftReport:
        shifts_df = self.load_shifts(game_id)
        if shifts_df.empty:
            return GameShiftReport(game_id, 0.0, [])

        # Reconstruct ice state at 1-second granularity
        timeline = self.build_ice_timeline(shifts_df, game_id)
        
        issues = []
        valid_seconds = 0
        total_seconds = len(timeline)

        for second, ice_state in timeline.items():
            # Basic 5v5 check logic (simplified for now)
            # In reality, we need to know the expected strength from penalties, but here we check for valid counts
            home_count = len(ice_state['home'])
            away_count = len(ice_state['away'])
            
            # Check for impossible states (e.g. > 6 players, < 3 players)
            if home_count > 6 or home_count < 3 or away_count > 6 or away_count < 3:
                 issues.append(IceStateIssue(
                    time=second,
                    expected_home=5, # Placeholder expectation
                    actual_home=home_count,
                    expected_away=5, # Placeholder expectation
                    actual_away=away_count,
                    strength_state=f"{home_count}v{away_count}"
                ))
            else:
                valid_seconds += 1
        
        coverage = valid_seconds / total_seconds if total_seconds > 0 else 0.0
        return GameShiftReport(game_id, coverage, issues)

    def load_shifts(self, game_id: int) -> pd.DataFrame:
        return self.db.execute("""
            SELECT player_id, team_id, period, start_seconds, end_seconds 
            FROM shifts 
            WHERE game_id = ?
            ORDER BY period, start_seconds
        """, [game_id]).fetchdf()

    def build_ice_timeline(self, shifts_df: pd.DataFrame, game_id: int) -> Dict[int, Dict[str, set]]:
        # Build a second-by-second map of who is on ice
        # Timeline covers 0 to 3600+ seconds (depending on OT)
        max_time = int(shifts_df['end_seconds'].max()) if not shifts_df.empty else 3600
        timeline = {}
        
        # Initialize timeline
        for t in range(max_time + 1):
            timeline[t] = {'home': set(), 'away': set()}

        # Populate timeline
        # Note: This is a slow iterative approach, optimized for clarity/validation not speed
        # For production validation on many games, vectorization would be better
        for _, row in shifts_df.iterrows():
            start = int(row['start_seconds'])
            end = int(row['end_seconds'])
            team_id = row['team_id']
            player_id = row['player_id']
            
            # Determine if home or away (need game info, assuming we can infer or pass it)
            # For now, we'll just store by team_id and map later if needed, 
            # or assume we can distinguish by team_id in the check
            # To make it simple, let's just use team_id as key
            
            for t in range(start, end):
                if t in timeline:
                    # We need to know which team is home/away to categorize correctly
                    # For this snippet, I'll use a placeholder logic or just store by team_id
                    # Let's assume we fetch home/away team IDs first
                    pass
        
        # RE-IMPLEMENTING with team awareness
        home_team_id, away_team_id = self.get_game_teams(game_id)
        
        # Re-init timeline
        timeline = {t: {'home': set(), 'away': set()} for t in range(max_time + 1)}

        for _, row in shifts_df.iterrows():
            start = int(row['start_seconds'])
            end = int(row['end_seconds'])
            team_id = int(row['team_id'])
            player_id = int(row['player_id'])
            
            target_set = 'home' if team_id == home_team_id else 'away' if team_id == away_team_id else None
            
            if target_set:
                for t in range(start, end):
                    if t in timeline:
                        timeline[t][target_set].add(player_id)
                        
        return timeline

    def get_game_teams(self, game_id: int):
        # Helper to get home/away team IDs
        res = self.db.execute("SELECT home_team_id, away_team_id FROM games WHERE game_id = ?", [game_id]).fetchone()
        if res:
            return res[0], res[1]
        return 0, 0
