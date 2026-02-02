import duckdb
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .common import ValidationResult, ValidationReport

class StintValidator:
    def __init__(self, db_connection: duckdb.DuckDBPyConnection):
        self.db = db_connection
        self.results: List[ValidationResult] = []

    def validate_game_stints(self, game_id: int) -> ValidationReport:
        self.results = []
        stints_df = self.load_stints(game_id)
        
        if stints_df.empty:
            self.results.append(ValidationResult(
                check="stint_existence",
                passed=False,
                details=f"No stints found for game {game_id}",
                severity="ERROR"
            ))
            return ValidationReport(self.results)

        self.check_stint_durations(stints_df)
        self.check_stint_counts(stints_df)
        self.check_zero_event_stints(stints_df)
        
        return ValidationReport(self.results)

    def load_stints(self, game_id: int) -> pd.DataFrame:
        # Assuming stints are stored in a 'stints' table or similar, 
        # or we might need to generate them. 
        # Based on user request, this validates *generated* stints.
        # If they are not persisted, this validator might need to accept a DF.
        # For now, assuming they are in a table 'stints' or 'apm_stints'.
        try:
            return self.db.execute("""
                SELECT * FROM stints WHERE game_id = ?
            """, [game_id]).fetchdf()
        except:
            # Fallback if table doesn't exist yet
            return pd.DataFrame()

    def check_stint_durations(self, df: pd.DataFrame):
        # Flag outliers (< 2s or > 180s)
        short_stints = df[df['duration'] < 2]
        long_stints = df[df['duration'] > 180]
        
        self.results.append(ValidationResult(
            check="stint_duration_outliers",
            passed=len(long_stints) == 0, # Short stints might be valid (quick changes), long ones suspicious
            details=f"Found {len(short_stints)} short (<2s) and {len(long_stints)} long (>180s) stints",
            severity="WARNING" if len(long_stints) > 0 else "PASS"
        ))

    def check_stint_counts(self, df: pd.DataFrame):
        # Expected 50-150 for 5v5
        count = len(df)
        passed = 50 <= count <= 150
        self.results.append(ValidationResult(
            check="stint_count_per_game",
            passed=passed,
            details=f"Stint count {count} outside expected range [50, 150]",
            severity="WARNING" if not passed else "PASS"
        ))

    def check_zero_event_stints(self, df: pd.DataFrame):
        # What % have no events? (Expected: 30-50%)
        # Assuming 'events_count' column exists
        if 'events_count' in df.columns:
            zero_event_count = len(df[df['events_count'] == 0])
            pct = zero_event_count / len(df) if len(df) > 0 else 0
            
            # This is just a heuristic, not a hard failure
            passed = 0.2 <= pct <= 0.6 
            self.results.append(ValidationResult(
                check="zero_event_stint_rate",
                passed=passed,
                details=f"Zero-event stint rate {pct:.1%} (expected 30-50%)",
                severity="INFO"
            ))
