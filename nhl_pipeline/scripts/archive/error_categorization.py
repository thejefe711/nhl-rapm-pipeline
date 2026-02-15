#!/usr/bin/env python3
"""
Error Categorization System - Fix categories, not individual games.

Automatically classifies pipeline failures so you fix the root cause,
not debug each game individually.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import traceback

@dataclass
class GameProcessingError:
    """Structured error classification."""
    category: str
    message: str
    severity: str  # 'fatal', 'retry', 'warning'
    game_id: str
    details: Dict = None
    traceback: str = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

class ErrorCategorizer:
    """Automatically categorizes pipeline errors."""

    def __init__(self):
        self.error_counts = defaultdict(int)
        self.error_samples = defaultdict(list)  # Sample games per error type
        self.max_samples = 3

    def categorize_error(self, game_id: str, exception: Exception,
                        context: Dict = None) -> GameProcessingError:
        """Categorize an exception into a structured error."""

        error_msg = str(exception)
        error_type = type(exception).__name__

        # Context helps with categorization
        context = context or {}

        # TIME SYSTEM ERRORS
        if any(keyword in error_msg.lower() for keyword in ['time', 'period', 'elapsed', 'remaining']):
            return GameProcessingError(
                category="TIME_CONVERSION",
                message="Time format mismatch between events and shifts",
                severity="fatal",
                game_id=game_id,
                details={
                    "error_msg": error_msg,
                    "likely_cause": "Events use time_remaining, shifts use time_elapsed"
                }
            )

        # DATA TYPE ERRORS
        elif any(keyword in error_msg.lower() for keyword in ['dtype', 'int64', 'string', 'object']):
            return GameProcessingError(
                category="DATA_TYPE_MISMATCH",
                message="Player ID or data type mismatch",
                severity="fatal",
                game_id=game_id,
                details={
                    "error_msg": error_msg,
                    "likely_cause": "Player IDs as string vs int, or column dtypes"
                }
            )

        # MERGE ERRORS (most common)
        elif any(keyword in error_msg.lower() for keyword in ['merge', 'join', 'no rows', 'empty']):
            return GameProcessingError(
                category="MERGE_FAILURE",
                message="Failed to merge events and shifts data",
                severity="fatal",
                game_id=game_id,
                details={
                    "error_msg": error_msg,
                    "events_rows": context.get('events_rows', 'unknown'),
                    "shifts_rows": context.get('shifts_rows', 'unknown'),
                    "likely_cause": "Time conversion or player ID mismatch"
                }
            )

        # MISSING DATA ERRORS
        elif any(keyword in error_msg.lower() for keyword in ['missing', 'nan', 'none', 'keyerror']):
            return GameProcessingError(
                category="MISSING_FIELDS",
                message="Required fields missing from API response",
                severity="fatal",
                game_id=game_id,
                details={
                    "error_msg": error_msg,
                    "likely_cause": "API schema changes or incomplete data"
                }
            )

        # API ERRORS
        elif any(str(code) in error_msg for code in [400, 401, 403, 404, 429, 500, 502, 503]):
            status_code = None
            for code in [400, 401, 403, 404, 429, 500, 502, 503]:
                if str(code) in error_msg:
                    status_code = code
                    break

            severity = "retry" if status_code in [429, 500, 502, 503] else "fatal"

            return GameProcessingError(
                category="API_ERROR",
                message=f"API request failed with status {status_code}",
                severity=severity,
                game_id=game_id,
                details={
                    "status_code": status_code,
                    "error_msg": error_msg,
                    "endpoint": context.get('endpoint', 'unknown')
                }
            )

        # PERFORMANCE ERRORS (O(n²) issues)
        elif any(keyword in error_msg.lower() for keyword in ['timeout', 'memory', 'killed', 'hang']):
            return GameProcessingError(
                category="PERFORMANCE_TIMEOUT",
                message="Processing timed out - likely O(n²) algorithm",
                severity="fatal",
                game_id=game_id,
                details={
                    "error_msg": error_msg,
                    "events_count": context.get('events_count', 'unknown'),
                    "shifts_count": context.get('shifts_count', 'unknown'),
                    "likely_cause": "Inefficient algorithm on large games"
                }
            )

        # FALLBACK - UNKNOWN ERROR
        else:
            return GameProcessingError(
                category="UNKNOWN_ERROR",
                message=f"Unclassified error: {error_type}",
                severity="fatal",
                game_id=game_id,
                details={
                    "error_type": error_type,
                    "error_msg": error_msg,
                    "traceback": traceback.format_exc()
                }
            )

    def record_error(self, error: GameProcessingError):
        """Record an error for reporting."""
        self.error_counts[error.category] += 1

        # Keep sample games for each error type
        if len(self.error_samples[error.category]) < self.max_samples:
            self.error_samples[error.category].append(error.game_id)

class PipelineRunner:
    """Wrapper for your existing pipeline with error categorization."""

    def __init__(self, max_retries: int = 3):
        self.categorizer = ErrorCategorizer()
        self.results = []
        self.max_retries = max_retries

    def process_game(self, game_id: str, season: str,
                    fetch_fn: Callable, parse_fn: Callable, merge_fn: Callable) -> Dict:
        """Process a single game with error handling."""

        result = {
            "game_id": game_id,
            "season": season,
            "status": "pending",
            "attempts": 0,
            "error": None
        }

        for attempt in range(self.max_retries):
            result["attempts"] = attempt + 1

            try:
                # Fetch data
                raw_data = fetch_fn(game_id, season)

                # Parse components
                parsed_data = parse_fn(raw_data)

                # Merge and validate
                final_data = merge_fn(parsed_data)

                # Success!
                result["status"] = "success"
                result["data"] = final_data
                break

            except Exception as e:
                # Categorize the error
                context = {
                    "attempt": attempt + 1,
                    "events_rows": len(parsed_data.get("events", [])) if 'parsed_data' in locals() else 0,
                    "shifts_rows": len(parsed_data.get("shifts", [])) if 'parsed_data' in locals() else 0,
                }

                categorized_error = self.categorizer.categorize_error(game_id, e, context)
                self.categorizer.record_error(categorized_error)

                result["error"] = categorized_error

                # Don't retry fatal errors
                if categorized_error.severity == "fatal":
                    result["status"] = "failed"
                    break

                # For retry-able errors, continue to next attempt
                continue

        self.results.append(result)
        return result

    def print_report(self):
        """Print comprehensive pipeline report."""

        total_games = len(self.results)
        successful = sum(1 for r in self.results if r["status"] == "success")
        failed = total_games - successful

        print("\nNHL PIPELINE RUN REPORT")
        print("=" * 40)
        print(f"\nOVERALL:")
        print(f"   Total games:  {total_games}")
        print(f"   Succeeded:    {successful} ({successful/total_games*100:.1f}%)")
        print(f"   Failed:       {failed} ({failed/total_games*100:.1f}%)")

        if self.categorizer.error_counts:
            print(f"\nERRORS BY CATEGORY:")
            print()

            # Sort by frequency
            sorted_errors = sorted(self.categorizer.error_counts.items(),
                                 key=lambda x: x[1], reverse=True)

            for category, count in sorted_errors:
                sample_games = self.categorizer.error_samples[category]
                print(f"   {category}: {count} games")
                print(f"      Sample games: {sample_games}")
                print()

    def suggest_fixes(self):
        """Suggest specific fixes for each error category."""

        print("SUGGESTED FIXES")
        print("=" * 20)

        fixes = {
            "TIME_CONVERSION": """
# FIX: Convert events time_remaining to seconds_elapsed
def convert_time_formats(events_df, shifts_df):
    def to_elapsed(period, time_remaining):
        # Convert "MM:SS" to seconds remaining, then to elapsed
        if ":" in time_remaining:
            m, s = time_remaining.split(":")
            remaining_sec = int(m) * 60 + int(s)
        else:
            remaining_sec = int(time_remaining)

        period_len = 1200 if period <= 3 else 300  # 20min OT, 5min shootout
        return period_len - remaining_sec

    events_df['sec_in_period'] = events_df.apply(
        lambda r: to_elapsed(r['period'], r['timeInPeriod']), axis=1
    )
    # shifts_df already has sec_in_period as elapsed time
    return events_df, shifts_df
""",

            "DATA_TYPE_MISMATCH": """
# FIX: Standardize player ID types before any operations
def standardize_player_ids(events_df, shifts_df):
    # Convert both to int64, handling NaN
    events_df['player_id'] = pd.to_numeric(events_df['player_id'], errors='coerce').astype('Int64')
    shifts_df['player_id'] = pd.to_numeric(shifts_df['player_id'], errors='coerce').astype('Int64')
    return events_df, shifts_df
""",

            "MERGE_FAILURE": """
# FIX: Ensure merge keys exist and are compatible
def safe_merge(events_df, shifts_df):
    # First standardize time and IDs
    events_df, shifts_df = convert_time_formats(events_df, shifts_df)
    events_df, shifts_df = standardize_player_ids(events_df, shifts_df)

    # Then merge on game_id, player_id, and sec_in_period
    merged = pd.merge(
        events_df, shifts_df,
        on=['game_id', 'player_id', 'sec_in_period'],
        how='left'
    )
    return merged
""",

            "MISSING_FIELDS": """
# FIX: Use safe field access with fallbacks
def safe_get_field(df, field_name, default=np.nan):
    \"\"\"Get field with fallback if missing.\"\"\"
    return df[field_name] if field_name in df.columns else pd.Series([default] * len(df))

# Usage:
df['primary_assist'] = safe_get_field(df, 'player_2_id', np.nan)
df['secondary_assist'] = safe_get_field(df, 'player_3_id', np.nan)
""",

            "API_ERROR": """
# FIX: Add retry logic with exponential backoff
import time
import requests

def fetch_with_retry(url, max_retries=5, base_delay=2):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code in [429, 500, 502, 503]:
                # Retry-able errors
                delay = base_delay * (2 ** attempt)
                print(f"Retry {attempt+1} in {delay}s...")
                time.sleep(delay)
                continue
            else:
                # Non-retry-able
                raise Exception(f"API Error {response.status_code}: {response.text}")
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt)
            print(f"Network error, retry {attempt+1} in {delay}s...")
            time.sleep(delay)

    raise Exception(f"Max retries exceeded for {url}")
""",

            "PERFORMANCE_TIMEOUT": """
# FIX: Replace O(n²) with O(n log n) using interval trees
from intervaltree import IntervalTree

def build_shift_lookup(shifts_df):
    \"\"\"Build efficient interval tree for shift lookups.\"\"\"
    trees = {}
    for game_id, game_shifts in shifts_df.groupby('game_id'):
        tree = IntervalTree()
        for _, shift in game_shifts.iterrows():
            # Add each player's shift interval
            tree[shift['shift_start']:shift['shift_end']] = {
                'player_id': shift['player_id'],
                'team_id': shift['team_id']
            }
        trees[game_id] = tree
    return trees

def get_players_at_time(game_id, sec_in_period, shift_trees):
    \"\"\"O(log n) lookup instead of O(n²).\"\"\"
    tree = shift_trees.get(game_id)
    if not tree:
        return []

    # Find all shifts active at this time
    active_shifts = tree[sec_in_period]
    return [shift.data for shift in active_shifts]
"""
        }

        # Show fixes for errors that occurred
        for category in self.categorizer.error_counts.keys():
            if category in fixes:
                print(f"\n{category}:")
                print(fixes[category])

    def get_failed_games_by_category(self, category: str) -> List[str]:
        """Get list of games that failed with a specific error category."""
        return [
            r["game_id"] for r in self.results
            if r["status"] == "failed" and r["error"].category == category
        ]

# Example usage wrapper for your existing pipeline
def example_pipeline_wrapper():
    """Example of how to wrap your existing pipeline functions."""

    def fetch_game_data(game_id, season):
        """Your existing fetch function."""
        # This would be your current fetch_game.py logic
        # For demo, we'll simulate API calls
        import time
        time.sleep(0.1)  # Simulate API delay

        # Simulate occasional API failures
        if int(game_id[-1]) % 7 == 0:  # Every 7th game fails
            raise Exception("API Error 429: Rate limit exceeded")

        return {"mock": "data"}

    def parse_game_data(raw_data):
        """Your existing parse functions."""
        # This would be your parse_pbp.py + parse_shifts.py logic
        return {
            "events": [{"game_id": "mock"}] * 100,
            "shifts": [{"game_id": "mock"}] * 50
        }

    def merge_and_validate(parsed_data):
        """Your existing merge logic."""
        # This would be your build_on_ice.py logic

        # Simulate different error types
        game_num = int(parsed_data["events"][0]["game_id"][-1]) if parsed_data["events"] else 0

        if game_num % 5 == 1:
            raise ValueError("No rows after merge - likely time format mismatch")
        elif game_num % 5 == 2:
            raise KeyError("Missing field 'player_id' - API schema changed")
        elif game_num % 5 == 3:
            raise Exception("429: Rate limit exceeded")

        return {"merged": "data"}

    # Run pipeline with error categorization
    runner = PipelineRunner()

    # Test with sample games
    test_games = [f"202402{i:04d}" for i in range(1, 21)]  # 20 games

    print("Testing error categorization with 20 games...")
    for game_id in test_games:
        result = runner.process_game(game_id, "20242025",
                                   fetch_game_data, parse_game_data, merge_and_validate)

    # Print results
    runner.print_report()
    runner.suggest_fixes()

if __name__ == "__main__":
    example_pipeline_wrapper()